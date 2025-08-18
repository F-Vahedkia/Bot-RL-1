# f02_data/data_handler.py
"""
Enhanced DataHandler for Bot-RL-1

Outputs:
    - fetch_for_symbol(symbol) -> Dict[timeframe -> pandas.DataFrame]
    - fetch_all() -> Dict[symbol -> Dict[timeframe -> DataFrame]]

Behavior:
    - Prefer MT5DataLoader if available, otherwise use MT5Connector.
    - Support multiple timeframes and 'n_candles' per timeframe.
    - If start_date/end_date provided, prefer range-fetching methods.
    - Standardize OHLC columns.
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# try to import loader / connector (may raise ImportError if not present)
try:
    from f02_data.mt5_data_loader import MT5DataLoader
except Exception:
    MT5DataLoader = None

try:
    from f02_data.mt5_connector import MT5Connector
except Exception:
    MT5Connector = None


def _ensure_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe has 'open','high','low','close' columns.
    If not present, attempt some common renames. If still missing, leave as-is and log.
    """
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(df)

    cols = set(df.columns.str.lower())
    mapping = {}
    # common alternatives
    if 'open' not in cols and 'o' in cols:
        mapping['o'] = 'open'
    if 'high' not in cols and 'h' in cols:
        mapping['h'] = 'high'
    if 'low' not in cols and 'l' in cols:
        mapping['l'] = 'low'
    if 'close' not in cols and 'c' in cols:
        mapping['c'] = 'close'

    if mapping:
        try:
            df = df.rename(columns=mapping)
        except Exception:
            logger.debug("Could not rename columns with mapping %s", mapping)

    # final check
    expected = {'open', 'high', 'low', 'close'}
    if not expected.issubset(set(df.columns.str.lower())):
        logger.debug("DataFrame missing some OHLC columns. cols=%s", df.columns.tolist())
    return df


class DataHandler:
    def __init__(self, config: Dict):
        self.config = config or {}
        self.symbols: List[str] = self.config.get("symbols", [])
        if isinstance(self.symbols, str):
            self.symbols = [self.symbols]

        # timeframes: accept string or list
        tfs = self.config.get("timeframes", self.config.get("timeframe", ["M1"]))
        if isinstance(tfs, str):
            tfs = [tfs]
        self.timeframes: List[str] = tfs

        # number of candles per timeframe to return (default 15)
        self.n_candles: int = int(self.config.get("n_candles", 15))

        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        if self.config.get("start_date"):
            self.start_date = datetime.strptime(self.config.get("start_date"), "%Y-%m-%d")
        if self.config.get("end_date"):
            self.end_date = datetime.strptime(self.config.get("end_date"), "%Y-%m-%d")

        # choose loader/connector with graceful fallback
        self.loader = None
        self.connector = None

        if MT5DataLoader is not None:
            try:
                self.loader = MT5DataLoader(self.config)
                logger.info("DataHandler: using MT5DataLoader")
            except Exception as ex:
                logger.debug("MT5DataLoader init failed: %s", ex)
                self.loader = None

        if self.loader is None and MT5Connector is not None:
            try:
                self.connector = MT5Connector()
                if hasattr(self.connector, "initialize"):
                    ok = self.connector.initialize()
                    if not ok:
                        raise RuntimeError("MT5Connector.initialize() returned False")
                logger.info("DataHandler: using MT5Connector")
            except Exception as ex:
                logger.exception("Failed to initialize MT5Connector: %s", ex)
                self.connector = None

        if self.loader is None and self.connector is None:
            logger.warning("No data backend available (MT5DataLoader nor MT5Connector). DataHandler cannot fetch remote data.")

        # simple in-memory cache per symbol for the current session/episode
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    def _fetch_with_loader(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # try several commonly-named methods on loader
        for m in ("get_last_candles", "fetch_last_candles", "load_last_candles", "get_candles", "fetch_candles"):
            if hasattr(self.loader, m):
                fn = getattr(self.loader, m)
                try:
                    # many loaders accept (symbol, timeframe, n) but signatures vary
                    return fn(symbol=symbol, timeframe=timeframe, n=self.n_candles)
                except TypeError:
                    # try alternative args
                    try:
                        return fn(symbol, timeframe, self.n_candles)
                    except Exception:
                        continue
        # fallback to a range method if loader supports it and start/end provided
        if hasattr(self.loader, "fetch_and_save_range") and self.start_date and self.end_date:
            try:
                # some loaders implement get_candles_range
                if hasattr(self.loader, "get_candles_range"):
                    return self.loader.get_candles_range(symbol=symbol, timeframe=timeframe, date_from=self.start_date, date_to=self.end_date)
            except Exception:
                pass
        raise RuntimeError("MT5DataLoader has no known fetch method")

    def _fetch_with_connector(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # if start/end provided, try range method first
        if self.start_date and self.end_date and hasattr(self.connector, "get_candles_range"):
            return self.connector.get_candles_range(symbol=symbol, timeframe=timeframe, date_from=self.start_date, date_to=self.end_date)

        # try get_candles / get_last_candles style functions
        for m in ("get_candles", "get_last_candles", "fetch_candles"):
            if hasattr(self.connector, m):
                fn = getattr(self.connector, m)
                try:
                    # try common signature names
                    return fn(symbol=symbol, timeframe=timeframe, num_candles=self.n_candles)
                except TypeError:
                    try:
                        return fn(symbol, timeframe, self.n_candles)
                    except Exception:
                        continue
        raise RuntimeError("MT5Connector has no known candle-fetching method")

    def fetch_for_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Return dict: { timeframe: DataFrame } for this symbol.
        Uses cache if present.
        """
        if symbol in self._cache:
            return self._cache[symbol]

        out: Dict[str, pd.DataFrame] = {}
        for tf in self.timeframes:
            try:
                if self.loader is not None:
                    df = self._fetch_with_loader(symbol, tf)
                elif self.connector is not None:
                    df = self._fetch_with_connector(symbol, tf)
                else:
                    df = pd.DataFrame()

                if df is None:
                    df = pd.DataFrame()

                df = _ensure_ohlc_cols(df)

                # Ensure datetime index if possible
                if 'time' in (c.lower() for c in df.columns):
                    try:
                        # normalize column name to 'time' lowercase detection above
                        time_col = [c for c in df.columns if c.lower() == 'time'][0]
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                        df = df.set_index(time_col)
                    except Exception:
                        pass

                # keep only last n_candles rows and pad top with NaNs if fewer rows
                if df.shape[0] >= self.n_candles:
                    df = df.tail(self.n_candles)
                else:
                    # create empty frame with expected columns and pad
                    cols = df.columns if not df.empty else ['open', 'high', 'low', 'close']
                    pad_rows = self.n_candles - max(0, df.shape[0])
                    pad_df = pd.DataFrame(np.nan, index=range(pad_rows), columns=cols)
                    df = pd.concat([pad_df, df], ignore_index=True)
                out[tf] = df.copy()
            except Exception as ex:
                logger.exception("Failed to fetch data for %s %s: %s", symbol, tf, ex)
                out[tf] = pd.DataFrame()

        # cache for reuse in the same run/episode
        self._cache[symbol] = out
        return out

    def fetch_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        for s in self.symbols:
            result[s] = self.fetch_for_symbol(s)
        return result

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        # close loader/connector gracefully
        try:
            if self.loader is not None and hasattr(self.loader, "close"):
                self.loader.close()
        except Exception:
            pass
        try:
            if self.connector is not None and hasattr(self.connector, "shutdown"):
                self.connector.shutdown()
        except Exception:
            pass
