from __future__ import annotations

"""
Feature/Indicators module for Bot-RL-1
-------------------------------------

- Implements a curated set of technical indicators with minimal 3rd‑party deps
  (pandas, numpy only) to keep the project portable.
- Designed to be deterministic, vectorized, and order‑preserving (left→right).
- Safe to import anywhere (no side effects). Logging is optional; if the main
  app configures logging, this module will use it, otherwise it stays quiet.

Expected input format
---------------------
OHLCV DataFrame with required columns:
    ['open', 'high', 'low', 'close']
Optional volume columns supported (in order of preference):
    'volume' (preferred), 'tick_volume' (MT5), 'tickvolume'
Index can be DatetimeIndex or RangeIndex. Rows must be in chronological order.

Public API (stable)
-------------------
- Indicators: collection of static methods that each append columns in-place
- FeaturePipeline: configurable batch computation with sensible defaults

All indicator outputs are prefixed to avoid collisions, e.g.,
    rsi_{period}, macd_line, macd_signal, macd_hist, bb_mid_{period}, ...

This file is designed to integrate with the existing project layout and
coding style (PEP8, type hints). It does NOT alter row ordering.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping of lowercase column name -> actual column name.

    Useful to support different column naming (e.g. 'tick_volume' from MT5).
    """
    return {c.lower(): c for c in df.columns}


def _validate_ohlc(df: pd.DataFrame, need_volume: bool = False) -> None:
    cols = _detect_columns(df)
    required = {"open", "high", "low", "close"}
    missing = required - set(cols.keys())
    if missing:
        raise ValueError(f"DataFrame is missing required OHLC columns: {missing}")

    if need_volume:
        # Accept 'volume' or MT5's 'tick_volume' (or common variants)
        if "volume" in cols:
            return
        # try MT5 style column names and create an alias 'volume' for convenience
        for candidate in ("tick_volume", "tickvolume", "volume_traded"):
            if candidate in cols:
                # create a normalized 'volume' column if not present
                df["volume"] = df[cols[candidate]]
                logger.debug("Created 'volume' alias from %s", cols[candidate])
                return
        raise ValueError(
            "Volume-based indicator requested, but no volume column found. "
            "Expected one of: 'volume', 'tick_volume'."
        )


def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period, min_periods=period).mean()


def _ema(s: pd.Series, period: int) -> pd.Series:
    # Pandas ewm is numerically stable and efficient
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def _rma(s: pd.Series, period: int) -> pd.Series:
    # Wilder's smoothing (a.k.a. RMA). Pandas ewm with alpha = 1/period
    alpha = 1.0 / float(period)
    return s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

class Indicators:
    """Namespace for indicator functions. Each method appends features in place.

    All methods accept a DataFrame "df" and return the same DataFrame for chaining.
    """

    # ---- Trend -------------------------------------------------------------
    @staticmethod
    def ema(df: pd.DataFrame, period: int = 50, price_col: str = "close") -> pd.DataFrame:
        _validate_ohlc(df)
        col = f"ema_{period}"
        df[col] = _ema(df[price_col].astype(float), period)
        return df

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
    ) -> pd.DataFrame:
        _validate_ohlc(df)
        price = df[price_col].astype(float)
        ema_fast = _ema(price, fast)
        ema_slow = _ema(price, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = _ema(macd_line, signal)
        macd_hist = macd_line - macd_signal
        df["macd_line"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        return df

    @staticmethod
    def supertrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """Compute SuperTrend indicator.

        Appends columns: 'supertrend', 'supertrend_dir' (1 = uptrend, -1 = downtrend).
        Requires ATR; computed internally using Wilder ATR.
        """
        _validate_ohlc(df)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        tr = _true_range(high, low, close)
        atr = _rma(tr, period)

        # Basic bands
        hl2 = (high + low) / 2.0
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        # Final bands (rolling logic)
        final_upper = upperband.copy()
        final_lower = lowerband.copy()

        for i in range(1, len(df)):
            # Upper band
            if (close.iloc[i - 1] <= final_upper.iloc[i - 1]) and (upperband.iloc[i] > final_upper.iloc[i - 1]):
                final_upper.iloc[i] = final_upper.iloc[i - 1]
            # Lower band
            if (close.iloc[i - 1] >= final_lower.iloc[i - 1]) and (lowerband.iloc[i] < final_lower.iloc[i - 1]):
                final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Supertrend line & direction
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = np.nan
                direction.iloc[i] = np.nan
                continue
            prev_st = supertrend.iloc[i - 1]
            prev_dir = direction.iloc[i - 1]

            if np.isnan(prev_st) or np.isnan(prev_dir):
                # Initialize trend based on close vs bands
                if close.iloc[i] > final_upper.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1.0
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1.0
                continue

            if prev_dir > 0:
                # If close crosses below final lower band, trend flips down
                if close.iloc[i] < final_lower.iloc[i]:
                    direction.iloc[i] = -1.0
                    supertrend.iloc[i] = final_upper.iloc[i]
                else:
                    direction.iloc[i] = 1.0
                    supertrend.iloc[i] = max(final_lower.iloc[i], prev_st)
            else:
                # If close crosses above final upper band, trend flips up
                if close.iloc[i] > final_upper.iloc[i]:
                    direction.iloc[i] = 1.0
                    supertrend.iloc[i] = final_lower.iloc[i]
                else:
                    direction.iloc[i] = -1.0
                    supertrend.iloc[i] = min(final_upper.iloc[i], prev_st)

        df["supertrend"] = supertrend
        df["supertrend_dir"] = direction
        return df

    # ---- Momentum ----------------------------------------------------------
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
        _validate_ohlc(df)
        price = df[price_col].astype(float)
        delta = price.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = _rma(gain, period)
        avg_loss = _rma(loss, period)
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df[f"rsi_{period}"] = rsi
        return df

    @staticmethod
    def stoch_rsi(
        df: pd.DataFrame,
        period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        price_col: str = "close",
    ) -> pd.DataFrame:
        _validate_ohlc(df)
        # First, RSI
        tmp = df.copy()
        Indicators.rsi(tmp, period=period, price_col=price_col)
        rsi_col = tmp[f"rsi_{period}"]

        min_rsi = rsi_col.rolling(window=period, min_periods=period).min()
        max_rsi = rsi_col.rolling(window=period, min_periods=period).max()
        stoch = (rsi_col - min_rsi) / (max_rsi - min_rsi)
        k = stoch.rolling(window=k_period, min_periods=k_period).mean() * 100.0
        d = k.rolling(window=d_period, min_periods=d_period).mean()
        df[f"stochrsi_k_{period}_{k_period}"] = k
        df[f"stochrsi_d_{period}_{k_period}_{d_period}"] = d
        return df

    # ---- Volatility --------------------------------------------------------
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        _validate_ohlc(df)
        tr = _true_range(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float))
        df[f"atr_{period}"] = _rma(tr, period)
        return df

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, n_std: float = 2.0, price_col: str = "close") -> pd.DataFrame:
        _validate_ohlc(df)
        price = df[price_col].astype(float)
        mid = _sma(price, period)
        std = price.rolling(window=period, min_periods=period).std(ddof=0)
        upper = mid + n_std * std
        lower = mid - n_std * std
        df[f"bb_mid_{period}"] = mid
        df[f"bb_upper_{period}_{int(n_std)}"] = upper
        df[f"bb_lower_{period}_{int(n_std)}"] = lower
        df[f"bb_width_{period}_{int(n_std)}"] = (upper - lower) / mid
        return df

    # ---- Volume-based ------------------------------------------------------
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.DataFrame:
        _validate_ohlc(df, need_volume=True)
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)
        # OBV increases by volume when price rises, decreases when price falls
        direction = np.sign(close.diff().fillna(0.0))
        df["obv"] = (direction * vol).fillna(0.0).cumsum()
        return df

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        _validate_ohlc(df, need_volume=True)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)
        tp = (high + low + close) / 3.0
        mf = tp * vol
        pos_mf = mf.where(tp > tp.shift(1), 0.0)
        neg_mf = mf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
        neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()
        mfr = pos_sum / neg_sum.replace(0.0, np.nan)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        df[f"mfi_{period}"] = mfi
        return df

    # ---- Others ------------------------------------------------------------
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        _validate_ohlc(df)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        tr = _true_range(high, low, close)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        atr = _rma(tr, period)
        plus_di = 100.0 * _rma(plus_dm, period) / atr
        minus_di = 100.0 * _rma(minus_dm, period) / atr

        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
        adx = _rma(dx, period)

        df[f"plus_di_{period}"] = plus_di
        df[f"minus_di_{period}"] = minus_di
        df[f"adx_{period}"] = adx
        return df

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        _validate_ohlc(df)
        tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        sma_tp = _sma(tp, period)
        mad = (tp - sma_tp).abs().rolling(window=period, min_periods=period).mean()
        cci = (tp - sma_tp) / (0.015 * mad)
        df[f"cci_{period}"] = cci
        return df


# ---------------------------------------------------------------------------
# Pipeline for batch computation
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSpec:
    name: str
    params: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class FeaturePipeline:
    """Build features based on a list of indicator specs.

    Example:
        specs = [
            IndicatorSpec("ema", {"period": 50}),
            IndicatorSpec("macd", {"fast": 12, "slow": 26, "signal": 9}),
            IndicatorSpec("rsi", {"period": 14}),
            IndicatorSpec("stoch_rsi", {"period": 14, "k_period": 3, "d_period": 3}),
            IndicatorSpec("atr", {"period": 14}),
            IndicatorSpec("bollinger_bands", {"period": 20, "n_std": 2}),
            IndicatorSpec("obv"),
            IndicatorSpec("mfi", {"period": 14}),
            IndicatorSpec("adx", {"period": 14}),
            IndicatorSpec("cci", {"period": 20}),
        ]
        fp = FeaturePipeline(specs)
        df = fp.apply(df)
    """

    specs: List[IndicatorSpec]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _validate_ohlc(df)
        out = df.copy()
        for spec in self.specs:
            fn = getattr(Indicators, spec.name, None)
            if fn is None or not callable(fn):
                raise AttributeError(f"Unknown indicator '{spec.name}'.")
            try:
                out = fn(out, **spec.params)
                logger.debug("Applied indicator %s with params %s", spec.name, spec.params)
            except TypeError as e:
                raise TypeError(f"Invalid params for indicator '{spec.name}': {spec.params}") from e
        return out


# ---------------------------------------------------------------------------
# Sensible default bundle matching user's requested list
# ---------------------------------------------------------------------------

def default_specs(include_supertrend: bool = False, include_volume: bool = True) -> List[IndicatorSpec]:
    specs: List[IndicatorSpec] = [
        # 1) Trend
        IndicatorSpec("ema", {"period": 50}),
        IndicatorSpec("ema", {"period": 200}),
        IndicatorSpec("macd", {"fast": 12, "slow": 26, "signal": 9}),
    ]
    if include_supertrend:
        specs.append(IndicatorSpec("supertrend", {"period": 10, "multiplier": 3.0}))

    # 2) Momentum
    specs.extend([
        IndicatorSpec("rsi", {"period": 14}),
        IndicatorSpec("stoch_rsi", {"period": 14, "k_period": 3, "d_period": 3}),
    ])

    # 3) Volatility
    specs.extend([
        IndicatorSpec("atr", {"period": 14}),
        IndicatorSpec("bollinger_bands", {"period": 20, "n_std": 2.0}),
    ])

    # 4) Volume-based (if volume is present)
    if include_volume:
        specs.extend([
            IndicatorSpec("obv"),
            IndicatorSpec("mfi", {"period": 14}),
        ])

    # 5) Others
    specs.extend([
        IndicatorSpec("adx", {"period": 14}),
        IndicatorSpec("cci", {"period": 20}),
    ])

    return specs


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    specs: Optional[Iterable[IndicatorSpec]] = None,
    include_supertrend: bool = False,
    include_volume: Optional[bool] = None,
) -> pd.DataFrame:
    """High-level helper to build the standard indicator set.

    - If `specs` is provided, it takes precedence.
    - If `specs` is None, uses `default_specs()` with toggles.
    - `include_volume=None` auto-detects based on 'volume' or 'tick_volume' column presence.
    """
    if specs is None:
        if include_volume is None:
            cols = _detect_columns(df)
            include_volume = any(k in cols for k in ("volume", "tick_volume", "tickvolume"))
        specs = default_specs(include_supertrend=include_supertrend, include_volume=include_volume)
    pipeline = FeaturePipeline(list(specs))
    return pipeline.apply(df)


__all__ = [
    "Indicators",
    "IndicatorSpec",
    "FeaturePipeline",
    "default_specs",
    "build_features",
]
