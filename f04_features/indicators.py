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
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

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

    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 14, smoothK: int = 3, smoothD: int = 3, price_col: str = "close") -> pd.DataFrame:
        """
        Compute the Stochastic Oscillator (%K and %D).
        Appends columns: '%k', '%d'.
        """
        _validate_ohlc(df)
        low_min = df["low"].rolling(window=period, min_periods=period).min()
        high_max = df["high"].rolling(window=period, min_periods=period).max()
        df["%k"] = 100 * (df[price_col] - low_min) / (high_max - low_min)
        df["%d"] = df["%k"].rolling(window=smoothK, min_periods=smoothK).mean()
        df["%d"] = df["%d"].rolling(window=smoothD, min_periods=smoothD).mean()
        return df

    @staticmethod
    def parabolic_sar(df: pd.DataFrame, acceleration: float = 0.02, max_acceleration: float = 0.2) -> pd.DataFrame:
        """
        Compute the Parabolic SAR (Stop and Reverse).
        Appends column: 'psar'.
        """
        _validate_ohlc(df)
        df["psar"] = np.nan
        df["psar"] = df["close"]
        df["psar"] = df["psar"].shift(1)
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

    # ---- Divergence --------------------------------------------------------
    @staticmethod
    def detect_divergences_extrema(df: pd.DataFrame, indicator_columns: list, price_col: str = "close", order: int = 5, min_diff_ratio: float = 0.001):
        """
        Detect bullish and bearish divergences using local extrema with price & indicator validation.

        Parameters:
        - df: DataFrame containing price and indicator data.
        - indicator_columns: List of indicator column names to check for divergence.
        - price_col: Column name of the price (default 'close').
        - order: Number of points to use for local extrema (default 5).
        - min_diff_ratio: Minimum relative difference to validate divergence (default 0.1%).

        Returns:
        - df: DataFrame with new boolean columns for divergences:
            - 'RSI_bullish_divergence', 'RSI_bearish_divergence', etc.
        """
        for ind in indicator_columns:
            bull_col = f"{ind}_bullish_divergence"
            bear_col = f"{ind}_bearish_divergence"

            df[bull_col] = False
            df[bear_col] = False

            # شناسایی نقاط کف و سقف برای قیمت و اندیکاتور
            price_min_idx = argrelextrema(df[price_col].values, np.less, order=order)[0]
            price_max_idx = argrelextrema(df[price_col].values, np.greater, order=order)[0]

            ind_min_idx = argrelextrema(df[ind].values, np.less, order=order)[0]
            ind_max_idx = argrelextrema(df[ind].values, np.greater, order=order)[0]

            # واگرایی صعودی: کف قیمت پایین‌تر، کف اندیکاتور بالاتر
            for pi in price_min_idx:
                closest_ind_min = ind_min_idx[ind_min_idx <= pi]
                if len(closest_ind_min) == 0:
                    continue
                ii = closest_ind_min[-1]

                # بررسی تغییر واقعی قیمت و اندیکاتور
                price_diff_ratio = (df[price_col].iloc[ii] - df[price_col].iloc[pi]) / df[price_col].iloc[ii]
                ind_diff_ratio = (df[ind].iloc[pi] - df[ind].iloc[ii]) / df[ind].iloc[ii]

                if price_diff_ratio > 0 and ind_diff_ratio > min_diff_ratio:
                    df.at[pi, bull_col] = True

            # واگرایی نزولی: سقف قیمت بالاتر، سقف اندیکاتور پایین‌تر
            for pi in price_max_idx:
                closest_ind_max = ind_max_idx[ind_max_idx <= pi]
                if len(closest_ind_max) == 0:
                    continue
                ii = closest_ind_max[-1]

                price_diff_ratio = (df[price_col].iloc[pi] - df[price_col].iloc[ii]) / df[price_col].iloc[ii]
                ind_diff_ratio = (df[ind].iloc[ii] - df[ind].iloc[pi]) / df[ind].iloc[ii]

                if price_diff_ratio > 0 and ind_diff_ratio > min_diff_ratio:
                    df.at[pi, bear_col] = True

        return df

    # ---- Volume-based (حجم/فلو) --------------------------------------------
    @staticmethod
    def adl(df: pd.DataFrame) -> pd.DataFrame:
        """Accumulation / Distribution Line (Chaikin ADL)."""
        _validate_ohlc(df, need_volume=True)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        # Money Flow Multiplier = (2*close - high - low) / (high - low)
        denom = (high - low).replace(0.0, np.nan)
        mfm = ((2.0 * close - high - low) / denom).fillna(0.0)
        mfv = mfm * vol
        df["adl"] = mfv.cumsum().fillna(method="ffill").fillna(0.0)
        return df

    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Chaikin Money Flow (CMF)."""
        _validate_ohlc(df, need_volume=True)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        vol = df["volume"].astype(float)

        denom = (high - low).replace(0.0, np.nan)
        mfm = ((2.0 * close - high - low) / denom).fillna(0.0)
        mfv = mfm * vol

        sum_mfv = mfv.rolling(window=period, min_periods=period).sum()
        sum_vol = vol.rolling(window=period, min_periods=period).sum().replace(0.0, np.nan)
        cmf = sum_mfv / sum_vol
        df[f"cmf_{period}"] = cmf.fillna(0.0)
        return df

    @staticmethod
    def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.DataFrame:
        """Chaikin Oscillator = EMA_fast(ADL) - EMA_slow(ADL)."""
        # Ensure ADL exists
        if "adl" not in df.columns:
            df = Indicators.adl(df)
        adl_series = df["adl"].astype(float)
        fast_ema = _ema(adl_series, fast)
        slow_ema = _ema(adl_series, slow)
        df[f"chaikin_osc_{fast}_{slow}"] = (fast_ema - slow_ema)
        return df

    # ---- Patterns (الگوهای شمعی) --------------------------------------------
    @staticmethod
    def candlestick_patterns(df: pd.DataFrame, patterns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect simple candlestick patterns and append boolean columns:
            - doji, hammer, shooting_star, bull_engulf, bear_engulf
        The detection rules are conservative/simple and parameterizable later.
        """
        _validate_ohlc(df)
        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        body = (close - open_).abs()
        rng = (high - low).replace(0.0, np.nan)
        upper_shadow = (high - np.maximum(close, open_))
        lower_shadow = (np.minimum(close, open_) - low)

        # Doji: very small body relative to range
        df["pattern_doji"] = (body / rng <= 0.1).fillna(False)

        # Hammer (bullish): long lower shadow, small body near top
        df["pattern_hammer"] = (
            (lower_shadow >= 2 * body) & (upper_shadow <= 0.5 * body)
        ).fillna(False)

        # Shooting star (bearish): long upper shadow, small body near bottom
        df["pattern_shooting_star"] = (
            (upper_shadow >= 2 * body) & (lower_shadow <= 0.5 * body)
        ).fillna(False)

        # Engulfing patterns (compare with previous candle)
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        prev_body = (prev_close - prev_open).abs()

        # Bullish engulfing: previous bearish, current bullish, current body engulfs prev body
        df["pattern_bull_engulf"] = (
            (prev_close < prev_open)
            & (close > open_)
            & (close > prev_open)
            & (open_ < prev_close)
        ).fillna(False)

        # Bearish engulfing: previous bullish, current bearish, current body engulfs prev body
        df["pattern_bear_engulf"] = (
            (prev_close > prev_open)
            & (close < open_)
            & (open_ > prev_close)
            & (close < prev_open)
        ).fillna(False)

        return df


class MovingAverageCross_old:
    """
    A flexible Moving Average Crossover detector.
    Supports both SMA and EMA with arbitrary period pairs.
    Generates binary signals: 
        +1 for Bullish crossover (Golden Cross),
        -1 for Bearish crossover (Death Cross),
        0 for no signal.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50, ma_type: str = "EMA"):
        """
        Initialize the crossover detector.
        
        Args:
            fast_period (int): Period for the fast moving average.
            slow_period (int): Period for the slow moving average.
            ma_type (str): Type of MA to use ("SMA" or "EMA").
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be smaller than slow period.")
        if ma_type not in ["SMA", "EMA"]:
            raise ValueError("ma_type must be either 'SMA' or 'EMA'.")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.upper()

    def _calc_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Helper: calculate SMA or EMA."""
        if self.ma_type == "SMA":
            return series.rolling(window=period, min_periods=period).mean()
        elif self.ma_type == "EMA":
            return series.ewm(span=period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Generate moving average crossover signals.

        Args:
            df (pd.DataFrame): Input OHLC dataframe with a 'close' column (or given price_col).
            price_col (str): Column name for price series.

        Returns:
            pd.DataFrame: Original dataframe with added columns:
                - fast_MA
                - slow_MA
                - MA_Cross_Signal  (+1, -1, or 0)
        """
        df = df.copy()

        # Calculate fast and slow MAs
        df["fast_MA"] = self._calc_ma(df[price_col], self.fast_period)
        df["slow_MA"] = self._calc_ma(df[price_col], self.slow_period)

        # Detect crossovers
        df["MA_Cross_Signal"] = 0
        df.loc[(df["fast_MA"] > df["slow_MA"]) & (df["fast_MA"].shift(1) <= df["slow_MA"].shift(1)), "MA_Cross_Signal"] = +1
        df.loc[(df["fast_MA"] < df["slow_MA"]) & (df["fast_MA"].shift(1) >= df["slow_MA"].shift(1)), "MA_Cross_Signal"] = -1

        return df


class MovingAverageCross:
    """
    Robust & extensible MA crossover detector.

    Usage:
        mac = MovingAverageCross(fast=12, slow=26, ma_type='ema',
                                 signal_col='ma_cross_signal',
                                 min_bars_between_signals=3,
                                 confirm_bars=1)
        df = mac.add_to_df(df)   # returns df with fast/slow MA cols and signal column
        signal = mac.last_signal(df)  # +1 buy, -1 sell, 0 none

    Parameters:
    - fast, slow: int or List[int] (if lists, multiple pairs will be computed)
    - ma_type: 'ema' or 'sma'
    - signal_col: name of final signal column
    - min_bars_between_signals: suppress repeat signals within this many bars
    - confirm_bars: require signal direction to hold for `confirm_bars` (>=1) bars to confirm
    - magnitude: if True returns magnitude of crossover (diff normalized) instead of only -1/0/+1
    """

    def __init__(
        self,
        fast: Union[int, List[int]] = 12,
        slow: Union[int, List[int]] = 26,
        ma_type: str = "ema",
        signal_col: str = "ma_cross_signal",
        min_bars_between_signals: int = 3,
        confirm_bars: int = 1,
        magnitude: bool = False,
        price_col: str = "close",
    ):
        self.fast = [fast] if isinstance(fast, int) else list(fast)
        self.slow = [slow] if isinstance(slow, int) else list(slow)
        if len(self.fast) != len(self.slow):
            # allow broadcasting: if one of lists length 1, broadcast it
            if len(self.fast) == 1:
                self.fast = self.fast * len(self.slow)
            elif len(self.slow) == 1:
                self.slow = self.slow * len(self.fast)
            else:
                raise ValueError("fast and slow must have same length or one must be scalar.")
        self.ma_type = ma_type.lower()
        if self.ma_type not in ("ema", "sma"):
            raise ValueError("ma_type must be 'ema' or 'sma'")
        self.signal_col = signal_col
        self.min_bars_between_signals = max(0, int(min_bars_between_signals))
        self.confirm_bars = max(1, int(confirm_bars))
        self.magnitude = bool(magnitude)
        self.price_col = price_col

    # ---------- helpers ----------
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    def _compute_ma(self, df: pd.DataFrame, period: int, prefix: str) -> pd.Series:
        s = df[self.price_col].astype(float)
        if self.ma_type == "ema":
            return self._ema(s, period).rename(f"{prefix}_{period}")
        else:
            return self._sma(s, period).rename(f"{prefix}_{period}")

    # ---------- core API ----------
    def add_to_df(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Compute MA columns and crossover signal column and append to df.

        Adds columns:
          - ma_fast_{p}, ma_slow_{q}  (for each pair)
          - ma_diff_{i}  (fast - slow)
          - {signal_col}  (final aggregated signal)

        Signal semantics:
          +1 : bullish crossover (fast crosses above slow)
          -1 : bearish crossover
           0 : no signal

        If magnitude=True, signal column contains signed normalized magnitude instead of +/-1.
        """
        working = df if inplace else df.copy()

        # compute each MA pair and diffs
        diffs = []
        for idx, (f, s) in enumerate(zip(self.fast, self.slow)):
            fast_name = f"ma_fast_{f}"
            slow_name = f"ma_slow_{s}"
            working[fast_name] = self._compute_ma(working, f, prefix="ma_fast")
            working[slow_name] = self._compute_ma(working, s, prefix="ma_slow")
            diff_col = f"ma_diff_{f}_{s}"
            working[diff_col] = working[fast_name] - working[slow_name]
            diffs.append(diff_col)

        # aggregate diffs (sum) to form a combined diff measure
        if len(diffs) == 1:
            working["_ma_agg_diff"] = working[diffs[0]]
        else:
            working["_ma_agg_diff"] = working[diffs].sum(axis=1)

        # raw sign of diff
        raw_sign = np.sign(working["_ma_agg_diff"]).astype(int)

        # detect cross points: compare sign with previous bar
        sign_shift = raw_sign.shift(1).fillna(0).astype(int)
        cross = raw_sign - sign_shift  # +2 means -1 -> +1, +1 means 0->+1, -1 etc.

        # normalize to -1/0/+1 only when crossing happens from opposite sign
        # true cross occurs where sign differs and change is non-zero
        signal_series = pd.Series(0, index=working.index, dtype=float)

        for i in range(len(working)):
            # skip first bars where no previous exists
            if i == 0:
                continue
            prev = sign_shift.iloc[i]
            curr = raw_sign.iloc[i]
            if prev == 0 and curr == 0:
                continue
            # bullish cross: prev <= 0 and curr > 0
            if prev <= 0 and curr > 0:
                signal_series.iat[i] = 1.0
            # bearish cross: prev >= 0 and curr < 0
            elif prev >= 0 and curr < 0:
                signal_series.iat[i] = -1.0

        # optional confirmation: require the direction to hold for confirm_bars
        if self.confirm_bars and self.confirm_bars > 1:
            confirmed = signal_series.copy()
            for idx in range(len(signal_series)):
                sig = signal_series.iat[idx]
                if sig == 0:
                    continue
                # check next confirm_bars-1 bars maintain same sign of _ma_agg_diff
                end = min(len(signal_series), idx + self.confirm_bars)
                window = working["_ma_agg_diff"].iloc[idx:end]
                if sig > 0:
                    if not (window > 0).all():
                        confirmed.iat[idx] = 0.0
                else:
                    if not (window < 0).all():
                        confirmed.iat[idx] = 0.0
            signal_series = confirmed

        # apply min_bars_between_signals suppression
        if self.min_bars_between_signals > 0:
            last_sig_idx = -9999
            suppressed = signal_series.copy()
            for idx in range(len(signal_series)):
                if signal_series.iat[idx] != 0:
                    if idx - last_sig_idx <= self.min_bars_between_signals:
                        suppressed.iat[idx] = 0.0
                    else:
                        last_sig_idx = idx
            signal_series = suppressed

        # if magnitude requested, scale by normalized diff
        if self.magnitude:
            # normalize by rolling std to prevent huge values
            denom = working["_ma_agg_diff"].rolling(window=max(5, max(self.fast + self.slow))).std().replace(0, np.nan)
            norm = (working["_ma_agg_diff"] / denom).fillna(0.0)
            signal_series = signal_series * norm

        # write columns to df
        working[self.signal_col] = signal_series.astype(float)
        # cleanup internal agg column if desired (keep for debugging)
        # working.drop(columns=["_ma_agg_diff"], inplace=True)  # optional

        return working

    def last_signal(self, df: pd.DataFrame) -> float:
        """Return last signal value (float)."""
        if self.signal_col not in df.columns:
            df = self.add_to_df(df, inplace=False)
        val = float(df[self.signal_col].iloc[-1])
        if np.isnan(val):
            return 0.0
        return val

    def cross_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame of cross points (index, signal, fast, slow, diff).
        Useful for analysis/backtesting.
        """
        if self.signal_col not in df.columns:
            df = self.add_to_df(df, inplace=False)
        points = df[df[self.signal_col] != 0][[self.signal_col]]
        # augment with fast/slow pair values and diff
        for f, s in zip(self.fast, self.slow):
            points[f"ma_fast_{f}"] = df[f"ma_fast_{f}"]
            points[f"ma_slow_{s}"] = df[f"ma_slow_{s}"]
        points["_ma_agg_diff"] = df["_ma_agg_diff"]
        return points

    # optional convenience: produce boolean buy/sell columns
    def as_binary_cols(self, df: pd.DataFrame, buy_col: str = "ma_buy", sell_col: str = "ma_sell"):
        if self.signal_col not in df.columns:
            df = self.add_to_df(df, inplace=False)
        df[buy_col] = (df[self.signal_col] > 0).astype(int)
        df[sell_col] = (df[self.signal_col] < 0).astype(int)
        return df

    def to_spec(self) -> Dict[str, Any]:
        """Return a small dict describing the current config (for logging or saving)."""
        return {
            "fast": self.fast,
            "slow": self.slow,
            "ma_type": self.ma_type,
            "signal_col": self.signal_col,
            "min_bars_between_signals": self.min_bars_between_signals,
            "confirm_bars": self.confirm_bars,
            "magnitude": self.magnitude,
            "price_col": self.price_col,
        }

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

# ------------------------------------------------------------------
# Example of usage inside indicators.py (later in FeaturePipeline)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Example dummy usage
    data = pd.DataFrame({
        "close": np.random.random(100) * 100 + 1800  # mock price series
    })
    
    ma_cross = MovingAverageCross(fast_period=20, slow_period=50, ma_type="EMA")
    result = ma_cross.generate_signals(data)

    print(result.tail(10)[["close", "fast_MA", "slow_MA", "MA_Cross_Signal"]])

__all__ = [
    "Indicators",
    "IndicatorSpec",
    "FeaturePipeline",
    "default_specs",
    "build_features",
]
