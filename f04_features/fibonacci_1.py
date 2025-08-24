# f04_features/fibonacci.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

class Fibonacci:
    """
    Utility class for generating Fibonacci retracement, extension,
    and projection levels.

    Notes:
        - All outputs are returned as Pandas DataFrames for consistency.
        - Designed to be easily expandable with new Fibonacci-based tools.
    """

    # Common Fibonacci ratios
    RETRACEMENT_RATIOS: List[float] = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSION_RATIOS: List[float] = [1.272, 1.618, 2.0, 2.618]
    PROJECTION_RATIOS: List[float] = [0.618, 1.0, 1.618]

    @staticmethod
    def calculate_retracement(high: float, low: float) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels between a swing high and low.

        Args:
            high (float): Swing high price.
            low (float): Swing low price.

        Returns:
            pd.DataFrame: retracement levels with ratio and price.
        """
        diff = high - low
        levels = [(ratio, high - ratio * diff) for ratio in Fibonacci.RETRACEMENT_RATIOS]

        return pd.DataFrame(
            levels, columns=["ratio", "price"]
        ).assign(method="retracement", description="Swing High-Low")

    @staticmethod
    def calculate_extension(high: float, low: float) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels (beyond retracement).

        Args:
            high (float): Swing high price.
            low (float): Swing low price.

        Returns:
            pd.DataFrame: extension levels with ratio and price.
        """
        diff = high - low
        levels = [(ratio, high + ratio * diff) for ratio in Fibonacci.EXTENSION_RATIOS]

        return pd.DataFrame(
            levels, columns=["ratio", "price"]
        ).assign(method="extension", description="Swing High-Low")

    @staticmethod
    def calculate_projection(A: float, B: float, C: float) -> pd.DataFrame:
        """
        Calculate Fibonacci projection levels (ABC pattern).

        Args:
            A (float): Point A price.
            B (float): Point B price.
            C (float): Point C price.

        Returns:
            pd.DataFrame: projection levels with ratio and price.
        """
        diff = B - A
        levels = [(ratio, C + ratio * diff) for ratio in Fibonacci.PROJECTION_RATIOS]

        return pd.DataFrame(
            levels, columns=["ratio", "price"]
        ).assign(method="projection", description="ABC Pattern")

    @staticmethod
    def multi_timeframe_confluence(data: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Calculate retracement levels for multiple timeframes
        and return combined results.

        Args:
            data (Dict[str, Tuple[float, float]]):
                Dictionary with timeframe as key and (high, low) as value.

        Example:
            {
                "1h": (1800, 1750),
                "4h": (1850, 1700)
            }

        Returns:
            pd.DataFrame: all levels with timeframe column included.
        """
        all_levels = []
        for timeframe, (high, low) in data.items():
            df = Fibonacci.calculate_retracement(high, low)
            df = df.assign(timeframe=timeframe)
            all_levels.append(df)

        return pd.concat(all_levels, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    retr = Fibonacci.calculate_retracement(2000, 1900)
    print("\nRetracement Levels:\n", retr)

    ext = Fibonacci.calculate_extension(2000, 1900)
    print("\nExtension Levels:\n", ext)

    proj = Fibonacci.calculate_projection(1900, 2000, 1950)
    print("\nProjection Levels:\n", proj)

    mtf = Fibonacci.multi_timeframe_confluence({"1h": (2000, 1900), "4h": (2050, 1850)})
    print("\nMulti-Timeframe Retracement:\n", mtf)








# f04_features/fibonacci.py
"""
Fibonacci Analysis Toolkit for Multi-Timeframe Trading
------------------------------------------------------
این ماژول شامل ابزارهای کامل فیبوناچی برای تحلیل چند-تایم‌فریم است:
- Retracement & Projection
- Cluster Detection
- Auto Swing Detection
- Golden Zone Features
- Risk/Reward Estimation
- News-aware Filtering
"""

class FibonacciAnalyzer:
    def __init__(self, price_data: pd.DataFrame, news_data: Optional[pd.DataFrame] = None):
        """
        :param price_data: کندل‌های OHLC
        :param news_data: دیتای اخبار (اختیاری: ستون‌های time, impact)
        """
        self.df = price_data
        self.news = news_data

    # ------------------------------
    # 1. Auto Swing Detection
    # ------------------------------
    def detect_swings(self, window: int = 3) -> List[Tuple[int, float, str]]:
        """
        تشخیص Swing High/Low ساده با روش Fractal
        :param window: تعداد کندل قبل و بعد برای تایید
        :return: لیست (index, price, نوع)
        """
        swings = []
        for i in range(window, len(self.df) - window):
            high = self.df['High'].iloc[i]
            low = self.df['Low'].iloc[i]
            if high == max(self.df['High'].iloc[i - window:i + window + 1]):
                swings.append((i, high, 'high'))
            if low == min(self.df['Low'].iloc[i - window:i + window + 1]):
                swings.append((i, low, 'low'))
        return swings

    # ------------------------------
    # 2. Fibonacci Retracement
    # ------------------------------
    def fib_levels(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        محاسبه سطوح فیبوناچی بین یک high و low
        """
        diff = swing_high - swing_low
        levels = {
            "0%": swing_high,
            "23.6%": swing_high - 0.236 * diff,
            "38.2%": swing_high - 0.382 * diff,
            "50%": swing_high - 0.5 * diff,
            "61.8%": swing_high - 0.618 * diff,
            "78.6%": swing_high - 0.786 * diff,
            "100%": swing_low,
        }
        return levels

    # ------------------------------
    # 3. Fibonacci Projection (Extension)
    # ------------------------------
    def fib_projection(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        محاسبه فیبو امتدادی (پیشرفته‌تر)
        """
        diff = swing_high - swing_low
        return {
            "127.2%": swing_high + 0.272 * diff,
            "161.8%": swing_high + 0.618 * diff,
            "200%": swing_high + 1.0 * diff,
            "261.8%": swing_high + 1.618 * diff,
        }

    # ------------------------------
    # 4. Cluster Detector
    # ------------------------------
    def cluster_zones(self, fib_sets: List[Dict[str, float]], tolerance: float = 0.002) -> List[Tuple[float, float]]:
        """
        یافتن نواحی همپوشانی چند سطح فیبو
        :param fib_sets: لیستی از دیکشنری‌های سطوح
        :param tolerance: درصد تحمل (۰.۲٪ مثلا)
        :return: لیست بازه‌های خوشه
        """
        all_levels = [price for fib in fib_sets for price in fib.values()]
        all_levels.sort()
        clusters = []
        start = all_levels[0]

        for i in range(1, len(all_levels)):
            if abs(all_levels[i] - all_levels[i - 1]) / all_levels[i] <= tolerance:
                continue
            else:
                clusters.append((start, all_levels[i - 1]))
                start = all_levels[i]
        return clusters

    # ------------------------------
    # 5. Golden Zone Features
    # ------------------------------
    def golden_zone_distance(self, price: float, swing_high: float, swing_low: float) -> float:
        """
        فاصله نرمالایز شده قیمت تا Golden Zone (61.8–78.6%)
        """
        levels = self.fib_levels(swing_high, swing_low)
        g_low, g_high = levels["78.6%"], levels["61.8%"]
        if price < g_low:
            return (g_low - price) / g_low
        elif price > g_high:
            return (price - g_high) / g_high
        else:
            return 0.0  # داخل ناحیه

    # ------------------------------
    # 6. Risk/Reward Tool
    # ------------------------------
    def risk_reward(self, entry: float, stop: float, target: float) -> float:
        """
        محاسبه نسبت ریسک به ریوارد
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return round(reward / risk, 2) if risk > 0 else np.inf

    # ------------------------------
    # 7. News-aware Filtering
    # ------------------------------
    def filter_levels_by_news(self, fib_levels: Dict[str, float], current_time: pd.Timestamp,
                              news_window: int = 30) -> Dict[str, float]:
        """
        حذف/تخفیف سطوح فیبو نزدیک به زمان اخبار مهم
        :param fib_levels: سطوح فیبو
        :param current_time: زمان فعلی
        :param news_window: دقیقه قبل/بعد خبر
        """
        if self.news is None:
            return fib_levels

        nearby_news = self.news[
            (abs((self.news['time'] - current_time).dt.total_seconds()) / 60) <= news_window
        ]
        if not nearby_news.empty:
            # سطوح را با برچسب "weak" برمی‌گردانیم
            return {lvl + "_weak": val for lvl, val in fib_levels.items()}
        return fib_levels
