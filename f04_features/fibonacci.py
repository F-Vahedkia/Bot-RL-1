# f04_features/fibonacci.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd


# =========================
# Core: محاسبات فیبوناچی
# =========================
class FibonacciCore:
    """هسته‌ی محاسبات فیبوناچی با خروجی یکنواخت DataFrame."""

    RETRACE_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTEND_RATIOS  = [1.272, 1.618, 2.0, 2.618]          # سطح 127.2% یعنی (r-1)=0.272
    PROJECT_RATIOS = [0.618, 1.0, 1.618]                 # ABC projection

    @staticmethod
    def _ensure_order(high: float, low: float) -> Tuple[float, float]:
        return (high, low) if high >= low else (low, high)

    @staticmethod
    def retracement(high: float, low: float, *, context: Optional[dict] = None) -> pd.DataFrame:
        """سطوح بازگشتی بین یک Swing High و Swing Low."""
        high, low = FibonacciCore._ensure_order(high, low)
        diff = high - low
        rows = []
        for r in FibonacciCore.RETRACE_RATIOS:
            price = high - r * diff  # 0%≈high … 100%≈low
            rows.append({"level_name": f"R_{r}", "ratio": r, "price": price, "type": "retracement"})
        df = pd.DataFrame(rows)
        if context:
            for k, v in context.items():
                df[k] = v
        return df

    @staticmethod
    def extension(high: float, low: float, *, context: Optional[dict] = None) -> pd.DataFrame:
        """سطوح امتدادی (targets) بر اساس همان سوییگ."""
        high, low = FibonacciCore._ensure_order(high, low)
        diff = high - low
        rows = []
        for r in FibonacciCore.EXTEND_RATIOS:
            price = high + (r - 1.0) * diff  # 127.2% = high + 0.272*diff
            rows.append({"level_name": f"E_{r}", "ratio": r, "price": price, "type": "extension"})
        df = pd.DataFrame(rows)
        if context:
            for k, v in context.items():
                df[k] = v
        return df

    @staticmethod
    def projection(A: float, B: float, C: float, *, context: Optional[dict] = None) -> pd.DataFrame:
        """Projection سطوح ABC: از بردار AB روی C پروجکت می‌کنیم."""
        diff = B - A
        rows = []
        for r in FibonacciCore.PROJECT_RATIOS:
            price = C + r * diff
            rows.append({"level_name": f"P_{r}", "ratio": r, "price": price, "type": "projection"})
        df = pd.DataFrame(rows)
        if context:
            for k, v in context.items():
                df[k] = v
        return df

    @staticmethod
    def multi_timeframe_retracement(swings: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        محاسبهٔ Retracement برای چند تایم‌فریم.
        swings = { "H1": (high, low), "H4": (high, low), ... }
        """
        frames = []
        for tf, (hi, lo) in swings.items():
            frames.append(FibonacciCore.retracement(hi, lo, context={"timeframe": tf}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["level_name", "ratio", "price", "type", "timeframe"]
        )


# ======================================
# Analyzer: ابزارهای تحلیلی و Featureها
# ======================================
class FibonacciAnalyzer:
    """
    نسخهٔ یکپارچه با حفظ تمام ۷ متد (همگی خروجی DataFrame):
      1) detect_swings
      2) fib_levels              (retracement)
      3) fib_extension           (قبلاً fib_projection نامیده بودی اما کار extension می‌کرد)
      4) cluster_zones
      5) golden_zone_distance
      6) risk_reward
      7) filter_levels_by_news
    """

    def __init__(self, price_data: pd.DataFrame, news_data: Optional[pd.DataFrame] = None):
        """
        price_data: DataFrame با ستون‌های ['Open','High','Low','Close'] و index زمانی
        news_data:  (اختیاری) DataFrame با ستون‌های ['time','impact']، time از نوع datetime64
        """
        self.df = price_data.copy()
        self.news = news_data.copy() if news_data is not None else None

        # سازگاری نام ستون‌ها
        cols = {c.lower(): c for c in self.df.columns}
        # اگر حروف کوچک بودند:
        if set(["open", "high", "low", "close"]).issubset(cols.keys()):
            self.df.rename(columns={cols["open"]: "Open",
                                    cols["high"]: "High",
                                    cols["low"]: "Low",
                                    cols["close"]: "Close"}, inplace=True)

        if self.news is not None and "time" in self.news.columns:
            self.news["time"] = pd.to_datetime(self.news["time"])

    # ------------------------------
    # 1) Auto Swing Detection
    # ------------------------------
    def detect_swings(self, window: int = 3) -> pd.DataFrame:
        """
        تشخیص سادهٔ Swing High/Low به روش فراکتال.
        خروجی: DataFrame با ستون‌های: ['index','time','price','kind']  kind ∈ {'high','low'}
        """
        highs = self.df["High"].values
        lows  = self.df["Low"].values
        idx = self.df.index

        rows = []
        n = len(self.df)
        rng = range(window, n - window)
        for i in rng:
            seg_hi = highs[i - window:i + window + 1]
            seg_lo = lows[i - window:i + window + 1]
            if highs[i] == np.max(seg_hi):
                rows.append({"index": i, "time": idx[i], "price": highs[i], "kind": "high"})
            if lows[i] == np.min(seg_lo):
                rows.append({"index": i, "time": idx[i], "price": lows[i], "kind": "low"})

        return pd.DataFrame(rows, columns=["index", "time", "price", "kind"])

    # ------------------------------
    # 2) Fibonacci Retracement (DF)
    # ------------------------------
    def fib_levels(self, swing_high: float, swing_low: float, *, context: Optional[dict] = None) -> pd.DataFrame:
        """
        محاسبهٔ سطوح بازگشتی (خروجی DataFrame). Wrapper روی FibonacciCore.retracement
        """
        return FibonacciCore.retracement(swing_high, swing_low, context=context)

    # ------------------------------
    # 3) Fibonacci Extension (DF)
    # ------------------------------
    def fib_extension(self, swing_high: float, swing_low: float, *, context: Optional[dict] = None) -> pd.DataFrame:
        """
        سطوح امتدادی (targets). (توجه: نام قبلی تو 'fib_projection' بود اما محاسبهٔ extension انجام می‌داد)
        """
        return FibonacciCore.extension(swing_high, swing_low, context=context)

    # ------------------------------
    # 4) Cluster Detector (DF)
    # ------------------------------
    def cluster_zones(
        self,
        fib_level_frames: Iterable[pd.DataFrame],
        tolerance: float = 0.002
    ) -> pd.DataFrame:
        """
        یافتن نواحی خوشه‌ای سطوح فیبوناچی.
        ورودی: iterable از DataFrameهایی که ستون 'price' دارند (مثل خروجی‌های retracement/extension/projection)
        tolerance: تحمل نسبی (مثلاً 0.002 = دو دهم درصد)

        خروجی: DataFrame با ستون‌های:
          ['cluster_id','cluster_low','cluster_high','center','count','type']
        """
        # جمع‌آوری همه سطوح
        prices = []
        for df in fib_level_frames:
            if not isinstance(df, pd.DataFrame) or "price" not in df.columns:
                continue
            prices.extend(df["price"].dropna().tolist())

        if not prices:
            return pd.DataFrame(columns=["cluster_id", "cluster_low", "cluster_high", "center", "count", "type"])

        prices = sorted(prices)
        clusters = []
        start = prices[0]
        current = [start]

        for p in prices[1:]:
            if abs(p - current[-1]) / current[-1] <= tolerance:
                current.append(p)
            else:
                if len(current) >= 2:
                    clusters.append((min(current), max(current), float(np.mean(current)), len(current)))
                current = [p]
        # آخرین بسته
        if len(current) >= 2:
            clusters.append((min(current), max(current), float(np.mean(current)), len(current)))

        out = pd.DataFrame(clusters, columns=["cluster_low", "cluster_high", "center", "count"])
        if out.empty:
            return pd.DataFrame(columns=["cluster_id", "cluster_low", "cluster_high", "center", "count", "type"])

        out.insert(0, "cluster_id", range(1, len(out) + 1))
        out["type"] = "cluster"
        return out[["cluster_id", "cluster_low", "cluster_high", "center", "count", "type"]]

    # ------------------------------
    # 5) Golden Zone Distance (DF)
    # ------------------------------
    def golden_zone_distance(self, price: float, swing_high: float, swing_low: float) -> pd.DataFrame:
        """
        فاصلهٔ نرمال‌شدهٔ قیمت تا ناحیهٔ طلایی (61.8%–78.6%) بر حسب نسبی.
        خروجی: ['g_low','g_high','price','distance','in_zone','type']
        """
        retr = FibonacciCore.retracement(swing_high, swing_low)
        # استخراج سطوح
        r618 = retr.loc[retr["level_name"] == "R_0.618", "price"].iloc[0]
        r786 = retr.loc[retr["level_name"] == "R_0.786", "price"].iloc[0]
        g_low, g_high = min(r618, r786), max(r618, r786)  # برای هر جهتِ سوییگ

        if price < g_low:
            distance = (g_low - price) / g_low
            in_zone = False
        elif price > g_high:
            distance = (price - g_high) / g_high
            in_zone = False
        else:
            distance = 0.0
            in_zone = True

        return pd.DataFrame([{
            "g_low": g_low,
            "g_high": g_high,
            "price": price,
            "distance": float(distance),
            "in_zone": bool(in_zone),
            "type": "golden_zone"
        }])

    # ------------------------------
    # 6) Risk/Reward Tool (DF)
    # ------------------------------
    def risk_reward(self, entry: float, stop: float, target: float) -> pd.DataFrame:
        """
        نسبت ریسک به ریوارد (RR). خروجی: ['entry','stop','target','rr','type']
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = float(np.inf) if risk == 0 else float(reward / risk)
        return pd.DataFrame([{
            "entry": entry, "stop": stop, "target": target,
            "rr": round(rr, 4) if np.isfinite(rr) else rr,
            "type": "metric_rr"
        }])

    # ------------------------------
    # 7) News-aware Filtering (DF)
    # ------------------------------
    def filter_levels_by_news(
        self,
        fib_levels_df: pd.DataFrame,
        current_time: pd.Timestamp,
        news_window_min: int = 30
    ) -> pd.DataFrame:
        """
        اگر نزدیک خبر مهم باشیم، شدت (strength) سطوح را 'weak' علامت بزن.
        ورودی: fib_levels_df باید ستون‌های ['level_name','price','type'] داشته باشد.
        خروجی: همان DataFrame با ستونِ جدید 'strength' در {'strong','weak'}
        """
        df = fib_levels_df.copy()
        if df.empty:
            df["strength"] = []
            return df

        if self.news is None or "time" not in self.news.columns:
            df["strength"] = "strong"
            return df

        # آیا خبری در بازه ±news_window_min داریم؟
        delta = self.news["time"].sub(pd.to_datetime(current_time)).abs().dt.total_seconds() / 60.0
        near = delta.le(news_window_min)
        is_near_news = bool(near.any())

        df["strength"] = "weak" if is_near_news else "strong"
        return df


# ------------------------------
# مثال استفاده‌ی سریع
# ------------------------------
if __name__ == "__main__":
    # دادهٔ ساختگی برای تست
    idx = pd.date_range("2025-08-01", periods=50, freq="H")
    price = pd.DataFrame({
        "Open":  np.random.normal(1950, 5, size=50),
        "High":  np.random.normal(1953, 5, size=50),
        "Low":   np.random.normal(1947, 5, size=50),
        "Close": np.random.normal(1950, 5, size=50),
    }, index=idx)
    price["High"] = price[["Open","High","Low","Close"]].max(axis=1)
    price["Low"]  = price[["Open","High","Low","Close"]].min(axis=1)

    news = pd.DataFrame({
        "time": pd.to_datetime(["2025-08-01 10:00", "2025-08-02 14:30"]),
        "impact": ["high", "medium"]
    })

    analyzer = FibonacciAnalyzer(price, news)

    swings = analyzer.detect_swings(window=2)
    retr = analyzer.fib_levels(2000, 1900, context={"src": "H1"})
    ext  = analyzer.fib_extension(2000, 1900, context={"src": "H1"})
    proj = FibonacciCore.projection(1900, 2000, 1950, context={"src": "ABC"})
    cluster = analyzer.cluster_zones([retr, ext, proj], tolerance=0.003)
    gz = analyzer.golden_zone_distance(price=1960, swing_high=2000, swing_low=1900)
    rr = analyzer.risk_reward(entry=1950, stop=1940, target=1975)
    filtered = analyzer.filter_levels_by_news(retr, current_time=pd.Timestamp("2025-08-01 09:55"), news_window_min=30)

    demo = {
        "swings.head": swings.head(),
        "retr": retr,
        "ext": ext,
        "proj": proj,
        "cluster": cluster,
        "golden_zone": gz,
        "rr": rr,
        "filtered": filtered
    }
    for k, v in demo.items():
        print(f"\n== {k} ==\n", v)
