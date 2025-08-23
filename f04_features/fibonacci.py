# f04_features/fibonacci.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Iterable, Union
import numpy as np
import pandas as pd
from __future__ import annotations
import math

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



class MultiTimeframeAutoFibonacci:
    def __init__(self, price_df, timeframes=("1H","4H","1D")):
        self.price_df = price_df
        self.timeframes = timeframes
        self.results = {}

    def _resample(self, tf):
        return self.price_df.resample(tf).agg({
            "Open":"first","High":"max","Low":"min","Close":"last"
        }).dropna()

    def _detect_swings(self, df, prominence=0.002, min_distance=5):
        # الگوریتم ترکیبی fractal + prominence + فاصله
        swings = []
        for i in range(min_distance, len(df)-min_distance):
            high, low = df["High"].iloc[i], df["Low"].iloc[i]
            if high == df["High"].iloc[i-min_distance:i+min_distance+1].max():
                swings.append(("high", df.index[i], high))
            elif low == df["Low"].iloc[i-min_distance:i+min_distance+1].min():
                swings.append(("low", df.index[i], low))
        return swings

    def run(self):
        for tf in self.timeframes:
            df_tf = self._resample(tf)
            swings = self._detect_swings(df_tf)
            if len(swings) >= 2:
                # آخرین Swing High و Swing Low را بگیریم
                low = min([p for t,_,p in swings if t=="low"])
                high = max([p for t,_,p in swings if t=="high"])
                fib = FibonacciCore.Retracement(high=high, low=low).levels()
                self.results[tf] = fib
        return self.results

'''
# ============================================================
# Schema reference (برای هماهنگی با کدهای قبلی)
# ------------------------------------------------------------
# انتظار می‌رود DataFrame سطوح فیبوناچی (levels_df) این ستون‌ها را داشته باشد:
# - timeframe: str        (مثل "M15", "H1", "H4", "D1")
# - method: str           ("retracement" | "extension" | "projection")
# - ratio: float          (مثل 0.618 یا 1.618)
# - price: float          (سطح قیمتی محاسبه‌شده)
# - leg_id: Optional[str] (شناسه‌ی موج/سوییگ؛ اختیاری اما مفید)
# - meta:  Optional[dict] (اطلاعات اضافه؛ اختیاری)
#
# نمونه‌ی تولید levels_df:
# df = FibonacciCore.retracement(high=H, low=L, context={"timeframe": "H1", "leg_id": "swing_12"})
# سپس می‌توانی ستون‌های "timeframe" و "leg_id" را از context به df اضافه کنی.
# ============================================================
'''

@dataclass
class ClusterParams:
    """
    پیکربندی انعطاف‌پذیر برای تشخیص خوشه‌های همپوشان (Confluence Zones).
    - دو نوع تلورانس را پشتیبانی می‌کند: نسبی (rel) بر حسب درصد/بِیسیس‌پوینت و مطلق (abs) بر حسب قیمت/پیپ.
    - می‌توان حداقل تعداد برخورد (min_hits) تعیین کرد تا خوشه‌های ضعیف فیلتر شوند.
    - وزن‌دهی اختیاری بر اساس method/ratio/timeframe برای محاسبه‌ی امتیاز.
    """
    tolerance_mode: str = "rel"        # "rel" یا "abs"
    rel_tolerance: float = 0.001       # 0.001 ≈ 0.1% (100 bps)
    abs_tolerance: Optional[float] = None  # اگر tolerance_mode="abs"، مثلاً 1.0 برای 1 دلار/پیپ
    min_hits: int = 2                  # حداقل تعداد سطوح در یک خوشه
    use_timeframe_diversity: bool = True  # اگر True، تنوع تایم‌فریم امتیاز را افزایش می‌دهد
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "retracement": 1.0,
        "extension": 0.9,
        "projection": 0.8
    })
    ratio_weights: Dict[float, float] = field(default_factory=lambda: {
        0.236: 0.7, 0.382: 0.9, 0.5: 1.0, 0.618: 1.2, 0.786: 1.1,
        1.272: 1.0, 1.618: 1.2, 2.0: 0.9, 2.618: 0.8
    })
    # امتیازدهی به تایم‌فریم‌ها (اختیاری). اگر خالی باشد، همه 1.0.
    timeframe_weights: Dict[str, float] = field(default_factory=dict)

'''
# ============================================================
# Schema reference (برای هماهنگی با کدهای قبلی)
# ------------------------------------------------------------
# انتظار می‌رود DataFrame سطوح فیبوناچی (levels_df) این ستون‌ها را داشته باشد:
# - timeframe: str        (مثل "M15", "H1", "H4", "D1")
# - method: str           ("retracement" | "extension" | "projection")
# - ratio: float          (مثل 0.618 یا 1.618)
# - price: float          (سطح قیمتی محاسبه‌شده)
# - leg_id: Optional[str] (شناسه‌ی موج/سوییگ؛ اختیاری اما مفید)
# - meta:  Optional[dict] (اطلاعات اضافه؛ اختیاری)
#
# نمونه‌ی تولید levels_df:
# df = FibonacciCore.retracement(high=H, low=L, context={"timeframe": "H1", "leg_id": "swing_12"})
# سپس می‌توانی ستون‌های "timeframe" و "leg_id" را از context به df اضافه کنی.
# ============================================================
'''

@dataclass
class ClusterParams:
    """
    پیکربندی انعطاف‌پذیر برای تشخیص خوشه‌های همپوشان (Confluence Zones).
    - دو نوع تلورانس را پشتیبانی می‌کند: نسبی (rel) بر حسب درصد/بِیسیس‌پوینت و مطلق (abs) بر حسب قیمت/پیپ.
    - می‌توان حداقل تعداد برخورد (min_hits) تعیین کرد تا خوشه‌های ضعیف فیلتر شوند.
    - وزن‌دهی اختیاری بر اساس method/ratio/timeframe برای محاسبه‌ی امتیاز.
    """
    tolerance_mode: str = "rel"        # "rel" یا "abs"
    rel_tolerance: float = 0.001       # 0.001 ≈ 0.1% (100 bps)
    abs_tolerance: Optional[float] = None  # اگر tolerance_mode="abs"، مثلاً 1.0 برای 1 دلار/پیپ
    min_hits: int = 2                  # حداقل تعداد سطوح در یک خوشه
    use_timeframe_diversity: bool = True  # اگر True، تنوع تایم‌فریم امتیاز را افزایش می‌دهد
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "retracement": 1.0,
        "extension": 0.9,
        "projection": 0.8
    })
    ratio_weights: Dict[float, float] = field(default_factory=lambda: {
        0.236: 0.7, 0.382: 0.9, 0.5: 1.0, 0.618: 1.2, 0.786: 1.1,
        1.272: 1.0, 1.618: 1.2, 2.0: 0.9, 2.618: 0.8
    })
    # امتیازدهی به تایم‌فریم‌ها (اختیاری). اگر خالی باشد، همه 1.0.
    timeframe_weights: Dict[str, float] = field(default_factory=dict)


class FibonacciClusterDetector:
    """
    تشخیص نواحی همپوشان سطوح فیبوناچی (Confluence / Cluster Zones)
    ---------------------------------------------------------------
    ورودی: DataFrame سطوح فیبوناچی چند-تایم‌فریم (levels_df) با اسکیمای بالا.
    خروجی: DataFrame خوشه‌ها شامل:
        - cluster_id
        - price_min, price_max, center, width
        - hits (تعداد سطوح)
        - timeframes (لیست یکتا)
        - methods (لیست یکتا)
        - ratios (لیست یکتا)
        - score (امتیاز قدرت خوشه)
    """

    def __init__(self, params: Optional[ClusterParams] = None):
        self.params = params or ClusterParams()

    # ---------- Helper: tolerance ----------
    def _within_tolerance(self, p1: float, p2: float) -> bool:
        if self.params.tolerance_mode == "abs":
            tol = self.params.abs_tolerance if self.params.abs_tolerance is not None else 0.0
            return abs(p1 - p2) <= tol
        # relative tolerance (around current price)
        base = max(abs(p1), abs(p2))
        return abs(p1 - p2) <= base * self.params.rel_tolerance

    # ---------- Helper: score of a single level ----------
    def _level_score(self, row: pd.Series) -> float:
        method_w = self.params.method_weights.get(str(row.get("method", "")).lower(), 1.0)
        ratio_w = self.params.ratio_weights.get(float(row.get("ratio", 0.0)), 1.0)
        tf = str(row.get("timeframe", "")).upper()
        tf_w = self.params.timeframe_weights.get(tf, 1.0)
        return method_w * ratio_w * tf_w

    # ---------- Helper: diversity boost ----------
    def _diversity_bonus(self, timeframes: Iterable[str]) -> float:
        if not self.params.use_timeframe_diversity:
            return 1.0
        uniq = len(set([str(t).upper() for t in timeframes if pd.notna(t)]))
        # تابعی نرم برای افزایش امتیاز با تعداد تایم‌فریم‌ها
        # 1 TF → 1.00x, 2 TF → 1.10x, 3 TF → 1.20x, ...
        return 1.0 + 0.10 * max(0, uniq - 1)

    # ---------- Public: cluster ----------
    def cluster_levels(
        self,
        levels_df: pd.DataFrame,
        sort_by: str = "score"
    ) -> pd.DataFrame:
        """
        خوشه‌بندی سطوح فیبوناچی.
        Args:
            levels_df: DataFrame با ستون‌های (timeframe, method, ratio, price, leg_id, meta)
            sort_by: "score" یا "center" (برای مرتب‌سازی خروجی)

        Returns:
            clusters_df: DataFrame با ستون‌های:
                cluster_id, price_min, price_max, center, width,
                hits, timeframes, methods, ratios, score
        """
        if levels_df is None or levels_df.empty:
            return pd.DataFrame(columns=[
                "cluster_id", "price_min", "price_max", "center", "width",
                "hits", "timeframes", "methods", "ratios", "score"
            ])

        df = levels_df.copy()
        # اطمینان از وجود ستون‌های کلیدی
        for col in ["price", "ratio", "method", "timeframe"]:
            if col not in df.columns:
                df[col] = np.nan

        # مرتب‌سازی بر اساس قیمت
        df = df.sort_values(by="price").reset_index(drop=True)

        clusters: List[Dict[str, Union[int, float, List, set]]] = []
        current_cluster: List[pd.Series] = []

        def finalize_cluster(levels_in_cluster: List[pd.Series]) -> Optional[Dict]:
            if not levels_in_cluster:
                return None
            prices = [float(r["price"]) for r in levels_in_cluster]
            tfs = [str(r.get("timeframe", "")) for r in levels_in_cluster]
            methods = [str(r.get("method", "")) for r in levels_in_cluster]
            ratios = [float(r.get("ratio", np.nan)) for r in levels_in_cluster]

            price_min = float(np.min(prices))
            price_max = float(np.max(prices))
            center = float(np.mean(prices))
            width = price_max - price_min

            # امتیاز پایه = مجموع امتیاز سطح‌ها
            base_score = float(np.sum([self._level_score(r) for r in levels_in_cluster]))
            # پاداش تنوع تایم‌فریم
            score = base_score * self._diversity_bonus(tfs)

            return dict(
                price_min=price_min,
                price_max=price_max,
                center=center,
                width=width,
                hits=len(levels_in_cluster),
                timeframes=sorted(list(set([tf for tf in tfs if tf]))),
                methods=sorted(list(set([m.lower() for m in methods if m]))),
                ratios=sorted(list(set([r for r in ratios if not np.isnan(r)]))),
                score=round(score, 4),
            )

        # پیمایش ترتیبی و ساخت خوشه‌ها با آستانه‌ی مجاز
        for _, row in df.iterrows():
            if not current_cluster:
                current_cluster = [row]
                continue

            # اگر قیمتِ جدید داخل تلورانس نسبت به مرکز خوشه فعلی بود → اضافه کن
            current_center = float(np.mean([float(r["price"]) for r in current_cluster]))
            if self._within_tolerance(current_center, float(row["price"])):
                current_cluster.append(row)
            else:
                # خوشه‌ی فعلی را نهایی کن
                cl = finalize_cluster(current_cluster)
                if cl and cl["hits"] >= self.params.min_hits:
                    clusters.append(cl)
                # خوشه‌ی جدید را با آیتم جاری آغاز کن
                current_cluster = [row]

        # آخرین خوشه
        cl = finalize_cluster(current_cluster)
        if cl and cl["hits"] >= self.params.min_hits:
            clusters.append(cl)

        if not clusters:
            return pd.DataFrame(columns=[
                "cluster_id", "price_min", "price_max", "center", "width",
                "hits", "timeframes", "methods", "ratios", "score"
            ])

        clusters_df = pd.DataFrame(clusters).reset_index().rename(columns={"index": "cluster_id"})

        # مرتب‌سازی خروجی
        if sort_by in {"score", "center", "price_min", "price_max", "hits", "width"}:
            ascending = (sort_by != "score" and sort_by != "hits")
            clusters_df = clusters_df.sort_values(by=sort_by, ascending=ascending, ignore_index=True)

        return clusters_df

    # ---------- Utility: build levels_df from multiple swings ----------
    @staticmethod
    def build_levels_df_from_swings(
        swings_by_tf: Dict[str, List[Tuple[float, float, Optional[str]]]],
        # مثال: {"H1": [(high1, low1, "s1"), (high2, low2, "s2")], "H4": [...]}
        method: str = "retracement",
        ratios: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        اگر خروجی‌ات از «استخراج سوییگ چندتایم‌فریم» به‌صورت دیکشنری high/low باشد،
        این تابع به‌سرعت آن‌ها را به یک levels_df استاندارد تبدیل می‌کند.

        Args:
            swings_by_tf: دیکشنری تایم‌فریم ← لیست (high, low, leg_id)
            method: نوع سطح ("retracement" | "extension" | "projection")
            ratios: اگر None باشد، برای retracement از ratios پیش‌فرض FibonacciCore استفاده می‌شود.

        Returns:
            levels_df: DataFrame استاندارد سطوح.
        """
        records: List[Dict] = []
        use_ratios = ratios
        if use_ratios is None:
            if method == "retracement":
                use_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            elif method == "extension":
                use_ratios = [1.272, 1.618, 2.0, 2.618]
            elif method == "projection":
                use_ratios = [0.618, 1.0, 1.618]
            else:
                use_ratios = []

        for tf, legs in swings_by_tf.items():
            for (high, low, leg_id) in legs:
                if method == "retracement":
                    diff = high - low
                    for r in use_ratios:
                        price = high - r * diff
                        records.append({
                            "timeframe": tf, "method": "retracement", "ratio": float(r),
                            "price": float(price), "leg_id": leg_id
                        })
                elif method == "extension":
                    diff = high - low
                    for r in use_ratios:
                        price = high + r * diff
                        records.append({
                            "timeframe": tf, "method": "extension", "ratio": float(r),
                            "price": float(price), "leg_id": leg_id
                        })
                elif method == "projection":
                    # برای projection باید A/B/C داشته باشیم؛ اینجا صرفاً اسکلت آماده است.
                    # اگر A/B/C در دسترس نیست، بهتر است از مسیر دیگر تولید شود.
                    continue

        return pd.DataFrame.from_records(records, columns=[
            "timeframe", "method", "ratio", "price", "leg_id"
        ])

'''
نمونه استفاده (خیلی کوتاه)

# 1) فرض: levels_df را از قبل ساخته‌ای (مثلاً با FibonacciCore.retracement + الحاق تایم‌فریم‌ها)
# یا با build_levels_df_from_swings(...)
params = ClusterParams(tolerance_mode="rel", rel_tolerance=0.001, min_hits=2)
detector = FibonacciClusterDetector(params)

clusters = detector.cluster_levels(levels_df, sort_by="score")
print(clusters.head())
# ستون‌ها: cluster_id, price_min, price_max, center, width, hits, timeframes, methods, ratios, score
'''



####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@dataclass
class FibLevel:
    """Represent a Fibonacci level observed on a timeframe.

    Attributes:
        tf: timeframe label (e.g. 'H1', 'H4')
        ratio: Fibonacci ratio (e.g. 0.618)
        price: price level (float)
        meta: optional dict for storing origin info (swing indices, timestamp...)
    """
    tf: str
    ratio: float
    price: float
    meta: Dict = field(default_factory=dict)


@dataclass
class Zone:
    low: float
    high: float
    members: List[FibLevel] = field(default_factory=list)
    strength: float = 0.0

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0

    def width(self) -> float:
        return self.high - self.low


class GoldenZoneDetector:
    """Detects Golden Zones (high-confluence Fibonacci retracement areas) and ranks them.

    Design goals:
    - Compatible with existing fibonacci retracement outputs: accepts per-timeframe Fibonacci levels
    - Produces merged "zones" where levels from multiple timeframes overlap
    - Assigns a strength score (count + proximity weight) so zones can be ranked
    - Highly configurable tolerances but sensible defaults for immediate use

    How to use:
    1) If you already have computed levels across timeframes, call detect_from_levels(levels).
       levels: List[dict-like] with keys: {'tf','ratio','price','meta'(optional)} or instances of FibLevel.
    2) Or call detect_from_retracement(swing_low, swing_high, ratios=[...]) to compute levels from a single retracement.
    3) After detection, use get_top_zones(n) to fetch the strongest zones.
    """

    def __init__(
        self,
        zone_width_pct: float = 0.006,
        # zone_width_pct: default half-width fraction around level price (0.006 -> ±0.3%)
        merge_gap_pct: float = 0.0025,
        # merge_gap_pct: if two zones are within this fraction they will be considered overlapping/merged
        min_members: int = 2,
        # minimum distinct timeframes (members) to consider a zone meaningful
    ) -> None:
        self.zone_width_pct = float(zone_width_pct)
        self.merge_gap_pct = float(merge_gap_pct)
        self.min_members = int(min_members)

    def _level_to_initial_zone(self, price: float) -> Tuple[float, float]:
        """Convert a level price to a small zone (low, high) using zone_width_pct."""
        half = price * (self.zone_width_pct / 2.0)
        return (price - half, price + half)

    def _zones_overlap(self, z1: Tuple[float, float], z2: Tuple[float, float]) -> bool:
        # Treat a small gap (merge_gap_pct) as overlap for fuzzy matching
        low1, high1 = z1
        low2, high2 = z2
        gap = max(low1, low2) - min(high1, high2)
        # gap <= 0 means overlap; allow small positive gap tolerance
        tol = (abs(high1 + low1) / 2.0) * self.merge_gap_pct
        return gap <= tol

    def detect_from_levels(self, levels: List[Dict]) -> List[Zone]:
        """Main entry: detect zones from a list of levels.

        `levels` can be a list of FibLevel instances OR dicts with keys: tf, ratio, price, meta(optional).
        Returns: merged and scored list of Zone objects sorted by descending strength.
        """
        fibs: List[FibLevel] = []
        for lv in levels:
            if isinstance(lv, FibLevel):
                fibs.append(lv)
            else:
                # assume dict-like
                fibs.append(FibLevel(tf=lv.get('tf', 'NA'), ratio=float(lv['ratio']), price=float(lv['price']), meta=lv.get('meta', {})))

        # create initial tiny zones for each level
        initial = []  # list of (low, high, FibLevel)
        for f in fibs:
            low, high = self._level_to_initial_zone(f.price)
            initial.append((low, high, f))

        # sort by low price
        initial.sort(key=lambda x: x[0])

        # merge overlapping/fuzzy zones into clusters
        merged: List[Zone] = []
        for low, high, f in initial:
            placed = False
            for mz in merged:
                if self._zones_overlap((low, high), (mz.low, mz.high)):
                    # expand merged zone bounds to include new zone
                    mz.low = min(mz.low, low)
                    mz.high = max(mz.high, high)
                    mz.members.append(f)
                    placed = True
                    break
            if not placed:
                merged.append(Zone(low=low, high=high, members=[f]))

        # after initial merge, do a second pass to merge zones that became overlapping after expansion
        merged = self._consolidate_merged_zones(merged)

        # compute strength for each zone
        for mz in merged:
            mz.strength = self._compute_zone_strength(mz)

        # filter out weak zones that don't meet min_members (by distinct timeframe)
        result = [z for z in merged if self._distinct_tf_count(z) >= self.min_members]

        # sort by strength desc
        result.sort(key=lambda z: z.strength, reverse=True)

        return result

    def _consolidate_merged_zones(self, zones: List[Zone]) -> List[Zone]:
        if not zones:
            return []
        # sort by low
        zones.sort(key=lambda z: z.low)
        out: List[Zone] = [zones[0]]
        for z in zones[1:]:
            last = out[-1]
            if self._zones_overlap((last.low, last.high), (z.low, z.high)):
                # merge
                last.low = min(last.low, z.low)
                last.high = max(last.high, z.high)
                last.members.extend(z.members)
            else:
                out.append(z)
        return out

    def _distinct_tf_count(self, zone: Zone) -> int:
        return len(set([m.tf for m in zone.members]))

    def _compute_zone_strength(self, zone: Zone) -> float:
        """Strength heuristic combining:
        - number of distinct timeframes (primary)
        - density: how concentrated members are around the center
        - higher weight to members whose ratio is near 0.618 (configurable policy)
        """
        if not zone.members:
            return 0.0
        center = zone.center
        # distinct tf count
        distinct_tfs = self._distinct_tf_count(zone)

        # proximity score: sum of inverse distance to center (clipped)
        prox = 0.0
        for m in zone.members:
            dist = abs(m.price - center)
            # convert to relative fraction to avoid scale issues
            rel = dist / max(1e-8, center)
            prox += 1.0 / (1.0 + rel * 100.0)  # distances quickly reduce contribution

        # golden affinity: give slight bonus to members close to 0.618 ratio
        golden_bonus = 0.0
        for m in zone.members:
            golden_bonus += max(0.0, (1.0 - abs(m.ratio - 0.618) / 0.1))  # full bonus when within 0.0 of 0.618, fades by 0.1

        # final combination (weights can be tuned or exposed)
        strength = distinct_tfs * 2.0 + prox + 0.5 * golden_bonus
        return float(strength)

    def get_top_zones(self, zones: List[Zone], n: int = 3) -> List[Zone]:
        return sorted(zones, key=lambda z: z.strength, reverse=True)[:n]

    def detect_from_retracement(self, swing_low: float, swing_high: float, ratios: Optional[List[float]] = None, tf: str = 'NA') -> List[Zone]:
        """Convenience helper: compute Fibonacci levels from a single swing and detect golden zones.

        swing_low, swing_high: float prices (low < high for bull retracement; handler will work either way)
        ratios: list of Fibonacci ratios to compute (default common set)
        tf: optional timeframe tag to attach to produced levels
        """
        if ratios is None:
            ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

        levels = []
        high = max(swing_high, swing_low)
        low = min(swing_high, swing_low)
        rng = high - low
        # for each ratio compute price = high - ratio * range (typical retracement formula)
        for r in ratios:
            price = high - r * rng
            levels.append(FibLevel(tf=tf, ratio=r, price=price, meta={'from': 'single_retracement'}))

        return self.detect_from_levels(levels)

'''
# ------------------------
# Example (non-executing) usage to integrate into fibonacci.py
# ------------------------
#
# detector = GoldenZoneDetector(zone_width_pct=0.006, merge_gap_pct=0.0025, min_members=2)
#
# # If you already have multi-timeframe levels (list of dicts):
# multi_tf_levels = [
#     {'tf':'H1','ratio':0.618,'price':1.2345},
#     {'tf':'H4','ratio':0.618,'price':1.2339},
#     {'tf':'D1','ratio':0.5,'price':1.2360},
# ]
# zones = detector.detect_from_levels(multi_tf_levels)
# top = detector.get_top_zones(zones, n=3)
#
# # Or compute from swing points:
# zones_from_retr = detector.detect_from_retracement(swing_low=1.2000, swing_high=1.2500, tf='H1')
#
# ------------------------
# End of GoldenZoneDetector
'''
