# ===== BEGIN FILE: f04_features/fibonacci_3.py =====
""" fibonacci.py یک ماژول جامع و توسعه‌پذیر فیبوناچی برای ربات تریدر ویژگی‌ها: - تشخیص سوئینگ (پیک/والی) با متد ZigZag/peak detection (پارامترهای prominence & distance) - محاسبهٔ سطوح retracement/extension/projection با لیست قابل پیکربندی نسبت‌ها - خوشه‌بندی (clustering) سطوح با DBSCAN و نرمال‌سازی فاصله بر اساس ATR - محاسبهٔ ناحیهٔ Gold/Golden zone با tolerance تنظیم‌شونده مبتنی بر ATR - متدهای تولید پلان ورود (entry plan) شامل تعیین SL/TP و اندازهٔ پوزیشن بر اساس risk % - ابزارهایی برای ارزیابی تاریخی سطوح (touch/hit metrics) - فیلتر اسکلت برای اخبار (news-aware) و جایگاه برای گسترش - export برای رسم و ذخیرهٔ سطوح - قابلیت‌هایی برای تولید فیچرهای آمادهٔ RL و آپدیت لحظه‌ای (real_time_update) Dependencies: - numpy, pandas, scipy, sklearn طراحی: ماژولار، تایپ شده، مستندسازی‌شده و همراه با comments برای گسترش آینده.
""" 
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# Configure module logger
logger = logging.getLogger("fibonacci")
logger.addHandler(logging.NullHandler())

# -----------------------------
# Config / Constants
# -----------------------------
DEFAULT_FIB_RATIOS = [23.6, 38.2, 50.0, 61.8, 78.6, 100.0, 127.2, 161.8]

# -----------------------------
# Utilities
# -----------------------------
def _validate_price_index(df: pd.DataFrame) -> None:
    """اطمینان از اینکه DataFrame دارای index datetime و مرتب است و ستون‌های OHLC وجود دارد."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex (timestamps)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in increasing order")
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns.str.lower())):
        raise ValueError(f"DataFrame must include columns: {required}")

def atr(df: pd.DataFrame, window: int = 14, use_high_low: bool = True) -> pd.Series:
    """محاسبه ATR ساده.
    خروجی بر اساس close-index aligned.
    پارامترها: df: DataFrame با ستون‌های high، low، close window: طول پنجره """
    _validate_price_index(df)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

# -----------------------------
# Swing Detection (advanced)
# -----------------------------
class SwingDetector:
    """شناسایی نقاط swing (قله و دره) با روش find_peaks از scipy یا ZigZag-like logic.
    خروجی تابع detect_swings: DataFrame با ستون‌های ['time','idx','price','kind','prominence','confidence'] where kind in {'peak','valley'}
    پارامترهای قابل تنظیم:
    - method: 'peaks' یا 'zigzag'
    - min_prominence: کمترین prominence برای پذیرفتن پیک
    - min_distance_bars: حداقل فاصله بین سوئینگ‌ها بر حسب کندل
    - use_wicks: آیا از wick (high/low) استفاده شود یا body (close) """
    def __init__(self, method: str = "peaks", min_prominence: Optional[float] = None, min_distance_bars: int = 3, use_wicks: str = "wicks"):
        # 'wicks'|'bodies'|'close'
        self.method = method
        self.min_prominence = min_prominence
        self.min_distance_bars = max(1, int(min_distance_bars))
        self.use_wicks = use_wicks

    def _price_series(self, df: pd.DataFrame) -> pd.Series:
        # NOTE: For peaks we use 'close' series as baseline.
        # If wicks logic needed,
        # caller can pass df['high'] / df['low'] separately and post-process.
        if self.use_wicks == "wicks":
            return df["close"]
        elif self.use_wicks == "bodies":
            return (df["open"] + df["close"]) / 2
        elif self.use_wicks == "close":
            return df["close"]
        else:
            raise ValueError("use_wicks must be one of 'wicks','bodies','close'")

    def detect_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشخیص نقاط قله و دره به همراه prominence.
        خروجی: DataFrame index reset شده با ستون‌های:
        - time (Timestamp)
        - idx (original integer index)
        - price
        - kind ('peak'|'valley')
        - prominence (float)
        - confidence (0..1 normalized) """
        _validate_price_index(df)
        price_series = self._price_series(df)
        price = price_series.values

        # Determine prominence threshold if not set
        if self.min_prominence is None:
            # Heuristic: use std of price scaled
            self.min_prominence = max(1e-8, float(np.nanstd(price) * 0.5))

        # Detect peaks and valleys
        peaks, props_peaks = find_peaks(price, distance=self.min_distance_bars, prominence=self.min_prominence)
        valleys, props_valleys = find_peaks(-price, distance=self.min_distance_bars, prominence=self.min_prominence)

        records = []
        for p, prom in zip(peaks, props_peaks.get("prominences", [None] * len(peaks))):
            records.append({
                "time": df.index[p],
                "idx": int(p),
                "price": float(price[p]),
                "kind": "peak",
                "prominence": float(prom) if prom is not None else np.nan,
            })
        for v, prom in zip(valleys, props_valleys.get("prominences", [None] * len(valleys))):
            records.append({
                "time": df.index[v],
                "idx": int(v),
                "price": float(price[v]),
                "kind": "valley",
                "prominence": float(prom) if prom is not None else np.nan,
            })

        swings = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
        # Confidence score: normalize prominence
        if not swings.empty and swings["prominence"].notna().any():
            max_prom = swings["prominence"].max()
            swings["confidence"] = swings["prominence"] / (max_prom + 1e-9)
        else:
            swings["confidence"] = 0.0
        return swings

# -----------------------------
# Fibonacci core calculations
# -----------------------------
class FibonacciCore:
    """توابع پایهٔ محاسبه سطوح فیبوناچی متدها:
    - retracement: محاسبه سطوح بین swing high و swing low
    - extension: محاسبه سطوح extension
    - projection: محاسبه پروجکشن (در صورت نیاز)
    - multi_timeframe_retracement: ترکیب retracementها از چند تایم‌فریم
    همهٔ نسبت‌ها قابل پیکربندی هستند.
    """
    def __init__(self, ratios: Optional[Iterable[float]] = None):
        self.ratios = list(ratios) if ratios is not None else DEFAULT_FIB_RATIOS

    @staticmethod
    def _ordered_pair(a: float, b: float) -> Tuple[float, float]:
        return (a, b) if a <= b else (b, a)

    def retracement(self, start_price: float, end_price: float, *, context: Optional[Dict] = None) -> pd.DataFrame:
        """محاسبه سطوح retracement نسبت به بازهٔ start->end.
        خروجی: DataFrame columns = ['ratio','price','type'] """
        records = []
        for r in self.ratios:
            frac = r / 100.0
            # convention: level = end + (start - end) * (1 - frac)
            level = end_price + (start_price - end_price) * (1 - frac)
            records.append({"ratio": float(r), "price": float(level), "type": "retracement"})
        return pd.DataFrame(records)

    def extension(self, start_price: float, end_price: float, *, context: Optional[Dict] = None) -> pd.DataFrame:
        """محاسبه سطوح extension نسبت به بازهٔ start->end.
        خروجی: DataFrame columns = ['ratio','price','type'] """
        records = []
        for r in self.ratios:
            factor = r / 100.0
            level = end_price + (end_price - start_price) * factor
            records.append({"ratio": float(r), "price": float(level), "type": "extension"})
        return pd.DataFrame(records)

    def projection(self, a: float, b: float, c: float, *, ratios: Optional[Iterable[float]] = None, context: Optional[Dict] = None) -> pd.DataFrame:
        """Projection typical for harmonic patterns: project move BC based on AB length.
        returns DataFrame ['ratio','price','type'] """
        if ratios is None:
            ratios = self.ratios
        records = []
        ab = b - a
        bc = c - b
        for r in ratios:
            price = float(c + bc * (r / 100.0))
            records.append({"ratio": float(r), "price": price, "type": "projection"})
        return pd.DataFrame(records)

    def multi_timeframe_retracement(self, swings: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """گیرندهٔ دیکشنری tf -> (high, low) و بازگرداندن retracementهای همهٔ تایم‌فریم‌ها یکپارچه شده."""
        frames = []
        for tf, (high, low) in swings.items():
            df = self.retracement(high, low)
            df["tf"] = tf
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["ratio", "price", "type", "tf"])

# -----------------------------
# Clustering levels (DBSCAN normalized by ATR)
# -----------------------------
@dataclass
class ClusterInfo:
    cluster_id: int
    price: float
    members: List[float]
    strength: float
    touch_count: int
    last_touch: Optional[pd.Timestamp]
    timeframes: Dict[str, int]

class LevelClusterer:
    """خوشه‌بندی سطوح فیبوناچی با DBSCAN که فاصله را بر اساس ATR نرمال می‌کند.
    ایده: eps = eps_atr_multiplier * ATR_at_level """
    def __init__(self, eps_atr_multiplier: float = 0.5, min_samples: int = 1, atr_window: int = 14):
        self.eps_atr_multiplier = eps_atr_multiplier
        self.min_samples = max(1, int(min_samples))
        self.atr_window = atr_window

    def _compute_eps(self, price: float, atr_value: float) -> float:
        return float(max(1e-8, self.eps_atr_multiplier * atr_value))

    def cluster_levels(self, levels: Iterable[float], prices_df: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
        """خوشه‌بندی یک لیست از سطوح نسبت به دیتافریم قیمت داده‌شده.
        خروجی: DataFrame هر ردیف یک خوشه با مقادیر center/price, members, strength, touch_count, last_touch, timeframes """
        levels = np.array(sorted(set(float(l) for l in levels)))
        if levels.size == 0:
            return pd.DataFrame()
        # Compute ATR series and mean ATR as baseline
        atr_series = atr(prices_df, window=self.atr_window)
        mean_atr = float(atr_series.mean())
        # Use a global eps based on mean ATR to allow DBSCAN to run in 1D
        eps = max(1e-8, self.eps_atr_multiplier * mean_atr)
        # reshape for sklearn
        X = levels.reshape(-1, 1)
        db = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = db.fit_predict(X)
        clusters = []
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            members = list(levels[mask])
            if cluster_id == -1:
                # noise, treat each as its own cluster
                for m in members:
                    clusters.append({
                        "cluster_id": -1,
                        "price": float(m),
                        "members": [m],
                        "strength": 0.0,
                        "touch_count": 0,
                        "last_touch": None,
                        "timeframes": {timeframe: 1},
                    })
            else:
                center = float(np.mean(members))
                clusters.append({
                    "cluster_id": int(cluster_id),
                    "price": center,
                    "members": members,
                    "strength": float(len(members)),
                    "touch_count": 0,
                    "last_touch": None,
                    "timeframes": {timeframe: len(members)},
                })
        df_clusters = pd.DataFrame(clusters)
        return df_clusters

# -----------------------------
# Level analytics & historical metrics
# -----------------------------
class LevelAnalyzer:
    """محاسبهٔ متریک‌های تاریخی برای هر سطح مانند touch rate, hit rate, avg bounce.
    به عنوان ورودی: list of levels و price series (pandas Series close) برای محاسبهٔ metrics """
    def compute_level_stats(self, levels: Iterable[float], prices: pd.Series, lookahead_bars: int = 50, tolerance: float = 0.0) -> pd.DataFrame:
        """برای هر سطح، متریک‌هایی تولید می‌کند.
        metrics: level, touch_count, hit_count, hit_rate, avg_bounce """
        # Ensure numeric
        levels = sorted(set(float(l) for l in levels))
        if prices.empty:
            return pd.DataFrame()
        # tolerance can be absolute price or percent; assume absolute here
        records = []
        arr = prices.values
        times = prices.index
        n = len(arr)
        for level in levels:
            touches = 0
            hits = 0
            bounces = []
            times_to_hit = []
            # scan through candles and look for price crossing level
            for i in range(n - lookahead_bars):
                window = slice(i, i + lookahead_bars)
                segment_high = prices.iloc[i: i + lookahead_bars].max()
                segment_low = prices.iloc[i: i + lookahead_bars].min()
                if (segment_low - tolerance) <= level <= (segment_high + tolerance):
                    touches += 1
                    # whether immediate penetration: check if any price beyond level within lookahead
                    if level <= segment_high and level >= segment_low:
                        hits += 1
                        # bounce: magnitude after touch (if exists)
                        max_up = segment_high - level if segment_high > level else 0
                        max_down = level - segment_low if segment_low < level else 0
                        bounces.append(max(max_up, max_down))
                        times_to_hit.append(0)  # placeholder; could be improved with index of first touch
            record = {
                "level": float(level),
                "touch_count": int(touches),
                "hit_count": int(hits),
                "hit_rate": float(hits / touches) if touches > 0 else 0.0,
                "avg_bounce": float(np.mean(bounces)) if bounces else 0.0,
            }
            records.append(record)
        return pd.DataFrame(records)

# -----------------------------
# Order planning (entry/stop/target) helper
# -----------------------------
@dataclass
class OrderSpec:
    side: str  # 'buy'|'sell'
    entry_price: float
    stop_price: float
    take_profit: Optional[float]
    lot_size: float
    order_type: str = "limit"  # or 'market'
    meta: Dict[str, Any] = None

class OrderPlanner:
    """تبدیل یک سطح به یک برنامهٔ سفارش (entry/SL/TP/lot) با توجه به ریسک و ATR.
    پارامترها و فرمول‌ها قابل پیکربندی هستند.
    """
    def __init__(self, risk_pct: float = 0.5,  # درصد ریسک حساب برای هر ترید
                 account_equity: float = 10000.0,
                 sl_atr_mult: float = 1.0,
                 tick_size: Optional[float] = None,
                 min_lot: float = 0.01,
                 pip_value: Optional[float] = None):
        self.risk_pct = float(risk_pct)
        self.account_equity = float(account_equity)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tick_size = tick_size
        self.min_lot = min_lot
        self.pip_value = pip_value

    def _calc_lot_by_risk(self, sl_distance: float) -> float:
        """محاسبه لات بر اساس ریسک دلخواه و فاصله SL به قیمتِ ورود.
        به سادگی: lot = (equity * risk_pct) / (sl_distance * pip_value) نیاز است pip_value صحیح تنظیم شود برای دارایی‌هایی مثل XAUUSD ممکن است بر حسب واحد متفاوت باشد.
        """
        risk_amount = (self.risk_pct / 100.0) * self.account_equity
        if self.pip_value is None or self.pip_value == 0:
            # Cannot calculate accurately; fallback: return min_lot
            logger.warning("pip_value is not set; returning min_lot")
            return float(self.min_lot)
        lot = risk_amount / (abs(sl_distance) * self.pip_value)
        lot = max(float(self.min_lot), float(lot))
        return lot

    def make_entry_plan(self, level_price: float, side: str, price_now: float, atr_value: float, tp_atr_mult: float = 2.0, sl_buffer: float = 0.0, order_type: str = "limit") -> OrderSpec:
        """ساخت یک OrderSpec بر اساس level (به عنوان نقطهٔ ورودی یا مرجع).
        - side: 'buy' یا 'sell'
        - price_now: قیمت فعلی بازار
        - atr_value: مقدار ATR برای تعیین SL default
        - tp_atr_mult: نسبت TP نسبت به SL (مثلاً 2 برابر)
        - sl_buffer: مقدار فاصلهٔ اضافی از سطح بر حسب همان واحد قیمت """
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        # determine entry price; common tactic: place limit slightly better than level (configurable)
        entry_price = float(level_price)
        # determine sl distance using atr
        sl_distance = atr_value * self.sl_atr_mult + sl_buffer
        if side == "buy":
            stop_price = float(entry_price - sl_distance)
            take_profit = float(entry_price + sl_distance * tp_atr_mult) if tp_atr_mult else None
        else:
            stop_price = float(entry_price + sl_distance)
            take_profit = float(entry_price - sl_distance * tp_atr_mult) if tp_atr_mult else None
        lot_size = self._calc_lot_by_risk(abs(entry_price - stop_price))
        return OrderSpec(side=side, entry_price=entry_price, stop_price=stop_price, take_profit=take_profit, lot_size=lot_size, order_type=order_type, meta={"atr": atr_value, "sl_atr_mult": self.sl_atr_mult})

# -----------------------------
# News-aware filter (skeleton)
# -----------------------------
class NewsFilter:
    """اسکلت filtr کردن سطوح نزدیک به اخبار مهم.
    برای نسخهٔ اولیه، این کلاس فقط یک API ساده ارائه می‌دهد. بعداً می‌توان آن را به datafeed اخبار متصل کرد.
    """
    def __init__(self, lookahead_minutes: int = 60):
        self.lookahead_minutes = int(lookahead_minutes)

    def filter_levels_by_news(self, levels_df: pd.DataFrame, news_events: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        """news_events هر ایونت باید شامل {'time': Timestamp, 'impact': 'low'|'medium'|'high'}
        منطق: سطوحی که داخل window مربوطه به news با impact بالا هستند را تضعیف یا علامت‌گذاری می‌کنیم.
        """
        if levels_df.empty:
            return levels_df
        # Add column
        levels_df = levels_df.copy()
        levels_df["news_risk"] = 0
        for ev in news_events:
            ev_time = pd.to_datetime(ev["time"]) if not isinstance(ev["time"], pd.Timestamp) else ev["time"]
            impact = ev.get("impact", "low")
            window_start = ev_time - pd.Timedelta(minutes=self.lookahead_minutes)
            window_end = ev_time + pd.Timedelta(minutes=self.lookahead_minutes)
            weight = {"low": 1, "medium": 2, "high": 3}.get(impact, 1)
            mask = (levels_df["time"] >= window_start) & (levels_df["time"] <= window_end)
            levels_df.loc[mask, "news_risk"] = levels_df.loc[mask, "news_risk"].add(weight).fillna(weight)
        return levels_df

# -----------------------------
# Export / Visualization helpers
# -----------------------------
class Exporter:
    @staticmethod
    def to_json(levels_df: pd.DataFrame) -> str:
        return levels_df.to_json(orient="records", date_format="iso")

    @staticmethod
    def to_csv(levels_df: pd.DataFrame, path: str) -> None:
        levels_df.to_csv(path, index=False)

    @staticmethod
    def prepare_plot_annotations(levels_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """ایجاد ساختار annotation برای استفاده در plotly/mpl (سرور یا dashboard).
        هر annotation حاوی price, label, strength, color-hint است.
        """
        ann = []
        for _, r in levels_df.iterrows():
            ann.append({
                "price": float(r.get("price", 0.0)),
                "label": str(r.get("label", "fib")),
                "strength": float(r.get("strength", 0.0)),
            })
        return ann

# -----------------------------
# Feature extraction for RL
# -----------------------------
class FeatureExtractor:
    """تبدیل خروجی clusters/levels به فیچر‌های عددی مناسب برای مدل RL.
    نکات:
    - برای نرمال‌سازی از ATR استفاده می‌کنیم تا فیچرها در واحد volatility قرار بگیرند.
    - خروجی می‌تواند DataFrame (برای هر کلاستر) و یا بردار numpy (ترکیبی) باشد.
    """
    def __init__(self, atr_series: Optional[pd.Series] = None):
        self.atr_series = atr_series

    @staticmethod
    def _safe_get(series: pd.Series, idx: int, default: float = 0.0) -> float:
        try:
            return float(series.iloc[idx])
        except Exception:
            return default

    def extract_features_from_cluster_row(self, row: pd.Series, current_price: float, atr_value: float) -> Dict[str, float]:
        """گرفتن یک ردیف کلاستر و تولید فیچرهای کلیدی برای آن ردیف."""
        price = float(row.get("price", current_price))
        strength = float(row.get("strength", 0.0))
        hit_rate = float(row.get("hit_rate", 0.0)) if row.get("hit_rate") is not None else 0.0
        avg_bounce = float(row.get("avg_bounce", 0.0)) if row.get("avg_bounce") is not None else 0.0
        members = row.get("members", [])
        members_count = len(members) if isinstance(members, (list, tuple, np.ndarray)) else 1
        tf_count = len(row.get("timeframes", {})) if isinstance(row.get("timeframes", {}), dict) else 1
        # distance features (absolute and pct), normalized by ATR where possible
        abs_dist = price - current_price
        pct_dist = (abs_dist / current_price) if current_price != 0 else 0.0
        atr_norm_dist = (abs_dist / atr_value) if atr_value and atr_value != 0 else abs_dist
        # cluster width heuristic: max(member)-min(member)
        width = float(np.max(members) - np.min(members)) if members_count > 1 else 0.0
        width_norm = (width / atr_value) if atr_value and atr_value != 0 else width
        # final features dictionary
        features = {
            "price": price,
            "abs_dist": abs_dist,
            "pct_dist": pct_dist,
            "atr_norm_dist": atr_norm_dist,
            "strength": strength,
            "hit_rate": hit_rate,
            "avg_bounce": avg_bounce,
            "members_count": float(members_count),
            "tf_count": float(tf_count),
            "width": width,
            "width_norm": width_norm,
        }
        return features

    def extract_dataframe(self, clusters_df: pd.DataFrame, current_price: float, atr_value: float) -> pd.DataFrame:
        """تبدیل clusters_df به DataFrame فیچر‌دار (هر ردیف یک کلاستر)."""
        if clusters_df is None or clusters_df.empty:
            return pd.DataFrame()
        rows = []
        for _, r in clusters_df.iterrows():
            rows.append(self.extract_features_from_cluster_row(r, current_price, atr_value))
        df = pd.DataFrame(rows)
        # add some derived normalized features
        if "atr_norm_dist" not in df.columns and atr_value:
            df["atr_norm_dist"] = df["abs_dist"] / atr_value
        return df

    def to_vector(self, features_df: pd.DataFrame, top_k: int = 5) -> np.ndarray:
        """تبدیل DataFrame فیچرها به بردار ثابت‌طول مناسب برای RL.
        - strategy: انتخاب top_k بر اساس strength و سپس flatten ستون‌های مشخص.
        - اگر تعداد کلاستر کمتر باشد با صفر-padding تکمیل می‌کند.
        """
        if features_df is None or features_df.empty:
            return np.zeros(top_k * 6, dtype=float)  # default feature size (6 features per cluster)
        # choose key features and sort by strength
        features_df = features_df.sort_values("strength", ascending=False).reset_index(drop=True)
        chosen = features_df.head(top_k)
        vec = []
        for _, r in chosen.iterrows():
            vec.extend([
                float(r.get("atr_norm_dist", 0.0)),
                float(r.get("strength", 0.0)),
                float(r.get("hit_rate", 0.0)),
                float(r.get("avg_bounce", 0.0)),
                float(r.get("members_count", 0.0)),
                float(r.get("width_norm", 0.0)),
            ])
        # pad if needed
        needed = top_k * 6 - len(vec)
        if needed > 0:
            vec.extend([0.0] * needed)
        return np.array(vec, dtype=float)

# -----------------------------
# High-level orchestrator
# -----------------------------
class FibonacciEngine:
    """کلاس سطح بالا که ترکیب SwingDetector, FibonacciCore, LevelClusterer و OrderPlanner است.
    برای استفاده در ربات تریدر: این کلاس متدهایی برای تولید سطوح با وزن‌دهی و تولید پلان‌های معاملاتی ارائه می‌دهد. همچنین امکاناتی برای آپدیت لحظه‌ای، تولید فیچر برای RL و نرمال‌سازی بر اساس ATR را فراهم می‌کند.
    """
    def __init__(self, df_price: pd.DataFrame, fib_ratios: Optional[Iterable[float]] = None, atr_window: int = 14, cluster_eps_atr_mult: float = 0.5):
        _validate_price_index(df_price)
        self.df = df_price.copy()
        self.core = FibonacciCore(ratios=fib_ratios)
        self.swing_detector = SwingDetector()
        self.clusterer = LevelClusterer(eps_atr_multiplier=cluster_eps_atr_mult, atr_window=atr_window)
        self.analyzer = LevelAnalyzer()
        self.news_filter = NewsFilter()
        self.atr_window = atr_window
        self.atr_series = atr(self.df, window=atr_window)
        self.feature_extractor = FeatureExtractor(atr_series=self.atr_series)
        # cache latest computed sets
        self._last_raw_levels: Optional[pd.DataFrame] = None
        self._last_clusters: Optional[pd.DataFrame] = None

    def generate_raw_levels(self) -> pd.DataFrame:
        """تشخیص سوئینگ و تولید سطوح فیبوناچی برای هر جفت swing (peak->valley و بالعکس).
        خروجی: DataFrame با سطرهایی از نوع: time, level_price, ratio, type """
        swings = self.swing_detector.detect_swings(self.df)
        records = []
        # iterate in ordered swings, pair consecutive swings as moves
        for i in range(len(swings) - 1):
            s1 = swings.iloc[i]
            s2 = swings.iloc[i + 1]
            start_price = s1["price"]
            end_price = s2["price"]
            # compute retracements and extensions for the move start->end
            retr_df = self.core.retracement(start_price, end_price)
            ext_df = self.core.extension(start_price, end_price)
            t = s2["time"]
            for _, r in retr_df.iterrows():
                records.append({"time": t, "level_price": r["price"], "ratio": r["ratio"], "type": r["type"]})
            for _, r in ext_df.iterrows():
                records.append({"time": t, "level_price": r["price"], "ratio": r["ratio"], "type": r["type"]})
        df_levels = pd.DataFrame(records)
        if not df_levels.empty:
            df_levels["time"] = pd.to_datetime(df_levels["time"])
            df_levels = df_levels.drop_duplicates(subset=["level_price", "ratio", "type"]).reset_index(drop=True)
        self._last_raw_levels = df_levels
        return df_levels

    def cluster_and_score(self, df_levels: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
        """خوشه‌بندی سطوح و اضافه کردن امتیاز strength بر اساس members و historical hits.
        خروجی: DataFrame خوشه‌ها با ستون price, strength, touch_count, last_touch.
        """
        if df_levels is None or df_levels.empty:
            self._last_clusters = pd.DataFrame()
            return pd.DataFrame()
        # cluster
        clusters_df = self.clusterer.cluster_levels(df_levels["level_price"].values, self.df, timeframe=timeframe)
        # compute historical metrics for cluster centers
        if clusters_df.empty:
            self._last_clusters = clusters_df
            return clusters_df
        stats = self.analyzer.compute_level_stats(clusters_df["price"].values, self.df["close"], lookahead_bars=50, tolerance=0.0)
        # merge
        if stats.empty:
            clusters_df["touch_count"] = 0
            clusters_df["hit_rate"] = 0.0
            clusters_df["avg_bounce"] = 0.0
        else:
            clusters_df = clusters_df.merge(stats, how="left", left_on="price", right_on="level")
            clusters_df["hit_rate"] = clusters_df["hit_rate"].fillna(0.0)
            clusters_df["avg_bounce"] = clusters_df["avg_bounce"].fillna(0.0)
        # compute composite strength
        clusters_df["strength"] = clusters_df["strength"] * (1.0 + clusters_df["hit_rate"])  # simple heuristic
        clusters_df = clusters_df.sort_values("strength", ascending=False).reset_index(drop=True)
        self._last_clusters = clusters_df
        return clusters_df

    def generate_entry_plans(self, clusters_df: pd.DataFrame, planner: OrderPlanner, lookback_atr: int = 14) -> List[OrderSpec]:
        """برای هر کلاستر، یک پلان سفارش تولید می‌کند (Buy if price > current? heuristics apply).
        این متد یک طرح اولیه می‌دهد. منطق side/confirmation باید در سطح strategy engine اعمال شود.
        """
        plans = []
        price_now = float(self.df["close"].iloc[-1])
        for _, row in clusters_df.iterrows():
            level_price = float(row["price"])
            # decide side: if level above current price -> sell resistance else buy support
            side = "sell" if level_price > price_now else "buy"
            # use recent ATR value
            atr_value = float(self.atr_series.iloc[-1]) if not self.atr_series.empty else 0.0
            plan = planner.make_entry_plan(level_price=level_price, side=side, price_now=price_now, atr_value=atr_value)
            plans.append(plan)
        return plans

    # -----------------------------
    # Real-time / incremental updates
    # -----------------------------
    def real_time_update(self, new_bar: Dict[str, Any]) -> None:
        """آپدیت دیتافریم قیمت با یک کندل/تیک جدید و بروز کردن ATR cache.
        new_bar باید شامل keys: ['time','open','high','low','close']، time قابل parse باشد. این متد دیتافریم را append کرده و atr_series را مجدداً محاسبه می‌کند.
        """
        # Validate input minimality
        try:
            t = pd.to_datetime(new_bar["time"])
            o = float(new_bar["open"])
            h = float(new_bar["high"])
            l = float(new_bar["low"])
            c = float(new_bar["close"])
        except Exception as e:
            logger.error("Invalid new_bar provided to real_time_update: %s", e)
            raise

        # Append or update final row if timestamp exists
        if t in self.df.index:
            self.df.loc[t, ["open", "high", "low", "close"]] = [o, h, l, c]
        else:
            # append
            row = pd.DataFrame([{"open": o, "high": h, "low": l, "close": c}], index=pd.DatetimeIndex([t]))
            self.df = pd.concat([self.df, row]).sort_index()
        # recompute atr_series tail (simple approach)
        self.atr_series = atr(self.df, window=self.atr_window)
        # invalidate caches
        self._last_raw_levels = None
        self._last_clusters = None

    # -----------------------------
    # Feature extraction helpers for RL
    # -----------------------------
    def compute_features_dataframe(self, clusters_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """برگرداندن DataFrame فیچرها برای تمام کلاسترها؛ اگر clusters_df None باشد از کش استفاده می‌کند."""
        if clusters_df is None:
            if self._last_clusters is None:
                # auto-generate clusters from raw levels
                raw = self._last_raw_levels if self._last_raw_levels is not None else self.generate_raw_levels()
                clusters_df = self.cluster_and_score(raw)
            else:
                clusters_df = self._last_clusters
        price_now = float(self.df["close"].iloc[-1])
        atr_value = float(self.atr_series.iloc[-1]) if not self.atr_series.empty else 0.0
        return self.feature_extractor.extract_dataframe(clusters_df, price_now, atr_value)

    def get_feature_vector_for_rl(self, clusters_df: Optional[pd.DataFrame] = None, top_k: int = 5) -> np.ndarray:
        """برگرداندن بردار عددی ثابت‌طول مناسب برای مدل RL."""
        df_feat = self.compute_features_dataframe(clusters_df)
        return self.feature_extractor.to_vector(df_feat, top_k=top_k)

    # -----------------------------
    # Golden zone quick helper (simple)
    # -----------------------------
    def detect_golden_zones(self, clusters_df: Optional[pd.DataFrame] = None, zone_width_atr_mult: float = 0.5) -> pd.DataFrame:
        """تشخیص سادهٔ Golden Zones براساس clusters: تبدیل هر کلاستر به zone around center با پهنای k*ATR."""
        if clusters_df is None:
            clusters_df = self._last_clusters if self._last_clusters is not None else self.cluster_and_score(self.generate_raw_levels())
        if clusters_df is None or clusters_df.empty:
            return pd.DataFrame()
        atr_value = float(self.atr_series.iloc[-1]) if not self.atr_series.empty else 0.0
        zones = []
        for _, r in clusters_df.iterrows():
            center = float(r["price"])
            half = zone_width_atr_mult * atr_value
            zones.append({"low": center - half, "high": center + half, "center": center, "strength": float(r.get("strength", 0.0))})
        return pd.DataFrame(zones).sort_values("strength", ascending=False).reset_index(drop=True)

# -----------------------------
# End of module
# -----------------------------
# NOTE:
# - این پیاده‌سازی روی readability و extensibility تمرکز دارد، نه حداکثر بهینه‌سازی سرعت.
# - برای محیط‌های با نیاز latency خیلی کم، باید بخش‌های detect_swings و cluster_levels برداری و یا به Cython/Numba منتقل شوند.
# - برای محاسبهٔ دقیق lot روی XAUUSD یا جفت‌های مختلف، pip_value و tick_size باید بر اساس قرارداد بروکر تنظیم شوند.
# - FeatureExtractor.to_vector strategy و feature set را می‌توان متناسب با مدل RL (مثلاً ورود continuous/ discrete) تغییر داد.
# - این نسخه همهٔ متدهای اصلی قبلی را حفظ کرده و قابلیت‌های feature-extraction و real-time update را به آن افزوده است.

# ===== END FILE: f04_features/fibonacci_3.py =====


# ===== BEGIN FILE: f04_features/fibonacci.py =====
# f04_features/fibonacci.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Iterable, Union, Any
import numpy as np
import pandas as pd

# =========================
# Core: محاسبات فیبوناچی
# =========================
class FibonacciCore:
    """هسته‌ی محاسبات فیبوناچی با خروجی یکنواخت DataFrame."""
    RETRACE_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTEND_RATIOS = [1.272, 1.618, 2.0, 2.618]  # سطح 127.2% یعنی (r-1)=0.272
    PROJECT_RATIOS = [0.618, 1.0, 1.618]  # ABC projection

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
        """ محاسبهٔ Retracement برای چند تایم‌فریم.
L1: swings = { "H1": (high, low), "H4": (high, low), ...
L2: } """
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
    """ نسخهٔ یکپارچه با حفظ تمام ۷ متد (همگی خروجی DataFrame):
    1) detect_swings
    2) fib_levels (retracement)
    3) fib_extension (قبلاً fib_projection نامیده بودی اما کار extension می‌کرد)
    4) cluster_zones
    5) golden_zone_distance
    6) risk_reward
    7) filter_levels_by_news
    """
    def __init__(self, price_data: pd.DataFrame, news_data: Optional[pd.DataFrame] = None):
        """ price_data: DataFrame با ستون‌های ['Open','High','Low','Close'] و index زمانی
            news_data: (اختیاری) DataFrame با ستون‌های ['time','impact']، time از نوع datetime64 """
        self.df = price_data.copy()
        self.news = news_data.copy() if news_data is not None else None
        # سازگاری نام ستون‌ها
        cols = {c.lower(): c for c in self.df.columns}
        # اگر حروف کوچک بودند:
        if set(["open", "high", "low", "close"]).issubset(cols.keys()):
            self.df.rename(columns={cols["open"]: "Open", cols["high"]: "High", cols["low"]: "Low", cols["close"]: "Close"}, inplace=True)
        if self.news is not None and "time" in self.news.columns:
            self.news["time"] = pd.to_datetime(self.news["time"])

    # ------------------------------
    # 1) Auto Swing Detection
    # ------------------------------
    def detect_swings(self, window: int = 3) -> pd.DataFrame:
        """ تشخیص سادهٔ Swing High/Low به روش فراکتال.
        خروجی: DataFrame با ستون‌های: ['index','time','price','kind'] kind ∈ {'high','low'} """
        highs = self.df["High"].values
        lows = self.df["Low"].values
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
        """ محاسبهٔ سطوح بازگشتی (خروجی DataFrame).
L4: Wrapper روی FibonacciCore.retracement """
        return FibonacciCore.retracement(swing_high, swing_low, context=context)

    # ------------------------------
    # 3) Fibonacci Extension (DF)
    # ------------------------------
    def fib_extension(self, swing_high: float, swing_low: float, *, context: Optional[dict] = None) -> pd.DataFrame:
        """ سطوح امتدادی (targets).
L5: (توجه: نام قبلی تو 'fib_projection' بود اما محاسبهٔ extension انجام می‌داد) """
        return FibonacciCore.extension(swing_high, swing_low, context=context)

    # ------------------------------
    # 4) Cluster Detector (DF)
    # ------------------------------
    def cluster_zones(
        self,
        fib_level_frames: Iterable[pd.DataFrame],
        tolerance: float = 0.002,
        extra_signals: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """ یافتن نواحی خوشه‌ای فیبوناچی + محاسبه Confluence Score.
        Parameters
        ----------
        fib_level_frames : iterable of pd.DataFrame لیستی از DataFrameها که شامل ستون "price" و متادیتا مثل "ratio","tf","type" هستند. مثال: خروجی متدهای retracement / extension / projection.
        tolerance : float آستانه خوشه‌بندی نسبی (۰.۰۰۲ یعنی ۰.۲٪).
        extra_signals : pd.DataFrame, optional اندیکاتورهای اضافی (MA/RSI/Support/Resistance). باید ستونی به نام "price" داشته باشد.
        weights : dict, optional وزن معیارها برای محاسبه نمره.
        اگر None، مقدار پیش‌فرض زیر استفاده می‌شود: { "count": 0.25, "timeframes": 0.25, "golden_ratio": 0.25, "density": 0.15, "extra_signals": 0.10 }
        Returns
        -------
        pd.DataFrame شامل ستون‌های: [ cluster_id, cluster_low, cluster_high, center, count, unique_timeframes, width, score, type ]
        """
        # -----------------------------
        # وزن‌های پیش‌فرض
        # -----------------------------
        if weights is None:
            weights = {
                "count": 0.25,
                "timeframes": 0.25,
                "golden_ratio": 0.25,
                "density": 0.15,
                "extra_signals": 0.10,
            }
        # -----------------------------
        # جمع‌آوری داده‌ها
        # -----------------------------
        levels = []
        for df in fib_level_frames:
            if not isinstance(df, pd.DataFrame) or "price" not in df.columns:
                continue
            for _, row in df.iterrows():
                levels.append(
                    {
                        "price": row["price"],
                        "ratio": row.get("ratio", None),
                        "tf": row.get("tf", None),
                        "type": row.get("type", None),
                    }
                )
        if not levels:
            return pd.DataFrame(
                columns=[
                    "cluster_id",
                    "cluster_low",
                    "cluster_high",
                    "center",
                    "count",
                    "unique_timeframes",
                    "width",
                    "score",
                    "type",
                ]
            )
        # -----------------------------
        # خوشه‌بندی بر اساس tolerance
        # -----------------------------
        prices = sorted([lvl["price"] for lvl in levels])
        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if abs(p - current[-1]) / current[-1] <= tolerance:
                current.append(p)
            else:
                if len(current) >= 2:
                    clusters.append(current)
                current = [p]
        if len(current) >= 2:
            clusters.append(current)
        if not clusters:
            return pd.DataFrame(
                columns=[
                    "cluster_id",
                    "cluster_low",
                    "cluster_high",
                    "center",
                    "count",
                    "unique_timeframes",
                    "width",
                    "score",
                    "type",
                ]
            )
        # -----------------------------
        # محاسبه متریک‌ها برای هر خوشه
        # -----------------------------
        out_rows: List[dict] = []
        for cid, cluster in enumerate(clusters, start=1):
            c_low = min(cluster)
            c_high = max(cluster)
            center = float(np.mean(cluster))
            width = c_high - c_low
            # فیلتراسیون داده‌های مربوط به این خوشه
            related = [lvl for lvl in levels if c_low <= lvl["price"] <= c_high]
            count = len(related)
            unique_timeframes = len(set([lvl["tf"] for lvl in related if lvl["tf"]]))
            # نزدیکی به سطح طلایی 0.618
            golden_level = 0.618
            golden_distance = min([abs(lvl["ratio"] - golden_level) for lvl in related if lvl["ratio"] is not None] or [1.0])
            golden_score = 1 - golden_distance  # هرچه نزدیک‌تر به 0.618 → بهتر
            # تراکم قیمتی (خوشه فشرده‌تر بهتر است)
            density_score = 1 / (1 + width / center)
            # بررسی هم‌پوشانی با extra_signals
            extra_score = 0
            if extra_signals is not None and "price" in extra_signals.columns:
                for sig_price in extra_signals["price"]:
                    if c_low <= sig_price <= c_high:
                        extra_score = 1
                        break
            # نرمال‌سازی و محاسبه score نهایی
            score = (
                weights["count"] * min(count / 10, 1.0)
                + weights["timeframes"] * min(unique_timeframes / 5, 1.0)
                + weights["golden_ratio"] * golden_score
                + weights["density"] * density_score
                + weights["extra_signals"] * extra_score
            )
            out_rows.append(
                {
                    "cluster_id": cid,
                    "cluster_low": c_low,
                    "cluster_high": c_high,
                    "center": center,
                    "count": count,
                    "unique_timeframes": unique_timeframes,
                    "width": width,
                    "score": round(score * 100, 2),  # نمره نهایی ۰–۱۰۰
                    "type": "cluster",
                }
            )
        return pd.DataFrame(out_rows)

    # ------------------------------
    # 5) Golden Zone Distance (DF)
    # ------------------------------
    def golden_zone_distance(
        self,
        price: float,
        *,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        retracement_df: Optional[pd.DataFrame] = None,
        ratios: Iterable[float] = (0.618, 0.786),
        normalize: str = "relative",  # "relative" or "absolute"
        level_name_fmt: str = "R_{:.3f}",
        ratio_match_tol: float = 1e-3,
    ) -> pd.DataFrame:
        """ محاسبهٔ فاصلهٔ قیمت تا \"Golden Zone\" تعریف‌شده توسط مجموعه‌ای از نسبت‌های فیبوناچی.
        رفتار:
        - اگر `retracement_df` داده شود، سعی می‌کند قیمتِ نسبت‌ها را از آن استخراج کند.
        - پذیرفته می‌شود که DataFrame شامل ستون‌های ('level_name','price') باشد یا شامل ستون‌های ('ratio','price') باشد.
        - اگر نسبت خاصی در retracement_df پیدا نشد و swing_high/low موجود باشد، آن نسبت را با فرمول مستقیم محاسبه می‌کند.
        - در صورتی که retracement_df داده نشده باشد ولی swing_high/low فراهم باشد، قیمت‌ها از high/low محاسبه می‌شوند.
        - اگر برای یک نسبت نه retracement و نه swing وجود داشته باشد، خطا می‌دهد.
        خروجی: یک DataFrame تک‌ردیفه شامل فیلدهای: ['g_low','g_high','center','price','distance_abs','distance_rel','in_zone', 'nearest_level','nearest_ratio','nearest_dist_abs','nearest_dist_rel','members']
        members: لیستی از دیکشنری‌هایی که منابع هر نسبت را نشان می‌دهد (مثلاً {'ratio':0.618,'price':1.2345,'source':'retracement'|'calc'})
        """
        # تبدیل ratios به لیست و اعتبارسنجی
        ratio_list = list(ratios)
        if not ratio_list:
            raise ValueError("`ratios` باید شامل حداقل یک نسبت فیبوناچی باشد.")

        # helper: امن‌سازی ترتیب high/low
        def _ensure_order(h: float, l: float) -> Tuple[float, float]:
            return (h, l) if h >= l else (l, h)

        # helper: محاسبه قیمت از high/low برای یک نسبت
        def _price_at_ratio_from_swings(r: float, high: float, low: float) -> float:
            return high - r * (high - low)

        # تلاش برای استخراج قیمت‌ها از retracement_df (در صورت موجود بودن)
        ratio_prices: Dict[float, float] = {}
        members = []
        if retracement_df is not None and not retracement_df.empty:
            df = retracement_df.copy()
            # normalize column names expectation
            # case 1: df has 'ratio' and 'price'
            if 'ratio' in df.columns and 'price' in df.columns:
                # convert ratios to float keys for matching
                for r in ratio_list:
                    # exact match on ratio column (allow small tol)
                    cand = df[ (df['ratio'].astype(float) - float(r)).abs() <= ratio_match_tol ]
                    if not cand.empty:
                        p = float(cand.iloc[0]['price'])
                        ratio_prices[r] = p
                        members.append({'ratio': r, 'price': p, 'source': 'retracement'})
            # case 2: df has 'level_name' and 'price' (e.g. 'R_0.618')
            if 'level_name' in df.columns and 'price' in df.columns:
                # attempt to parse float from level_name when possible
                # try two formats: exact formatted level_name_fmt or extract last numeric part
                for r in ratio_list:
                    target_name = level_name_fmt.format(r)
                    cand = df[df['level_name'] == target_name]
                    if not cand.empty:
                        p = float(cand.iloc[0]['price'])
                        ratio_prices[r] = p
                        # avoid duplicate member entry if already recorded
                        if not any(m['ratio'] == r for m in members):
                            members.append({'ratio': r, 'price': p, 'source': 'retracement'})

            # fallback: try to extract numeric suffix from level_name and match by tolerance
            if len(ratio_prices) < len(ratio_list):
                def _extract_ratio_from_name(name: str):
                    import re
                    nums = re.findall(r"[0-9]*\.?[0-9]+", str(name))
                    if nums:
                        try:
                            return float(nums[-1])
                        except Exception:
                            return None
                    return None
                for _, row in df[['level_name','price']].iterrows():
                    parsed = _extract_ratio_from_name(row['level_name'])
                    if parsed is None:
                        continue
                    for r in ratio_list:
                        if r in ratio_prices:
                            continue
                        if abs(parsed - r) <= ratio_match_tol:
                            p = float(row['price'])
                            ratio_prices[r] = p
                            members.append({'ratio': r, 'price': p, 'source': 'retracement'})
                            break

        # اگر برای بعضی نسبت‌ها قیمت نداریم، و swing_high/low موجود است، محاسبه کن
        if (len(ratio_prices) < len(ratio_list)) and (swing_high is not None and swing_low is not None):
            high, low = _ensure_order(swing_high, swing_low)
            for r in ratio_list:
                if r in ratio_prices:
                    continue
                p = float(_price_at_ratio_from_swings(r, high, low))
                ratio_prices[r] = p
                members.append({'ratio': r, 'price': p, 'source': 'calc'})

        # اگر هنوز برای بعضی نسبت‌ها مقدار نداریم -> خطا
        missing = [r for r in ratio_list if r not in ratio_prices]
        if missing:
            raise ValueError(f"نتوانستم قیمت برای نسبت‌های {missing} بیابم.\nلطفاً retracement_df یا swing_high/swing_low را تأمین کن.")

        # محاسبهٔ ناحیهٔ طلایی
        prices = list(ratio_prices.values())
        g_low = float(min(prices))
        g_high = float(max(prices))
        center = (g_low + g_high) / 2.0

        # فاصلهٔ مطلق و تشخیص درون/بیرون بودن
        if price < g_low:
            distance_abs = g_low - price
            in_zone = False
        elif price > g_high:
            distance_abs = price - g_high
            in_zone = False
        else:
            distance_abs = 0.0
            in_zone = True

        # فاصلهٔ نسبی (نسبت به center) یا absolute
        if normalize == "absolute":
            distance_rel = float(distance_abs)
        else:
            distance_rel = float(distance_abs / center) if center != 0 else float("nan")

        # نزدیک‌ترین سطح (از بین ratio_list)
        nearest_ratio = None
        nearest_level_price = None
        nearest_dist_abs = None
        nearest_dist_rel = None
        if ratio_prices:
            nearest_ratio = min(ratio_prices.keys(), key=lambda r: abs(ratio_prices[r] - price))
            nearest_level_price = float(ratio_prices[nearest_ratio])
            nearest_dist_abs = abs(nearest_level_price - price)
            nearest_dist_rel = (nearest_dist_abs / nearest_level_price) if nearest_level_price != 0 else float("nan")

        # جمع‌بندی خروجی — members را هم به‌عنوان تاریخچه/منبع برمی‌گردانیم
        out = {
            "g_low": g_low,
            "g_high": g_high,
            "center": float(center),
            "price": float(price),
            "distance_abs": float(distance_abs),
            "distance_rel": float(distance_rel),
            "in_zone": bool(in_zone),
            "nearest_level": nearest_level_price,
            "nearest_ratio": float(nearest_ratio) if nearest_ratio is not None else None,
            "nearest_dist_abs": float(nearest_dist_abs) if nearest_dist_abs is not None else None,
            "nearest_dist_rel": float(nearest_dist_rel) if nearest_dist_rel is not None else None,
            "members": members,  # useful for debugging / downstream weighting
            "type": "golden_zone",
        }
        return pd.DataFrame([out])

    # ------------------------------
    # 6) Risk/Reward Tool (DF)
    # ------------------------------
    def risk_reward(self, entry: float, stop: float, target: float) -> pd.DataFrame:
        """ نسبت ریسک به ریوارد (RR).
L12: خروجی: ['entry','stop','target','rr','type'] """
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = float(np.inf) if risk == 0 else float(reward / risk)
        return pd.DataFrame([{
            "entry": entry,
            "stop": stop,
            "target": target,
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
        """ اگر نزدیک خبر مهم باشیم، شدت (strength) سطوح را 'weak' علامت بزن.
        ورودی: fib_levels_df باید ستون‌های ['level_name','price','type'] داشته باشد.
        خروجی: همان DataFrame با ستونِ جدید 'strength' در {'strong','weak'} """
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
        "Open": np.random.normal(1950, 5, size=50),
        "High": np.random.normal(1953, 5, size=50),
        "Low": np.random.normal(1947, 5, size=50),
        "Close": np.random.normal(1950, 5, size=50),
    }, index=idx)
    price["High"] = price[["Open","High","Low","Close"]].max(axis=1)
    price["Low"] = price[["Open","High","Low","Close"]].min(axis=1)
    news = pd.DataFrame({
        "time": pd.to_datetime(["2025-08-01 10:00", "2025-08-02 14:30"]),
        "impact": ["high", "medium"]
    })
    analyzer = FibonacciAnalyzer(price, news)
    swings = analyzer.detect_swings(window=2)
    retr = analyzer.fib_levels(2000, 1900, context={"src": "H1"})
    ext = analyzer.fib_extension(2000, 1900, context={"src": "H1"})
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
                lows = min([p for t,_,p in swings if t=="low"])
                highs = max([p for t,_,p in swings if t=="high"])
                if not highs or not lows:
                    continue
                # یا ثبت لاگ و عبور از این timeframe
                low = min(lows)
                high = max(highs)
                fib = FibonacciCore.Retracement(high=high, low=low, context={"timeframe": tf}) #.levels()
                self.results[tf] = fib
        return self.results

'''
# ============================================================
# Schema reference (برای هماهنگی با کدهای قبلی)
# ------------------------------------------------------------
# انتظار می‌رود DataFrame سطوح فیبوناچی (levels_df) این ستون‌ها را داشته باشد:
# - timeframe: str (مثل "M15", "H1", "H4", "D1")
# - method: str ("retracement" | "extension" | "projection")
# - ratio: float (مثل 0.618 یا 1.618)
# - price: float (سطح قیمتی محاسبه‌شده)
# - leg_id: Optional[str] (شناسه‌ی موج/سوییگ؛ اختیاری اما مفید)
# - meta: Optional[dict] (اطلاعات اضافه؛ اختیاری)
#
# نمونه‌ی تولید levels_df:
# df = FibonacciCore.retracement(high=H, low=L, context={"timeframe": "H1", "leg_id": "swing_12"})
# سپس می‌توانی ستون‌های "timeframe" و "leg_id" را از context به df اضافه کنی.
# ============================================================
'''
@dataclass
class ClusterParams:
    """ پیکربندی انعطاف‌پذیر برای تشخیص خوشه‌های همپوشان (Confluence Zones).
    - دو نوع تلورانس را پشتیبانی می‌کند: نسبی (rel) بر حسب درصد/بِیسیس‌پوینت و مطلق (abs) بر حسب قیمت/پیپ.
    - می‌توان حداقل تعداد برخورد (min_hits) تعیین کرد تا خوشه‌های ضعیف فیلتر شوند.
    - وزن‌دهی اختیاری بر اساس method/ratio/timeframe برای محاسبه‌ی امتیاز.
    """
    tolerance_mode: str = "rel"  # "rel" یا "abs"
    rel_tolerance: float = 0.001  # 0.001 ≈ 0.1% (100 bps)
    abs_tolerance: Optional[float] = None  # اگر tolerance_mode="abs"، مثلاً 1.0 برای 1 دلار/پیپ
    min_hits: int = 2  # حداقل تعداد سطوح در یک خوشه
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
    # امتیازدهی به تایم‌فریم‌ها (اختیاری).
    # اگر خالی باشد، همه 1.0.
    timeframe_weights: Dict[str, float] = field(default_factory=dict)

# ============================================================
# END FILE: f04_features/fibonacci.py
# ===== END MERGED FILE =====
