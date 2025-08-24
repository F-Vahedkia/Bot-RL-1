"""
fibonacci.py

یک ماژول جامع و توسعه‌پذیر فیبوناچی برای ربات تریدر

ویژگی‌ها:
- تشخیص سوئینگ (پیک/والی) با متد ZigZag/peak detection (پارامترهای prominence & distance)
- محاسبهٔ سطوح retracement/extension/projection با لیست قابل پیکربندی نسبت‌ها
- خوشه‌بندی (clustering) سطوح با DBSCAN و نرمال‌سازی فاصله بر اساس ATR
- محاسبهٔ ناحیهٔ Gold/Golden zone با tolerance تنظیم‌شونده مبتنی بر ATR
- متدهای تولید پلان ورود (entry plan) شامل تعیین SL/TP و اندازهٔ پوزیشن بر اساس risk %
- ابزارهایی برای ارزیابی تاریخی سطوح (touch/hit metrics)
- فیلتر اسکلت برای اخبار (news-aware) و جایگاه برای گسترش
- export برای رسم و ذخیرهٔ سطوح
- قابلیت‌هایی برای تولید فیچرهای آمادهٔ RL و آپدیت لحظه‌ای (real_time_update)

Dependencies:
- numpy, pandas, scipy, sklearn

طراحی: ماژولار، تایپ شده، مستندسازی‌شده و همراه با comments برای گسترش آینده.

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
    """محاسبه ATR ساده. خروجی بر اساس close-index aligned.

    پارامترها:
        df: DataFrame با ستون‌های high، low، close
        window: طول پنجره
    """
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

    خروجی تابع detect_swings: DataFrame با ستون‌های ['time','idx','price','kind','prominence','confidence']
    where kind in {'peak','valley'}

    پارامترهای قابل تنظیم:
    - method: 'peaks' یا 'zigzag'
    - min_prominence: کمترین prominence برای پذیرفتن پیک
    - min_distance_bars: حداقل فاصله بین سوئینگ‌ها بر حسب کندل
    - use_wicks: آیا از wick (high/low) استفاده شود یا body (close)
    """

    def __init__(self,
                 method: str = "peaks",
                 min_prominence: Optional[float] = None,
                 min_distance_bars: int = 3,
                 use_wicks: str = "wicks"):  # 'wicks'|'bodies'|'close'
        self.method = method
        self.min_prominence = min_prominence
        self.min_distance_bars = max(1, int(min_distance_bars))
        self.use_wicks = use_wicks

    def _price_series(self, df: pd.DataFrame) -> pd.Series:
        # NOTE: For peaks we use 'close' series as baseline. If wicks logic needed,
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
          - confidence (0..1 normalized)
        """
        _validate_price_index(df)

        price_series = self._price_series(df)
        price = price_series.values

        # Determine prominence threshold if not set
        if self.min_prominence is None:
            # Heuristic: use std of price scaled
            self.min_prominence = max(1e-8, float(np.nanstd(price) * 0.5))

        # Detect peaks and valleys
        peaks, props_peaks = find_peaks(price,
                                        distance=self.min_distance_bars,
                                        prominence=self.min_prominence)
        valleys, props_valleys = find_peaks(-price,
                                           distance=self.min_distance_bars,
                                           prominence=self.min_prominence)

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
    """توابع پایهٔ محاسبه سطوح فیبوناچی

    متدها:
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
        خروجی: DataFrame columns = ['ratio','price','type']
        """
        records = []
        for r in self.ratios:
            frac = r / 100.0
            # convention: level = end + (start - end) * (1 - frac)
            level = end_price + (start_price - end_price) * (1 - frac)
            records.append({"ratio": float(r), "price": float(level), "type": "retracement"})
        return pd.DataFrame(records)

    def extension(self, start_price: float, end_price: float, *, context: Optional[Dict] = None) -> pd.DataFrame:
        """محاسبه سطوح extension نسبت به بازهٔ start->end.
        خروجی: DataFrame columns = ['ratio','price','type']
        """
        records = []
        for r in self.ratios:
            factor = r / 100.0
            level = end_price + (end_price - start_price) * factor
            records.append({"ratio": float(r), "price": float(level), "type": "extension"})
        return pd.DataFrame(records)

    def projection(self, a: float, b: float, c: float, *, ratios: Optional[Iterable[float]] = None,
                   context: Optional[Dict] = None) -> pd.DataFrame:
        """Projection typical for harmonic patterns: project move BC based on AB length.
        returns DataFrame ['ratio','price','type']
        """
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

    ایده: eps = eps_atr_multiplier * ATR_at_level
    """

    def __init__(self, eps_atr_multiplier: float = 0.5, min_samples: int = 1, atr_window: int = 14):
        self.eps_atr_multiplier = eps_atr_multiplier
        self.min_samples = max(1, int(min_samples))
        self.atr_window = atr_window

    def _compute_eps(self, price: float, atr_value: float) -> float:
        return float(max(1e-8, self.eps_atr_multiplier * atr_value))

    def cluster_levels(self,
                       levels: Iterable[float],
                       prices_df: pd.DataFrame,
                       timeframe: str = "1m") -> pd.DataFrame:
        """خوشه‌بندی یک لیست از سطوح نسبت به دیتافریم قیمت داده‌شده.

        خروجی: DataFrame هر ردیف یک خوشه با مقادیر center/price, members, strength, touch_count, last_touch, timeframes
        """
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

    به عنوان ورودی: list of levels و price series (pandas Series close) برای محاسبهٔ metrics
    """

    def compute_level_stats(self,
                            levels: Iterable[float],
                            prices: pd.Series,
                            lookahead_bars: int = 50,
                            tolerance: float = 0.0) -> pd.DataFrame:
        """برای هر سطح، متریک‌هایی تولید می‌کند.

        metrics: level, touch_count, hit_count, hit_rate, avg_bounce
        """
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

    def __init__(self,
                 risk_pct: float = 0.5,  # درصد ریسک حساب برای هر ترید
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
        """محاسبه لات بر اساس ریسک دلخواه و فاصله SL به قیمتِ ورود. به سادگی: lot = (equity * risk_pct) / (sl_distance * pip_value)

        نیاز است pip_value صحیح تنظیم شود برای دارایی‌هایی مثل XAUUSD ممکن است بر حسب واحد متفاوت باشد.
        """
        risk_amount = (self.risk_pct / 100.0) * self.account_equity
        if self.pip_value is None or self.pip_value == 0:
            # Cannot calculate accurately; fallback: return min_lot
            logger.warning("pip_value is not set; returning min_lot")
            return float(self.min_lot)
        lot = risk_amount / (abs(sl_distance) * self.pip_value)
        lot = max(float(self.min_lot), float(lot))
        return lot

    def make_entry_plan(self,
                        level_price: float,
                        side: str,
                        price_now: float,
                        atr_value: float,
                        tp_atr_mult: float = 2.0,
                        sl_buffer: float = 0.0,
                        order_type: str = "limit") -> OrderSpec:
        """ساخت یک OrderSpec بر اساس level (به عنوان نقطهٔ ورودی یا مرجع).

        - side: 'buy' یا 'sell'
        - price_now: قیمت فعلی بازار
        - atr_value: مقدار ATR برای تعیین SL default
        - tp_atr_mult: نسبت TP نسبت به SL (مثلاً 2 برابر)
        - sl_buffer: مقدار فاصلهٔ اضافی از سطح بر حسب همان واحد قیمت
        """
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

        return OrderSpec(side=side,
                         entry_price=entry_price,
                         stop_price=stop_price,
                         take_profit=take_profit,
                         lot_size=lot_size,
                         order_type=order_type,
                         meta={"atr": atr_value, "sl_atr_mult": self.sl_atr_mult})


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

    برای استفاده در ربات تریدر: این کلاس متدهایی برای تولید سطوح با وزن‌دهی و تولید پلان‌های معاملاتی ارائه می‌دهد.
    همچنین امکاناتی برای آپدیت لحظه‌ای، تولید فیچر برای RL و نرمال‌سازی بر اساس ATR را فراهم می‌کند.
    """

    def __init__(self,
                 df_price: pd.DataFrame,
                 fib_ratios: Optional[Iterable[float]] = None,
                 atr_window: int = 14,
                 cluster_eps_atr_mult: float = 0.5):
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

        خروجی: DataFrame با سطرهایی از نوع: time, level_price, ratio, type
        """
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
        stats = self.analyzer.compute_level_stats(clusters_df["price"].values, self.df["close"], lookahead_bars=50,
                                                  tolerance=0.0)
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
            plan = planner.make_entry_plan(level_price=level_price,
                                           side=side,
                                           price_now=price_now,
                                           atr_value=atr_value)
            plans.append(plan)
        return plans

    # -----------------------------
    # Real-time / incremental updates
    # -----------------------------
    def real_time_update(self, new_bar: Dict[str, Any]) -> None:
        """آپدیت دیتافریم قیمت با یک کندل/تیک جدید و بروز کردن ATR cache.
        new_bar باید شامل keys: ['time','open','high','low','close']، time قابل parse باشد.
        این متد دیتافریم را append کرده و atr_series را مجدداً محاسبه می‌کند.
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

# Example usage (برای توسعه‌دهنده):
# from fibonacci import FibonacciEngine, OrderPlanner
# df = pd.read_csv('price.csv', parse_dates=['time'], index_col='time')
# eng = FibonacciEngine(df)
# raw = eng.generate_raw_levels()
# clusters = eng.cluster_and_score(raw)
# planner = OrderPlanner(risk_pct=0.5, account_equity=10000.0, pip_value=1.0)
# plans = eng.generate_entry_plans(clusters, planner)
# feats_df = eng.compute_features_dataframe(clusters)
# rl_vec = eng.get_feature_vector_for_rl(clusters, top_k=5)
# eng.real_time_update({"time": "2025-08-24 21:00:00", "open": 1950, "high": 1955, "low": 1948, "close": 1952})
