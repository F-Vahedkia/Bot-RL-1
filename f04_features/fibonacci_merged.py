"""
fibonacci_merged.py

فایل نهایی تلفیقی فیبوناچی — مناسب برای ربات تریدر مبتنی بر Reinforcement Learning (RL)

این ماژول ترکیبی از دو نسخهٔ قبلی (fibonacci.py و fibonacci_3.py) است و تلاش شده:
- متدهای بهتر و عملیاتی‌تر انتخاب شوند
- قابلیت‌های مفید برای RL اضافه گردد (feature extraction, fixed-length vector)
- قابلیت‌های real-time و cache برای اجرای روی VPS فراهم شود
- کلاس‌ها و متدها مستندسازی و کامنت‌گذاری شده‌اند تا قابل توسعه باشند

نکته‌ها:
- این فایل "مینیمال" نیست؛ قابلیت‌ها و متدهای احتمالی آینده نیز آورده شده‌اند مگر اینکه احتمال کمی برای استفاده داشته باشند.
- برای عملکرد production، برخی پارامترها باید متناسب با قرارداد بروکر (pip_value, tick_size) تنظیم شوند.

Dependencies: numpy, pandas, scipy, scikit-learn
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# Module logger
logger = logging.getLogger("fibonacci_merged")
logger.addHandler(logging.NullHandler())

# -----------------------------
# Constants / Defaults
# -----------------------------
DEFAULT_FIB_RATIOS = [23.6, 38.2, 50.0, 61.8, 78.6, 100.0, 127.2, 161.8]
DEFAULT_ATR_WINDOW = 14

# -----------------------------
# Utilities
# -----------------------------

def _validate_price_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex (timestamps)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in increasing order")
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns.str.lower())):
        raise ValueError(f"DataFrame must include columns: {required}")


def atr(df: pd.DataFrame, window: int = DEFAULT_ATR_WINDOW) -> pd.Series:
    """Compute ATR series (simple moving average of True Range).

    Returns series aligned with df index.
    """
    _validate_price_index(df)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class FibLevel:
    tf: str
    ratio: float
    price: float
    source: str = "retracement"  # retracement|extension|projection
    meta: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class ClusterParams:
    tolerance_mode: str = "rel"  # 'rel' or 'abs'
    rel_tolerance: float = 0.002  # relative tolerance (fraction)
    abs_tolerance: float = 0.0
    min_hits: int = 1
    use_timeframe_diversity: bool = True
    method_weights: Dict[str, float] = field(default_factory=lambda: {"retracement": 1.0, "extension": 1.0, "projection": 1.0})
    ratio_weights: Dict[float, float] = field(default_factory=dict)
    timeframe_weights: Dict[str, float] = field(default_factory=dict)


# -----------------------------
# Swing detection
# -----------------------------
class SwingDetector:
    """Detects swing highs and lows using scipy.find_peaks or a zigzag-like approach.

    Configuration options allow tuning for noisy/timeframe-specific data.
    """

    def __init__(self, method: str = "peaks", min_prominence: Optional[float] = None,
                 min_distance_bars: int = 3, use_wicks: str = "close"):
        self.method = method
        self.min_prominence = min_prominence
        self.min_distance_bars = max(1, int(min_distance_bars))
        if use_wicks not in ("close", "bodies", "wicks"):
            raise ValueError("use_wicks must be 'close','bodies' or 'wicks'")
        self.use_wicks = use_wicks

    def _series(self, df: pd.DataFrame) -> pd.Series:
        if self.use_wicks == "close":
            return df["close"]
        if self.use_wicks == "bodies":
            return (df["open"] + df["close"]) / 2.0
        # wicks: we still return close but caller can use high/low if needed
        return df["close"]

    def detect_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns: ['time','idx','price','kind','prominence','confidence']"""
        _validate_price_index(df)
        series = self._series(df)
        price = series.values

        if self.min_prominence is None:
            # heuristic: factor of price std
            self.min_prominence = max(1e-9, float(np.nanstd(price) * 0.4))

        peaks, props_peaks = find_peaks(price, distance=self.min_distance_bars, prominence=self.min_prominence)
        valleys, props_valleys = find_peaks(-price, distance=self.min_distance_bars, prominence=self.min_prominence)

        recs = []
        for p, prom in zip(peaks, props_peaks.get("prominences", [])):
            recs.append({"time": df.index[p], "idx": int(p), "price": float(price[p]), "kind": "peak", "prominence": float(prom)})
        for v, prom in zip(valleys, props_valleys.get("prominences", [])):
            recs.append({"time": df.index[v], "idx": int(v), "price": float(price[v]), "kind": "valley", "prominence": float(prom)})

        swings = pd.DataFrame(recs).sort_values("idx").reset_index(drop=True)
        if swings.empty:
            swings = pd.DataFrame(columns=["time", "idx", "price", "kind", "prominence", "confidence"])
            return swings
        max_prom = swings["prominence"].max() if swings["prominence"].notna().any() else 1.0
        swings["confidence"] = swings["prominence"] / (max_prom + 1e-9)
        return swings


# -----------------------------
# Fibonacci core
# -----------------------------
class FibonacciCore:
    """Core functions to compute retracement, extension and projection levels.

    Returns DataFrame for each method to keep outputs homogeneous.
    """

    def __init__(self, ratios: Optional[Iterable[float]] = None):
        self.ratios = list(ratios) if ratios is not None else list(DEFAULT_FIB_RATIOS)

    def _compute_levels(self, start: float, end: float, kind: str) -> pd.DataFrame:
        recs = []
        for r in self.ratios:
            frac = r / 100.0
            if kind == "retracement":
                price = end + (start - end) * (1 - frac)
            elif kind == "extension":
                price = end + (end - start) * frac
            else:
                price = np.nan
            recs.append({"ratio": float(r), "price": float(price), "type": kind})
        return pd.DataFrame(recs)

    def retracement(self, high: float, low: float, *, context: Optional[Dict] = None) -> pd.DataFrame:
        return self._compute_levels(high, low, "retracement")

    def extension(self, high: float, low: float, *, context: Optional[Dict] = None) -> pd.DataFrame:
        return self._compute_levels(high, low, "extension")

    def projection(self, a: float, b: float, c: float, *, ratios: Optional[Iterable[float]] = None,
                   context: Optional[Dict] = None) -> pd.DataFrame:
        if ratios is None:
            ratios = self.ratios
        recs = []
        bc = c - b
        for r in ratios:
            price = float(c + bc * (r / 100.0))
            recs.append({"ratio": float(r), "price": price, "type": "projection"})
        return pd.DataFrame(recs)

    def multi_timeframe_retracement(self, swings: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        frames = []
        for tf, (high, low) in swings.items():
            df = self.retracement(high, low)
            df["tf"] = tf
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["ratio", "price", "type", "tf"]) 


# -----------------------------
# Clustering & scoring
# -----------------------------
class LevelClusterer:
    """Clustering using DBSCAN with eps based on ATR (volatility-aware).

    Returns DataFrame of clusters with center price and members.
    """

    def __init__(self, eps_atr_multiplier: float = 0.5, min_samples: int = 1, atr_window: int = DEFAULT_ATR_WINDOW):
        self.eps_atr_multiplier = float(eps_atr_multiplier)
        self.min_samples = max(1, int(min_samples))
        self.atr_window = int(atr_window)

    def cluster_levels(self, levels: Iterable[float], prices_df: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
        levels_arr = np.array(sorted(set(float(l) for l in levels)))
        if levels_arr.size == 0:
            return pd.DataFrame()
        atr_series = atr(prices_df, window=self.atr_window)
        mean_atr = float(atr_series.mean()) if not atr_series.empty else 0.0
        eps = max(1e-9, self.eps_atr_multiplier * (mean_atr if mean_atr > 0 else 1.0))
        X = levels_arr.reshape(-1, 1)
        db = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = db.fit_predict(X)

        clusters = []
        for cid in sorted(set(labels)):
            mask = labels == cid
            members = list(levels_arr[mask])
            if cid == -1:
                # treat noise as individual clusters
                for m in members:
                    clusters.append({"cluster_id": -1, "price": float(m), "members": [m], "strength": 0.0, "touch_count": 0, "last_touch": None, "timeframes": {timeframe: 1}})
            else:
                center = float(np.mean(members))
                clusters.append({"cluster_id": int(cid), "price": center, "members": members, "strength": float(len(members)), "touch_count": 0, "last_touch": None, "timeframes": {timeframe: len(members)}})
        return pd.DataFrame(clusters)


class FibonacciClusterDetector:
    """A higher-level detector that groups levels according to tolerance and produces a scored cluster list.

    This class implements explainable scoring (weights, timeframe diversity, ratio proximity) based on ClusterParams.
    """

    def __init__(self, params: Optional[ClusterParams] = None):
        self.params = params or ClusterParams()

    def _within_tolerance(self, p1: float, p2: float) -> bool:
        if self.params.tolerance_mode == "abs":
            return abs(p1 - p2) <= self.params.abs_tolerance
        # relative mode
        tol = self.params.rel_tolerance * max(abs(p1), abs(p2), 1.0)
        return abs(p1 - p2) <= tol

    def _level_score(self, group: List[Dict[str, Any]]) -> float:
        # group: list of dicts with keys: price, tf, type, ratio, meta
        base = 0.0
        tf_set = set()
        method_score = 0.0
        ratio_score = 0.0
        for item in group:
            method = item.get("type", "retracement")
            ratio = float(item.get("ratio", 0.0))
            tf = item.get("tf", "NA")
            tf_set.add(tf)
            method_score += self.params.method_weights.get(method, 1.0)
            # ratio weight if set
            ratio_score += self.params.ratio_weights.get(ratio, 1.0)
        # diversity bonus
        diversity = float(len(tf_set)) if self.params.use_timeframe_diversity else 1.0
        # simple composite
        score = (method_score + ratio_score) * diversity
        return float(score)

    def _diversity_bonus(self, timeframes: Iterable[str]) -> float:
        if not timeframes:
            return 0.0
        distinct = len(set(timeframes))
        return float(1.0 + 0.1 * (distinct - 1))

    @staticmethod
    def _merge_group_to_cluster(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        prices = [float(x["price"]) for x in group]
        center = float(np.mean(prices))
        members = group
        tfs = {g.get("tf", "NA"): 1 for g in group}
        return {"price": center, "members": members, "timeframes": tfs}

    def cluster_levels(self, levels_df: pd.DataFrame, sort_by: str = "score") -> pd.DataFrame:
        """Group levels sequentially according to tolerance and compute cluster score.

        levels_df expected columns: ['price','ratio','type','tf']
        """
        if levels_df is None or levels_df.empty:
            return pd.DataFrame()
        # sort levels by price
        df = levels_df.copy().sort_values("price").reset_index(drop=True)
        groups: List[List[Dict[str, Any]]] = []
        for _, row in df.iterrows():
            price = float(row["price"])
            item = {"price": price, "ratio": row.get("ratio"), "type": row.get("type"), "tf": row.get("tf", "NA"), "meta": {}}
            placed = False
            for g in groups:
                # check against group's center
                center = np.mean([float(x["price"]) for x in g])
                if self._within_tolerance(center, price):
                    g.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])
        clusters = []
        for g in groups:
            merged = self._merge_group_to_cluster(g)
            score = self._level_score(g)
            clusters.append({"price": merged["price"], "members": merged["members"], "timeframes": merged["timeframes"], "score": float(score), "hits": len(g)})
        cdf = pd.DataFrame(clusters)
        if cdf.empty:
            return cdf
        cdf = cdf.sort_values(sort_by, ascending=False).reset_index(drop=True)
        return cdf

    @staticmethod
    def build_levels_df_from_swings(swings_by_tf: Dict[str, List[Tuple[float, float]]], method: str = "retracement", ratios: Optional[Iterable[float]] = None) -> pd.DataFrame:
        """Convert swings_by_tf: {tf: [(high,low), ...]} into a levels_df standardized.

        Returns columns: ['price','ratio','type','tf']
        """
        core = FibonacciCore(ratios=ratios)
        frames = []
        for tf, swings in swings_by_tf.items():
            for (high, low) in swings:
                if method == "retracement":
                    df = core.retracement(high, low)
                elif method == "extension":
                    df = core.extension(high, low)
                else:
                    df = core.retracement(high, low)
                df["tf"] = tf
                frames.append(df)
        if frames:
            df_all = pd.concat(frames, ignore_index=True)
            df_all = df_all.rename(columns={"price": "price", "ratio": "ratio", "type": "type"})
            return df_all[["price", "ratio", "type", "tf"]]
        return pd.DataFrame(columns=["price", "ratio", "type", "tf"])


# -----------------------------
# Historical analytics
# -----------------------------
class LevelAnalyzer:
    """Compute touch/hit/avg_bounce metrics for each price level over price series."""

    def compute_level_stats(self, levels: Iterable[float], prices: pd.Series, lookahead_bars: int = 50, tolerance: float = 0.0) -> pd.DataFrame:
        levels_sorted = sorted(set(float(l) for l in levels))
        if prices.empty:
            return pd.DataFrame()
        records = []
        arr = prices.values
        n = len(arr)
        for level in levels_sorted:
            touches = 0
            hits = 0
            bounces = []
            for i in range(max(0, n - lookahead_bars)):
                segment = prices.iloc[i: i + lookahead_bars]
                seg_high = segment.max()
                seg_low = segment.min()
                if (seg_low - tolerance) <= level <= (seg_high + tolerance):
                    touches += 1
                    if level <= seg_high and level >= seg_low:
                        hits += 1
                    max_up = seg_high - level if seg_high > level else 0
                    max_down = level - seg_low if seg_low < level else 0
                    bounces.append(max(max_up, max_down))
            records.append({"level": level, "touch_count": touches, "hit_count": hits, "hit_rate": float(hits / touches) if touches else 0.0, "avg_bounce": float(np.mean(bounces)) if bounces else 0.0})
        return pd.DataFrame(records)


# -----------------------------
# Order planning
# -----------------------------
@dataclass
class OrderSpec:
    side: str
    entry_price: float
    stop_price: float
    take_profit: Optional[float]
    lot_size: float
    order_type: str = "limit"
    meta: Dict[str, Any] = field(default_factory=dict)


class OrderPlanner:
    def __init__(self, risk_pct: float = 0.5, account_equity: float = 10000.0, sl_atr_mult: float = 1.0, min_lot: float = 0.01, pip_value: Optional[float] = None):
        self.risk_pct = float(risk_pct)
        self.account_equity = float(account_equity)
        self.sl_atr_mult = float(sl_atr_mult)
        self.min_lot = float(min_lot)
        self.pip_value = pip_value

    def _calc_lot_by_risk(self, sl_distance: float) -> float:
        risk_amount = (self.risk_pct / 100.0) * self.account_equity
        if not self.pip_value:
            logger.warning("pip_value not set; returning min_lot")
            return float(self.min_lot)
        lot = risk_amount / (abs(sl_distance) * self.pip_value) if sl_distance and self.pip_value else float(self.min_lot)
        return max(float(self.min_lot), float(lot))

    def make_entry_plan(self, level_price: float, side: str, price_now: float, atr_value: float, tp_atr_mult: float = 2.0, sl_buffer: float = 0.0, order_type: str = "limit") -> OrderSpec:
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        entry = float(level_price)
        sl_distance = atr_value * self.sl_atr_mult + sl_buffer
        if side == "buy":
            stop = entry - sl_distance
            tp = entry + sl_distance * tp_atr_mult if tp_atr_mult else None
        else:
            stop = entry + sl_distance
            tp = entry - sl_distance * tp_atr_mult if tp_atr_mult else None
        lot = self._calc_lot_by_risk(abs(entry - stop))
        return OrderSpec(side=side, entry_price=entry, stop_price=stop, take_profit=tp, lot_size=lot, order_type=order_type, meta={"atr": atr_value, "sl_mult": self.sl_atr_mult})


# -----------------------------
# News filter
# -----------------------------
class NewsFilter:
    def __init__(self, lookahead_minutes: int = 60):
        self.lookahead_minutes = int(lookahead_minutes)

    def filter_levels_by_news(self, levels_df: pd.DataFrame, news_events: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        if levels_df is None or levels_df.empty:
            return levels_df
        out = levels_df.copy()
        out["news_risk"] = 0
        for ev in news_events:
            ev_time = pd.to_datetime(ev.get("time")) if ev.get("time") is not None else None
            if ev_time is None:
                continue
            impact = ev.get("impact", "low")
            weight = {"low": 1, "medium": 2, "high": 3}.get(impact, 1)
            window_start = ev_time - pd.Timedelta(minutes=self.lookahead_minutes)
            window_end = ev_time + pd.Timedelta(minutes=self.lookahead_minutes)
            mask = (out["time"] >= window_start) & (out["time"] <= window_end)
            out.loc[mask, "news_risk"] = out.loc[mask, "news_risk"].add(weight).fillna(weight)
        return out


# -----------------------------
# Feature extraction for RL
# -----------------------------
class FeatureExtractor:
    def __init__(self, atr_series: Optional[pd.Series] = None):
        self.atr_series = atr_series

    def extract_features_from_cluster_row(self, row: pd.Series, current_price: float, atr_value: float) -> Dict[str, float]:
        price = float(row.get("price", current_price))
        strength = float(row.get("strength", 0.0)) if row.get("strength") is not None else float(row.get("score", 0.0))
        hit_rate = float(row.get("hit_rate", 0.0)) if row.get("hit_rate") is not None else 0.0
        avg_bounce = float(row.get("avg_bounce", 0.0)) if row.get("avg_bounce") is not None else 0.0
        members = row.get("members", [])
        members_count = len(members) if isinstance(members, (list, tuple, np.ndarray)) else 1
        tf_count = len(row.get("timeframes", {})) if isinstance(row.get("timeframes", {}), dict) else 1
        abs_dist = price - current_price
        pct_dist = (abs_dist / current_price) if current_price else 0.0
        atr_norm = (abs_dist / atr_value) if atr_value else abs_dist
        width = float(np.max(members) - np.min(members)) if members_count > 1 else 0.0
        width_norm = (width / atr_value) if atr_value else width
        return {"price": price, "abs_dist": abs_dist, "pct_dist": pct_dist, "atr_norm_dist": atr_norm, "strength": strength, "hit_rate": hit_rate, "avg_bounce": avg_bounce, "members_count": float(members_count), "tf_count": float(tf_count), "width": width, "width_norm": width_norm}

    def extract_dataframe(self, clusters_df: pd.DataFrame, current_price: float, atr_value: float) -> pd.DataFrame:
        if clusters_df is None or clusters_df.empty:
            return pd.DataFrame()
        rows = []
        for _, r in clusters_df.iterrows():
            rows.append(self.extract_features_from_cluster_row(r, current_price, atr_value))
        df = pd.DataFrame(rows)
        return df

    def to_vector(self, features_df: pd.DataFrame, top_k: int = 5) -> np.ndarray:
        if features_df is None or features_df.empty:
            return np.zeros(top_k * 6, dtype=float)
        features_df = features_df.sort_values("strength", ascending=False).reset_index(drop=True)
        chosen = features_df.head(top_k)
        vec = []
        for _, r in chosen.iterrows():
            vec.extend([float(r.get("atr_norm_dist", 0.0)), float(r.get("strength", 0.0)), float(r.get("hit_rate", 0.0)), float(r.get("avg_bounce", 0.0)), float(r.get("members_count", 0.0)), float(r.get("width_norm", 0.0))])
        needed = top_k * 6 - len(vec)
        if needed > 0:
            vec.extend([0.0] * needed)
        return np.array(vec, dtype=float)


# -----------------------------
# Golden Zone Detector
# -----------------------------
class GoldenZoneDetector:
    def __init__(self, zone_width_pct: float = 0.006, merge_gap_pct: float = 0.0025, min_members: int = 2):
        self.zone_width_pct = float(zone_width_pct)
        self.merge_gap_pct = float(merge_gap_pct)
        self.min_members = int(min_members)

    def _level_to_zone(self, price: float, reference_price: float) -> Tuple[float, float]:
        half = abs(reference_price) * self.zone_width_pct
        return (price - half, price + half)

    def _zones_overlap(self, z1: Tuple[float, float], z2: Tuple[float, float]) -> bool:
        return not (z1[1] < z2[0] - self.merge_gap_pct * max(abs(z1[0]), abs(z2[0])) or z2[1] < z1[0] - self.merge_gap_pct * max(abs(z1[0]), abs(z2[0])))

    def detect_from_levels(self, levels: List[Dict[str, Any]], reference_price: float) -> List[Zone]:
        # build initial zones
        initial = []
        for l in levels:
            p = float(l.get("price"))
            low, high = self._level_to_zone(p, reference_price)
            initial.append(Zone(low=low, high=high, members=[FibLevel(tf=l.get("tf","NA"), ratio=float(l.get("ratio",0.0)), price=p, source=l.get("type","retracement"))], strength=1.0))
        # merge zones by overlap
        merged = []
        for z in initial:
            placed = False
            for m in merged:
                if self._zones_overlap((z.low, z.high), (m.low, m.high)):
                    # merge
                    m.members.extend(z.members)
                    m.low = min(m.low, z.low)
                    m.high = max(m.high, z.high)
                    placed = True
                    break
            if not placed:
                merged.append(z)
        # compute strength
        for z in merged:
            z.strength = float(len(z.members)) * (1.0 + 0.1 * len(set([m.tf for m in z.members])))
        # filter by min_members
        merged = [z for z in merged if len(z.members) >= self.min_members]
        merged.sort(key=lambda x: x.strength, reverse=True)
        return merged


# -----------------------------
# Exporter
# -----------------------------
class Exporter:
    @staticmethod
    def to_json(df: pd.DataFrame) -> str:
        return df.to_json(orient="records", date_format="iso")

    @staticmethod
    def to_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)


# -----------------------------
# High-level Engine
# -----------------------------
class FibonacciEngine:
    """Engine that orchestrates swing detection, level generation, clustering, scoring and feature extraction.

    Designed to be RL-ready: provides get_feature_vector_for_rl and real_time_update.
    """

    def __init__(self, df_price: pd.DataFrame, fib_ratios: Optional[Iterable[float]] = None, atr_window: int = DEFAULT_ATR_WINDOW, cluster_eps_atr_mult: float = 0.5, cluster_params: Optional[ClusterParams] = None):
        _validate_price_index(df_price)
        self.df = df_price.copy()
        self.atr_window = int(atr_window)
        self.atr_series = atr(self.df, window=self.atr_window)
        self.swing_detector = SwingDetector(min_distance_bars=3)
        self.core = FibonacciCore(ratios=fib_ratios)
        self.clusterer = LevelClusterer(eps_atr_multiplier=cluster_eps_atr_mult, atr_window=self.atr_window)
        self.cluster_detector = FibonacciClusterDetector(params=cluster_params)
        self.analyzer = LevelAnalyzer()
        self.news_filter = NewsFilter()
        self.feature_extractor = FeatureExtractor(atr_series=self.atr_series)
        self.golden_detector = GoldenZoneDetector()
        # caches
        self._last_raw_levels: Optional[pd.DataFrame] = None
        self._last_clusters: Optional[pd.DataFrame] = None

    def generate_raw_levels(self) -> pd.DataFrame:
        swings = self.swing_detector.detect_swings(self.df)
        recs = []
        for i in range(len(swings) - 1):
            s1 = swings.iloc[i]
            s2 = swings.iloc[i + 1]
            start = s1["price"]
            end = s2["price"]
            t = s2["time"]
            retr = self.core.retracement(start, end)
            ext = self.core.extension(start, end)
            for _, r in retr.iterrows():
                recs.append({"time": t, "price": r["price"], "ratio": r["ratio"], "type": r["type"], "tf": "native"})
            for _, r in ext.iterrows():
                recs.append({"time": t, "price": r["price"], "ratio": r["ratio"], "type": r["type"], "tf": "native"})
        df_levels = pd.DataFrame(recs)
        if not df_levels.empty:
            df_levels["time"] = pd.to_datetime(df_levels["time"])
            df_levels = df_levels.drop_duplicates(subset=["price", "ratio", "type"]).reset_index(drop=True)
        self._last_raw_levels = df_levels
        return df_levels

    def cluster_and_score(self, levels_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if levels_df is None:
            levels_df = self._last_raw_levels if self._last_raw_levels is not None else self.generate_raw_levels()
        if levels_df is None or levels_df.empty:
            self._last_clusters = pd.DataFrame()
            return pd.DataFrame()
        # create clusters via DBSCAN (volatility-aware)
        clusters_df = self.clusterer.cluster_levels(levels_df["price"].values, self.df, timeframe="native")
        if clusters_df.empty:
            self._last_clusters = clusters_df
            return clusters_df
        # align stats
        stats = self.analyzer.compute_level_stats(clusters_df["price"].values, self.df["close"], lookahead_bars=50, tolerance=0.0)
        if not stats.empty:
            clusters_df = clusters_df.merge(stats, how="left", left_on="price", right_on="level")
            clusters_df["hit_rate"] = clusters_df["hit_rate"].fillna(0.0)
            clusters_df["avg_bounce"] = clusters_df["avg_bounce"].fillna(0.0)
        else:
            clusters_df["hit_rate"] = 0.0
            clusters_df["avg_bounce"] = 0.0
        # simple composite strength heuristic
        clusters_df["strength"] = clusters_df["strength"] * (1.0 + clusters_df["hit_rate"]) 
        clusters_df = clusters_df.sort_values("strength", ascending=False).reset_index(drop=True)
        self._last_clusters = clusters_df
        return clusters_df

    def generate_entry_plans(self, clusters_df: Optional[pd.DataFrame], planner: OrderPlanner) -> List[OrderSpec]:
        if clusters_df is None or clusters_df.empty:
            return []
        plans = []
        price_now = float(self.df["close"].iloc[-1])
        atr_value = float(self.atr_series.iloc[-1]) if not self.atr_series.empty else 0.0
        for _, row in clusters_df.iterrows():
            level_price = float(row["price"])
            side = "sell" if level_price > price_now else "buy"
            plan = planner.make_entry_plan(level_price=level_price, side=side, price_now=price_now, atr_value=atr_value)
            plans.append(plan)
        return plans

    # Real-time update: add new bar or update existing
    def real_time_update(self, new_bar: Dict[str, Any]) -> None:
        try:
            t = pd.to_datetime(new_bar["time"]) if not isinstance(new_bar.get("time"), pd.Timestamp) else new_bar.get("time")
            o = float(new_bar["open"])
            h = float(new_bar["high"])
            l = float(new_bar["low"])
            c = float(new_bar["close"])
        except Exception as e:
            logger.error("Invalid new_bar: %s", e)
            raise
        if t in self.df.index:
            self.df.loc[t, ["open", "high", "low", "close"]] = [o, h, l, c]
        else:
            row = pd.DataFrame([{"open": o, "high": h, "low": l, "close": c}], index=pd.DatetimeIndex([t]))
            self.df = pd.concat([self.df, row]).sort_index()
        self.atr_series = atr(self.df, window=self.atr_window)
        self._last_raw_levels = None
        self._last_clusters = None

    # RL feature helpers
    def compute_features_dataframe(self, clusters_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if clusters_df is None:
            clusters_df = self._last_clusters if self._last_clusters is not None else self.cluster_and_score(self._last_raw_levels)
        price_now = float(self.df["close"].iloc[-1])
        atr_value = float(self.atr_series.iloc[-1]) if not self.atr_series.empty else 0.0
        return self.feature_extractor.extract_dataframe(clusters_df, price_now, atr_value)

    def get_feature_vector_for_rl(self, clusters_df: Optional[pd.DataFrame] = None, top_k: int = 5) -> np.ndarray:
        df_feat = self.compute_features_dataframe(clusters_df)
        return self.feature_extractor.to_vector(df_feat, top_k=top_k)

    # Golden zones
    def detect_golden_zones(self, levels_df: Optional[pd.DataFrame] = None, reference_price: Optional[float] = None) -> List[Zone]:
        if levels_df is None:
            levels_df = self._last_raw_levels if self._last_raw_levels is not None else self.generate_raw_levels()
        if levels_df is None or levels_df.empty:
            return []
        levels = levels_df.to_dict(orient="records")
        ref = float(reference_price) if reference_price is not None else float(self.df["close"].iloc[-1])
        zones = self.golden_detector.detect_from_levels(levels, reference_price=ref)
        return zones


# -----------------------------
# If needed: small demo (kept minimal intentionally)
# -----------------------------
if __name__ == "__main__":
    print("This module defines a RL-ready Fibonacci engine. Import and use in your bot.")

# End of file
