# file: f04_features/support_resistance.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

class SupportResistanceMultiTF:
    """
    استخراج سطوح حمایت و مقاومت به صورت Multi-Timeframe با الگوریتم‌های کلاسیک و مدرن.
    """

    def __init__(
        self,
        lookback: int = 5,
        cluster_method: str = "kmeans",
        n_clusters: int = 5,
        prominence: float = 0.001
    ):
        """
        :param lookback: تعداد کندل‌ها برای تشخیص fractals و swing points
        :param cluster_method: روش خوشه‌بندی (kmeans, dbscan, gmm یا None)
        :param n_clusters: تعداد خوشه‌ها برای kmeans یا gmm
        :param prominence: حساسیت برای DBSCAN یا فیلتر سطوح نزدیک
        """
        self.lookback = lookback
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.prominence = prominence

    # ---------------------------
    # الگوریتم‌های کلاسیک
    # ---------------------------

    def detect_fractals(self, df: pd.DataFrame):
        """
        تشخیص fractals بالا و پایین از روی کندل‌ها
        """
        highs, lows = df['High'].values, df['Low'].values
        fractal_highs, fractal_lows = [], []

        for i in range(self.lookback, len(df) - self.lookback):
            if highs[i] == max(highs[i - self.lookback:i + self.lookback + 1]):
                fractal_highs.append((i, highs[i]))
            if lows[i] == min(lows[i - self.lookback:i + self.lookback + 1]):
                fractal_lows.append((i, lows[i]))

        return fractal_highs, fractal_lows

    def detect_swing_points(self, df: pd.DataFrame):
        """
        تشخیص swing highs و swing lows
        """
        highs, lows = df['High'].values, df['Low'].values
        swing_highs, swing_lows = [], []

        for i in range(1, len(df) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append((i, lows[i]))

        return swing_highs, swing_lows

    def calculate_pivots(self, df: pd.DataFrame):
        """
        Pivot Points استاندارد
        """
        high, low, close = df['High'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2}

    # ---------------------------
    # الگوریتم‌های مدرن (ML/Clustering)
    # ---------------------------

    def cluster_levels(self, levels):
        """
        خوشه‌بندی سطوح حمایت/مقاومت
        """
        if len(levels) < 2:
            return levels

        levels_array = np.array(levels).reshape(-1, 1)

        if self.cluster_method == "kmeans":
            model = KMeans(n_clusters=min(self.n_clusters, len(levels)))
            model.fit(levels_array)
            return sorted(model.cluster_centers_.flatten())

        elif self.cluster_method == "dbscan":
            model = DBSCAN(eps=self.prominence, min_samples=2)
            labels = model.fit_predict(levels_array)
            clusters = {}
            for lbl, val in zip(labels, levels_array.flatten()):
                if lbl == -1:
                    continue
                clusters.setdefault(lbl, []).append(val)
            return [np.mean(v) for v in clusters.values()]

        elif self.cluster_method == "gmm":
            model = GaussianMixture(n_components=min(self.n_clusters, len(levels)))
            model.fit(levels_array)
            return sorted(model.means_.flatten())

        return levels

    # ---------------------------
    # Multi-Timeframe Processing
    # ---------------------------

    def get_levels_multitf(self, df_dict: dict):
        """
        استخراج سطوح حمایت/مقاومت Multi-Timeframe
        :param df_dict: دیکشنری {timeframe: DataFrame}
        :return: دیکشنری {timeframe: {levels, fractals, swings, pivots}}, و تلفیق سطوح قوی
        """
        result = {}
        all_levels = []

        for tf, df in df_dict.items():
            fractal_highs, fractal_lows = self.detect_fractals(df)
            swing_highs, swing_lows = self.detect_swing_points(df)
            pivots = self.calculate_pivots(df)

            raw_levels = [p[1] for p in fractal_highs + fractal_lows + swing_highs + swing_lows]
            clustered_levels = self.cluster_levels(raw_levels)

            result[tf] = {
                "levels": clustered_levels,
                "fractals": {"highs": fractal_highs, "lows": fractal_lows},
                "swings": {"highs": swing_highs, "lows": swing_lows},
                "pivots": pivots
            }

            all_levels.extend(clustered_levels)

        # تلفیق سطوح از همه تایم‌فریم‌ها
        combined_levels = self.cluster_levels(all_levels)

        return result, combined_levels
