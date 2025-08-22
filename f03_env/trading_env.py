# f03_env/trading_env.py
"""
TradingEnv (Gymnasium) — robust implementation integrated with project files.

Design goals:
- Use DataHandler.fetch_for_symbol(...) (or fallback) for multi-timeframe data.
- Use Executor to place/close orders if available; otherwise simulate.
- Use RiskManager functions if present; otherwise use fallback simple sizing.
- Action space: structured Dict: {type, volume_frac, sl_frac, tp_frac, trailing, riskfree}
- Observation: flattened windows of OHLC (n_candles_per_tf per timeframe) + basic account info.
- Reward: equity change (realized + unrealized); easy to extend.
"""

from __future__ import annotations
from datetime import datetime
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# try to import project modules with graceful fallback
try:
    from f10_utils.config_loader import ConfigLoader
except Exception:
    ConfigLoader = None

try:
    from f02_data.data_handler import DataHandler
except Exception:
    DataHandler = None

try:
    from f09_execution.executor import Executor
except Exception:
    Executor = None

# Risk manager may expose a class or functions; import module as fallback
try:
    from f10_utils.risk_manager import RiskManager
    _RISK_MANAGER_IS_CLASS = True
except Exception:
    try:
        import f10_utils.risk_manager as risk_manager_module  # type: ignore
        RiskManager = None
        _RISK_MANAGER_IS_CLASS = False
    except Exception:
        risk_manager_module = None
        RiskManager = None
        _RISK_MANAGER_IS_CLASS = False

# logging (try project logging setup if exists)
try:
    from f10_utils.logging_cfg import setup_logging

    logger = setup_logging("TradingEnv")
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TradingEnv")

# -------- Adding Divergence -------------------
from f04_features.indicators import Indicators
from f04_features.support_resistance import SupportResistanceMultiTF

# ------------------------------
# Helper utilities
# ------------------------------
def _get_cfg_dict_from_loader(cfg_path: Optional[str]) -> Dict[str, Any]:
    """
    Load config in a best-effort way using ConfigLoader if present,
    otherwise expect cfg_path to be a dict (caller responsibility).
    """
    if cfg_path is None:
        return {}
    if ConfigLoader is None:
        # if user passed a dict path (actually a dict), return it
        if isinstance(cfg_path, dict):
            return cfg_path
        logger.warning("ConfigLoader not available; expecting a dict config passed instead of path.")
        return {}
    # ConfigLoader exists - try common attributes
    try:
        loader = ConfigLoader(cfg_path) if isinstance(cfg_path, str) else ConfigLoader(cfg_path)
    except Exception:
        try:
            loader = ConfigLoader()
        except Exception:
            logger.warning("ConfigLoader could not be instantiated; returning empty config.")
            return {}
    # try to extract config dict from loader
    if hasattr(loader, "get_all"):
        try:
            return loader.get_all() or {}
        except Exception:
            pass
    if hasattr(loader, "config"):
        try:
            return getattr(loader, "config") or {}
        except Exception:
            pass
    if hasattr(loader, "load"):
        try:
            return loader.load() or {}
        except Exception:
            pass
    logger.warning("ConfigLoader present but no known access method; returning empty config.")
    return {}


def _safe_get(obj: Any, path: List[str], default: Any = None) -> Any:
    """Safe nested getter for dict-like configs."""
    cur = obj
    for p in path:
        if not cur:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, default)
        else:
            return default
    return cur if cur is not None else default


class TrailingStopManager:
    """
    TrailingStopManager — provides parametrized trailing-stop calculations:
     - chandelier exit (long: highest_high - atr_mult * ATR)
     - atr-based trailing (current_price - direction * atr_mult * ATR)

    Usage:
      manager = TrailingStopManager(cfg)
      new_sl = manager.compute_new_sl(pos, base_df, current_ts, current_price)
      if new_sl is not None: pos['sl'] = new_sl
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        # default params (can be overridden via cfg['trailing_defaults'])
        td = (self.cfg.get("env_defaults") or {}).get("trailing_defaults", {}) if isinstance(self.cfg, dict) else {}
        self.default_mode = td.get("mode", "chandelier")  # "chandelier" or "atr"
        self.chandelier_period = int(td.get("chandelier_period", 22))
        self.chandelier_atr_mult = float(td.get("chandelier_atr_mult", 3.0))
        self.atr_period = int(td.get("atr_period", 14))
        self.atr_mult = float(td.get("atr_mult", 1.5))
        # minimum atr required to update (to avoid nan early in series)
        self.min_atr_non_nan = int(td.get("min_atr_non_nan", max(self.chandelier_period, self.atr_period)))

    @staticmethod
    def _compute_atr(series_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Robust ATR calculation returning a pandas Series aligned with series_df.index.
        Uses Wilder's smoothing (RMA).
        """
        if series_df is None or series_df.empty:
            return pd.Series([], dtype=float)
        high = series_df["high"].astype(float)
        low = series_df["low"].astype(float)
        close = series_df["close"].astype(float)

        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder RMA (exponential smoothing with alpha = 1/period)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        return atr

    def _find_index_for_timestamp(self, df: pd.DataFrame, ts: Optional[pd.Timestamp]) -> int:
        """
        Return integer index in df that corresponds to timestamp ts (ffill).
        If df has no DatetimeIndex, return last index.
        """
        if df is None or df.empty:
            return -1
        if ts is None:
            return len(df) - 1
        if isinstance(df.index, pd.DatetimeIndex):
            pos = df.index.get_indexer([ts], method="ffill")[0]
            if pos == -1:
                # fallback: last available <= ts not found, return last index
                return max(0, len(df) - 1)
            return int(pos)
        # if time column exists, try to parse
        if "time" in df.columns:
            times = pd.to_datetime(df["time"], errors="coerce")
            idx = np.searchsorted(times.values, np.datetime64(ts), side="right") - 1
            idx = max(0, min(idx, len(times) - 1))
            return int(idx)
        # fallback
        return len(df) - 1

    def compute_new_sl(
        self,
        pos: Dict[str, Any],
        base_df: pd.DataFrame,
        current_ts: Optional[pd.Timestamp] = None,
        current_price: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute a new trailing SL for `pos` using base_df (the base timeframe DataFrame)
        and the timestamp/current_price. Returns new SL price or None if not applicable.

        pos: dict must contain at least 'direction' (1 or -1). Optional keys:
             - 'trailing_mode' : "chandelier" or "atr" (overrides default)
             - 'trailing_period', 'atr_mult', 'chandelier_period', 'chandelier_atr_mult'
        """
        if base_df is None or base_df.empty:
            return None
        # decide method
        mode = pos.get("trailing_mode", self.default_mode)
        # compute index for current timestamp
        idx = self._find_index_for_timestamp(base_df, current_ts)
        if idx < 0 or idx >= len(base_df):
            return None

        # compute ATR series once
        # choose the maximum period to ensure we have values
        period_for_atr = int(pos.get("atr_period", self.atr_period))
        atr_series = self._compute_atr(base_df, period=period_for_atr)
        atr_val = None
        try:
            atr_val = float(atr_series.iloc[idx])
        except Exception:
            atr_val = None

        # if ATR not available yet, skip
        if atr_val is None or np.isnan(atr_val):
            return None

        direction = int(pos.get("direction", 1))
        # method-specific logic
        if mode == "chandelier":
            per = int(pos.get("chandelier_period", self.chandelier_period))
            atr_mult = float(pos.get("chandelier_atr_mult", self.chandelier_atr_mult))
            start_idx = max(0, idx - per + 1)
            window = base_df.iloc[start_idx : idx + 1]
            if window.empty:
                return None
            highest = float(window["high"].max())
            lowest = float(window["low"].min())
            if direction == 1:
                # long: CE = highest_high - atr_mult * ATR
                new_sl = highest - atr_mult * atr_val
            else:
                # short: CE = lowest_low + atr_mult * ATR
                new_sl = lowest + atr_mult * atr_val
            return float(new_sl)
        else:
            # default to ATR base trailing: distance = atr_mult * ATR
            atr_mult = float(pos.get("atr_mult", self.atr_mult))
            if current_price is None or np.isnan(current_price):
                # fallback to close price at idx
                try:
                    current_price = float(base_df["close"].iloc[idx])
                except Exception:
                    return None
            new_sl = current_price - direction * (atr_mult * atr_val)
            return float(new_sl)

# -------------------------
# Helper for time features
# -------------------------
def _add_time_features(obs_dict: Dict[str, float], ts: pd.Timestamp):
    """
    Modify obs_dict in-place with time-based features:
      - hour_of_day
      - day_of_week
      - session flags: is_london, is_newyork, is_asia
      - minute_of_hour
    ts: pandas Timestamp in UTC
    """
    if ts is None or pd.isna(ts):
        return
    hour = ts.hour
    obs_dict["hour_norm"] = hour / 23.0
    obs_dict["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    obs_dict["hour_cos"] = math.cos(2 * math.pi * hour / 24)

    dow = ts.dayofweek  # Monday=0..Sunday=6
    obs_dict["dow_norm"] = dow / 6.0
    obs_dict["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    obs_dict["dow_cos"] = math.cos(2 * math.pi * dow / 7)

    minute = ts.minute
    obs_dict["min_norm"] = minute / 59.0

    # session flags in UTC
    obs_dict["is_london"] = 1.0 if 7 <= hour < 16 else 0.0
    obs_dict["is_newyork"] = 1.0 if 12 <= hour < 20 else 0.0
    obs_dict["is_asia"] = 1.0 if (0 <= hour < 7) or (16 <= hour < 24) else 0.0

# ------------------------------
# TradingEnv
# ------------------------------
class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg: Optional[Any] = None):
        """
        cfg: either a path accepted by ConfigLoader, or a config dict.
        """
        super().__init__()

        # ------ load config (best-effort) ------
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            self.cfg = _get_cfg_dict_from_loader(cfg)

        # fallback small config defaults
        self.symbol: str = _safe_get(self.cfg, ["env_defaults", "symbol"], _safe_get(self.cfg, ["trading", "symbol"], "XAUUSD"))
        self.timeframes: List[str] = _safe_get(self.cfg, ["env_defaults", "timeframes"], _safe_get(self.cfg, ["trading", "timeframes"], ["M1", "M5", "M30", "H4", "D1", "W1"]))
        if isinstance(self.timeframes, str):
            self.timeframes = [self.timeframes]
        self.n_candles_per_tf: int = int(_safe_get(self.cfg, ["env_defaults", "n_candles_per_tf"], _safe_get(self.cfg, ["data_fetch_defaults", "n_candles"], _safe_get(self.cfg, ["n_candles"], 15))))
        # execution/risk defaults
        self.initial_balance: float = float(_safe_get(self.cfg, ["env_defaults", "initial_balance"], _safe_get(self.cfg, ["account", "initial_balance"], 10000.0)))
        self.max_lot: float = float(_safe_get(self.cfg, ["env_defaults", "max_lot"], 1.0))
        # sl/tp mode: 'pips' or 'fraction' (fraction of price); default pips
        self.sl_tp_mode: str = str(_safe_get(self.cfg, ["env_defaults", "sl_tp_mode"], "pips"))
        self.pip_size: float = float(_safe_get(self.cfg, ["env_defaults", "pip_size"], 0.01))  # pip value in price units for symbol; tune for XAUUSD etc.
        # market frictions (optional)
        self.spread: float = float(_safe_get(self.cfg, ["market", "spread"], 0.0))
        self.commission: float = float(_safe_get(self.cfg, ["market", "commission"], 0.0))
        self.slippage: float = float(_safe_get(self.cfg, ["market", "slippage"], 0.0))

        # risk limits
        self.max_drawdown: float = float(_safe_get(self.cfg, ["risk", "max_drawdown"], 0.2 * self.initial_balance))

        # ------ components init (best-effort) ------
        # DataHandler (prefer instance that takes config dict)
        if DataHandler is None:
            logger.error("DataHandler not available. TradingEnv cannot load market data.")
            raise RuntimeError("DataHandler missing")
        
        #---------------قطعه کد اضافه شده --------------------------
        # normalize config so DataHandler and TradingEnv agree
        df = self.config.get("data_fetch_defaults", {}) or {}
        # prefer explicit top-level, else nested
        if "n_candles_per_tf" in self.config:
            self.config["n_candles"] = int(self.config["n_candles_per_tf"])
        elif "n_candles" not in self.config and "n_candles" in df:
            self.config["n_candles"] = int(df.get("n_candles"))

        if "start_date" not in self.config and df.get("start_date"):
            self.config["start_date"] = df.get("start_date")
        if "end_date" not in self.config and df.get("end_date"):
            self.config["end_date"] = df.get("end_date")

        # ensure DataHandler uses same timeframes
        self.config["timeframes"] = self.timeframes
        #---------------------------------------------------------------
        
        try:
            # prefer ctor that accepts dict
            self.data_handler = DataHandler(self.cfg)
        except Exception:
            try:
                self.data_handler = DataHandler(config=self.cfg)
            except Exception as ex:
                logger.exception("Failed to instantiate DataHandler: %s", ex)
                raise

        # Executor: try common ctor signatures
        self.executor = None
        if Executor is not None:
            try:
                # try Executor(cfg=...)
                try:
                    self.executor = Executor(cfg=self.cfg)
                except TypeError:
                    try:
                        self.executor = Executor(self.cfg)
                    except TypeError:
                        self.executor = Executor()
                logger.info("Executor instantiated")
            except Exception as ex:
                logger.exception("Executor instantiation failed: %s", ex)
                self.executor = None
        else:
            logger.info("Executor not available; env will simulate orders internally.")

        # Risk manager
        self.risk = None
        if _RISK_MANAGER_IS_CLASS:
            try:
                self.risk = RiskManager(self.cfg)
            except Exception as ex:
                logger.exception("RiskManager class init failed: %s", ex)
                self.risk = None
        else:
            # module fallback (risk_manager_module may be None)
            self.risk = risk_manager_module if 'risk_manager_module' in globals() else None

        # ------ load market data for the symbol (multi-tf) ------
        # prefer fetch_for_symbol() -> {tf: DataFrame}
        self.market_data: Dict[str, pd.DataFrame] = {}
        try:
            if hasattr(self.data_handler, "fetch_for_symbol"):
                self.market_data = self.data_handler.fetch_for_symbol(self.symbol) or {}
            elif hasattr(self.data_handler, "fetch_all"):
                all_ = self.data_handler.fetch_all()
                self.market_data = all_.get(self.symbol, {}) if isinstance(all_, dict) else {}
            elif hasattr(self.data_handler, "fetch_for_symbol"):
                # fallback: use single timeframe from data_handler
                tf = getattr(self.data_handler, "timeframe", self.timeframes[0])
                df = self.data_handler.fetch_for_symbol(self.symbol)
                self.market_data[tf] = df if df is not None else pd.DataFrame()
            else:
                logger.warning("DataHandler has no known fetch method; market_data empty.")
        except Exception as ex:
            logger.exception("Error fetching market data: %s", ex)
            self.market_data = {}

        # normalize frames: lowercase columns and ensure at least open/high/low/close
        for tf, df in list(self.market_data.items()):
            if not isinstance(df, pd.DataFrame):
                self.market_data[tf] = pd.DataFrame()
                continue
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            self.market_data[tf] = df

        # compute max_steps based on shortest timeframe series (sliding windows)
        lengths = []
        for tf in self.timeframes:
            df = self.market_data.get(tf)
            lengths.append(len(df) if isinstance(df, pd.DataFrame) else 0)
        # if some series empty, we allow env to run but max_steps will be 0 => immediate done
        possible = [max(0, L - self.n_candles_per_tf + 1) for L in lengths] if lengths else [0]
        self.max_steps = int(min(possible)) if possible else 0

        # observation/ action spaces
        self.n_features = 4  # open, high, low, close (volume excluded for now)
        self.candles_flat_size = len(self.timeframes) * self.n_candles_per_tf * self.n_features
        #------------------------------------- Time Features
        self.time_feature_names = [
            "hour_norm", "hour_sin", "hour_cos","dow_norm", "dow_sin", "dow_cos",
            "min_norm", "is_london", "is_newyork", "is_asia"
        ]
        self.extra_features = 5 + len(self.time_feature_names)  # account info + time
        obs_size = self.candles_flat_size + self.extra_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )        
        #-------------------------------------
        self.extra_features = 5  # balance, equity, free_margin (placeholder), total_volume, last_dir
        obs_size = self.candles_flat_size + self.extra_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Action: structured dict for extensibility
        # type: 0=hold,1=buy,2=sell,3=close_last
        self.action_space = spaces.Dict({
            "type": spaces.Discrete(4),
            "volume": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # fraction of max_lot
            "sl": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # fraction or pips depending on mode
            "tp": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "trailing": spaces.Discrete(2),
            "riskfree": spaces.Discrete(2),
        })

        # trading internals
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.equity = float(self.initial_balance)
        self.positions: List[Dict[str, Any]] = []
        self.done = False
        self._last_equity = self.equity

        # ----- for support_resistanceMultiTF -----
        self.sr_module = SupportResistanceMultiTF(
            lookback=5,
            cluster_method="kmeans",
            n_clusters=5,
            prominence=0.001
        )

        # ----- trailing manager (Chandelier / ATR) -----
        self.trailing_manager = TrailingStopManager(cfg=self.cfg)


    # ---------------- Gym API ----------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.equity = float(self.initial_balance)
        self.positions = []
        self.done = False
        self._last_equity = self.equity
        # clear data handler cache if exists
        if hasattr(self.data_handler, "clear_cache"):
            try:
                self.data_handler.clear_cache()
            except Exception:
                pass
        # re-fetch market data to start fresh
        try:
            if hasattr(self.data_handler, "fetch_for_symbol"):
                self.market_data = self.data_handler.fetch_for_symbol(self.symbol) or {}
            elif hasattr(self.data_handler, "fetch_all"):
                all_ = self.data_handler.fetch_all()
                self.market_data = all_.get(self.symbol, {}) if isinstance(all_, dict) else {}
        except Exception as ex:
            logger.exception("reset: failed fetching market data: %s", ex)
            self.market_data = {}
        # normalize
        for tf, df in list(self.market_data.items()):
            if isinstance(df, pd.DataFrame):
                df.columns = [c.lower() for c in df.columns]
                self.market_data[tf] = df
            else:
                self.market_data[tf] = pd.DataFrame()
        # recompute max_steps
        lengths = [len(self.market_data.get(tf, pd.DataFrame())) for tf in self.timeframes]
        poss = [max(0, L - self.n_candles_per_tf + 1) for L in lengths] if lengths else [0]
        self.max_steps = int(min(poss)) if poss else 0

        # -------- افزودن محاسبه واگرایی‌ها --------
        try:
            df = self.data_handler.fetch_for_symbol(self.symbol, self.timeframes)
            df = Indicators.detect_divergences_extrema(
                df, indicator_columns=['RSI', 'Stochastic RSI', 'MACD', 'CCI', 'MFI']
            )
            # اینجا می‌توانید ستون‌های واگرایی را به self.market_data یا observation اضافه کنید
            # مثال ساده (می‌توانید طبق ساختار observation خودتان تغییر دهید):
            self.divergences = df[[col for col in df.columns if 'divergence' in col.lower()]]
        except Exception as ex:
            logger.exception("reset: failed computing divergences: %s", ex)
            self.divergences = pd.DataFrame()

        obs = self._get_state()
        return obs, {}

    def step(self, action: Dict[str, Any]):
        if self.done:
            raise RuntimeError("step() called after done=True; call reset()")

        # -----------------------------
        # ۱. اعمال اکشن
        # -----------------------------
        self._apply_action(action)

        # -----------------------------
        # ۲. تعیین قیمت فعلی (استفاده از تایم‌فریم دقیق‌ترین)
        # -----------------------------
        price = self._get_price_for_step(
            self.timeframes[0], self.current_step + self.n_candles_per_tf - 1
        )

        # -----------------------------
        # ۳. به‌روزرسانی پوزیشن‌ها و محاسبه PnL واقعی
        # -----------------------------
        self._update_positions(price)

        # -----------------------------
        # ۴. پیشروی گام زمانی
        # -----------------------------
        self.current_step += 1

        # -----------------------------
        # ۵. محاسبه state و reward با استفاده از _get_state() اصلاح شده
        # -----------------------------
        state = self._get_state()
        reward = self._calculate_reward()
        self._last_equity = self.equity
        self.done = self._check_done()

        # -----------------------------
        # ۶. اطلاعات اضافی
        # -----------------------------
        info = {
            "balance": self.balance,
            "equity": self.equity,
            "positions": self.positions,
            "current_step": self.current_step,
        }

        # -----------------------------
        # ۷. بازگشت مقادیر
        # -----------------------------
        return state, float(reward), bool(self.done), False, info

    def render(self, mode: str = "human"):
        logger.info("Step %d / %d | Balance: %.4f | Equity: %.4f | Open pos: %d",
                    self.current_step, self.max_steps, self.balance, self.equity, len(self.positions))

    def close(self):
        try:
            if hasattr(self.data_handler, "close"):
                self.data_handler.close()
        except Exception:
            pass
        try:
            if self.executor and hasattr(self.executor, "close"):
                self.executor.close()
        except Exception:
            pass

    # ---------------- internal helpers ----------------
    def _get_state_old(self) -> Dict[str, Any]:
        """
        محاسبه وضعیت فعلی محیط (state) برای عامل RL
        شامل داده‌های پوزیشن‌ها، بالانس، اندیکاتورها و واگرایی‌ها
        """

        # -----------------------------
        # ۱. ساختار پایه state
        # -----------------------------
        state = {
            "balance": self.balance,
            "equity": self.equity,
            "positions": self.positions.copy(),   # کپی تا از تغییر ناخواسته جلوگیری بشه
            "current_step": self.current_step,
        }

        # -----------------------------
        # ۲. داده‌های اصلی قیمت‌ها و اندیکاتورها
        # -----------------------------
        try:
            df = self.data_handler.fetch_for_symbol(self.symbol, self.timeframes)
        except Exception as e:
            raise RuntimeError(f"خطا در دریافت داده برای {self.symbol}: {e}")

        # اگر دیتافریم خالی بود
        if df is None or df.empty:
            return state

        # -----------------------------
        # ۳. محاسبه واگرایی‌ها
        # -----------------------------
        try:
            df = Indicators.detect_divergences_extrema(
                df,
                indicator_columns=["RSI", "Stochastic RSI", "MACD", "CCI", "MFI"]
            )
        except Exception as e:
            raise RuntimeError(f"خطا در محاسبه واگرایی‌ها: {e}")

        # آخرین ردیف (جدیدترین کندل) برای استخراج فیچرها
        latest = df.iloc[-1].to_dict()

        # -----------------------------
        # ۴. افزودن فیچرها به state
        # -----------------------------
        for col, val in latest.items():
            state[col] = float(val) if pd.notnull(val) else 0.0

        # --- اضافه کردن سطوح حمایت/مقاومت Multi-Timeframe ---
        if hasattr(self, "sr_module") and hasattr(self.data_handler, "fetch_for_symbol"):
            # داده‌های چند تایم‌فریم را بگیریم
            df_dict = self.data_handler.fetch_for_symbol(self.symbol, self.timeframes)
            sr_dict, combined_levels = self.sr_module.get_levels_multitf(df_dict)

            # برای ساده‌سازی، فقط سطوح تلفیقی را به state اضافه می‌کنیم
            state.extend(combined_levels)

            # در صورت نیاز می‌توان سطوح هر تایم‌فریم را هم اضافه کرد
            # برای tf, vals in sr_dict.items():
            #     state.extend(vals['levels'])

        return np.array(state, dtype=np.float32)

    def _get_state(self) -> np.ndarray:
        """
        محاسبه وضعیت فعلی محیط (state) برای عامل RL
        شامل داده‌های پوزیشن‌ها، بالانس، اندیکاتورها، واگرایی‌ها و سطوح حمایت/مقاومت Multi-Timeframe
        """

        # -----------------------------
        # ۱. ساختار پایه state
        # -----------------------------
        state = {
            "balance": self.balance,
            "equity": self.equity,
            "positions": self.positions.copy(),  # جلوگیری از تغییر ناخواسته
            "current_step": self.current_step,
        }

        # -----------------------------
        # ۲. داده‌های اصلی قیمت‌ها و اندیکاتورها
        # -----------------------------
        try:
            df = self.data_handler.fetch_for_symbol(self.symbol, self.timeframes)
        except Exception as e:
            raise RuntimeError(f"خطا در دریافت داده برای {self.symbol}: {e}")

        if df is None or df.empty:
            return np.array(list(state.values()), dtype=np.float32)

        # -----------------------------
        # ۳. محاسبه واگرایی‌ها
        # -----------------------------
        try:
            df = Indicators.detect_divergences_extrema(
                df,
                indicator_columns=["RSI", "Stochastic RSI", "MACD", "CCI", "MFI"]
            )
        except Exception as e:
            raise RuntimeError(f"خطا در محاسبه واگرایی‌ها: {e}")

        latest = df.iloc[-1].to_dict()

        # -----------------------------
        # ۴. افزودن فیچرهای اندیکاتورها و واگرایی‌ها
        # -----------------------------
        for col, val in latest.items():
            state[col] = float(val) if pd.notnull(val) else 0.0

        # -----------------------------
        # ۵. افزودن سطوح حمایت/مقاومت Multi-Timeframe
        # -----------------------------
        if hasattr(self, "sr_module") and hasattr(self.data_handler, "fetch_for_symbol"):
            df_dict = self.data_handler.fetch_for_symbol(self.symbol, self.timeframes)
            sr_dict, combined_levels = self.sr_module.get_levels_multitf(df_dict)

            # اضافه کردن سطوح تلفیقی به state با کلیدهای sr_0, sr_1, ...
            for i, level in enumerate(combined_levels):
                state[f"sr_{i}"] = float(level)

        # -----------------------------
        # ۶. تبدیل نهایی به numpy array
        # -----------------------------
        return np.array(list(state.values()), dtype=np.float32)

    def _get_base_timestamp_for_step(self) -> Optional[pd.Timestamp]:
        """Return the timestamp (end time) of the base TF at current_step.
        Assumes self.timeframes[0] is the most-granular timeframe.
        """
        base_tf = self.timeframes[0]
        df = self.market_data.get(base_tf, pd.DataFrame())
        if df is None or df.empty:
            return None
        # prefer datetime index, else try 'time' column
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            pos = self.current_step + self.n_candles_per_tf - 1
            pos = max(0, min(pos, len(df)-1))
            return df.index[pos]
        if 'time' in df.columns:
            ts = pd.to_datetime(df['time'], errors='coerce')
            pos = self.current_step + self.n_candles_per_tf - 1
            pos = max(0, min(pos, len(ts)-1))
            return ts.iloc[pos]
        return None

    def _slice_window_for_tf(self, tf: str, start: int) -> np.ndarray:
        """
        Align window for tf using base timestamp for current step.
        Returns flattened (n_candles_per_tf x 4) array padded with NaN when needed.
        """
        df = self.market_data.get(tf, pd.DataFrame())
        cols = ['open', 'high', 'low', 'close']
        if df.empty:
            return np.full((self.n_candles_per_tf * 4,), np.nan, dtype=float)

        # get base timestamp
        base_ts = self._get_base_timestamp_for_step()
        # find the index in this tf which is the last <= base_ts
        if base_ts is not None:
            if isinstance(df.index, pd.DatetimeIndex):
                pos = df.index.get_indexer([base_ts], method='ffill')[0]
            elif 'time' in df.columns:
                times = pd.to_datetime(df['time'], errors='coerce')
                pos = np.searchsorted(times.values, np.datetime64(base_ts), side='right') - 1
            else:
                pos = start + self.n_candles_per_tf - 1  # fallback to original index logic
        else:
            pos = start + self.n_candles_per_tf - 1

        # now compute window start/end on this tf
        end_idx = int(max(0, min(pos, len(df)-1)))
        start_idx = end_idx - self.n_candles_per_tf + 1
        if start_idx < 0:
            pad_rows = -start_idx
            start_idx = 0
        else:
            pad_rows = 0

        slice_df = df.iloc[start_idx:end_idx+1]
        # extract columns in desired order, pad missing columns with NaN
        available = [c for c in cols if c in slice_df.columns]
        if slice_df.empty:
            vals = np.empty((0,4))
        else:
            vals = slice_df[available].values
        if vals.shape[1] < 4:
            mat = np.full((vals.shape[0], 4), np.nan, dtype=float)
            if vals.size > 0:
                mat[:, :vals.shape[1]] = vals
            vals = mat
        if pad_rows > 0:
            pad = np.full((pad_rows, 4), np.nan, dtype=float)
            vals = np.vstack([pad, vals])
        # ensure rows == n_candles_per_tf
        if vals.shape[0] < self.n_candles_per_tf:
            extra = self.n_candles_per_tf - vals.shape[0]
            pad2 = np.full((extra,4), np.nan, dtype=float)
            vals = np.vstack([pad2, vals])
        return vals.flatten()

    def _get_price_for_step(self, timeframe: str, idx: int) -> float:
        df = self.market_data.get(timeframe, pd.DataFrame())
        if df.empty:
            return float("nan")
        base_ts = self._get_base_timestamp_for_step()
        if base_ts is not None:
            if isinstance(df.index, pd.DatetimeIndex):
                pos = df.index.get_indexer([base_ts], method='ffill')[0]
            elif 'time' in df.columns:
                times = pd.to_datetime(df['time'], errors='coerce')
                pos = np.searchsorted(times.values, np.datetime64(base_ts), side='right') - 1
            else:
                pos = idx
        else:
            pos = idx
        # clamp and return close/open...
        pos = max(0, min(pos, len(df)-1))
        row = df.iloc[pos]
        if 'close' in row.index:
            return float(row['close'])
        for c in ['open','high','low']:
            if c in row.index:
                return float(row[c])
        return float("nan")

    def _apply_action(self, action: Dict[str, Any]):
        """
        Interpret and execute action.
        Tries to use Executor if available; otherwise simulates a position.
        """
        # normalize action dict (accept numpy arrays)
        typ = int(action.get("type", 0)) if isinstance(action, dict) else int(action[0])
        vol_frac = float(np.asarray(action.get("volume", 0.0)).reshape(-1)[0]) if isinstance(action, dict) else float(action[1])
        sl_frac = float(np.asarray(action.get("sl", 0.0)).reshape(-1)[0]) if isinstance(action, dict) else float(action[2])
        tp_frac = float(np.asarray(action.get("tp", 0.0)).reshape(-1)[0]) if isinstance(action, dict) else float(action[3])
        trailing_flag = bool(int(action.get("trailing", 0))) if isinstance(action, dict) else bool(int(action[4]))
        riskfree_flag = bool(int(action.get("riskfree", 0))) if isinstance(action, dict) else False

        lot = max(0.0, min(1.0, vol_frac)) * self.max_lot
        # price at entry
        price = self._get_price_for_step(self.timeframes[0], self.current_step + self.n_candles_per_tf - 1)

        if typ == 0:
            # hold
            return
        if typ == 3:
            # close last position
            if self.positions:
                last = self.positions.pop()
                exit_price = price
                pnl = last["direction"] * (exit_price - last["entry_price"]) * last["volume"]
                self.balance += pnl
                logger.debug("Closed last position (sim): pnl=%.6f", pnl)
            return

        direction = 1 if typ == 1 else -1
        # compute sl/tp absolute price if provided (sl_frac, tp_frac)
        sl_price = None
        tp_price = None
        if sl_frac and not math.isnan(price):
            if self.sl_tp_mode == "pips":
                sl_price = price - direction * (sl_frac * self.pip_size)
            else:
                sl_price = price - direction * (sl_frac * price)
        if tp_frac and not math.isnan(price):
            if self.sl_tp_mode == "pips":
                tp_price = price + direction * (tp_frac * self.pip_size)
            else:
                tp_price = price + direction * (tp_frac * price)

        #------ قطعه مد اصافه شده -------------------------
        # pip size fallback: از self.pip_size استفاده کن (موجود در cfg) یا 0.0001
        pip_size = float(getattr(self, "pip_size", 0.0001))
        sl_pips = None
        tp_pips = None
        if sl_price is not None and not math.isnan(price):
            sl_pips = abs(price - sl_price) / pip_size
        if tp_price is not None and not math.isnan(price):
            tp_pips = abs(price - tp_price) / pip_size
        #--------------------------------------------------

        # try to place order via Executor
        placed = False
        order_info = {}
        if self.executor is not None:
            # try common method names
            try_methods = [
                ("place_market_order", {"symbol": self.symbol, "side": "buy" if direction == 1 else "sell", "lot": lot, "sl": sl_price, "tp": tp_price, "sl_pips": sl_pips, "tp_pips": tp_pips, "trailing": trailing_flag, "deviation": int(getattr(self.executor, "default_deviation", 20))}),
                ("execute_order",      {"symbol": self.symbol, "side": "buy" if direction == 1 else "sell", "lot": lot, "sl": sl_price, "tp": tp_price, "sl_pips": sl_pips, "tp_pips": tp_pips,}),
                ("open_position",      {"symbol": self.symbol, "direction": direction,                      "lot": lot, "sl": sl_price, "tp": tp_price, "sl_pips": sl_pips, "tp_pips": tp_pips,}),
            ]
            
            for m, kwargs in try_methods:
                if hasattr(self.executor, m):
                    try:
                        fn = getattr(self.executor, m)
                        res = fn(**kwargs)
                        # interpret result: expect dict with success flag or truthy result
                        if isinstance(res, dict):
                            # old & Changed:  ok = res.get("success", True)  # default True if no explicit key
                            ok = bool(res.get("ok", res.get("success", False)))
                            order_info = res
                        else:
                            ok = bool(res)
                            order_info = {"result": res}
                        if ok:
                            placed = True
                            break
                    except Exception as ex:
                        logger.debug("Executor method %s raised: %s", m, ex)
                        continue
        # if not placed, simulate position locally
        if not placed:
            pos = {
                "entry_price": price,
                "entry_step": self.current_step,
                "volume": lot,
                "direction": direction,
                "sl": sl_price,
                "tp": tp_price,
                "trailing": bool(trailing_flag),
                "riskfree": bool(riskfree_flag),
            }
            self.positions.append(pos)
            logger.debug("Simulated open position: %s", pos)
        else:
            # if executor placed and returned order_info but no position details, create simulated wrapper
            if order_info and ("entry_price" in order_info or "price" in order_info or "filled_price" in order_info):
                entry_p = order_info.get("entry_price") or order_info.get("price") or order_info.get("filled_price") or price
                pos = {
                    "entry_price": float(entry_p),
                    "entry_step": self.current_step,
                    "volume": float(order_info.get("volume", lot)),
                    "direction": int(order_info.get("direction", direction)),
                    "sl": order_info.get("sl", sl_price),
                    "tp": order_info.get("tp", tp_price),
                    "trailing": bool(order_info.get("trailing", trailing_flag)),
                    "riskfree": bool(order_info.get("riskfree", riskfree_flag)),
                }
                self.positions.append(pos)

    def _update_positions(self, current_price: float):
        """
        Update positions: check SL/TP/trailing/riskfree, realize PnL for closed positions,
        and update balance/equity accordingly.
        """
        if current_price is None or (isinstance(current_price, float) and math.isnan(current_price)):
            # can't update
            return
        realized = 0.0
        survivors = []
        for pos in self.positions:
            vol = pos.get("volume", 0.0)
            dir_ = pos.get("direction", 1)
            entry = pos.get("entry_price", current_price)
            u_pnl = dir_ * (current_price - entry) * vol
            closed = False
            # TP
            if pos.get("tp") is not None:
                if (dir_ == 1 and current_price >= pos["tp"]) or (dir_ == -1 and current_price <= pos["tp"]):
                    realized += u_pnl
                    closed = True
            # SL
            if pos.get("sl") is not None and not closed:
                if (dir_ == 1 and current_price <= pos["sl"]) or (dir_ == -1 and current_price >= pos["sl"]):
                    realized += u_pnl
                    closed = True
            if closed:
                logger.debug("Position closed by SL/TP: entry=%.6f cur=%.6f pnl=%.6f", entry, current_price, u_pnl)
                continue
            # riskfree: move SL to breakeven if profit threshold reached
            if pos.get("riskfree"):
                threshold = float(_safe_get(self.cfg, ["env_defaults", "riskfree_profit_threshold"], 0.005))
                if u_pnl >= threshold * entry * vol:
                    new_sl = entry
                    if pos.get("sl") is None or (dir_ == 1 and new_sl > pos["sl"]) or (dir_ == -1 and new_sl < pos["sl"]):
                        pos["sl"] = new_sl
                        logger.debug("Riskfree applied: new SL=%.6f", new_sl)
            
            # trailing (use advanced TrailingStopManager if available)
            if pos.get("trailing"):
                try:
                    base_df = self.market_data.get(self.timeframes[0], pd.DataFrame())
                    # obtain base timeframe timestamp for this step
                    base_ts = self._get_base_timestamp_for_step()
                    # compute new SL using manager (returns None if not applicable)
                    new_sl = None
                    if hasattr(self, "trailing_manager") and self.trailing_manager is not None:
                        new_sl = self.trailing_manager.compute_new_sl(
                            pos, base_df, current_ts=base_ts, current_price=current_price
                        )
                    # fallback to old simple trailing if manager didn't return a SL
                    if new_sl is None:
                        trigger = float(_safe_get(self.cfg, ["env_defaults", "trailing_trigger"], 0.01))
                        trailing_distance = float(_safe_get(self.cfg, ["env_defaults", "trailing_distance"], 0.005))
                        if u_pnl >= trigger * entry * vol:
                            new_sl = current_price - dir_ * (trailing_distance * current_price)
                    # apply new_sl if it's better/closer in direction of locking profit
                    if new_sl is not None:
                        if pos.get("sl") is None or (dir_ == 1 and new_sl > pos["sl"]) or (dir_ == -1 and new_sl < pos["sl"]):
                            pos["sl"] = new_sl
                            logger.debug("Trailing SL updated to %.6f", new_sl)
                except Exception as ex:
                    logger.exception("Trailing manager error: %s", ex)


            survivors.append(pos)
        # realize pnl to balance
        if realized != 0.0:
            self.balance += realized
        self.positions = survivors
        # update equity: balance + unrealized
        unreal = sum(p.get("direction", 1) * (current_price - p.get("entry_price", current_price)) * p.get("volume", 0.0) for p in self.positions)
        self.equity = self.balance + unreal

    def _calculate_reward(self) -> float:
        """
        Simplified reward: equity change since last step.
        """
        new_equity = self.balance  # simplified assumption
        reward = new_equity - self.equity
        self.equity = new_equity
        return reward

    def _check_done(self) -> bool:
        if self.balance <= 0:
            logger.info("Done: balance depleted")
            return True
        if self.max_steps is not None and self.current_step >= self.max_steps:
            logger.info("Done: max_steps reached")
            return True
        # drawdown
        if (self.initial_balance - self.equity) > float(self.max_drawdown):
            logger.info("Done: max_drawdown exceeded")
            return True
        return False

    def _estimate_free_margin(self) -> float:
        """
        Robust estimate of free margin.

        Strategy (priority):
        1. If self.risk is a RiskManager instance with get_free_margin(...) -> use it.
        2. If self.risk is the risk_manager module exposing used_margin_from_positions(...) and get_free_margin(...) -> use them.
        3. Fallback: compute rough used margin from open positions:
            used_margin += (lot * contract_size * price) / leverage
        where contract_size is taken from risk.get_symbol_info() if available, else sensible defaults.
        Returns free margin = max(0.0, balance - used_margin).
        The function is defensive (catches exceptions) and logs warnings on failure.
        """
        try:
            # 1) If RiskManager class instance with get_free_margin(balance, positions, ...) exists
            if self.risk is not None and _RISK_MANAGER_IS_CLASS and hasattr(self.risk, "get_free_margin"):
                try:
                    # RiskManager.get_free_margin(self, balance, positions, symbol_info_map=None)
                    return float(self.risk.get_free_margin(self.balance, self.positions, None))
                except TypeError:
                    # try alternative signature get_free_margin(balance, used_margin)
                    try:
                        used = self.risk.used_margin_from_positions(self.positions, leverage=float(_safe_get(self.cfg, ["env_defaults", "leverage"], 100)))
                        return float(self.risk.get_free_margin(self.balance, used))
                    except Exception:
                        logger.debug("RiskManager.get_free_margin fallback attempt failed.", exc_info=True)

            # 2) If risk is a module exposing used_margin_from_positions(...) and get_free_margin(...)
            if self.risk is not None and not _RISK_MANAGER_IS_CLASS:
                if hasattr(self.risk, "used_margin_from_positions") and hasattr(self.risk, "get_free_margin"):
                    try:
                        leverage = float(_safe_get(self.cfg, ["env_defaults", "leverage"], _safe_get(self.cfg, ["account", "leverage"], 100)))
                        used = self.risk.used_margin_from_positions(self.positions, leverage=leverage)
                        return float(self.risk.get_free_margin(self.balance, used))
                    except Exception:
                        logger.debug("risk module margin functions raised; will fallback.", exc_info=True)

            # 3) Fallback manual estimate
            used_margin = 0.0
            leverage = float(_safe_get(self.cfg, ["env_defaults", "leverage"], _safe_get(self.cfg, ["account", "leverage"], 100)))
            # get a helper for symbol info if available
            symbol_info_fn = None
            if self.risk is not None and hasattr(self.risk, "get_symbol_info"):
                symbol_info_fn = getattr(self.risk, "get_symbol_info")
            elif "risk_manager_module" in globals() and hasattr(globals()["risk_manager_module"], "get_symbol_info"):
                symbol_info_fn = getattr(globals()["risk_manager_module"], "get_symbol_info")

            # a helper to get a reasonable price for a position
            def _pos_price(p):
                return float(p.get("entry_price") or p.get("price") or p.get("filled_price") or
                            self._get_price_for_step(self.timeframes[0], self.current_step + self.n_candles_per_tf - 1) or 0.0)

            for p in (self.positions or []):
                try:
                    lot = float(p.get("volume") or p.get("lot") or p.get("lots") or 0.0)
                    if lot <= 0:
                        continue
                    symbol = p.get("symbol") or self.symbol
                    price = _pos_price(p)
                    # default contract size
                    contract = 100000.0
                    if symbol_info_fn is not None:
                        try:
                            info = symbol_info_fn(symbol)
                            if isinstance(info, dict) and info.get("contract_size") is not None:
                                contract = float(info.get("contract_size"))
                        except Exception:
                            logger.debug("symbol_info_fn failed for %s", symbol, exc_info=True)
                    # required margin approximation
                    used_margin += (lot * contract * max(1.0, price)) / max(1.0, leverage)
                except Exception:
                    logger.debug("Error estimating used margin for position %s", p, exc_info=True)
                    continue

            free = max(0.0, float(self.balance) - float(used_margin))
            return float(free)

        except Exception as ex:
            # final fallback: return balance (safe, non-negative)
            try:
                logger.warning("Failed to estimate free margin robustly: %s. Returning balance as free margin (fallback).", ex)
            except Exception:
                pass
            return float(max(0.0, getattr(self, "balance", 0.0)))

    def _get_base_timestamp_for_step(self) -> pd.Timestamp:
        """
        Fetch the timestamp for current step.
        Uses the latest candle of the lowest timeframe.
        """
        try:
            df = self.data_handler.get_candles(self.symbol, [self.timeframes[0]], self.n_candles_per_tf)
            if isinstance(df, np.ndarray):
                df = df.reshape(-1, 5)  # OHLCV
            # assume last row index is timestamp
            ts = pd.Timestamp.now(tz="UTC")
            return ts
        except Exception as e:
            self.logger.warning(f"Timestamp fetch failed: {e}")
            return pd.NaT

    # ----------------- utilities -----------------
    def get_state_size(self) -> int:
        return int(self.candles_flat_size + self.extra_features)

    def get_action_space(self) -> spaces.Space:
        return self.action_space

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframes": self.timeframes,
            "n_candles_per_tf": self.n_candles_per_tf,
            "obs_size": self.get_state_size(),
            "max_steps": self.max_steps,
        }

    # --------------------------------------------------
    # summary
    # --------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframes": self.timeframes,
            "n_candles_per_tf": self.n_candles_per_tf,
            "base_obs_size": self.candles_flat_size + 5,
            "time_features": self.time_feature_names,
            "obs_size": self.candles_flat_size + self.extra_features,
            "max_steps": self.max_steps,
        }