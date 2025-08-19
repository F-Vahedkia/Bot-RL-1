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

        obs = self._get_state()
        return obs, {}

    def step(self, action: Dict[str, Any]):
        if self.done:
            raise RuntimeError("step() called after done=True; call reset()")

        # apply action
        self._apply_action(action)

        # determine current price (use most granular timeframe first in list)
        price = self._get_price_for_step(self.timeframes[0], self.current_step + self.n_candles_per_tf - 1)

        # update positions: check sl/tp/trailing/riskfree and compute realized pnl
        self._update_positions(price)

        # advance time
        self.current_step += 1

        # compute new state and reward
        state = self._get_state()
        reward = self._calculate_reward()
        self._last_equity = self.equity
        self.done = self._check_done()

        info = {
            "balance": self.balance,
            "equity": self.equity,
            "positions": self.positions,
            "current_step": self.current_step,
        }
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
    def _get_state(self) -> np.ndarray:
        """
        Build flattened observation:
        [tf1_window_flat, tf2_window_flat, ..., balance, equity, free_margin, total_volume, last_dir]
        """
        windows = []
        for tf in self.timeframes:
            w = self._slice_window_for_tf(tf, self.current_step)
            windows.append(w)
        candles = np.concatenate(windows).astype(np.float32) if windows else np.full((self.candles_flat_size,), np.nan, dtype=np.float32)
        total_volume = float(sum(p.get("volume", 0.0) for p in self.positions))
        last_dir = float(self.positions[-1]["direction"]) if self.positions else 0.0
        free_margin = self._estimate_free_margin()
        account_info = np.array([float(self.balance), float(self.equity), float(free_margin), total_volume, last_dir], dtype=np.float32)
        obs = np.concatenate([candles, account_info])
        return obs

    def _slice_window_for_tf(self, tf: str, start: int) -> np.ndarray:
        """
        Return flattened n_candles_per_tf x 4 (open,high,low,close) array for timeframe tf.
        Pads top with NaN if insufficient rows.
        """
        df = self.market_data.get(tf, pd.DataFrame())
        cols = ['open', 'high', 'low', 'close']
        if df.empty:
            pad = np.full((self.n_candles_per_tf, 4), np.nan, dtype=float)
            return pad.flatten()
        # use iloc slice
        start_idx = int(start)
        end_idx = start_idx + self.n_candles_per_tf
        # clamp
        if start_idx < 0:
            start_idx = 0
            end_idx = self.n_candles_per_tf
        slice_df = df.iloc[start_idx:end_idx]
        # ensure columns exist and in order
        available_cols = [c for c in cols if c in slice_df.columns]
        if slice_df.empty:
            pad = np.full((self.n_candles_per_tf, 4), np.nan, dtype=float)
            return pad.flatten()
        vals = slice_df[available_cols].values
        # if missing some columns, pad those columns
        if vals.shape[1] < 4:
            mat = np.full((vals.shape[0], 4), np.nan, dtype=float)
            mat[:, :vals.shape[1]] = vals
            vals = mat
        # if fewer rows than required, pad top
        if vals.shape[0] < self.n_candles_per_tf:
            pad_rows = self.n_candles_per_tf - vals.shape[0]
            pad = np.full((pad_rows, 4), np.nan, dtype=float)
            vals = np.vstack([pad, vals])
        return vals.flatten()

    def _get_price_for_step(self, timeframe: str, idx: int) -> float:
        df = self.market_data.get(timeframe, pd.DataFrame())
        if df.empty:
            return float("nan")
        # clamp index
        if idx < 0:
            idx = 0
        if idx >= len(df):
            idx = len(df) - 1
        row = df.iloc[idx]
        if 'close' in row.index:
            return float(row['close'])
        for c in ['open', 'high', 'low']:
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
                            ok = res.get("success", True)  # default True if no explicit key
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
            # trailing
            if pos.get("trailing"):
                trigger = float(_safe_get(self.cfg, ["env_defaults", "trailing_trigger"], 0.01))
                trailing_distance = float(_safe_get(self.cfg, ["env_defaults", "trailing_distance"], 0.005))
                if u_pnl >= trigger * entry * vol:
                    new_sl = current_price - dir_ * (trailing_distance * current_price)
                    if pos.get("sl") is None or (dir_ == 1 and new_sl > pos["sl"]) or (dir_ == -1 and new_sl < pos["sl"]):
                        pos["sl"] = new_sl
                        logger.debug("Trailing SL updated to %.6f", new_sl)
            survivors.append(pos)
        # realize pnl to balance
        if realized != 0.0:
            self.balance += realized
        self.positions = survivors
        # update equity: balance + unrealized
        unreal = sum(p.get("direction", 1) * (current_price - p.get("entry_price", current_price)) * p.get("volume", 0.0) for p in self.positions)
        self.equity = self.balance + unreal

    def _calculate_reward(self) -> float:
        # simple reward: change in equity
        return float(self.equity - self._last_equity)

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

    def _estimate_free_margin_old(self) -> float:
        # best-effort: try risk manager module/class for accurate calc
        try:
            if self.risk is None:
                return 0.0
            if _RISK_MANAGER_IS_CLASS and hasattr(self.risk, "get_free_margin"):
                return float(self.risk.get_free_margin(balance=self.balance, equity=self.equity))
            elif not _RISK_MANAGER_IS_CLASS and hasattr(self.risk, "get_free_margin"):
                return float(self.risk.get_free_margin(self.balance, self.equity))
            # fallback simplistic estimate
            used_margin = sum(p.get("volume", 0.0) * 1000.0 for p in self.positions)  # placeholder
            return max(0.0, self.balance - used_margin)
        except Exception:
            return 0.0

    def _estimate_free_margin(self) -> float:
        try:
            # اگر risk یک کلاس با متد get_free_margin داره، ازش استفاده کن
            if self.risk is not None and _RISK_MANAGER_IS_CLASS and hasattr(self.risk, "get_free_margin"):
                return float(self.risk.get_free_margin(balance=self.balance, equity=self.equity))

            # اگر risk ماژولیه و تابع pip_value_per_lot یا _get_symbol_info داره، سعی می‌کنیم مارجین را برآورد کنیم
            if self.risk is not None and not _RISK_MANAGER_IS_CLASS:
                # estimate used margin by contract_size / leverage if possible
                try:
                    # try to read symbol info helper if exposed
                    if hasattr(self.risk, "_get_symbol_info"):
                        info = self.risk._get_symbol_info(self.symbol)
                        contract = float(info.get("contract_size", 100000.0))
                    else:
                        contract = 100000.0
                    leverage = float(_safe_get(self.cfg, ["env_defaults", "leverage"], _safe_get(self.cfg, ["account", "leverage"], 100)))
                    used_margin = 0.0
                    for p in self.positions:
                        entry = p.get("entry_price", 0.0) or 0.0
                        vol = float(p.get("volume", 0.0))
                        # approximate margin = (volume * contract * entry_price) / leverage
                        used_margin += (vol * contract * entry) / max(1.0, leverage)
                    free = max(0.0, self.balance - used_margin)
                    return float(free)
                except Exception:
                    pass

            # final fallback: conservative estimate (used margin ~ sum volume * 1000)
            used_margin = sum(p.get("volume", 0.0) * 1000.0 for p in self.positions)
            return max(0.0, self.balance - used_margin)
        except Exception:
            return 0.0

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
