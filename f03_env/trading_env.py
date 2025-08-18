# f03_env/trading_env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Any, Optional

from f02_data.data_handler import DataHandler

import logging
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment using DataHandler as source of market data.
    - Expects DataHandler.fetch_for_symbol(symbol) -> Dict[timeframe -> DataFrame]
    - Each DataFrame should contain at least columns ['open','high','low','close'] (or mappable ones).
    - The env makes sliding windows of `n_candles` for each timeframe.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config or {}

        # ---- config basics ----
        self.symbol: str = self.config.get("symbol", self.config.get("symbols", ["XAUUSD"])[0])
        self.timeframes: List[str] = self.config.get("timeframes", ["M1", "M5", "M30", "H4", "D1", "W1"])
        self.n_candles_per_tf: int = int(self.config.get("n_candles_per_tf", 15))
        self.initial_balance: float = float(self.config.get("initial_balance", 10000.0))
        self.max_steps: Optional[int] = None  # computed after loading data

        # ---- Data backend ----
        self.data_handler = DataHandler(self.config)

        # Preload market data (dict: timeframe -> DataFrame)
        self.market_data: Dict[str, pd.DataFrame] = self.data_handler.fetch_for_symbol(self.symbol)
        # standardize column names to lower-case for safety
        for tf, df in list(self.market_data.items()):
            if isinstance(df, pd.DataFrame):
                df.columns = [c.lower() for c in df.columns]
            else:
                self.market_data[tf] = pd.DataFrame()

        # compute available length and max_steps (sliding windows)
        lengths = []
        for tf in self.timeframes:
            df = self.market_data.get(tf)
            if df is None:
                lengths.append(0)
            else:
                lengths.append(len(df))
        if lengths:
            # max possible start index such that we can take n_candles window
            possible = [max(0, L - self.n_candles_per_tf + 1) for L in lengths]
            # if any timeframe has zero length -> environment will still run but steps limited by min(possible)
            self.max_steps = min(possible) if possible else 0
        else:
            self.max_steps = 0

        # ---- Observation space ----
        n_features = 4  # open, high, low, close
        # flattened candles: n_timeframes * n_candles_per_tf * n_features
        self.candles_flat_size = len(self.timeframes) * self.n_candles_per_tf * n_features

        # extra account features (balance, equity, free_margin, pos_volume_total, pos_direction_last)
        self.extra_features = 5
        obs_size = self.candles_flat_size + self.extra_features

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # ---- Action space: structured for extensibility ----
        # action['type']: 0=hold,1=buy,2=sell,3=close_last
        # action['volume']: fraction of max_lot (0..1)
        # action['sl'], action['tp']: normalized fraction of price (0 means ignore)
        # action['trailing_enable'], action['riskfree_enable']: binary flags
        self.action_space = spaces.Dict({
            "type": spaces.Discrete(4),
            "volume": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "sl": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "tp": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "trailing": spaces.Discrete(2),
            "riskfree": spaces.Discrete(2),
        })

        # ---- trading internal state ----
        self.current_step: int = 0  # sliding window start index
        self.balance: float = self.initial_balance
        self.equity: float = self.initial_balance
        self.positions: List[Dict[str, Any]] = []  # stack of open positions
        self.done: bool = False

        # parameters for simple execution rules (can be tuned via config)
        self.max_lot: float = float(self.config.get("max_lot", 1.0))
        self.trailing_distance: float = float(self.config.get("trailing_distance", 0.005))  # price fraction (e.g., 0.005 = 0.5%)
        self.trailing_trigger: float = float(self.config.get("trailing_trigger", 0.01))  # profit fraction to enable trailing
        self.riskfree_profit_threshold: float = float(self.config.get("riskfree_profit_threshold", 0.005))  # profit fraction to move SL to breakeven

        # For reward bookkeeping
        self._last_equity: float = self.equity

        # build initial state
        self.state = self._get_state()

    def reset(self):
        """
        Reset env to start of a new episode.
        If you want random episodes, you can randomize current_step here within valid range.
        """
        # clear positions and balances
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.done = False
        self._last_equity = self.equity

        # reload market data (clear cache if needed)
        # keep the same DataHandler instance but clear its cache so fresh fetch if necessary
        if hasattr(self.data_handler, "clear_cache"):
            try:
                self.data_handler.clear_cache()
            except Exception:
                pass
        self.market_data = self.data_handler.fetch_for_symbol(self.symbol)
        for tf, df in list(self.market_data.items()):
            if isinstance(df, pd.DataFrame):
                df.columns = [c.lower() for c in df.columns]
            else:
                self.market_data[tf] = pd.DataFrame()

        # recompute max_steps
        lengths = [len(self.market_data.get(tf, pd.DataFrame())) for tf in self.timeframes]
        possible = [max(0, L - self.n_candles_per_tf + 1) for L in lengths]
        self.max_steps = min(possible) if possible else 0

        self.state = self._get_state()
        return self.state

    def step(self, action: Dict[str, Any]):
        """
        Execute one step:
        - interpret action
        - update positions (mark-to-market)
        - check SL/TP/trailing/riskfree
        - advance sliding window (current_step += 1)
        - compute reward and done
        """
        if self.done:
            raise RuntimeError("Step called after done=True. Call reset() to start a new episode.")

        # interpret and apply action
        self._apply_action(action)

        # update positions PnL based on current price (we use the most granular timeframe's close)
        current_price = self._get_price_for_step(self.timeframes[0], self.current_step + self.n_candles_per_tf - 1)
        self._update_positions(current_price)

        # advance market window
        self.current_step += 1

        # new state & reward
        self.state = self._get_state()
        reward = self._calculate_reward()
        self._last_equity = self.equity
        self.done = self._check_done()

        info = {
            "balance": self.balance,
            "equity": self.equity,
            "positions": self.positions,
            "current_step": self.current_step,
        }
        return self.state, float(reward), bool(self.done), info

    # ----------------- internal helpers -----------------

    def _get_price_for_step(self, timeframe: str, idx: int) -> float:
        """
        Return a 'price' for given timeframe at index idx (iloc).
        If not available, fallback to last available close or NaN.
        """
        df = self.market_data.get(timeframe)
        if df is None or df.empty:
            return float("nan")
        # clamp idx
        if idx < 0:
            idx = 0
        if idx >= len(df):
            idx = len(df) - 1
        row = df.iloc[idx]
        # prefer 'close' column
        if 'close' in row.index:
            return float(row['close'])
        # fallback to last numeric column
        for v in ['open', 'high', 'low']:
            if v in row.index:
                return float(row[v])
        # if none available
        return float("nan")

    def _slice_window_for_tf(self, tf: str, start: int) -> np.ndarray:
        """
        Return flattened OHLC window of length n_candles_per_tf for timeframe tf,
        padded with NaN rows at top if source shorter than required.
        Output shape: (n_candles_per_tf * 4,)
        """
        df = self.market_data.get(tf, pd.DataFrame())
        # prepare empty matrix
        cols = ['open', 'high', 'low', 'close']
        if df.empty:
            pad = np.full((self.n_candles_per_tf, 4), np.nan, dtype=float)
            return pad.flatten()

        # determine slice indices (use iloc)
        start_idx = int(start)
        end_idx = start_idx + self.n_candles_per_tf
        # if df shorter than end_idx, pad at top
        if start_idx < 0:
            # pad beginning
            needed_top = min(-start_idx, self.n_candles_per_tf)
            pad = np.full((needed_top, 4), np.nan, dtype=float)
            slice_df = df.iloc[0:end_idx] if end_idx > 0 else df.iloc[0:0]
            vals = slice_df[[c for c in cols if c in slice_df.columns]].values
            # if columns missing, pad missing columns with NaN columns
            if vals.shape[1] < 4:
                # build matrix with correct cols
                mat = np.full((vals.shape[0], 4), np.nan, dtype=float)
                # fill matching columns left-to-right
                mat[:, :vals.shape[1]] = vals
                vals = mat
            arr = np.vstack([pad, vals])
        else:
            # normal case
            slice_df = df.iloc[start_idx:end_idx]
            vals = slice_df[[c for c in cols if c in slice_df.columns]].values if not slice_df.empty else np.empty((0, 4))
            # pad at top if not enough rows
            if vals.shape[0] < self.n_candles_per_tf:
                pad_rows = self.n_candles_per_tf - vals.shape[0]
                pad = np.full((pad_rows, 4), np.nan, dtype=float)
                # ensure vals has 4 columns
                if vals.shape[1] < 4:
                    mat = np.full((vals.shape[0], 4), np.nan, dtype=float)
                    if vals.size > 0:
                        mat[:, :vals.shape[1]] = vals
                    vals = mat
                arr = np.vstack([pad, vals])
            else:
                arr = vals
        return arr.flatten()

    def _get_state(self) -> np.ndarray:
        """
        Build the flattened observation vector:
        [tf1_window (flat), tf2_window (flat), ..., account_info(5)]
        account_info: balance, equity, free_margin (placeholder 0), total_pos_volume, last_pos_direction
        """
        windows = []
        for tf in self.timeframes:
            w = self._slice_window_for_tf(tf, self.current_step)
            windows.append(w)
        candles = np.concatenate(windows).astype(np.float32)

        total_volume = float(sum(p.get("volume", 0.0) for p in self.positions))
        last_dir = float(self.positions[-1]["direction"]) if self.positions else 0.0
        account_info = np.array([
            float(self.balance),
            float(self.equity),
            0.0,  # free margin (placeholder)
            total_volume,
            last_dir,
        ], dtype=np.float32)
        obs = np.concatenate([candles, account_info])
        return obs

    def _apply_action(self, action: Dict[str, Any]):
        """
        Interpret structured action and execute:
        - type: 0 hold, 1 buy, 2 sell, 3 close last position
        - volume: fraction of self.max_lot
        - sl/tp: normalized fractions -> translated into absolute price distance
        - trailing, riskfree: flags toggling behavior stored in position dict
        """
        # defensive conversions
        typ = int(action.get("type", 0))
        volume_frac = float(action.get("volume", [0.0])[0]) if hasattr(action.get("volume"), "__iter__") else float(action.get("volume", 0.0))
        sl_frac = float(action.get("sl", [0.0])[0]) if hasattr(action.get("sl"), "__iter__") else float(action.get("sl", 0.0))
        tp_frac = float(action.get("tp", [0.0])[0]) if hasattr(action.get("tp"), "__iter__") else float(action.get("tp", 0.0))
        trailing_flag = bool(int(action.get("trailing", 0)))
        riskfree_flag = bool(int(action.get("riskfree", 0)))

        volume = max(0.0, min(1.0, volume_frac)) * self.max_lot
        # current market price (use most granular tf close of current window end)
        price = self._get_price_for_step(self.timeframes[0], self.current_step + self.n_candles_per_tf - 1)

        if typ == 1 or typ == 2:
            direction = 1 if typ == 1 else -1
            pos = {
                "entry_price": price,
                "volume": volume,
                "direction": direction,
                "sl": None,
                "tp": None,
                "trailing": trailing_flag,
                "riskfree": riskfree_flag,
                "open_step": self.current_step,
            }
            # translate sl/tp fractions to absolute price distances if provided (>0)
            if sl_frac > 0 and price and not np.isnan(price):
                # sl_frac is fraction of price (e.g., 0.001 => 0.1%)
                pos["sl"] = price - direction * (sl_frac * price)
            if tp_frac > 0 and price and not np.isnan(price):
                pos["tp"] = price + direction * (tp_frac * price)
            # append to positions
            self.positions.append(pos)
            logger.debug("Opened position: %s", pos)
        elif typ == 3:
            # close last position (LIFO). A better logic could choose which to close.
            if self.positions:
                last = self.positions.pop()
                # realize PnL into balance (simple calculation)
                exit_price = price
                pnl = last["direction"] * (exit_price - last["entry_price"]) * last["volume"]
                self.balance += pnl
                logger.debug("Closed position: pnl=%.6f, new_balance=%.6f", pnl, self.balance)
        else:
            # hold -> nothing to do
            pass

    def _update_positions(self, current_price: float):
        """
        Update mark-to-market equity and check SL/TP/trailing/riskfree for each position.
        For closed positions we realize PnL into balance.
        """
        if current_price is None or np.isnan(current_price):
            # cannot update
            return

        realized = 0.0
        new_positions = []
        for pos in self.positions:
            # compute unrealized pnl (very simplified)
            u_pnl = pos["direction"] * (current_price - pos["entry_price"]) * pos["volume"]

            # check TP
            if pos.get("tp") is not None:
                if (pos["direction"] == 1 and current_price >= pos["tp"]) or (pos["direction"] == -1 and current_price <= pos["tp"]):
                    # take profit -> realize
                    realized += u_pnl
                    logger.debug("TP hit for pos %s: pnl=%.6f", pos, u_pnl)
                    continue  # do not keep this pos

            # check SL
            if pos.get("sl") is not None:
                if (pos["direction"] == 1 and current_price <= pos["sl"]) or (pos["direction"] == -1 and current_price >= pos["sl"]):
                    # stop loss -> realize
                    realized += u_pnl
                    logger.debug("SL hit for pos %s: pnl=%.6f", pos, u_pnl)
                    continue

            # riskfree: if profit above threshold and flag set, move SL to breakeven
            if pos.get("riskfree") and u_pnl >= (self.riskfree_profit_threshold * pos["entry_price"] * pos["volume"]):
                # set SL to entry_price (breakeven) if it would improve current SL
                new_sl = pos["entry_price"]
                if pos.get("sl") is None or (pos["direction"] == 1 and new_sl > pos["sl"]) or (pos["direction"] == -1 and new_sl < pos["sl"]):
                    pos["sl"] = new_sl
                    logger.debug("Riskfree applied, new SL=%s", pos["sl"])

            # trailing: simple rule - if profit exceed trigger, set SL to current_price - direction * trailing_distance*price
            if pos.get("trailing"):
                if u_pnl >= (self.trailing_trigger * pos["entry_price"] * pos["volume"]):
                    new_sl = current_price - pos["direction"] * (self.trailing_distance * current_price)
                    # update SL if it moves in favorable direction
                    if pos.get("sl") is None or (pos["direction"] == 1 and new_sl > pos["sl"]) or (pos["direction"] == -1 and new_sl < pos["sl"]):
                        pos["sl"] = new_sl
                        logger.debug("Trailing SL updated to %s for pos", pos["sl"])

            # keep position
            new_positions.append(pos)

        # replace positions with survivors
        self.positions = new_positions

        # update equity: equity = balance + unrealized pnl of open positions
        unrealized_total = sum(p["direction"] * (current_price - p["entry_price"]) * p["volume"] for p in self.positions)
        self.equity = self.balance + unrealized_total

        # if any realized pnl occurred, add to balance
        if realized != 0.0:
            self.balance += realized
            self.equity = self.balance + sum(p["direction"] * (current_price - p["entry_price"]) * p["volume"] for p in self.positions)

    def _calculate_reward(self) -> float:
        """
        Reward: change in equity since last step (simple). Can be replaced with
        risk-adjusted metrics (sharpe proxy, drawdown penalty, etc.)
        """
        return float(self.equity - self._last_equity)

    def _check_done(self) -> bool:
        """
        End episode if:
        - we exhausted the sliding windows (current_step >= max_steps)
        - or balance depleted
        """
        if self.balance <= 0:
            return True
        if self.max_steps is not None and self.current_step >= self.max_steps:
            return True
        return False

    def render(self, mode="human"):
        print(f"Step: {self.current_step}/{self.max_steps}, Balance: {self.balance:.4f}, Equity: {self.equity:.4f}, Positions: {len(self.positions)}")

    def close(self):
        try:
            if hasattr(self.data_handler, "close"):
                self.data_handler.close()
        except Exception:
            pass
