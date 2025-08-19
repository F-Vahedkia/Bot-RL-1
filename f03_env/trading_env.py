# f03_env/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from f02_data.data_handler import DataHandler
from f09_execution.executor import Executor
from f10_utils.risk_manager import RiskManager
from f10_utils.config_loader import ConfigLoader
from f10_utils.logging_cfg import setup_logging

class TradingEnv(gym.Env):
    """
    Forex Trading Environment for RL agents.
    Integrates OHLCV, indicators, account info, and execution logic.

    State space:
        - OHLCV + indicators (from DataHandler)
        - Account info (balance, equity, margin, free_margin)

    Action space:
        - [0] Buy / Sell / Hold
        - [1] Lot size (multiples of 0.01)
        - [2] Stop Loss (pips)
        - [3] Take Profit (pips)
        - [4] Trailing Stop (pips)

    Reward:
        - Based on realized/unrealized PnL
        - Penalizes excessive drawdown
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config_path="f01_config/config_example.yaml"):
        super(TradingEnv, self).__init__()

        # --- Load config & logging ---
        self.config = ConfigLoader(config_path).config
        self.logger = setup_logging("TradingEnv")

        # --- Initialize components ---
        self.data_handler = DataHandler(config_path=config_path)
        self.executor = Executor(config_path=config_path)
        self.risk_manager = RiskManager(config_path=config_path)

        # --- Market & trading params ---
        self.symbol = self.config["trading"]["symbol"]
        self.initial_balance = self.config["account"]["initial_balance"]
        self.spread = self.config["market"]["spread"]
        self.commission = self.config["market"]["commission"]
        self.slippage = self.config["market"]["slippage"]

        # --- Spaces ---
        obs_size = self._get_observation_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, 5, 5, 0]),  # action params
            high=np.array([2, 1.0, 200, 200, 100]),
            dtype=np.float32,
        )

        # --- Internal state ---
        self.current_step = 0
        self.account_balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_positions = []
        self.done = False
        self.history = []

    # -----------------------------
    # Core Gym methods
    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.account_balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_positions = []
        self.done = False
        self.history = []

        self.data_handler.reset()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        action = [side, lot_size, sl, tp, trailing]
        side: 0=Hold, 1=Buy, 2=Sell
        """
        self.current_step += 1
        side, lot_size, sl, tp, trailing = self._parse_action(action)

        # --- Execute trade ---
        reward = 0
        trade_info = None
        if side in [1, 2]:  # Buy or Sell
            if self.risk_manager.validate_order(self.account_balance, lot_size):
                trade_info = self.executor.execute_order(
                    symbol=self.symbol,
                    side="buy" if side == 1 else "sell",
                    lot_size=lot_size,
                    sl=sl,
                    tp=tp,
                    trailing=trailing,
                )
                if trade_info["success"]:
                    self.open_positions.append(trade_info)

        # --- Update market data ---
        obs = self._get_observation()

        # --- Update account equity ---
        unrealized_pnl = self._calculate_unrealized_pnl()
        realized_pnl = self._close_positions_if_hit()
        self.equity = self.account_balance + unrealized_pnl

        # --- Reward function ---
        reward = realized_pnl + unrealized_pnl
        if self._max_drawdown_exceeded():
            reward -= 10
            self.done = True

        # --- Check episode end ---
        if self.current_step >= len(self.data_handler.data) - 1:
            self.done = True

        info = {"step": self.current_step, "equity": self.equity}
        return obs, reward, self.done, False, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.account_balance:.2f}, Equity: {self.equity:.2f}"
        )

    def close(self):
        self.data_handler = None
        self.executor = None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_observation_size(self):
        # OHLCV (5) + indicators (from data_handler) + account info (4)
        dummy_obs = self.data_handler.get_observation(0)
        obs_size = len(dummy_obs) + 4
        return obs_size

    def _get_observation(self):
        market_obs = self.data_handler.get_observation(self.current_step)
        account_obs = [
            self.account_balance,
            self.equity,
            self.risk_manager.get_margin(self.account_balance),
            self.risk_manager.get_free_margin(self.account_balance, self.equity),
        ]
        return np.array(market_obs + account_obs, dtype=np.float32)

    def _parse_action(self, action):
        side = int(round(action[0]))
        lot_size = round(action[1], 2)
        sl = int(round(action[2]))
        tp = int(round(action[3]))
        trailing = int(round(action[4]))
        return side, lot_size, sl, tp, trailing

    def _calculate_unrealized_pnl(self):
        """Sum unrealized PnL for all open positions (simplified)."""
        pnl = 0
        for pos in self.open_positions:
            current_price = self.data_handler.get_price(self.current_step)
            if pos["side"] == "buy":
                pnl += (current_price - pos["price"]) * pos["lots"] * 100000
            elif pos["side"] == "sell":
                pnl += (pos["price"] - current_price) * pos["lots"] * 100000
        return pnl

    def _close_positions_if_hit(self):
        """Close positions if SL/TP reached."""
        closed_pnl = 0
        current_price = self.data_handler.get_price(self.current_step)
        remaining_positions = []
        for pos in self.open_positions:
            hit = False
            if pos["sl"] and (
                (pos["side"] == "buy" and current_price <= pos["sl"])
                or (pos["side"] == "sell" and current_price >= pos["sl"])
            ):
                hit = True
            if pos["tp"] and (
                (pos["side"] == "buy" and current_price >= pos["tp"])
                or (pos["side"] == "sell" and current_price <= pos["tp"])
            ):
                hit = True

            if hit:
                if pos["side"] == "buy":
                    closed_pnl += (current_price - pos["price"]) * pos["lots"] * 100000
                else:
                    closed_pnl += (pos["price"] - current_price) * pos["lots"] * 100000
                self.account_balance += closed_pnl
            else:
                remaining_positions.append(pos)
        self.open_positions = remaining_positions
        return closed_pnl

    def _max_drawdown_exceeded(self):
        max_dd = self.config["risk"]["max_drawdown"]
        return (self.initial_balance - self.equity) > max_dd
