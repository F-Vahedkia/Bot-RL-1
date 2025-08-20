# f10_utils/risk_manager.py
"""
Risk manager utilities for Bot-RL-1

Provides:
- RiskManager class (recommended) and module-level helper functions for backward compatibility.
- get_symbol_info(symbol) -> dict(point, pip, contract_size, base, quote)
- pip_value_per_lot(symbol, lot=1.0, account_currency=None) -> value of 1 pip for provided lot in account currency (best-effort)
- round_lot(lot, lot_step) and validate_lot(...)
- compute_lot_size_by_risk(equity, risk_percent, sl_pips, symbol, ...) -> (lot, estimated_risk_amount, pip_value)
- required_margin(lot, price, leverage, contract_size) -> margin required
- used_margin_from_positions(positions, leverage, symbol_info_map) -> used margin estimate
- get_free_margin(balance, used_margin) -> free margin
- suggest_lot_by_risk(...) convenience wrapper
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, List
import math
import logging
import re

logger = logging.getLogger(__name__)

# optional MetaTrader5 usage
try:
    import MetaTrader5 as mt5  # type: ignore
    _HAS_MT5 = True
except Exception:
    mt5 = None  # type: ignore
    _HAS_MT5 = False

# Try to import config_loader.config if available (backwards compat)
try:
    from f10_utils.config_loader import config as _CONFIG  # type: ignore
except Exception:
    _CONFIG = {}  # type: ignore

# ----------------------------
# symbol helpers
# ----------------------------
_SYMBOL_RE = re.compile(r"^([A-Za-z]+)")

def _normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    m = _SYMBOL_RE.match(sym)
    return m.group(1).upper() if m else sym.upper()

def _parse_symbol(sym: str) -> Tuple[Optional[str], Optional[str]]:
    s = _normalize_symbol(sym)
    if len(s) >= 6:
        return s[:3], s[3:6]
    # handle common commodity letters like XAUUSD -> base=XAU quote=USD
    if len(s) == 6 or len(s) >= 5:
        # try split heuristics: last 3 are quote if possible
        base = s[:-3]
        quote = s[-3:]
        return base, quote
    return None, None

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Return a dictionary with keys:
      - point: minimal price increment
      - pip: value of one pip in price units (usually point * 10 for 5-digit quotes)
      - contract_size: units per lot (e.g. 100000 for forex, 100 for XAU)
      - base, quote: currency codes or None
    Uses MT5 when available; otherwise returns conservative defaults.
    """
    sym = _normalize_symbol(symbol)
    base, quote = _parse_symbol(sym)
    # sensible defaults
    default_point = 0.0001
    default_contract = 100000.0  # forex standard
    # handle XAU/XAG or metal-like tick size
    if sym.startswith("XAU") or "XAU" in sym or sym.startswith("GOLD"):
        # many brokers quote XAUUSD with 2 decimal places (0.01) or 1/10 of cent
        default_point = 0.01
        default_contract = 100.0
    # JPY pairs typically 0.01 point
    if quote == "JPY" or "JPY" in (quote or ""):
        default_point = 0.01

    # Compute pip default (commonly point * 10 for 5-digit quotes)
    def _default_pip(point):
        try:
            return float(point) * 10.0 if float(point) < 0.001 else float(point)
        except Exception:
            return float(point)

    if not _HAS_MT5:
        logger.debug("MT5 not available; using defaults for symbol %s", sym)
        return {"point": float(default_point), "pip": float(_default_pip(default_point)),
                "contract_size": float(default_contract), "base": base, "quote": quote}

    try:
        info = mt5.symbol_info(sym)
        if info is None:
            logger.warning("mt5.symbol_info(%s) returned None; using defaults", sym)
            return {"point": float(default_point), "pip": float(_default_pip(default_point)),
                    "contract_size": float(default_contract), "base": base, "quote": quote}
        point = getattr(info, "point", None) or getattr(info, "min_distance", None) or default_point
        # determine pip
        point = float(point)
        pip = float(point * 10.0) if point < 0.001 else float(point)
        contract = getattr(info, "trade_contract_size", None) or getattr(info, "contract_size", None) or default_contract
        try:
            contract = float(contract)
        except Exception:
            contract = float(default_contract)
        return {"point": point, "pip": pip, "contract_size": contract, "base": base, "quote": quote}
    except Exception as ex:
        logger.exception("Error reading symbol_info for %s: %s", sym, ex)
        return {"point": float(default_point), "pip": float(_default_pip(default_point)),
                "contract_size": float(default_contract), "base": base, "quote": quote}

# ----------------------------
# rounding / validation
# ----------------------------
def round_lot(lot: float, lot_step: float) -> float:
    if lot_step <= 0:
        raise ValueError("lot_step must be > 0")
    steps = math.floor(lot / lot_step)
    rounded = round(steps * lot_step, 8)
    return rounded

def validate_lot(lot: float, min_lot: float = 0.01, lot_step: float = 0.01, max_lot: Optional[float] = None) -> float:
    """
    Normalize a raw lot value to allowed broker increments:
    - If lot <= 0 -> returns 0.0
    - If lot < min_lot: if lot >= (min_lot/2) -> round up to min_lot; else -> 0.0
    - If lot >= min_lot: floor to nearest lot_step, clip to max_lot if provided
    """
    try:
        lot = float(lot)
    except Exception:
        return 0.0
    if lot <= 0.0:
        return 0.0
    if lot < min_lot:
        if lot >= (min_lot / 2.0):
            # round up to min_lot (or nearest multiple)
            steps = math.ceil(min_lot / lot_step)
            up = round(steps * lot_step, 8)
            if max_lot is not None and up > max_lot:
                return round_lot(max_lot, lot_step)
            return up
        return 0.0
    # floor to nearest multiple
    steps = math.floor(lot / lot_step)
    rounded = round(steps * lot_step, 8)
    if rounded < min_lot:
        steps = math.ceil(min_lot / lot_step)
        up = round(steps * lot_step, 8)
        if max_lot is not None and up > max_lot:
            return round_lot(max_lot, lot_step)
        return up
    if max_lot is not None and rounded > max_lot:
        return round_lot(max_lot, lot_step)
    return rounded

# ----------------------------
# pip value and currency conversion
# ----------------------------
def _mid_price_of_symbol(sym: str) -> Optional[float]:
    if not _HAS_MT5:
        return None
    try:
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            return None
        bid = getattr(tick, "bid", None)
        ask = getattr(tick, "ask", None)
        if bid is None or ask is None:
            last = getattr(tick, "last", None)
            return float(last) if last is not None else None
        return (float(bid) + float(ask)) / 2.0
    except Exception:
        logger.exception("Error reading tick for %s", sym)
        return None

def pip_value_per_lot(symbol: str, lot: float = 1.0, account_currency: Optional[str] = None) -> float:
    """
    Return value of one pip for `lot` in account_currency (if convertible).
    Algorithm:
      pip_value_in_quote = pip_size * contract_size * lot
    If account currency differs from quote, try to get conversion rate via MT5 (direct or inverse pair).
    If conversion not possible, return pip value in quote currency (best-effort).
    """
    info = get_symbol_info(symbol)
    pip_size = float(info.get("pip", 0.0))
    contract = float(info.get("contract_size", 1.0))
    base = info.get("base")
    quote = info.get("quote") or ""
    pip_value_in_quote = pip_size * contract * float(lot)

    acct_cur = account_currency
    if acct_cur is None and _HAS_MT5:
        try:
            ai = mt5.account_info()
            acct_cur = getattr(ai, "currency", None) if ai is not None else None
        except Exception:
            acct_cur = None

    if acct_cur is None:
        # cannot convert; return value in quote currency
        logger.debug("Account currency unknown; returning pip value in quote currency (%s)", quote)
        return float(pip_value_in_quote)

    acct_cur = acct_cur.upper()
    quote = (quote or "").upper()
    if quote == acct_cur:
        return float(pip_value_in_quote)

    # try direct pair (quote->acct) then inverse (acct->quote)
    pair_direct = f"{quote}{acct_cur}"
    pair_inverse = f"{acct_cur}{quote}"

    if _HAS_MT5:
        try:
            if mt5.symbol_info(pair_direct) is not None:
                rate = _mid_price_of_symbol(pair_direct)
                if rate:
                    return float(pip_value_in_quote * rate)
            if mt5.symbol_info(pair_inverse) is not None:
                rate = _mid_price_of_symbol(pair_inverse)
                if rate:
                    return float(pip_value_in_quote / rate)
        except Exception:
            logger.exception("Error while attempting currency conversion for pip value.")

    logger.warning("Could not convert pip value from %s to %s (pairs %s/%s missing). Returning pip in quote.", quote, acct_cur, pair_direct, pair_inverse)
    return float(pip_value_in_quote)

# ----------------------------
# margin and lot sizing
# ----------------------------
def required_margin(lot: float, price: float, leverage: float = 100.0, contract_size: Optional[float] = None, symbol: Optional[str] = None) -> float:
    """
    Estimate required margin for opening a position of `lot` at `price`.
    margin = (lot * contract_size * price) / leverage
    If contract_size not provided, try symbol info.
    """
    try:
        lot = float(lot)
        price = float(price)
        leverage = float(leverage) if leverage and float(leverage) > 0 else 100.0
    except Exception:
        return 0.0

    if contract_size is None and symbol is not None:
        info = get_symbol_info(symbol)
        contract_size = float(info.get("contract_size", 100000.0))
    if contract_size is None:
        contract_size = 100000.0
    try:
        margin = (lot * float(contract_size) * price) / max(1.0, leverage)
        return float(margin)
    except Exception:
        return 0.0

def used_margin_from_positions(positions: List[Dict[str, Any]], leverage: float = 100.0, symbol_info_map: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
    """
    Rough estimate of used margin given open positions.
    positions: list of dicts with keys: symbol, volume/lot, entry_price
    symbol_info_map: optional precomputed symbol->info dict
    """
    total = 0.0
    for p in positions:
        try:
            symbol = p.get("symbol") or p.get("s") or p.get("instrument") or p.get("symbol_name")
            lot = p.get("volume") or p.get("lot") or p.get("lots") or p.get("size") or 0.0
            price = p.get("entry_price") or p.get("price") or p.get("filled_price") or p.get("entry") or 0.0
            if symbol_info_map and symbol in symbol_info_map:
                info = symbol_info_map[symbol]
            else:
                info = get_symbol_info(symbol) if symbol else {"contract_size": 100000.0}
            contract = float(info.get("contract_size", 100000.0))
            margin = required_margin(lot, price or _mid_price_of_symbol(symbol) or price, leverage=leverage, contract_size=contract)
            total += margin
        except Exception:
            logger.exception("Error computing used margin for position: %s", p)
    return float(total)

def get_free_margin(balance: float, used_margin: float) -> float:
    try:
        return max(0.0, float(balance) - float(used_margin))
    except Exception:
        return 0.0

def compute_lot_size_by_risk(equity: float, risk_percent: float, sl_pips: float, symbol: str, *,
                             min_lot: Optional[float] = None,
                             lot_step: Optional[float] = None,
                             max_lot: Optional[float] = None,
                             pip_price: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Compute lot size so that (lot * sl_pips * pip_value) ~= risk_amount where
    risk_amount = equity * risk_percent / 100

    Returns (lot_validated, actual_risk_amount, pip_value_per_lot)
    """
    cfg = _CONFIG or {}
    vs = {}
    # try to find volume settings in config
    if isinstance(cfg, dict):
        vs = cfg.get("volume_settings", {}) or cfg.get("env_defaults", {}) or {}

    default_min = vs.get("min_volume", 0.01)
    default_step = vs.get("volume_step", 0.01)
    default_max = vs.get("max_volume", None)

    min_lot = default_min if min_lot is None else min_lot
    lot_step = default_step if lot_step is None else lot_step
    max_lot = default_max if max_lot is None else max_lot

    if sl_pips <= 0:
        raise ValueError("sl_pips must be > 0")

    risk_amount = float(equity) * float(risk_percent) / 100.0
    pip_val = float(pip_price) if pip_price is not None else pip_value_per_lot(symbol, lot=1.0)

    if pip_val <= 0:
        raise RuntimeError("pip value computed as <= 0")

    raw_lot = risk_amount / (sl_pips * pip_val)
    lot = validate_lot(raw_lot, min_lot=min_lot, lot_step=lot_step, max_lot=max_lot)
    actual_risk = lot * sl_pips * pip_val
    logger.info("compute_lot_size_by_risk: equity=%s risk_pct=%s sl_pips=%s raw_lot=%s validated_lot=%s actual_risk=%s",
                equity, risk_percent, sl_pips, raw_lot, lot, actual_risk)
    return float(lot), float(actual_risk), float(pip_val)

def suggest_lot_by_risk(balance: float, risk_percent: float, sl_pips: float, symbol: str, **kwargs) -> float:
    lot, _actual, _pv = compute_lot_size_by_risk(balance, risk_percent, sl_pips, symbol, **kwargs)
    return lot

# ----------------------------
# Convenience RiskManager class
# ----------------------------
class RiskManager:
    """
    Wrapper class that exposes the functionality with a simple instance API.
    Accepts optional cfg dict to read defaults (volume_settings, leverage, etc.)
    """
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or _CONFIG or {}
        vs = self.cfg.get("volume_settings", {}) or {}
        self.min_lot = float(vs.get("min_volume", 0.01))
        self.lot_step = float(vs.get("volume_step", 0.01))
        self.max_lot = vs.get("max_volume", None)
        self.leverage = float(self.cfg.get("env_defaults", {}).get("leverage", self.cfg.get("account", {}).get("leverage", 100.0)))

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        return get_symbol_info(symbol)

    def pip_value_per_lot(self, symbol: str, lot: float = 1.0, account_currency: Optional[str] = None) -> float:
        return pip_value_per_lot(symbol, lot=lot, account_currency=account_currency)

    def round_lot(self, lot: float) -> float:
        return round_lot(lot, self.lot_step)

    def validate_lot(self, lot: float) -> float:
        return validate_lot(lot, min_lot=self.min_lot, lot_step=self.lot_step, max_lot=self.max_lot)

    def compute_lot_size_by_risk(self, equity: float, risk_percent: float, sl_pips: float, symbol: str, **kwargs) -> Tuple[float, float, float]:
        return compute_lot_size_by_risk(equity, risk_percent, sl_pips, symbol, min_lot=self.min_lot, lot_step=self.lot_step, max_lot=self.max_lot, **kwargs)

    def required_margin(self, lot: float, price: float, contract_size: Optional[float] = None) -> float:
        return required_margin(lot, price, leverage=self.leverage, contract_size=contract_size)

    def used_margin_from_positions(self, positions: List[Dict[str, Any]], symbol_info_map: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
        return used_margin_from_positions(positions, leverage=self.leverage, symbol_info_map=symbol_info_map)

    def get_free_margin(self, balance: float, positions: List[Dict[str, Any]], symbol_info_map: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
        used = self.used_margin_from_positions(positions, symbol_info_map=symbol_info_map)
        return get_free_margin(balance, used)

# ----------------------------
# Backwards-compatible short names (module-level)
# ----------------------------
__all__ = [
    "get_symbol_info", "pip_value_per_lot", "round_lot", "validate_lot",
    "compute_lot_size_by_risk", "required_margin", "used_margin_from_positions",
    "get_free_margin", "suggest_lot_by_risk", "RiskManager",
]
