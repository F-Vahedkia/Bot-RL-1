# f10_utils/risk_manager.py
"""
Risk manager — consolidated, safer implementation.

وظایف:
- compute_lot_size_by_risk(...)
- pip_value_per_lot(...)
- round_lot, validate_lot
- fallback به مقادیر پیش‌فرض در صورت نبودن MetaTrader5
"""
from typing import Tuple, Optional, Dict, Any
import math
import logging
import re

logger = logging.getLogger(__name__)

# تلاش برای بارگذاری mt5 در صورت وجود (اختیاری)
try:
    import MetaTrader5 as mt5  # type: ignore
    _HAS_MT5 = True
except Exception:
    mt5 = None
    _HAS_MT5 = False

# کوشش برای گرفتن کانفیگ در صورت وجود (اختیاری)
try:
    from f10_utils.config_loader import config
except Exception:
    config = {}

# ---------- کمکی‌ها ----------
def _normalize_symbol(sym: str) -> str:
    """Keep leading alphabetical part (EURUSD.met -> EURUSD)."""
    m = re.match(r"([A-Za-z]+)", sym)
    return m.group(1) if m else sym

def _parse_symbol(sym: str) -> Tuple[Optional[str], Optional[str]]:
    """
    بازگرداندن base, quote از اسم نماد استاندارد 6 حرفی مثل EURUSD.
    اگر نتوان تشخیص داد (نام کوتاه‌تر یا غیر استاندارد) None برمی‌گرداند.
    """
    s = _normalize_symbol(sym).upper()
    if len(s) >= 6:
        return s[:3], s[3:6]
    return None, None

def _get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    خواندن info از MT5 یا بازگردانی مقادیر پیش‌فرض.
    خروجی keys: point, pip (calculated), contract_size, base, quote
    """
    sym = _normalize_symbol(symbol).upper()
    base, quote = _parse_symbol(sym)

    # default
    default_point = 0.0001
    default_contract = 100000.0

    # JPY pair default point
    if quote == "JPY" or "JPY" in sym:
        default_point = 0.01

    if not _HAS_MT5:
        logger.debug("MT5 not available, using defaults for %s", sym)
        pip = default_point if default_point >= 0.001 else default_point * 10.0
        return {"point": default_point, "pip": pip, "contract_size": default_contract, "base": base, "quote": quote}

    try:
        info = mt5.symbol_info(sym)
        if info is None:
            logger.warning("mt5.symbol_info(%s) is None; using defaults", sym)
            pip = default_point if default_point >= 0.001 else default_point * 10.0
            return {"point": default_point, "pip": pip, "contract_size": default_contract, "base": base, "quote": quote}

        point = getattr(info, "point", None) or getattr(info, "min_distance", None) or default_point
        # determine pip: many brokers use 5-digit quotes where 1 pip = point * 10
        if float(point) < 0.001:
            pip = float(point) * 10.0
        else:
            pip = float(point)

        contract = getattr(info, "trade_contract_size", None) or getattr(info, "contract_size", None) or default_contract
        contract = float(contract)
        return {"point": float(point), "pip": float(pip), "contract_size": contract, "base": base, "quote": quote}
    except Exception as e:
        logger.exception("Error getting symbol_info(%s): %s", symbol, e)
        pip = default_point if default_point >= 0.001 else default_point * 10.0
        return {"point": default_point, "pip": pip, "contract_size": default_contract, "base": base, "quote": quote}

def _mid_price_of_symbol(sym: str) -> Optional[float]:
    """میانگین bid/ask برای نماد؛ محافظت شده در برابر نبود mt5."""
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

# ---------- گرد کردن و اعتبارسنجی لات ----------
def round_lot(lot: float, lot_step: float) -> float:
    if lot_step <= 0:
        raise ValueError("lot_step must be > 0")
    steps = math.floor(lot / lot_step)
    rounded = round(steps * lot_step, 8)
    return rounded


def validate_lot(lot: float, min_lot: float = 0.01, lot_step: float = 0.01, max_lot: Optional[float] = None) -> float:
    """
    رفتار جدید:
    - اگر lot <= 0 -> 0.0
    - اگر lot < min_lot:
        - اگر lot >= 0.5 * min_lot -> مقدار به min_lot (همراستا با lot_step، یعنی بالا گرد به نزدیک‌ترین مضرب مجاز)
        - در غیر این صورت -> 0.0
    - اگر lot >= min_lot:
        - مقدار را به پایین‌ترین مضرب مجاز (floor) گرد می‌کنیم؛ سپس محدود به max_lot در صورت وجود.
    """
    if lot <= 0:
        return 0.0

    # اگر کمتر از min_lot باشیم، براساس آستانهٔ نصف تصمیم می‌گیریم (پیش از گرد کردن)
    if lot < min_lot:
        if lot >= (min_lot / 2.0):
            # بالاگرد به نزدیک‌ترین مضرب مجاز که >= min_lot
            steps = math.ceil(min_lot / lot_step)
            up = round(steps * lot_step, 8)
            if max_lot is not None and up > max_lot:
                return round_lot(max_lot, lot_step)
            return up
        else:
            return 0.0

    # حالا lot >= min_lot: گرد کردن به پایین (floor) به نزدیک‌ترین گام مجاز
    rounded = round_lot(lot, lot_step)
    # اگر گرد کردن باعث شد به زیر min_lot برگردیم، بالاگرد به min_lot
    if rounded < min_lot:
        steps = math.ceil(min_lot / lot_step)
        up = round(steps * lot_step, 8)
        if max_lot is not None and up > max_lot:
            return round_lot(max_lot, lot_step)
        return up

    if max_lot is not None and rounded > max_lot:
        return round_lot(max_lot, lot_step)

    return rounded

# ---------- ارزش پیپ ----------
def pip_value_per_lot(symbol: str, lot: float = 1.0, account_currency: Optional[str] = None) -> float:
    """
    بازگرداندن ارزش یک پیپ برای `lot` لات بر حسب ارز حساب (در صورت امکان).
    الگوریتم:
      pip_in_quote = pip_size * contract_size * lot
      اگر quote != account_currency، سعی می‌کنیم نرخ تبدیل بین quote و account_currency
      را با پیدا کردن جفت مستقیم یا معکوس بیابیم.
    """
    info = _get_symbol_info(symbol)
    pip_size = info["pip"]            # مقدار یک pip به واحد quote
    contract = info["contract_size"]
    base = info.get("base")
    quote = info.get("quote")

    pip_value_in_quote = pip_size * contract * float(lot)

    # determine account currency
    acct_cur = account_currency
    if acct_cur is None and _HAS_MT5:
        try:
            ai = mt5.account_info()
            acct_cur = getattr(ai, "currency", None) if ai is not None else None
        except Exception:
            acct_cur = None

    if acct_cur is None:
        # نمی‌توانیم تبدیل ارز انجام دهیم؛ مقدار بر حسب quote را بازگردان
        logger.debug("Account currency unknown or MT5 unavailable; returning pip in quote currency (%s).", quote)
        return float(pip_value_in_quote)

    acct_cur = acct_cur.upper()
    quote = (quote or "").upper()
    if quote == acct_cur:
        return float(pip_value_in_quote)

    # try to find conversion pair
    pair_direct = f"{quote}{acct_cur}"
    pair_inverse = f"{acct_cur}{quote}"

    # direct
    if _HAS_MT5 and mt5.symbol_info(pair_direct) is not None:
        rate = _mid_price_of_symbol(pair_direct)
        if rate:
            return float(pip_value_in_quote * rate)

    # inverse
    if _HAS_MT5 and mt5.symbol_info(pair_inverse) is not None:
        rate = _mid_price_of_symbol(pair_inverse)
        if rate:
            return float(pip_value_in_quote / rate)

    # fallback: cannot convert — return value in quote
    logger.warning("Could not convert pip value from %s to %s (pairs %s/%s missing). Returning unconverted.", quote, acct_cur, pair_direct, pair_inverse)
    return float(pip_value_in_quote)

# ---------- محاسبه لات بر اساس ریسک ----------
def compute_lot_size_by_risk(
    equity: float,
    risk_percent: float,
    sl_pips: float,
    symbol: str,
    *,
    min_lot: Optional[float] = None,
    lot_step: Optional[float] = None,
    max_lot: Optional[float] = None,
    pip_price: Optional[float] = None
) -> Tuple[float, float]:
    # default from config if available
    cfg = config if isinstance(config, dict) else {}
    vs = cfg.get("volume_settings", {}) if cfg else {}
    default_min = vs.get("min_volume", 0.01)
    default_step = vs.get("volume_step", 0.01)
    default_max = vs.get("max_volume", None)

    if min_lot is None:
        min_lot = default_min
    if lot_step is None:
        lot_step = default_step
    if max_lot is None:
        max_lot = default_max

    if sl_pips <= 0:
        raise ValueError("sl_pips must be > 0")

    risk_amount = float(equity) * float(risk_percent) / 100.0

    pip_val = float(pip_price) if pip_price is not None else pip_value_per_lot(symbol, lot=1.0)
    if pip_val <= 0:
        raise RuntimeError("pip value computed as <= 0")

    raw_lot = risk_amount / (sl_pips * pip_val)
    lot = validate_lot(raw_lot, min_lot=min_lot, lot_step=lot_step, max_lot=max_lot)
    actual_risk = lot * sl_pips * pip_val

    logger.info("Computed lot=%s (raw=%s) for symbol=%s; actual_risk=%s", lot, raw_lot, symbol, actual_risk)
    return lot, actual_risk

# ---------- quick demo ----------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--sl", type=float, default=20.0)
    args = parser.parse_args()

    lot, actual = compute_lot_size_by_risk(args.equity, args.risk, args.sl, args.symbol)
    print(f"Lot={lot}, estimated risk amount={actual:.2f}")
