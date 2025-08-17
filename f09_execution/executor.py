# f09_execution/executor.py
"""
Executor — رابط ارسال و مدیریت سفارشات زنده به MetaTrader5
مسئولیت‌ها:
- اتصال امن به MT5 (از MT5Connector استفاده می‌کند)
- ارسال سفارشات بازار (market orders) با تعیین حجم به صورت صریح یا بر اساس ریسک
- بستن موقعیت‌ها (کامل یا جزئی)
- افزایش/کاهش حجم موقعیت باز با روش close-partial + reopen (اگر لازم باشد)
- اعمال SL / TP / trailing (حداقل SL/TP پایه)
- مانیتور نتیجه order_send و گزارش لاگ / هشدار

نکات:
- این ماژول فرض می‌کند logging قبلاً مقداردهی شده است (configure_logging()).
  اگر نه، آن را لود می‌کند (با حفاظت در برابر double-init).
- برای محاسبه لات و ارزش پیپ از f10_utils.risk_manager استفاده می‌شود.
"""
from __future__ import annotations

import logging
#import time
from typing import Optional, Tuple, Dict, Any

import MetaTrader5 as mt5  # type: ignore

from f10_utils.config_loader import config, load_config
from f02_data.mt5_connector import MT5Connector
from f10_utils.risk_manager import compute_lot_size_by_risk, pip_value_per_lot, validate_lot
from f10_utils.logging_cfg import configure_logging

logger = logging.getLogger(__name__)

# محافظت: اگر logging هنوز مقداردهی نشده، مقداردهی می‌کنیم (ایمن و idempotent)
try:
    # configure_logging ممکن است قبلاً اجرا شده باشد؛ اجرای مجدد مشکلی ایجاد نمی‌کند
    configure_logging()
except Exception:
    # اگر configure_logging خطا داد، حداقل یک basicConfig ساده تنظیم می‌کنیم
    logging.basicConfig(level=logging.INFO)


class Executor:
    def __init__(self, connector: Optional[MT5Connector] = None, cfg: Optional[dict] = None):
        """
        connector: نمونه MT5Connector (اگر None ساخته می‌شود)
        cfg: دیکشنری کانفیگ؛ اگر None از load_config() استفاده می‌شود.
        """
        self.config = cfg or (config if isinstance(config, dict) else load_config())
        self.connector = connector or MT5Connector()
        # default deviation برای slippage در درخواست‌ها (قابل override)
        self.default_deviation = int(self.config.get("execution", {}).get("deviation", 20))
        # magic number برای تفکیک سفارشات ربات (می‌توانید در config قرار دهید)
        self.magic = int(self.config.get("execution", {}).get("magic", 123456))

    # -------- کمک: وضعیت حساب و equity --------
    def _get_account_equity(self) -> float:
        """دریافت equity فعلی از حساب MT5؛ اگر ناموفق شد استثنا می‌اندازد."""
        ai = mt5.account_info()
        if ai is None:
            logger.error("Failed to read account_info from MT5")
            raise ConnectionError("MT5 account_info() returned None")
        return float(getattr(ai, "equity", getattr(ai, "balance", 0.0)))

    # -------- کمک: تبدیل SL/TP به قیمت --------
    def _price_from_pips(self, symbol: str, side: str, pips: float) -> float:
        """
        تبدیل مقدار پیپ (sl/ tp) به قیمت (قیمت مطلق)
        side: 'buy' or 'sell' (برای تعیین مبنای ask/bid)
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick not available for {symbol}")
        mid = (getattr(tick, "bid", 0.0) + getattr(tick, "ask", 0.0)) / 2.0
        # خواندن pip size
        # استفاده از risk_manager._get_symbol_info غیرعمومی نیست، پس از pip_value_per_lot برای خواندن pip نمی‌کنیم.
        # اینجا برای اندازه پیپ از نماد info استفاده می‌کنیم:
        si = mt5.symbol_info(symbol)
        if si is None:
            raise RuntimeError(f"symbol_info not available for {symbol}")

        point = getattr(si, "point", None)
        if point is None:
            # fallback: اگر JPY در نماد باشد پیپ = 0.01
            point = 0.01 if "JPY" in symbol else 0.0001

        # در بسیاری از بروکرها پیپ واقعی برای قیمت‌های 5 رقمی point*10 است
        pip_size = float(point) * 10.0 if float(point) < 0.001 else float(point)

        # بسته به طرف معامله، SL/TP را از قیمت بازار کم/جمع می‌کنیم
        if side.lower() == "buy":
            price = float(getattr(tick, "ask", mid))  # entry price برای خرید
            sl_price = price - pips * pip_size
            tp_price = price + pips * pip_size
        else:
            price = float(getattr(tick, "bid", mid))
            sl_price = price + pips * pip_size
            tp_price = price - pips * pip_size

        return price, sl_price, tp_price

    # -------- بررسی اجتناب از اخبار (placeholder hook) --------
    def _is_news_avoid(self, symbol: str) -> bool:
        """
        اگر در کانفیگ news_avoidance فعال باشد می‌توان این تابع را گسترش داد
        تا با استفاده از منبع فاندامنتال/تقویم بررسی کند در پنجره اجتناب قرار داریم یا نه.
        فعلاً فقط مقدار کانفیگ را خوانده و False برمی‌گرداند.
        """
        na = self.config.get("news_avoidance", {}) or {}
        if not na.get("enabled", False):
            return False
        # TODO: اتصال به ماژول فاندامنتال/تقویم و بررسی زمان انتشار رویدادهای با impact بالا
        logger.debug("news_avoidance is enabled but no calendar check implemented; allowing trade by default")
        return False

    # -------- ارسال سفارش مارکت (اصلی) --------
    def place_market_order(
        self,
        symbol: str,
        side: str,
        lot: Optional[float] = None,
        risk_percent: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        deviation: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ارسال سفارش باز (market deal).
        - اگر lot مشخص نشود و risk_percent داده شده باشد، از compute_lot_size_by_risk استفاده می‌شود.
        - sl_pips و tp_pips به پیپ داده می‌شوند (در صورت None از config یا عدم تعیین صرفنظر می‌شود).
        - side: 'buy' یا 'sell'
        بازگرداند: dict نتیجه (شامل 'retcode', 'order' اگر موفق، و 'comment')
        """
        if self._is_news_avoid(symbol):
            msg = f"Trade skipped due to news avoidance for {symbol}"
            logger.info(msg)
            return {"ok": False, "reason": "news_avoid", "message": msg}

        # مطمئن شو اتصال برقرار است
        if not self.connector.ensure_connection():
            raise ConnectionError("MT5 connector not connected")

        deviation = deviation or self.default_deviation
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # اگر lot تعیین نشده، از ریسک استفاده می‌کنیم
        if lot is None:
            if risk_percent is None:
                # اگر هیچ‌کدام داده نشده، خطا
                raise ValueError("Either 'lot' or 'risk_percent' must be provided")
            # equity را می‌گیریم
            equity = self._get_account_equity()
            # اگر sl_pips نداشته باشیم، نمی‌توانیم بر اساس ریسک مقدار دهی کنیم
            if sl_pips is None:
                raise ValueError("sl_pips must be provided when using risk_percent")
            lot, actual_risk = compute_lot_size_by_risk(equity, risk_percent, sl_pips, symbol)
            logger.debug("Computed lot by risk: %s (actual_risk=%s)", lot, actual_risk)

        # اعتبارسنجی و گرد کردن لات نهایی بر اساس min/step config
        vs = self.config.get("volume_settings", {}) or {}
        min_lot = vs.get("min_volume", 0.01)
        lot_step = vs.get("volume_step", 0.01)
        max_lot = vs.get("max_volume", None)
        lot = validate_lot(lot, min_lot=min_lot, lot_step=lot_step, max_lot=max_lot)
        if lot <= 0:
            msg = f"Computed lot {lot} invalid (below min or zero)"
            logger.warning(msg)
            return {"ok": False, "reason": "lot_too_small", "message": msg}

        # قیمت entry و محاسبه SL/TP بر حسب پیپ (اگر ارائه شده)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick info for {symbol}")

        if side == "buy":
            price = float(getattr(tick, "ask", 0.0))
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = float(getattr(tick, "bid", 0.0))
            order_type = mt5.ORDER_TYPE_SELL

        sl_price = None
        tp_price = None
        if sl_pips is not None:
            _, sl_price_calc, _ = self._price_from_pips(symbol, side, sl_pips)
            sl_price = sl_price_calc
        if tp_pips is not None:
            _, _, tp_price_calc = self._price_from_pips(symbol, side, tp_pips)
            tp_price = tp_price_calc

        # آماده‌سازی request برای order_send
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl_price) if sl_price is not None else 0.0,
            "tp": float(tp_price) if tp_price is not None else 0.0,
            "deviation": int(deviation),
            "magic": int(self.magic),
            "comment": comment or "Bot-RL-1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        logger.info("Sending market order: %s %s lot=%s price=%s sl=%s tp=%s", side.upper(), symbol, lot, price, sl_price, tp_price)
        result = mt5.order_send(request)
        # result may be a structure with retcode, comment, order, etc.
        if result is None:
            logger.error("mt5.order_send returned None for request: %s", request)
            return {"ok": False, "reason": "no_response", "request": request}

        # بررسی نتیجه
        ret = getattr(result, "retcode", None)
        if ret is None:
            # برخی نسخه‌ها return dict-like
            logger.debug("order_send result: %s", result)
            ok = True
            return {"ok": True, "raw": result}

        if ret == mt5.TRADE_RETCODE_DONE or str(ret).upper().endswith("DONE") or ret == 10009:
            # کدهای retcode متفاوتند؛ این شرط عمومی است — لاگ کامل می‌کنیم
            logger.info("Order succeeded: retcode=%s result=%s", ret, result)
            return {"ok": True, "retcode": int(ret), "result": result._asdict() if hasattr(result, "_asdict") else result}
        else:
            logger.error("Order failed: retcode=%s result=%s", ret, result)
            return {"ok": False, "retcode": int(ret), "result": result._asdict() if hasattr(result, "_asdict") else result}

    # -------- بستن پوزیشن (کامل یا جزئی) --------
    def close_position(self, ticket_or_symbol: str | int, volume: Optional[float] = None) -> Dict[str, Any]:
        """
        بستن پوزیشن:
        - اگر ticket_or_symbol یک عدد باشد فرض می‌کنیم ticket است و آن پوزیشن را می‌بندیم.
        - در غیر این صورت اگر رشته بود به عنوان symbol در نظر گرفته و تمام پوزیشن‌های آن نماد را می‌بندیم یا بر حسب volume جزئی می‌بندیم.
        - volume اگر None باشد پوزیشن کامل بسته می‌شود؛ در غیر این صورت همان حجم بسته می‌شود.
        """
        if not self.connector.ensure_connection():
            raise ConnectionError("MT5 connector not connected")

        # جستجوی پوزیشن‌ها
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            logger.info("No open positions found")
            return {"ok": False, "reason": "no_positions"}

        # پیدا کردن پوزیشن مورد نظر
        target_positions = []
        if isinstance(ticket_or_symbol, int):
            for p in positions:
                if getattr(p, "ticket", None) == ticket_or_symbol:
                    target_positions = [p]
                    break
        else:
            sym = str(ticket_or_symbol)
            for p in positions:
                if getattr(p, "symbol", "") == sym:
                    target_positions.append(p)

        if not target_positions:
            logger.warning("No matching positions found for %s", ticket_or_symbol)
            return {"ok": False, "reason": "not_found"}

        results = []
        for p in target_positions:
            pos_volume = float(getattr(p, "volume", 0.0))
            pos_type = int(getattr(p, "type", 0))  # 0=buy,1=sell
            symbol = getattr(p, "symbol")
            # decide closing volume
            close_vol = pos_volume if volume is None else min(pos_volume, float(volume))
            if close_vol <= 0:
                continue

            # build close request (opposite type)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error("No tick for %s", symbol)
                continue
            if pos_type == 0:  # buy -> close with SELL at bid
                price = float(getattr(tick, "bid", 0.0))
                order_type = mt5.ORDER_TYPE_SELL
            else:
                price = float(getattr(tick, "ask", 0.0))
                order_type = mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(close_vol),
                "type": order_type,
                "position": int(getattr(p, "ticket", 0)),
                "price": price,
                "deviation": self.default_deviation,
                "magic": int(self.magic),
                "comment": "close_by_bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }

            res = mt5.order_send(request)
            results.append(res)
            logger.info("Sent close request for %s vol=%s res=%s", symbol, close_vol, res)

        return {"ok": True, "results": results}

    # -------- افزایش حجم پوزیشن (open additional) --------
    def increase_position(self, symbol: str, side: str, add_lot: float, comment: Optional[str] = None) -> Dict[str, Any]:
        """
        افزایش حجم: ارسال سفارش جدید هم‌جهت با پوزیشن (باز کردن پوزیشن جدید یا افزودن)
        - در بسیاری از بروکرها جمع کردن پوزیشن انجام می‌شود (pos های هم‌جهت ترکیب می‌شوند)
        """
        return self.place_market_order(symbol=symbol, side=side, lot=add_lot, comment=comment)

    # -------- کاهش حجم پوزیشن (partial close) --------
    def decrease_position(self, ticket_or_symbol: str | int, reduce_lot: float) -> Dict[str, Any]:
        """
        کاهش حجم: close_position با volume=reduce_lot
        """
        return self.close_position(ticket_or_symbol, volume=reduce_lot)

    # -------- مانیتور پوزیشن‌ها وگزارش وضعیت ساده --------
    def list_positions(self) -> list:
        """
        بازگرداندن لیست پوزیشن‌های باز (هر پوزیشن به صورت dict ساده).
        """
        ps = mt5.positions_get()
        out = []
        if ps is None:
            return out
        for p in ps:
            out.append({
                "ticket": getattr(p, "ticket", None),
                "symbol": getattr(p, "symbol", None),
                "volume": float(getattr(p, "volume", 0.0)),
                "type": int(getattr(p, "type", 0)),
                "price_open": float(getattr(p, "price_open", 0.0)),
                "sl": float(getattr(p, "sl", 0.0)),
                "tp": float(getattr(p, "tp", 0.0)),
                "profit": float(getattr(p, "profit", 0.0))
            })
        return out
