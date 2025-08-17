# f10_utils/logging_cfg.py
"""
تنظیمات متمرکز لاگ‌گیری برای پروژه Bot-RL-1
- لاگ‌گیری در کنسول و فایل چرخشی (Rotating File)
- ارسال هشدار تلگرام برای پیام‌های خطا (ERROR) و بحرانی (CRITICAL)
نسخهٔ اصلاح‌شده:
- ایمن در برابر مقداردهی دوباره (idempotent)
- محافظت در برابر خطاهای ارسال تلگرام (جلوگیری از recursion)
- استفاده از سطوح لاگ به‌صورت عددی
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import requests
from typing import Optional

# دقت: load_config ممکن است exceptions پرتاب کند — در این ماژول از آن محافظت می‌کنیم
try:
    from f10_utils.config_loader import load_config
except Exception:
    # اگر load_config در import دچار مشکل شد، تابع placeholder تعریف می‌کنیم
    def load_config():
        return {}

logger = logging.getLogger(__name__)


class TelegramHandler(logging.Handler):
    """
    هندلر سفارشی برای ارسال لاگ‌های خطا و بحرانی به تلگرام.
    دقت: این هندلر طوری طراحی شده که در صورت شکست ارسال، هیچ لاگ جدیدی
    به root logger نفرستد تا از حلقهٔ بازگشتی (recursion) جلوگیری شود.
    """

    def __init__(self, bot_token: str, chat_id: str, level: int = logging.ERROR):
        super().__init__(level)
        self.bot_token = bot_token
        self.chat_id = chat_id
        # یک لاگر داخلی با نام مشخص برای لاگ‌گیری داخلی این هندلر (غیر از root)
        self._internal = logging.getLogger("telegram_handler_internal")
        # از پراپگیت جلوگیری می‌کنیم تا لاگ‌های داخلی این لاگر به root نرود
        self._internal.propagate = False

    def emit(self, record: logging.LogRecord) -> None:
        """
        ارسال پیام لاگ به تلگرام.
        در صورت بروز خطا کاملاً ساکت می‌ماند (هیچ استثنایی بیرون نمی‌دهد).
        """
        try:
            # از formatter موجود استفاده کن؛ اگر نبود از یک فرمت ساده fallback بزن
            try:
                msg = self.format(record)
            except Exception:
                msg = f"{record.levelname}: {record.getMessage()}"

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": f"🚨 Bot Alert 🚨\n{msg}",
                "parse_mode": "HTML"
            }
            # درخواست با timeout کوتاه؛ اگر fail شود، ما آن را می‌بلعیم
            try:
                resp = requests.post(url, data=payload, timeout=5)
                # اگر پاسخ غیر موفق بود، لاگ‌گیری داخلی در سطح DEBUG داشته باشیم (به root نخواهد رفت)
                if resp is not None and resp.status_code >= 400:
                    self._internal.debug("Telegram sendMessage returned status %s: %s", resp.status_code, getattr(resp, "text", ""))
            except Exception:
                # بلعیدن خطاها — جدا از لاگ root
                self._internal.debug("TelegramHandler failed to send message (exception swallowed)")
        except Exception:
            # هر استثنای دیگری را هم نادیده می‌گیریم تا از ایجاد حلقه لاگ جلوگیری شود
            try:
                # مطمئن شویم این هم به root لاگ نمی‌رسد
                self._internal.debug("TelegramHandler unexpected failure in emit()", exc_info=True)
            except Exception:
                pass


def configure_logging() -> None:
    """
    پیکربندی سیستم لاگ‌گیری برای کل پروژه.
    - ایمن (idempotent): اگر root logger قبلاً handler داشته باشد، دوباره مقداردهی انجام نمی‌شود.
    - باید قبل از ماژول‌هایی که لاگ می‌نویسند صدا زده شود (اما اگر نشد باز هم محافظت دارد).
    """
    # اگر قبلاً مقداردهی شده، از دوباره‌کاری جلوگیری کن
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # از debug استفاده می‌کنیم تا در لاگ اصلی فقط یک خط نشود؛ این تابع باید کم‌هیاهو باشد
        root_logger.debug("configure_logging: root logger already configured; skipping reconfiguration")
        return

    # بارگذاری کانفیگ با محافظت
    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}

    # مقادیر پیش‌فرض
    log_dir = str(cfg.get("logging", {}).get("log_dir", "logs"))
    log_level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    log_file = str(cfg.get("logging", {}).get("log_file", "bot.log"))

    # تبدیل نام سطح به مقدار عددی ایمن
    log_level = getattr(logging, log_level_name, logging.INFO)

    # ایجاد پوشه لاگ در صورت نیاز (اگر نتواند فقط ادامه می‌دهیم)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # اگر ایجاد مسیر شکست خورد، از مسیر فعلی استفاده می‌کنیم
        log_dir = "."

    # مسیر فایل لاگ
    log_path = os.path.join(log_dir, log_file)

    # فرمت لاگ
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # File rotating handler (همیشه سطح DEBUG برای جمع‌آوری لاگ‌های کامل)
    try:
        file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=7, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    except Exception:
        file_handler = None
        # اگر ساخت فایل هندلر موفق نبود، لاگ را در کنسول ادامه می‌دهیم

    # اضافه کردن هندلرها به root logger
    root_logger.setLevel(logging.DEBUG)  # اجازه دهیم همه پیام‌ها دسترس باشند؛ نمایش کنترل‌شده توسط handlerهاست
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    # فعال‌سازی هشدار تلگرام در صورت وجود تنظیمات مناسب
    try:
        telegram_cfg = cfg.get("alerts", {}).get("telegram", {}) if isinstance(cfg, dict) else {}
        bot_token = telegram_cfg.get("bot_token")
        chat_id = telegram_cfg.get("chat_id")
        if bot_token and chat_id:
            telegram_handler = TelegramHandler(bot_token=bot_token, chat_id=str(chat_id), level=logging.ERROR)
            telegram_handler.setFormatter(formatter)
            root_logger.addHandler(telegram_handler)
            root_logger.info("هندلر هشدار تلگرام فعال شد.")
    except Exception:
        # از خطای احتمالی در هندلر تلگرام صرف‌نظر کنیم — نباید باعث متوقف شدن برنامه شود
        root_logger.debug("Failed to configure Telegram handler (ignored)", exc_info=True)

    # کاهش نویز لاگ برخی کتابخانه‌ها
    try:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("MetaTrader5").setLevel(logging.WARNING)
    except Exception:
        pass

    root_logger.info("لاگ‌گیری مقداردهی شد. سطح: %s، مسیر: %s", logging.getLevelName(log_level), log_dir)
