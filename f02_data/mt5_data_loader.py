# f02_data/mt5_data_loader.py
"""
MT5DataLoader
- استفاده از MT5Connector برای دانلود داده‌های تاریخیِ چند نماد و چند تایم‌فریم
- ذخیره CSV برای هر جفت نماد-تایم‌فریم
- مدیریت خطا و ادامه (fail-safe): اگر یک دانلود شکست خورد، بقیه ادامه می‌یابد
- قابلیت resume/skip_existing، تنظیمات retry/backoff برای هر دانلود
- مقادیر پیش‌فرض از config خوانده می‌شوند (download_defaults، data_paths)
نسخهٔ بهبود‌یافته:
- مدیریت روشن مالکیت کانکتور (_owns_connector)
- محافظت در برابر double-initialize
- پیام‌های لاگ کامل‌تر و امن‌تر
- متد close() برای بستن ایمن کانکتور در صورت نیاز
"""

import logging
import time
import pandas as pd
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Set
from f02_data.mt5_connector import MT5Connector
from f10_utils.config_loader import config

logger = logging.getLogger(__name__)
# کاهش صدای لاگ داخلی MT5 (در صورت نیاز)
try:
    logging.getLogger("MetaTrader5").setLevel(logging.WARNING)
except Exception:
    pass

class MT5DataLoader:
    def __init__(self, connector: Optional[MT5Connector] = None, cfg: Optional[Dict[str, Any]] = None):
        """
        connector: در صورت داده نشدن، یک MT5Connector جدید ساخته می‌شود اما initialize انجام نمی‌شود.
                   اگر connector از بیرون داده شود، loader مالک آن نیست و shutdown آن را انجام نخواهد داد.
        cfg: دیکشنری کانفیگ؛ در صورت None از f10_utils.config_loader.config استفاده می‌کند.
        """
        self.config = cfg or config or {}
        self.symbols = list(self.config.get("symbols", []) or [])
        self.timeframes = list(self.config.get("timeframes", []) or [])

        # مالکیت کانکتور: اگر ما کانکتور ساختیم، در پایان آن را shutdown می‌کنیم
        self._owns_connector = connector is None
        self.connector = connector or MT5Connector()

        # --- خواندن مسیر خروجی به‌صورت ایمن ---
        data_paths = {}
        if isinstance(self.config, dict):
            data_paths = self.config.get("data_paths", {}) or {}

        price_path = data_paths.get("price_data")
        if price_path:
            # اگر مسیر به صورت Path یا str باشد، آن را به Path تبدیل می‌کنیم
            try:
                self.output_dir = Path(price_path)
            except Exception:
                logger.exception("Invalid price_data path in config: %r. Falling back to default.", price_path)
                self.output_dir = Path("f02_data/price")
        else:
            # fallback اگر در config نبود
            self.output_dir = Path("f02_data/price")

    def _ensure_connector_ready(self, init_opts: Dict[str, int]) -> bool:
        """
        اطمینان از اینکه کانکتور برای دانلود آماده است.
        - اگر loader مالک کانکتور است، تلاش به initialize می‌کند (با init_opts).
        - اگر کانکتور خارجی داده شده و از قبل متصل است از آن استفاده می‌کند؛ در غیر این صورت ensure_connection امتحان می‌شود.
        باز می‌گرداند True اگر کانکتور آماده است، در غیر این صورت False.
        """
        logger.debug("_ensure_connector_ready: owns_connector=%s", self._owns_connector)
        if self._owns_connector:
            # ما کانکتور را ساخته‌ایم => باید initialize کنیم (اما از double-init جلوگیری می‌کنیم)
            try:
                # اگر کانکتور خودش وضعیت connected را مدیریت می‌کند، از آن استفاده کن
                if getattr(self.connector, "connected", False):
                    logger.debug("Connector already connected (owns_connector=True).")
                    return True
                ok = self.connector.initialize(**(init_opts or {}))
                if not ok:
                    logger.critical("Unable to initialize connector (owns_connector=True).")
                return bool(ok)
            except Exception:
                logger.exception("Exception while initializing owned connector")
                return False
        else:
            # کانکتور خارجی: سعی می‌کنیم آن را ensure کنیم
            try:
                # اگر صریحاً فِلَگ connected وجود دارد و True است، فرض می‌کنیم آماده است
                if getattr(self.connector, "connected", False):
                    logger.debug("External connector already connected.")
                    return True
                # در غیر این صورت از ensure_connection استفاده کن (ممکن است initialize را خودش انجام دهد)
                ok = self.connector.ensure_connection()
                if not ok:
                    logger.error("External connector.ensure_connection() returned False.")
                return bool(ok)
            except Exception:
                logger.exception("Error while ensuring external connector")
                return False

    def fetch_and_save_all(  
        self,
        num_bars: Optional[int] = None,
        initialize_retry: Optional[Dict[str, int]] = None,
        skip_existing: bool = False,
        per_symbol_retries: int = 2,
        per_symbol_backoff: float = 2.0,
        skip_symbols: Optional[list[str]] = None
    ) -> bool:
        """
        دانلود دسته‌ای برای همه نمادها و تایم‌فریم‌ها و ذخیرهٔ CSV.

        پارامترها:
          - num_bars: تعداد کندل برای هر دانلود؛ اگر None از config.download_defaults.num_bars استفاده می‌شود.
          - initialize_retry: دیکشنری شامل max_retries و retry_delay برای اتصال اولیه MT5.
                               اگر None از config.download_defaults.initialize_retry استفاده می‌شود.
          - skip_existing: اگر True و فایل CSV وجود داشت، دانلود برای آن فایل رد شود.
          - per_symbol_retries: تعداد تلاش‌های مجدد برای دانلود یک نماد/تایم‌فریم در صورت خطا.
          - per_symbol_backoff: تاخیر بین تلاش‌های مجدد (ثانیه).
          - skip_symbols: لیستی از نمادها که باید نادیده گرفته شوند (اختیاری).
        باز می‌گرداند True اگر روند کلی اجرا شد (حتی اگر برخی دانلودها با خطا مواجه شده باشند).
        """
        cfg_defaults = self.config.get("download_defaults", {}) if isinstance(self.config, dict) else {}
        num_bars = int(num_bars or cfg_defaults.get("num_bars", 3000))
        init_opts = initialize_retry or cfg_defaults.get("initialize_retry", {"max_retries": 60, "retry_delay": 5})

        skip_symbols_set: Set[str] = set(skip_symbols or [])

        # اگر نمادها یا تایم‌فریم‌ها خالی‌اند خبر می‌دهیم و سریع خروج می‌دهیم (هیچ کاری انجام نمی‌شود)
        if not self.symbols or not self.timeframes:
            logger.warning("No symbols or timeframes configured for MT5DataLoader. symbols=%r timeframes=%r",
                           self.symbols, self.timeframes)
            return True

        # اتصال: بررسی/initialize کانکتور
        logger.info("Preparing MT5 connector (init_opts=%s, owns_connector=%s)...", init_opts, self._owns_connector)
        if not self._ensure_connector_ready(init_opts):
            logger.critical("MT5 connector not ready — aborting download.")
            return False

        # پوشه خروجی — اطمینان از اینکه output_dir یک Path است و قابل mkdir است
        try:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.exception("Failed to create output directory %s: %s", self.output_dir, e)
            # در این حالت بلافاصله بازگردانده می‌شود چون نوشتن فایل ممکن نیست
            # همچنین اگر ما مالک کانکتور هستیم آن را می‌بندیم
            if self._owns_connector:
                try:
                    self.connector.shutdown()
                except Exception:
                    logger.exception("Error while shutting down connector after directory failure")
            return False

        # حلقه دانلود
        for symbol in self.symbols:
            if symbol in skip_symbols_set:
                logger.info("Skipping symbol (user requested): %s", symbol)
                continue

            for tf in self.timeframes:
                out_file = self.output_dir / f"{symbol}_{tf}.csv"

                if skip_existing and out_file.exists():
                    logger.info("Skipping existing file %s (skip_existing=True)", out_file)
                    continue

                attempt = 0
                while attempt <= per_symbol_retries:
                    try:
                        attempt += 1
                        logger.info("Downloading %s %s (attempt %d/%d) — bars=%d", symbol, tf, attempt, per_symbol_retries + 1, num_bars)
                        df = self.connector.get_candles(symbol=symbol, timeframe=tf, num_candles=num_bars)

                        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                            logger.warning("Empty or no data returned for %s %s on attempt %d", symbol, tf, attempt)
                            # در صورتی که داده خالی است، تکرار دوباره منطقی نیست مگر خطا باشد
                            break

                        # مرتب‌سازی و ایمن‌سازی ایندکس
                        try:
                            if "time" in df.columns:
                                df = df.sort_values("time").set_index("time")
                            else:
                                # اگر ایندکس قبلاً زمان بود
                                df.sort_index(inplace=True)
                        except Exception:
                            # اگر DataFrame ساختار غیرمنتظره‌ای داشت، لاگ کن و شکست بخور
                            logger.exception("Unexpected DataFrame structure for %s %s; aborting this pair.", symbol, tf)
                            break

                        # ذخیره CSV — index_label مشخص شده تا خوانایی بیشتر شود
                        try:
                            df.to_csv(out_file, index=True, index_label="time")
                        except Exception:
                            logger.exception("Failed to write CSV for %s %s to %s", symbol, tf, out_file)
                            raise

                        logger.info("Saved %s", out_file)
                        # موفق شدیم — خروج از حلقه retry
                        break

                    except Exception as ex:
                        logger.exception("Failed to download/save %s %s on attempt %d: %s", symbol, tf, attempt, ex)
                        if attempt > per_symbol_retries:
                            logger.error("Exceeded retries for %s %s — giving up and continuing with next pair", symbol, tf)
                            break
                        else:
                            sleep_time = per_symbol_backoff * attempt
                            logger.info("Waiting %.1f seconds before retrying %s %s", sleep_time, symbol, tf)
                            time.sleep(sleep_time)

        # در انتها فقط اگر ما مالکِ connector بوده‌ایم آن را می‌بندیم
        if self._owns_connector:
            try:
                self.connector.shutdown()
            except Exception:
                logger.exception("Error while shutting down connector")

        return True

    def fetch_range_and_save(self, symbol: str, timeframe: str, date_from, date_to, out_path: str) -> bool:
        """
        دانلود بین دو تاریخ و ذخیره به out_path (CSV).
        توجه: date_from/date_to باید اشیاء datetime یا timezone-aware باشند (بهتر است timezone-aware استفاده شود).
        """
        try:
            logger.info("Downloading range %s %s from %s to %s", symbol, timeframe, date_from, date_to)
            if not self.connector.ensure_connection():
                logger.error("Connector not connected for range download")
                raise ConnectionError("Connector not connected")

            df = self.connector.get_candles_range(symbol, timeframe, date_from, date_to)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logger.warning("No data in range for %s %s", symbol, timeframe)
                return False

            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=True, index_label="time")
            logger.info("Saved range file %s", out_path)
            return True

        except Exception as ex:
            logger.exception("Error in fetch_range_and_save: %s", ex)
            return False

    # --- convenience wrapper for single pair download (useful for tests) ---
    def fetch_one_and_save(self, symbol: str, timeframe: str, num_bars: Optional[int] = None, out_file: Optional[str] = None) -> bool:
        """
        دانلود یک نماد-تایم‌فریم و ذخیره؛ برای تست سریع یا مثال‌ها مفید است.
        """
        nb = int(num_bars or self.config.get("download_defaults", {}).get("num_bars", 3000))
        out_file = out_file or str(self.output_dir / f"{symbol}_{timeframe}.csv")
        return self.fetch_range_and_save_wrapper(symbol, timeframe, nb, out_file)

    def fetch_range_and_save_wrapper(self, symbol: str, timeframe: str, num_bars: int, out_file: str) -> bool:
        """
        دانلود num_bars آخر برای نماد و ذخیره به out_file.
        این متد برای سازگاری با تست‌های ساده نوشته شده است.
        """
        try:
            if not self.connector.ensure_connection():
                logger.error("Connector not connected for fetch_range_and_save_wrapper")
                raise ConnectionError("Connector not connected")

            df = self.connector.get_candles(symbol=symbol, timeframe=timeframe, num_candles=num_bars)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logger.warning("Empty data for %s %s", symbol, timeframe)
                return False

            out_path = Path(out_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if "time" in df.columns:
                df = df.sort_values("time").set_index("time")
            df.to_csv(out_path, index=True, index_label="time")
            logger.info("Saved %s", out_path)
            return True

        except Exception as ex:
            logger.exception("Error in fetch_range_and_save_wrapper: %s", ex)
            return False

    def close(self) -> None:
        """
        بستن ایمنِ connector در صورت مالکیت.
        می‌توانید این متد را از خارج فراخوانی کنید تا loader به صورت صریح cleanup کند.
        """
        if self._owns_connector:
            try:
                self.connector.shutdown()
            except Exception:
                logger.exception("Error while closing owned connector")
