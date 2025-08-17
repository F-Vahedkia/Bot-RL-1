# data/mt5_connector.py
"""
MT5Connector
- مدیریت اتصال/قطع اتصال به MetaTrader5
- گرفتن کندل‌ها (تعداد معین یا در یک بازه‌ی زمانی)
- تضمین (ensure) اتصال و retry هنگام قطع
"""
import time
import logging
import pandas as pd
import MetaTrader5 as mt5
from f10_utils.config_loader import config

logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self.connected = False
        # انتظار می‌رود config ماژول از قبل بارگذاری شده باشد
        self.login_info = config["mt5_credentials"]

    def initialize(self, max_retries: int = 60, retry_delay: int = 5) -> bool:
        """
        تلاش برای اتصال و لاگین به MT5 با مکانیزم retry.
        برمی‌گرداند True در صورت موفقیت.
        """
        for attempt in range(1, max_retries + 1):
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed (attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
                continue

            authorized = mt5.login(
                login=self.login_info["login"],
                password=self.login_info["password"],
                server=self.login_info["server"]
            )

            if authorized:
                self.connected = True
                logger.info("Connected to MetaTrader5")
                return True

            logger.error(f"MT5 login failed (attempt {attempt}/{max_retries})")
            mt5.shutdown()
            time.sleep(retry_delay)

        logger.critical("All attempts to connect to MT5 failed.")
        return False

    def ensure_connection(self) -> bool:
        """
        اگر اتصال قطع بود، تلاش می‌کند دوباره وصل شود (با استفاده از initialize).
        """
        if not self.connected:
            logger.warning("MT5 not connected — attempting reconnect...")
            return self.initialize()
        return True

    def shutdown(self):
        """قطع اتصال از MT5 و بروزرسانی وضعیت داخلی."""
        try:
            mt5.shutdown()
        except Exception as e:
            logger.exception("Error while shutting down MT5: %s", e)
        finally:
            self.connected = False
            logger.info("MT5 shutdown complete")

    def get_candles(self, symbol: str, timeframe: str, num_candles: int = 1000) -> pd.DataFrame:
        """
        دریافت آخرین `num_candles` کندل برای نماد و تایم‌فریم مشخص.
        خروجی: pandas.DataFrame با index از نوع datetime و ستون‌های open/high/low/close/tick_volume/spread
        """
        if not self.ensure_connection():
            raise ConnectionError("Cannot connect to MT5")

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M30": mt5.TIMEFRAME_M30,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }

        tf = tf_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_candles)
        if rates is None or len(rates) == 0:
            logger.warning("No rates for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def get_candles_range(self, symbol: str, timeframe: str, date_from, date_to) -> pd.DataFrame:
        """
        دریافت کندل‌ها بین دو تاریخ (inclusive).
        date_from / date_to باید از نوع datetime.datetime باشند.
        """
        if not self.ensure_connection():
            raise ConnectionError("Cannot connect to MT5")

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M30": mt5.TIMEFRAME_M30,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }

        tf = tf_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            logger.warning("No rates (range) for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df
