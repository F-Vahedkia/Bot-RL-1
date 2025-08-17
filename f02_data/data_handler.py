# f02_data/data_handler.py
"""
DataHandler:
- دریافت داده‌های تاریخی برای نمادها در بازه‌ی مشخص
- تبدیل به pandas.DataFrame و مرتب‌سازی
- این کلاس نباید مسئول اتصال مستقیم و مدیریت ذخیره‌ی CSV باشد (آن وظیفه MT5DataLoader است)
"""
import pandas as pd
from datetime import datetime
from f02_data.mt5_connector import MT5Connector
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, config):
        """
        config: دیکشنریِ کانفیگ (معمولاً از config_loader.get_all())
        انتظار کلیدها:
          - symbols: list of str
          - timeframe: str e.g. "M5"
          - start_date, end_date: "YYYY-MM-DD"
        """
        self.config = config
        self.symbols = config.get('symbols', [])
        self.timeframe = config.get('timeframe', 'M1')
        # تبدیل تاریخ‌های ورودی به datetime
        self.start_date = datetime.strptime(config.get('start_date'), "%Y-%m-%d")
        self.end_date = datetime.strptime(config.get('end_date'), "%Y-%m-%d")
        self.connector = MT5Connector()
        if not self.connector.initialize():
            raise ConnectionError("Failed to initialize MT5 connector in DataHandler")

    def fetch_all(self):
        """گرفتن داده برای همه نمادها در بازه مشخص"""
        data = {}
        for s in self.symbols:
            df = self.fetch_ohlc(s)
            if df is not None and not df.empty:
                data[s] = df
        return data

    def fetch_ohlc(self, symbol: str) -> pd.DataFrame:
        """گرفتن داده OHLC یک نماد در بازه start_date..end_date"""
        try:
            df = self.connector.get_candles_range(
                symbol=symbol,
                timeframe=self.timeframe,
                date_from=self.start_date,
                date_to=self.end_date
            )
            if df.empty:
                logger.warning("No data for %s in range", symbol)
                return pd.DataFrame()
            return df
        except Exception as ex:
            logger.exception("Error fetching OHLC for %s: %s", symbol, ex)
            return pd.DataFrame()

    def close(self):
        self.connector.shutdown()
