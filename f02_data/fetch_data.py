# f02_data/fetch_data.py
"""
اجرای فرآیند دانلود دسته‌ای داده‌ها (daily/weekly) — این فایل را کرون یا scheduler فراخوانی کند.
"""
import logging
from f02_data.mt5_data_loader import MT5DataLoader

# ساده و مؤثر: تنظیم لاگ کلی برنامه
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting MT5 data fetch...")
    loader = MT5DataLoader()
    loader.fetch_and_save_all(num_bars=3000)
    logger.info("MT5 data fetch finished.")

if __name__ == "__main__":
    main()
