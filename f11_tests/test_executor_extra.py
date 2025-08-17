# f11_tests/test_executor_extra.py
import os
import tempfile
import pandas as pd
import logging

import pytest

# مسیرهای ماژول‌ها ممکن است بسته به ساختار پروژه شما متفاوت باشد.
# در پروژه‌ی شما این مسیرها طبق صحبت‌ها هستند:
from f13_risk import risk_manager
from f02_data.mt5_data_loader import MT5DataLoader

logging.basicConfig(level=logging.INFO)


def test_compute_lot_returns_zero_when_raw_below_half_min(monkeypatch):
    """
    اگر raw_lot < 0.5 * min_lot -> باید 0.0 برگردد.
    ما pip_value_per_lot را mock می‌کنیم تا وابستگی به MT5 حذف شود.
    """
    monkeypatch.setattr(risk_manager, "pip_value_per_lot", lambda symbol, lot=1.0, account_currency=None: 10.0)
    lot, actual = risk_manager.compute_lot_size_by_risk(
        equity=100.0, risk_percent=1.0, sl_pips=50.0, symbol="EURUSD"
    )
    assert lot == 0.0
    assert actual == 0.0  # چون lot=0.0، ریسک واقعی صفر است


def test_compute_lot_rounds_up_to_min_if_between_half_and_min(monkeypatch):
    """
    اگر raw_lot بین 0.5*min_lot و min_lot باشد -> باید به min_lot بالاگرد شود.
    مثال: با pip_value=10 و sl_pips=12 -> raw_lot ~= 0.00833 -> با min_lot=0.01 باید 0.01 شود.
    """
    monkeypatch.setattr(risk_manager, "pip_value_per_lot", lambda symbol, lot=1.0, account_currency=None: 10.0)
    lot, actual = risk_manager.compute_lot_size_by_risk(
        equity=100.0, risk_percent=1.0, sl_pips=12.0, symbol="EURUSD"
    )
    assert pytest.approx(lot, rel=1e-9) == 0.01
    assert actual > 0.0


def test_pip_value_fallback_without_mt5(monkeypatch):
    """
    اگر MT5 در دسترس نباشد، pip_value_per_lot باید مقدار معقول (بزرگتر از صفر) بازگرداند.
    این تست _HAS_MT5 را موقتاً False می‌کند.
    """
    monkeypatch.setattr(risk_manager, "_HAS_MT5", False)
    # فراخوانی تابع باید بدون exception باشد و مقدار مثبت بدهد
    val = risk_manager.pip_value_per_lot("EURUSD", lot=1.0, account_currency=None)
    assert isinstance(val, float)
    assert val > 0.0


def test_mt5_dataloader_continues_on_empty_df(tmp_path, monkeypatch):
    """
    اگر connector.get_candles یک DataFrame خالی برگرداند، MT5DataLoader باید آن را رد کند و ادامه دهد
    (بدون خطا و بدون نوشتن فایل).
    ما یک connector stub تعریف می‌کنیم.
    """
    class StubConnectorEmpty:
        def initialize(self, **kwargs):
            return True
        def get_candles(self, symbol, timeframe, num_candles=0):
            return pd.DataFrame()  # خالی
        def shutdown(self):
            pass

    outdir = tmp_path / "price"
    outdir.mkdir()
    loader = MT5DataLoader(connector=StubConnectorEmpty())
    # override symbols/timeframes محلی برای عدم وابستگی به config
    loader.symbols = ["FAKEPAIR"]
    loader.timeframes = ["M1"]
    loader.output_dir = str(outdir)

    # نباید استثنا پرتاب کند و نباید فایلی ایجاد شود
    loader.fetch_and_save_all(num_bars=10)
    files = list(outdir.glob("*"))
    assert len(files) == 0


def test_mt5_dataloader_saves_csv_when_data_present(tmp_path, monkeypatch):
    """
    اگر connector.get_candles داده واقعی برگرداند، فایل CSV در output_dir ذخیره شود.
    """
    class StubConnectorData:
        def initialize(self, **kwargs):
            return True
        def get_candles(self, symbol, timeframe, num_candles=0):
            # تولید df نمونه با چند ردیف
            idx = pd.date_range("2025-01-01", periods=3, freq="min")
            df = pd.DataFrame({
                "open": [1.0, 1.1, 1.2],
                "high": [1.2, 1.2, 1.3],
                "low":  [0.9, 1.0, 1.05],
                "close":[1.05, 1.15, 1.25],
                "tick_volume":[10,11,12],
                "spread":[1,1,1],
            }, index=idx)
            df.index.name = "time"
            return df
        def shutdown(self):
            pass

    outdir = tmp_path / "price2"
    outdir.mkdir()
    loader = MT5DataLoader(connector=StubConnectorData())
    loader.symbols = ["FAKEPAIR2"]
    loader.timeframes = ["M1"]
    loader.output_dir = str(outdir)

    loader.fetch_and_save_all(num_bars=3)
    files = list(outdir.glob("*.csv"))
    assert len(files) == 1
    # basic check محتویات فایل
    df = pd.read_csv(files[0], index_col="time", parse_dates=True)
    assert not df.empty
    assert "open" in df.columns
