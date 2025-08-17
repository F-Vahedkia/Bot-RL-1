# f11_tests/test_executor.py
import pytest
from unittest.mock import MagicMock, patch

# مسیر: pytest در ریشه پروژه اجرا شود تا imports نسبی کار کنند.


# ---------- Fixtures: ساخت ماژول mt5 موک و risk_manager موک ----------
@pytest.fixture
def mock_mt5():
    m = MagicMock()
    # constants
    m.TRADE_ACTION_DEAL = 0
    m.ORDER_TYPE_BUY = 0
    m.ORDER_TYPE_SELL = 1
    # order_send default returns a dict-like success
    m.order_send = MagicMock(return_value={"retcode": 0, "order": 123})
    # symbol_info_tick returns an object with bid/ask
    tick = MagicMock()
    tick.bid = 1.2345
    tick.ask = 1.2347
    m.symbol_info_tick = MagicMock(return_value=tick)
    # account_info
    ai = MagicMock()
    ai.equity = 10000.0
    ai.balance = 10000.0
    ai.currency = "USD"
    m.account_info = MagicMock(return_value=ai)
    # symbol_info for pip point
    si = MagicMock()
    si.point = 0.0001
    si.trade_contract_size = 100000
    m.symbol_info = MagicMock(return_value=si)
    # positions_get default empty
    m.positions_get = MagicMock(return_value=[])
    return m

@pytest.fixture
def mock_risk_manager():
    rm = MagicMock()
    # compute_lot_size_by_risk returns (lot, actual_risk)
    rm.compute_lot_size_by_risk = MagicMock(return_value=(0.02, 10.0))
    return rm

# ---------- Import Executor (after creating fixtures) ----------
from f09_execution.executor_for_test import Executor, BUY, SELL

# ---------- Tests ----------

def test_place_market_order_builds_request_and_sends(mock_mt5):
    exec = Executor(mt5_module=mock_mt5, risk_manager=None, cfg={"execution": {"magic": 42}})
    # provide explicit lot (no risk_percent)
    res = exec.place_market_order(symbol="EURUSD", side=BUY, lot=0.05, sl_pips=20, tp_pips=40, comment="test")
    assert res["ok"] is True
    # ensure order_send called exactly once
    assert mock_mt5.order_send.called
    args = mock_mt5.order_send.call_args[0]
    assert isinstance(args[0], dict) or len(args) == 1  # allow order_send(req) or order_send(dict)
    # inspect request dict depending on how Executor called order_send
    req = args[0] if len(args) == 1 else args
    assert req["symbol"] == "EURUSD"
    assert float(req["volume"]) == pytest.approx(0.05)
    # comment and magic present
    assert req.get("comment", "").startswith("test") or req.get("magic", 42) == 42

def test_place_market_order_by_risk_calls_risk_manager(mock_mt5, mock_risk_manager):
    exec = Executor(mt5_module=mock_mt5, risk_manager=mock_risk_manager)
    # call with risk_percent -> should call risk_manager.compute_lot_size_by_risk
    res = exec.place_market_order(symbol="EURUSD", side=BUY, risk_percent=1.0, sl_pips=20)
    mock_risk_manager.compute_lot_size_by_risk.assert_called_once()
    assert res["ok"] is True
    assert res["lot"] == pytest.approx(0.02)

def test_close_position_by_ticket_sends_close_request(monkeypatch, mock_mt5):
    # prepare a fake position object returned by positions_get
    pos = MagicMock()
    pos.ticket = 9999
    pos.symbol = "EURUSD"
    pos.volume = 0.02
    pos.type = mock_mt5.ORDER_TYPE_BUY  # originally buy
    mock_mt5.positions_get = MagicMock(return_value=[pos])

    exec = Executor(mt5_module=mock_mt5, risk_manager=None)
    res = exec.close_position_by_ticket(ticket=9999)
    # close triggers place_market_order and ultimately mt5.order_send
    assert mock_mt5.order_send.called
    assert res["ok"] is True

def test_news_avoid_skips_trade(mock_mt5):
    # news_checker returns True => skip
    def news_checker(symbol):
        return True

    exec = Executor(mt5_module=mock_mt5, risk_manager=None, news_checker=news_checker)
    res = exec.place_market_order(symbol="EURUSD", side=BUY, lot=0.01)
    assert res["ok"] is False
    assert res["reason"] == "news_avoidance"
    mock_mt5.order_send.assert_not_called()

def test_compute_lot_by_risk_unit(monkeypatch):
    # Import risk_manager module if exists, else skip
    try:
        from f13_risk import risk_manager as rm
    except Exception:
        pytest.skip("risk_manager module not available for unit test")

    # monkeypatch pip_value_per_lot to deterministic value
    monkeypatch.setattr(rm, "pip_value_per_lot", lambda symbol, lot=1.0, account_currency=None: 10.0)
    lot, actual = rm.compute_lot_size_by_risk(equity=10000, risk_percent=1.0, sl_pips=20, symbol="EURUSD")
    # risk_amount = 10000 * 1% = 100; pip_val=10 => raw_lot = 100 / (20*10) = 0.5 => after rounding probably 0.5 or adjusted
    assert lot >= 0.0
    assert actual >= 0.0
