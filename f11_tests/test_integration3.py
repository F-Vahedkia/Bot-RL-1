# f11_tests/test_integration_full.py
"""
Integration test (safe) for core modules:
 - f10_utils/config_loader.py (config)
 - f10_utils/logging_cfg.py
 - f02_data/mt5_connector.py
 - f02_data/mt5_data_loader.py
 - f02_data/data_handler.py  (if available)
 - f13_risk/risk_manager.py
 - f09_execution/executor_3.py  (only read-only methods)
This script *uses real MT5 connection* but DOES NOT send any trade orders.
Run from project root:
    python f11_tests/test_integration_full.py
"""
from __future__ import annotations

import sys
import os
import traceback
from datetime import datetime, timedelta
import logging

# ensure project root is on path
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# --- load config + logging (make logging init idempotent) ---
from f10_utils.config_loader import config

# Configure logging only if not already configured (prevents duplicate handlers/messages)
try:
    from f10_utils.logging_cfg import configure_logging
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging()
    else:
        # already configured by other imports; nothing to do
        logging.getLogger(__name__).debug("Logging already configured; skipping configure_logging()")
except Exception as e:
    # fallback minimal logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning("configure_logging() failed; fallback basicConfig used: %s", e)

# imports from your project
from f02_data.mt5_connector import MT5Connector
from f02_data.mt5_data_loader import MT5DataLoader

# DataHandler optional
try:
    from f02_data.data_handler import DataHandler
except Exception:
    DataHandler = None

from f10_utils.risk_manager import compute_lot_size_by_risk, pip_value_per_lot

# executor_3 is your current executor file (safe: we will not call order_send)
try:
    from f09_execution import executor_3 as executor_mod
except Exception:
    executor_mod = None

import MetaTrader5 as mt5  # used only for safe read operations

LOG = logging.getLogger(__name__)


def check_config_keys(cfg, required_paths):
    missing = []
    for path in required_paths:
        node = cfg
        for p in path.split("."):
            if not isinstance(node, dict) or p not in node:
                missing.append(path)
                break
            node = node[p]
    return missing


def safe_call(fn, *a, **kw):
    """helper: call fn and print exception but don't raise further"""
    try:
        return fn(*a, **kw)
    except Exception:
        traceback.print_exc()
        return None


def main():
    print("=== Integration test (safe) start ===")
    # quick config checks
    required = [
        "mt5_credentials.login",
        "mt5_credentials.password",
        "mt5_credentials.server",
    ]
    missing = check_config_keys(config, required)
    if missing:
        print("Missing required config keys (populate via .env or config.yaml):", missing)
        print("Aborting integration test.")
        return 2

    # create a single connector and initialize it once
    print("-> testing MT5Connector.initialize()")
    connector = MT5Connector()
    init_retry = config.get("download_defaults", {}).get("initialize_retry", {"max_retries": 10, "retry_delay": 2})
    try:
        ok = connector.initialize(**init_retry)
    except Exception as e:
        print("MT5Connector.initialize raised exception:", e)
        traceback.print_exc()
        return 3

    if not ok:
        print("MT5Connector failed to initialize (check credentials/MT5 terminal).")
        return 4
    print("  MT5 connected OK")

    # 2) fetch a few candles with MT5Connector (safe read)
    symbols = config.get("symbols", ["EURUSD"])
    timeframes = config.get("timeframes", ["M1"])
    symbol = symbols[0]
    tf = timeframes[0]
    print(f"-> fetching a few candles: {symbol} {tf}")
    try:
        df = connector.get_candles(symbol, tf, num_candles=10)
    except Exception as e:
        print("get_candles raised exception:", e)
        traceback.print_exc()
        # try shut down and exit gracefully
        try:
            connector.shutdown()
        except Exception:
            pass
        return 5

    if df is None or df.empty:
        print("get_candles returned no data. This might be normal (market closed) or feed issue.")
    else:
        # df.index may be a DatetimeIndex or there may be 'time' column
        try:
            first_idx = df.index[0] if hasattr(df, "index") and len(df.index) > 0 else df.iloc[0].get("time", None)
            last_idx = df.index[-1] if hasattr(df, "index") and len(df.index) > 0 else df.iloc[-1].get("time", None)
            print(f"  fetched {len(df)} rows; first index: {first_idx} last index: {last_idx}")
        except Exception:
            print(f"  fetched {len(df)} rows")

    # 3) MT5DataLoader.download (small batch)
    print("-> testing MT5DataLoader.fetch_and_save_all (num_bars=10) -- will create files in price_data path")
    loader = MT5DataLoader(connector=connector, cfg=config)

    # IMPORTANT: many loader implementations call connector.initialize() and connector.shutdown()
    # to avoid double-initialize/shutdown when we intentionally reuse the same connector,
    # temporarily replace connector.initialize and connector.shutdown with no-ops for the loader call.
    # (This prevents duplicate 'Connected' logs and premature shutdown.)
    need_restore_init = False
    need_restore_shutdown = False
    saved_init = None
    saved_shutdown = None

    if getattr(loader, "connector", None) is connector:
        # patch only for this call
        if hasattr(connector, "initialize"):
            saved_init = connector.initialize
            connector.initialize = lambda **kw: True  # no-op (assume already initialized)
            need_restore_init = True
        if hasattr(connector, "shutdown"):
            saved_shutdown = connector.shutdown
            connector.shutdown = lambda: None  # defer actual shutdown to end of test
            need_restore_shutdown = True

    try:
        # call loader; even if it internally attempts initialize/shutdown we've made them safe
        safe_call(loader.fetch_and_save_all, num_bars=10, initialize_retry=init_retry, skip_existing=False)
        print("  fetch_and_save_all completed (check configured price_data folder).")
    except Exception as e:
        print("MT5DataLoader.fetch_and_save_all raised exception:", e)
        traceback.print_exc()
        # continue; not fatal
    finally:
        # restore patched methods
        if need_restore_init and saved_init is not None:
            connector.initialize = saved_init
        if need_restore_shutdown and saved_shutdown is not None:
            connector.shutdown = saved_shutdown

    # 4) DataHandler (optional) - call a non-destructive fetch if available
    if DataHandler is not None:
        print("-> testing DataHandler.fetch_all (if available)")
        try:
            # try construct a minimal cfg for DataHandler if it expects that
            dh_cfg = {
                "symbols": [symbol],
                "timeframe": tf,
                "start_date": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "end_date": datetime.utcnow().strftime("%Y-%m-%d"),
            }
            # If DataHandler accepts connector param, pass the same connector (best-effort)
            try:
                handler = DataHandler(dh_cfg, connector=connector)  # some implementations accept connector
            except TypeError:
                handler = DataHandler(dh_cfg)

            # many implementations provide fetch_all or fetch_range; try safe methods
            if hasattr(handler, "fetch_all"):
                data_map = safe_call(handler.fetch_all)
                if not data_map:
                    print("  DataHandler.fetch_all returned empty dict or None.")
                else:
                    for s, df2 in data_map.items():
                        print(f"  DataHandler fetched {len(df2)} rows for {s}")
            elif hasattr(handler, "fetch_range_and_save"):
                print("  DataHandler has fetch_range_and_save only; calling with small range to temp file")
                tmp_file = os.path.join("f02_data", "temp_test_data.csv")
                success = safe_call(handler.fetch_range_and_save, symbol, tf, None, None, tmp_file)
                print("  fetch_range_and_save result:", success)
            else:
                print("  DataHandler does not expose fetch_all or fetch_range_and_save; skipping.")
            # try to close handler if it has close()
            try:
                handler.close()
            except Exception:
                pass
        except Exception as e:
            print("  DataHandler test raised exception (non-fatal):", e)
            traceback.print_exc()
    else:
        print("-> DataHandler not found; skipping DataHandler tests")

    # 5) Risk manager quick checks (compute lot using on-account equity)
    print("-> testing risk manager and executor read-only methods")
    try:
        # ensure MT5 is active (re-initialize the connector if needed)
        try:
            # if connector has ensure_connection, use it; else try initialize again
            if hasattr(connector, "ensure_connection"):
                ok_conn = connector.ensure_connection()
                if not ok_conn and hasattr(connector, "initialize"):
                    connector.initialize(**init_retry)
            else:
                # best-effort
                connector.initialize(**init_retry)
        except Exception:
            # ignore; we'll try to read account_info anyway
            LOG.debug("Re-init attempt for connector failed (continuing)")

        ai = mt5.account_info()
        if ai is None:
            print("  account_info() returned None; cannot compute lot_by_risk, skipping")
        else:
            equity = float(getattr(ai, "equity", getattr(ai, "balance", 0.0)))
            print(f"  account equity read: {equity:.2f}")
            # compute a sample lot (1% risk, 20 pips)
            lot, actual = compute_lot_size_by_risk(equity, 1.0, 20.0, symbol)
            print(f"  compute_lot_size_by_risk -> lot={lot}, estimated_risk_amount={actual:.2f}")
            # pip value sample
            pip_val = pip_value_per_lot(symbol, lot=1.0)
            print(f"  pip_value_per_lot(symbol,1.0) = {pip_val}")
    except Exception as e:
        print("  risk manager checks raised exception (non-fatal):", e)
        traceback.print_exc()

    # 6) Executor (executor_3) read-only checks (no order_send)
    if executor_mod is not None:
        print("-> testing executor_3 read-only methods (no order_send)")
        try:
            # create executor with same connector (so it uses existing mt5 connection)
            ExecutorClass = getattr(executor_mod, "Executor", None)
            if ExecutorClass is not None:
                exec_obj = ExecutorClass(connector=connector, cfg=config)
                # list positions (safe read)
                positions = safe_call(exec_obj.list_positions) or []
                print(f"  executor.list_positions returned {len(positions)} entries")
                # account equity via executor method if present
                if hasattr(exec_obj, "_get_account_equity"):
                    try:
                        eq2 = exec_obj._get_account_equity()
                        print(f"  executor._get_account_equity -> {eq2:.2f}")
                    except Exception as e:
                        print("  executor._get_account_equity raised:", e)
                else:
                    print("  executor does not implement _get_account_equity()")
            else:
                print("  Executor class not found in executor_3 module")
        except Exception as e:
            print("  executor_3 checks raised exception (non-fatal):", e)
            traceback.print_exc()
    else:
        print("-> executor_3 module not found; skipping executor checks")

    # 7) shutdown connector cleanly (once)
    try:
        # if connector has is_connected/ensure_connection, check before shutdown
        try:
            if hasattr(connector, "ensure_connection"):
                if connector.ensure_connection():
                    connector.shutdown()
                else:
                    # not connected; nothing to do
                    pass
            else:
                # best-effort shutdown
                connector.shutdown()
        except Exception:
            # still attempt shutdown
            try:
                connector.shutdown()
            except Exception:
                pass
        print("-> MT5 connector shutdown OK")
    except Exception as e:
        print("-> connector.shutdown raised exception:", e)
        traceback.print_exc()

    print("=== Integration test finished ===")
    return 0


if __name__ == "__main__":
    rc = main()
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)
