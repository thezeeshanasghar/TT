
# mt5_fib_snr_scalper.py
# Live MT5 scalping bot: Fibonacci retracement + Support/Resistance confluence
# Risk controls: 1% per-trade risk, daily loss cap, max effective leverage, max open positions, optional FIFO/no-hedging.
# Notes:
# - Live trading only; no backtests (per client constraints). A dry-run toggle is provided for safe forward testing.
# - Default example: EURUSD on M1. Adjust CONFIG as needed.
# - Ensure your MT5 terminal is running and logged in before executing this script.
# - U.S.-style constraints: set CONFIG["nfa_fifo_no_hedging"] = True to avoid hedging and enforce FIFO closes.

import MetaTrader5 as mt5
import time
import math
import statistics
import json
import os
from datetime import datetime, timedelta, timezone

# ========== CONFIGURATION ==========
CONFIG = {
    # Trading
    "symbol": "XAUUSD",
    "timeframe": mt5.TIMEFRAME_M1,       # M1 default
    "lookback_bars": 600,                # bars to pull each loop
    "poll_interval_sec": 0.5,            # loop sleep time

    # Fib/SNR mechanics
    "swing_left_right": 5,               # fractal swing pivot width (bars on each side)
    "snr_pivot_width": 3,                # pivot width for S/R map (smaller than swing width to capture micro levels)
    "snr_lookback": 300,                 # bars to scan for S/R pivot levels
    "confluence_tolerance_pips": 0.5,    # price must be within this to both a fib level and S/R
    "min_impulse_pips": 5,               # minimum size of the last impulse leg to anchor fibs
    "atr_period": 14,                    # for volatility sanity checks
    "min_atr_pips": 0.4,                 # avoid ultra-dead markets

    # Entries/Exits
    "rr_ratio": 1.2,                     # take-profit = rr_ratio * risk (distance to stop)
    "stop_buffer_pips": 1.0,             # extra beyond swing
    "entry_mode": "market",              # "market" or "limit"
    "limit_improve_pips": 0.0,           # if using limit, improve price by this amount toward level

    # Risk & compliance
    "risk_per_trade_pct": 1.0,           # 1% risk per trade (of equity)
    "daily_loss_cap_pct": 3.0,           # stop new trades after -3% realized for the day
    "max_effective_leverage": 5.0,       # cap notional/equity
    "max_open_positions": 3,             # across all symbols
    "max_spread_pips": 0.6,              # skip if spread exceeds this
    "nfa_fifo_no_hedging": True,         # U.S.-style behavior: no hedging, FIFO closes
    "account_is_hedging": False,         # set True if your account supports hedging and you want to allow it

    # Execution
    "slippage_points": 10,               # deviation for order_send (points)
    "retry_on_requote": 2,
    "dry_run": False,                    # if True, log actions but do not send orders

    # Daily PnL tracking (uses local midnight by default; adjust if you want broker/server midnight)
    "daily_state_file": "daily_state.json",
}

# ========== UTILS ==========

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("No MT5 account info. Ensure MT5 terminal is running and logged in.")
    print(f"[INIT] Connected to MT5. Account: {acc.login}, Equity: {acc.equity:.2f}, Balance: {acc.balance:.2f}")

def select_symbol(symbol: str):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol}")
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        raise RuntimeError(f"Symbol info not available or not visible for {symbol}")
    return info

def now():
    return datetime.now()

def load_daily_state(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_daily_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)

def today_key():
    return datetime.now().strftime("%Y-%m-%d")

def points_per_pip(symbol_info):
    digits = symbol_info.digits
    if digits in (5, 3):
        return 10
    return 1

def pips_to_points(symbol_info, pips):
    return pips * points_per_pip(symbol_info)

def points_to_price(symbol_info, points):
    return points * symbol_info.point

def price_to_points(symbol_info, price_diff):
    return price_diff / symbol_info.point

def pips_to_price(symbol_info, pips):
    return points_to_price(symbol_info, pips_to_points(symbol_info, pips))

def get_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("Failed to get tick")
    return tick

def fetch_rates(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates for {symbol}")
    return list(rates)

def atr(values_high, values_low, values_close, period):
    trs = []
    for i in range(1, len(values_close)):
        hl = values_high[i] - values_low[i]
        hc = abs(values_high[i] - values_close[i-1])
        lc = abs(values_low[i] - values_close[i-1])
        trs.append(max(hl, hc, lc))
    if len(trs) < period:
        return None
    return statistics.fmean(trs[-period:])

def spread_in_pips(symbol_info, tick):
    sp = (tick.ask - tick.bid) / symbol_info.point
    return sp / points_per_pip(symbol_info)

def list_positions():
    poss = mt5.positions_get()
    return [] if poss is None else list(poss)

def count_open_positions():
    poss = list_positions()
    return len(poss)

def positions_for_symbol(symbol):
    return [p for p in list_positions() if p.symbol == symbol]

# ========== S/R AND SWING DETECTION ==========

def pivots(series_high, series_low, width):
    pivs = []
    n = len(series_high)
    for i in range(width, n - width):
        window_h = series_high[i - width:i + width + 1]
        window_l = series_low[i - width:i + width + 1]
        if series_high[i] == max(window_h) and window_h.count(series_high[i]) == 1:
            pivs.append((i, series_high[i], "H"))
        if series_low[i] == min(window_l) and window_l.count(series_low[i]) == 1:
            pivs.append((i, series_low[i], "L"))
    return pivs

def last_impulse_from_swings(highs, lows, width, min_pips, symbol_info):
    piv = pivots(highs, lows, width)
    if len(piv) < 2:
        return None
    piv.sort(key=lambda x: x[0])
    last = piv[-1]
    prev = piv[-2]
    if last[2] == "H" and prev[2] == "L":
        leg_pips = (last[1] - prev[1]) / symbol_info.point / points_per_pip(symbol_info)
        if leg_pips >= min_pips:
            return {"dir": "up", "low_idx": prev[0], "low": prev[1], "high_idx": last[0], "high": last[1]}
    if last[2] == "L" and prev[2] == "H":
        leg_pips = (prev[1] - last[1]) / symbol_info.point / points_per_pip(symbol_info)
        if leg_pips >= min_pips:
            return {"dir": "down", "high_idx": prev[0], "high": prev[1], "low_idx": last[0], "low": last[1]}
    return None

def fib_levels_from_leg(leg):
    lo = leg["low"]
    hi = leg["high"]
    if leg["dir"] == "up":
        length = hi - lo
        return {
            "23.6": hi - 0.236 * length,
            "38.2": hi - 0.382 * length,
            "50.0": hi - 0.500 * length,
            "61.8": hi - 0.618 * length,
            "78.6": hi - 0.786 * length
        }
    else:
        length = hi - lo
        return {
            "23.6": lo + 0.236 * length,
            "38.2": lo + 0.382 * length,
            "50.0": lo + 0.500 * length,
            "61.8": lo + 0.618 * length,
            "78.6": lo + 0.786 * length
        }

def snr_levels_from_pivots(highs, lows, width, lookback):
    piv = pivots(highs[-lookback:], lows[-lookback:], width)
    levels = []
    for idx, price, kind in piv:
        levels.append(price)
    levels = sorted(levels)
    clustered = []
    cluster_tol = (max(highs[-lookback:]) - min(lows[-lookback:])) * 0.001
    for lv in levels:
        if not clustered or abs(lv - clustered[-1]) > cluster_tol:
            clustered.append(lv)
        else:
            clustered[-1] = (clustered[-1] + lv) / 2.0
    return clustered

def nearest_confluence(price, fibs, snr_levels, tol_price):
    best = None
    for name, fprice in fibs.items():
        if abs(price - fprice) <= tol_price:
            for s in snr_levels:
                if abs(fprice - s) <= tol_price:
                    priority = {"78.6": 3, "61.8": 3, "50.0": 2, "38.2": 1, "23.6": 0}.get(name, 0)
                    score = priority - abs(price - fprice) / tol_price
                    if best is None or score > best["score"]:
                        best = {"name": name, "level": fprice, "sr": s, "score": score}
    return best

# ========== RISK & SIZING ==========
def calc_pip_value_per_lot(symbol_info):
    pips_points = points_per_pip(symbol_info)
    factor = pips_points / (symbol_info.trade_tick_size / symbol_info.point if symbol_info.trade_tick_size > 0 else 1.0)
    return symbol_info.trade_tick_value * factor

def round_volume(symbol_info, lots):
    step = symbol_info.volume_step
    minv = symbol_info.volume_min
    maxv = symbol_info.volume_max
    lots = max(minv, min(maxv, math.floor(lots / step) * step))
    return lots

def calc_position_size_lots(symbol_info, equity, risk_pct, stop_distance_pips, max_effective_leverage, price):
    if stop_distance_pips <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    pip_value = calc_pip_value_per_lot(symbol_info)
    lots_by_risk = risk_amount / (stop_distance_pips * pip_value)

    contract = symbol_info.trade_contract_size
    notional_per_lot = price * contract
    if equity <= 0:
        return 0.0
    lots_leverage_cap = (equity * max_effective_leverage) / max(notional_per_lot, 1e-12)

    lots = min(lots_by_risk, lots_leverage_cap)
    return round_volume(symbol_info, lots)

def daily_pnl_and_start_equity(state, acc_equity):
    key = today_key()
    today = state.get(key, {})
    if "start_equity" not in today:
        today["start_equity"] = acc_equity
        today["realized_pnl"] = 0.0
        state[key] = today
        return 0.0, today["start_equity"]
    return today.get("realized_pnl", 0.0), today["start_equity"]

def update_daily_realized_pnl(state, realized_delta):
    key = today_key()
    today = state.get(key, {})
    today["realized_pnl"] = today.get("realized_pnl", 0.0) + realized_delta
    state[key] = today

def realized_pnl_today_from_history():
    start = datetime.combine(datetime.now().date(), datetime.min.time())
    deals = mt5.history_deals_get(start, datetime.now())
    realized = 0.0
    if deals is not None:
        for d in deals:
            realized += d.profit
    return realized

def fifo_block_hedge(symbol, direction):
    poss = positions_for_symbol(symbol)
    if not poss:
        return False
    if CONFIG["nfa_fifo_no_hedging"] or not CONFIG["account_is_hedging"]:
        for p in poss:
            if direction == "buy" and p.type == mt5.POSITION_TYPE_SELL:
                return True
            if direction == "sell" and p.type == mt5.POSITION_TYPE_BUY:
                return True
    return False

# ========== ORDER ROUTING ==========
def send_order(symbol_info, direction, entry_price, sl_price, tp_price, lots):
    symbol = symbol_info.name
    if lots <= 0:
        return False, "Zero lots after sizing"
    deviation = CONFIG["slippage_points"]

    if CONFIG["dry_run"]:
        print(f"[DRY] {direction.upper()} {symbol} {lots:.2f} @ {entry_price:.5f} SL {sl_price:.5f} TP {tp_price:.5f}")
        return True, "dry-run"

    if CONFIG["entry_mode"] == "limit":
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
        price = entry_price
    else:
        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        price = entry_price

    request = {
        "action": mt5.TRADE_ACTION_DEAL if order_type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL) else mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": deviation,
        "type_filling": symbol_info.trade_fill_mode,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "fib_snr_scalper",
    }

    for attempt in range(1 + CONFIG["retry_on_requote"]):
        result = mt5.order_send(request)
        if result is None:
            return False, f"order_send failed: {mt5.last_error()}"
        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            print(f"[EXEC] {direction.upper()} {symbol} lots={lots:.2f} sl={sl_price:.5f} tp={tp_price:.5f} ret={result.retcode}")
            return True, f"retcode={result.retcode}"
        elif result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED):
            print(f"[RETRY] Requote/price changed (attempt {attempt+1}) ret={result.retcode}")
            time.sleep(0.2)
            continue
        else:
            return False, f"order_send ret={result.retcode} comment={result.comment}"
    return False, "requote loop exhausted"

# ========== MAIN SIGNAL PIPELINE ==========
def run():
    initialize_mt5()
    sym = CONFIG["symbol"]
    info = select_symbol(sym)
    acc = mt5.account_info()
    state = load_daily_state(CONFIG["daily_state_file"])

    realized_today = realized_pnl_today_from_history()
    daily_pnl, start_equity = daily_pnl_and_start_equity(state, acc.equity)
    update_daily_realized_pnl(state, realized_today - daily_pnl)
    save_daily_state(CONFIG["daily_state_file"], state)

    tol_price = pips_to_price(info, CONFIG["confluence_tolerance_pips"])

    print(f"[START] {sym} timeframe=M1 risk={CONFIG['risk_per_trade_pct']}% daily_cap={CONFIG['daily_loss_cap_pct']}% max_lev={CONFIG['max_effective_leverage']} max_pos={CONFIG['max_open_positions']}")

    while True:
        try:
            acc = mt5.account_info()
            if acc is None:
                raise RuntimeError("Lost MT5 account connection")

            # Daily cap check
            realized_today = realized_pnl_today_from_history()
            start_equity = load_daily_state(CONFIG["daily_state_file"]).get(today_key(), {}).get("start_equity", acc.equity)
            cap_currency = start_equity * (CONFIG["daily_loss_cap_pct"] / 100.0)
            if realized_today <= -cap_currency:
                print(f"[HALT] Daily loss cap reached: {realized_today:.2f} <= -{cap_currency:.2f}. No new trades today.")
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Max open positions
            if count_open_positions() >= CONFIG["max_open_positions"]:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Market data
            rates = fetch_rates(sym, CONFIG["timeframe"], CONFIG["lookback_bars"])
            if len(rates) < max(CONFIG["snr_lookback"], CONFIG["atr_period"]) + 10:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            highs = [r['high'] for r in rates]
            lows  = [r['low']  for r in rates]
            closes= [r['close']for r in rates]

            # ATR and spread filters
            cur_atr = atr(highs, lows, closes, CONFIG["atr_period"])
            if cur_atr is None:
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            atr_pips = cur_atr / info.point / points_per_pip(info)
            if atr_pips < CONFIG["min_atr_pips"]:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            tick = get_tick(sym)
            spr = spread_in_pips(info, tick)
            if spr > CONFIG["max_spread_pips"]:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Build last impulse and fibs
            leg = last_impulse_from_swings(highs, lows, CONFIG["swing_left_right"], CONFIG["min_impulse_pips"], info)
            if not leg:
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            fibs = fib_levels_from_leg(leg)

            # Build S/R map
            snr = snr_levels_from_pivots(highs, lows, CONFIG["snr_pivot_width"], CONFIG["snr_lookback"])

            # Determine price reference for entry check
            bid, ask, last = tick.bid, tick.ask, tick.last
            mid = (bid + ask) / 2.0

            # Confluence near current tradable side
            conf = nearest_confluence(mid, fibs, snr, tol_price)
            if not conf:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Directional logic: only trade pullbacks toward the impulse base
            direction = None
            entry_price = None
            if leg["dir"] == "up":
                if mid <= conf["level"] + tol_price:
                    direction = "buy"
                    entry_price = ask if CONFIG["entry_mode"] == "market" else conf["level"] - pips_to_price(info, CONFIG["limit_improve_pips"])
                    sl_price = leg["low"] - pips_to_price(info, CONFIG["stop_buffer_pips"])
                    stop_pips = (entry_price - sl_price) / info.point / points_per_pip(info)
                    tp_price = entry_price + pips_to_price(info, stop_pips * CONFIG["rr_ratio"])
            else:
                if mid >= conf["level"] - tol_price:
                    direction = "sell"
                    entry_price = bid if CONFIG["entry_mode"] == "market" else conf["level"] + pips_to_price(info, CONFIG["limit_improve_pips"])
                    sl_price = leg["high"] + pips_to_price(info, CONFIG["stop_buffer_pips"])
                    stop_pips = (sl_price - entry_price) / info.point / points_per_pip(info)
                    tp_price = entry_price - pips_to_price(info, stop_pips * CONFIG["rr_ratio"])

            if direction is None or stop_pips <= 0:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # FIFO/no-hedging guard
            if fifo_block_hedge(sym, direction):
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Size position
            equity = acc.equity
            lots = calc_position_size_lots(info, equity, CONFIG["risk_per_trade_pct"], stop_pips, CONFIG["max_effective_leverage"], mid)
            if lots <= 0:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Final spread check again just before send
            tick = get_tick(sym)
            spr = spread_in_pips(info, tick)
            if spr > CONFIG["max_spread_pips"]:
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Send order
            ok, msg = send_order(info, direction, entry_price, sl_price, tp_price, lots)
            if not ok:
                print(f"[ORDER_FAIL] {msg}")

        except KeyboardInterrupt:
            print("[EXIT] KeyboardInterrupt")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1.0)

    mt5.shutdown()

if __name__ == "__main__":
    run()