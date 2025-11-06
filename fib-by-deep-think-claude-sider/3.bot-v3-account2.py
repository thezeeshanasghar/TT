
# mt5_fib_snr_scalper.py
# Live MT5 scalping bot: Fibonacci retracement + Support/Resistance confluence
# GOLD (XAUUSDm) optimized version with detailed diagnostics

import MetaTrader5 as mt5
import time
import math
import statistics
import json
import os
from datetime import datetime, timedelta, timezone

# ========== CONFIGURATION ==========
CONFIG = {
    # MT5 Account Credentials
    "mt5_login": 261525020,               # Your MT5 account number
    "mt5_password": "Ae!8bfb666",      # Your MT5 password
    "mt5_server": "Exness-MT5Trial16",   # Your broker's server name (e.g., "ICMarkets-Demo")
    "mt5_path": None,                    # Path to MT5 terminal (None = auto-detect)
    "mt5_timeout": 60000,                # Connection timeout in milliseconds
    
    # Trading
    "symbol": "XAUUSDm",  # Changed from XAUUSDm to XAUUSD for Exness
    "timeframe": mt5.TIMEFRAME_M1,       # M1 default
    "lookback_bars": 600,                # bars to pull each loop
    "poll_interval_sec": 0.5,            # loop sleep time

    # Fib/SNR mechanics
    "swing_left_right": 3,               # fractal swing pivot width (reduced to catch recent swings)
    "snr_pivot_width": 2,                # pivot width for S/R map
    "snr_lookback": 200,                 # bars to scan for S/R pivot levels
    "confluence_tolerance_pips": 15.0,   # GOLD: increased tolerance to find more confluence
    "min_impulse_pips": 20,              # GOLD: reduced to catch smaller recent moves
    "atr_period": 14,                    # for volatility sanity checks
    "min_atr_pips": 5.0,                 # GOLD: avoid ultra-dead markets

    # Entries/Exits
    "rr_ratio": 1.2,                     # take-profit = rr_ratio * risk
    "stop_buffer_pips": 5.0,             # GOLD: extra beyond swing
    "entry_mode": "market",              # "market" or "limit"
    "limit_improve_pips": 0.0,           # if using limit, improve price by this amount

    # Risk & compliance
    "risk_per_trade_pct": 1.0,           # 1% risk per trade
    "daily_loss_cap_pct": 3.0,           # stop new trades after -3% realized
    "max_effective_leverage": 5.0,       # cap notional/equity
    "max_open_positions": 3,             # across all symbols
    "max_spread_pips": 50.0,             # GOLD: ignore spread (increased from 3.0)
    "nfa_fifo_no_hedging": True,         # U.S.-style: no hedging
    "account_is_hedging": True,          # set True if account supports hedging

    # Execution
    "slippage_points": 100,              # GOLD: increased deviation (points)
    "retry_on_requote": 2,
    "dry_run": False,                    # if True, log actions but do not send orders

    # Daily PnL tracking
    "daily_state_file": "daily_state.json",
    
    # Diagnostics
    "verbose_diagnostics": True,         # Enable detailed filter diagnostics
    "diagnostic_interval_sec": 30,       # Print diagnostic summary every N seconds
}

# ========== UTILS ==========

def initialize_mt5():
    """Initialize MT5 connection with specific account credentials"""
    # Step 1: Initialize MT5 terminal
    if CONFIG["mt5_path"] is not None:
        if not mt5.initialize(path=CONFIG["mt5_path"]):
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {error}. Check the MT5 path.")
    else:
        if not mt5.initialize():
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialize() failed: {error}. Ensure MT5 terminal is installed.")
    
    # Step 2: Login to specific account
    authorized = mt5.login(
        login=CONFIG["mt5_login"],
        password=CONFIG["mt5_password"],
        server=CONFIG["mt5_server"],
        timeout=CONFIG["mt5_timeout"]
    )
    
    if not authorized:
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed: {error}. Check your login ({CONFIG['mt5_login']}), password, and server ({CONFIG['mt5_server']}).")
    
    acc = mt5.account_info()
    if acc is None:
        mt5.shutdown()
        raise RuntimeError("No MT5 account info after login. Connection failed.")
    
    print(f"[INIT] Connected to MT5")
    print(f"  Server: {CONFIG['mt5_server']}")
    print(f"  Account: {acc.login}")
    print(f"  Equity: ${acc.equity:.2f}")
    print(f"  Balance: ${acc.balance:.2f}")
    print(f"  Leverage: 1:{acc.leverage}")

def find_gold_symbols():
    """Find all available Gold/XAU symbols in the account"""
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return []
    
    gold_symbols = []
    for s in all_symbols:
        name_upper = s.name.upper()
        if "XAU" in name_upper or "GOLD" in name_upper:
            gold_symbols.append(s.name)
    
    return gold_symbols

def select_symbol(symbol: str):
    if not mt5.symbol_select(symbol, True):
        # Try to find available Gold symbols
        gold_symbols = find_gold_symbols()
        error_msg = f"Could not select symbol '{symbol}'."
        if gold_symbols:
            error_msg += f"\n  Available Gold symbols in your account: {', '.join(gold_symbols)}"
            error_msg += f"\n  Update CONFIG['symbol'] to one of these symbols."
        else:
            error_msg += "\n  No Gold/XAU symbols found. Check symbol name in MT5."
        raise RuntimeError(error_msg)
    
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
    """
    For GOLD (XAUUSDm): 
    - Typically 2-3 digits (e.g., 2654.32)
    - 1 pip = $0.10 = 10 points
    """
    digits = symbol_info.digits
    symbol_name = symbol_info.name.upper()
    
    # Gold-specific handling
    if "XAU" in symbol_name or "GOLD" in symbol_name:
        if digits == 2:
            return 10  # 1 pip = 10 points (0.10)
        elif digits == 3:
            return 10  # 1 pip = 10 points (0.010)
    
    # Forex standard
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

def price_to_pips(symbol_info, price_diff):
    """Convert price difference to pips"""
    points = price_to_points(symbol_info, abs(price_diff))
    return points / points_per_pip(symbol_info)

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
        leg_pips = price_to_pips(symbol_info, last[1] - prev[1])
        if leg_pips >= min_pips:
            return {"dir": "up", "low_idx": prev[0], "low": prev[1], "high_idx": last[0], "high": last[1], "pips": leg_pips}
    if last[2] == "L" and prev[2] == "H":
        leg_pips = price_to_pips(symbol_info, prev[1] - last[1])
        if leg_pips >= min_pips:
            return {"dir": "down", "high_idx": prev[0], "high": prev[1], "low_idx": last[0], "low": last[1], "pips": leg_pips}
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
    if not levels:
        return []
    levels = sorted(levels)
    clustered = []
    cluster_tol = (max(highs[-lookback:]) - min(lows[-lookback:])) * 0.002  # Increased clustering tolerance
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
                    score = priority - abs(price - fprice) / (tol_price + 1e-9)
                    if best is None or score > best["score"]:
                        best = {"name": name, "level": fprice, "sr": s, "score": score}
    return best

# ========== RISK & SIZING ==========
def calc_pip_value_per_lot(symbol_info):
    """
    For Gold: pip value is typically $1.00 per 0.10 move on 0.01 lot
    """
    # Use trade_tick_value and trade_tick_size from symbol
    if symbol_info.trade_tick_value > 0 and symbol_info.trade_tick_size > 0:
        pips_per_tick = symbol_info.trade_tick_size * points_per_pip(symbol_info) / symbol_info.point
        pip_value = symbol_info.trade_tick_value / max(pips_per_tick, 1e-9)
        return pip_value
    
    # Fallback: estimate
    return 1.0  # For Gold, typically $1 per pip per 0.01 lot

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
        print(f"[DRY] {direction.upper()} {symbol} {lots:.2f} @ {entry_price:.2f} SL {sl_price:.2f} TP {tp_price:.2f}")
        return True, "dry-run"

    if CONFIG["entry_mode"] == "limit":
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
        price = entry_price
    else:
        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        price = entry_price

    filling_type = symbol_info.filling_mode
    # Choose appropriate filling mode
    if filling_type & 1:  # FOK
        fill_mode = mt5.ORDER_FILLING_FOK
    elif filling_type & 2:  # IOC
        fill_mode = mt5.ORDER_FILLING_IOC
    else:
        fill_mode = mt5.ORDER_FILLING_RETURN

    request = {
        "action": mt5.TRADE_ACTION_DEAL if order_type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL) else mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": deviation,
        "type_filling": fill_mode,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "fib_snr_scalper",
        "magic": 234567,
    }

    for attempt in range(1 + CONFIG["retry_on_requote"]):
        result = mt5.order_send(request)
        if result is None:
            return False, f"order_send failed: {mt5.last_error()}"
        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            print(f"[EXEC] {direction.upper()} {symbol} lots={lots:.2f} sl={sl_price:.2f} tp={tp_price:.2f} ret={result.retcode}")
            return True, f"retcode={result.retcode}"
        elif result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED):
            print(f"[RETRY] Requote/price changed (attempt {attempt+1}) ret={result.retcode}")
            time.sleep(0.2)
            continue
        else:
            return False, f"order_send ret={result.retcode} comment={result.comment}"
    return False, "requote loop exhausted"

# ========== DIAGNOSTICS ==========
class DiagnosticStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loops = 0
        self.atr_fails = 0
        self.spread_fails = 0
        self.no_impulse = 0
        self.no_snr = 0
        self.no_confluence = 0
        self.wrong_direction = 0
        self.fifo_blocked = 0
        self.zero_lots = 0
        self.signals_generated = 0
        self.last_atr_pips = None
        self.last_spread_pips = None
        self.last_leg_pips = None
        
    def print_summary(self):
        print("\n" + "="*60)
        print(f"[DIAGNOSTIC] Summary (last {self.loops} loops)")
        print("="*60)
        
        # Format optional values properly
        atr_str = f"{self.last_atr_pips:.1f}" if self.last_atr_pips is not None else "N/A"
        spread_str = f"{self.last_spread_pips:.2f}" if self.last_spread_pips is not None else "N/A"
        leg_str = f"{self.last_leg_pips:.1f}" if self.last_leg_pips is not None else "N/A"
        
        print(f"ATR too low:          {self.atr_fails:4d} | Last ATR: {atr_str} pips (min: {CONFIG['min_atr_pips']})")
        print(f"Spread too wide:      {self.spread_fails:4d} | Last spread: {spread_str} pips (max: {CONFIG['max_spread_pips']})")
        print(f"No valid impulse:     {self.no_impulse:4d} | Last leg: {leg_str} pips (min: {CONFIG['min_impulse_pips']})")
        print(f"No S/R levels found:  {self.no_snr:4d}")
        print(f"No confluence:        {self.no_confluence:4d} | Tolerance: {CONFIG['confluence_tolerance_pips']} pips")
        print(f"Wrong direction:      {self.wrong_direction:4d}")
        print(f"FIFO blocked:         {self.fifo_blocked:4d}")
        print(f"Zero lot size:        {self.zero_lots:4d}")
        print(f"Valid signals:        {self.signals_generated:4d}")
        print("="*60 + "\n")

# ========== MAIN SIGNAL PIPELINE ==========
def run():
    initialize_mt5()
    sym = CONFIG["symbol"]
    info = select_symbol(sym)
    
    # Print symbol info for debugging
    print(f"\n[SYMBOL INFO] {sym}")
    print(f"  Digits: {info.digits}")
    print(f"  Point: {info.point}")
    print(f"  Points per pip: {points_per_pip(info)}")
    print(f"  Tick size: {info.trade_tick_size}")
    print(f"  Tick value: {info.trade_tick_value}")
    print(f"  Contract size: {info.trade_contract_size}")
    print(f"  Volume step: {info.volume_step}, Min: {info.volume_min}, Max: {info.volume_max}")
    print(f"  Pip value per lot: ${calc_pip_value_per_lot(info):.2f}\n")
    
    acc = mt5.account_info()
    state = load_daily_state(CONFIG["daily_state_file"])

    realized_today = realized_pnl_today_from_history()
    daily_pnl, start_equity = daily_pnl_and_start_equity(state, acc.equity)
    update_daily_realized_pnl(state, realized_today - daily_pnl)
    save_daily_state(CONFIG["daily_state_file"], state)

    tol_price = pips_to_price(info, CONFIG["confluence_tolerance_pips"])

    print(f"[START] {sym} timeframe=M1 risk={CONFIG['risk_per_trade_pct']}% daily_cap={CONFIG['daily_loss_cap_pct']}% max_lev={CONFIG['max_effective_leverage']} max_pos={CONFIG['max_open_positions']}")
    print(f"[CONFIG] Confluence tolerance: {CONFIG['confluence_tolerance_pips']} pips = {tol_price:.2f} price units\n")

    loop_count = 0
    last_log_time = time.time()
    last_diagnostic_time = time.time()
    stats = DiagnosticStats()

    while True:
        loop_count += 1
        stats.loops += 1
        current_time = time.time()
        
        # Log heartbeat every 30 seconds
        if current_time - last_log_time >= 30:
            print(f"[ALIVE] Loop #{loop_count} at {now().strftime('%H:%M:%S')} - Scanning market...")
            last_log_time = current_time
        
        # Print diagnostic summary
        if CONFIG["verbose_diagnostics"] and current_time - last_diagnostic_time >= CONFIG["diagnostic_interval_sec"]:
            stats.print_summary()
            stats.reset()
            last_diagnostic_time = current_time
        
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

            # ATR filter
            cur_atr = atr(highs, lows, closes, CONFIG["atr_period"])
            if cur_atr is None:
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            
            atr_pips = price_to_pips(info, cur_atr)
            stats.last_atr_pips = atr_pips
            
            if atr_pips < CONFIG["min_atr_pips"]:
                stats.atr_fails += 1
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Spread filter
            tick = get_tick(sym)
            spr = spread_in_pips(info, tick)
            stats.last_spread_pips = spr
            
            if spr > CONFIG["max_spread_pips"]:
                stats.spread_fails += 1
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Build last impulse and fibs
            leg = last_impulse_from_swings(highs, lows, CONFIG["swing_left_right"], CONFIG["min_impulse_pips"], info)
            if not leg:
                # Try to get best available leg for diagnostics
                piv = pivots(highs, lows, CONFIG["swing_left_right"])
                if len(piv) >= 2:
                    piv.sort(key=lambda x: x[0])
                    last = piv[-1]
                    prev = piv[-2]
                    if last[2] == "H" and prev[2] == "L":
                        stats.last_leg_pips = price_to_pips(info, last[1] - prev[1])
                    elif last[2] == "L" and prev[2] == "H":
                        stats.last_leg_pips = price_to_pips(info, prev[1] - last[1])
                stats.no_impulse += 1
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            
            stats.last_leg_pips = leg["pips"]
            if CONFIG["verbose_diagnostics"]:
                print(f"[LEG] {leg['dir'].upper()} impulse: {leg['pips']:.1f} pips | Low: {leg['low']:.2f} High: {leg['high']:.2f}")
            
            fibs = fib_levels_from_leg(leg)
            
            # Log Fib levels occasionally
            if CONFIG["verbose_diagnostics"] and loop_count % 60 == 1:
                print(f"[FIB LEVELS] 23.6%: {fibs['23.6']:.2f} | 38.2%: {fibs['38.2']:.2f} | 50.0%: {fibs['50.0']:.2f} | 61.8%: {fibs['61.8']:.2f} | 78.6%: {fibs['78.6']:.2f}")

            # Build S/R map
            snr = snr_levels_from_pivots(highs, lows, CONFIG["snr_pivot_width"], CONFIG["snr_lookback"])
            if not snr:
                stats.no_snr += 1
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            
            if CONFIG["verbose_diagnostics"]:
                print(f"[SNR] Found {len(snr)} S/R levels")

            # Determine price reference for entry check
            bid, ask, last = tick.bid, tick.ask, tick.last
            mid = (bid + ask) / 2.0

            # Confluence near current tradable side
            conf = nearest_confluence(mid, fibs, snr, tol_price)
            if not conf:
                stats.no_confluence += 1
                # Log occasionally why no confluence found
                if loop_count % 60 == 1:
                    print(f"[NO CONF] Current price: {mid:.2f} | Tolerance: {CONFIG['confluence_tolerance_pips']} pips ({tol_price:.2f})")
                    print(f"          Nearest Fib 61.8%: {fibs['61.8']:.2f} (dist: {abs(mid - fibs['61.8']):.2f})")
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            
            if CONFIG["verbose_diagnostics"]:
                print(f"[CONF] {conf['name']} @ {conf['level']:.2f} near S/R {conf['sr']:.2f} | Current: {mid:.2f}")

            # Directional logic: only trade pullbacks toward the impulse base
            direction = None
            entry_price = None
            stop_pips = 0
            
            if leg["dir"] == "up":
                if mid <= conf["level"] + tol_price:
                    direction = "buy"
                    entry_price = ask if CONFIG["entry_mode"] == "market" else conf["level"] - pips_to_price(info, CONFIG["limit_improve_pips"])
                    sl_price = leg["low"] - pips_to_price(info, CONFIG["stop_buffer_pips"])
                    stop_pips = price_to_pips(info, entry_price - sl_price)
                    tp_price = entry_price + pips_to_price(info, stop_pips * CONFIG["rr_ratio"])
                else:
                    stats.wrong_direction += 1
            else:
                if mid >= conf["level"] - tol_price:
                    direction = "sell"
                    entry_price = bid if CONFIG["entry_mode"] == "market" else conf["level"] + pips_to_price(info, CONFIG["limit_improve_pips"])
                    sl_price = leg["high"] + pips_to_price(info, CONFIG["stop_buffer_pips"])
                    stop_pips = price_to_pips(info, sl_price - entry_price)
                    tp_price = entry_price - pips_to_price(info, stop_pips * CONFIG["rr_ratio"])
                else:
                    stats.wrong_direction += 1

            if direction is None or stop_pips <= 0:
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            
            stats.signals_generated += 1
            print(f"\n[SIGNAL] {direction.upper()} @ {entry_price:.2f} | SL: {sl_price:.2f} ({stop_pips:.1f}p) | TP: {tp_price:.2f}")

            # FIFO/no-hedging guard
            if fifo_block_hedge(sym, direction):
                stats.fifo_blocked += 1
                print(f"[BLOCKED] FIFO/no-hedging rule prevents {direction.upper()} trade")
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Size position
            equity = acc.equity
            lots = calc_position_size_lots(info, equity, CONFIG["risk_per_trade_pct"], stop_pips, CONFIG["max_effective_leverage"], mid)
            if lots <= 0:
                stats.zero_lots += 1
                print(f"[SIZE FAIL] Zero lots: equity={equity:.2f}, stop={stop_pips:.1f}p, pip_value=${calc_pip_value_per_lot(info):.2f}")
                time.sleep(CONFIG["poll_interval_sec"])
                continue
            
            print(f"[SIZE] Lots: {lots:.2f} | Risk: ${equity * CONFIG['risk_per_trade_pct'] / 100:.2f} | Equity: ${equity:.2f}")

            # Final spread check again just before send
            tick = get_tick(sym)
            spr = spread_in_pips(info, tick)
            if spr > CONFIG["max_spread_pips"]:
                print(f"[SPREAD] Spread widened to {spr:.2f} pips before execution")
                time.sleep(CONFIG["poll_interval_sec"])
                continue

            # Send order
            print(f"[TRADE] Executing {direction.upper()} order...")
            ok, msg = send_order(info, direction, entry_price, sl_price, tp_price, lots)
            if not ok:
                print(f"[ORDER_FAIL] {msg}")
            else:
                print(f"[SUCCESS] Order placed: {msg}\n")

        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)
        
        time.sleep(CONFIG["poll_interval_sec"])

    mt5.shutdown()

if __name__ == "__main__":
    run()
