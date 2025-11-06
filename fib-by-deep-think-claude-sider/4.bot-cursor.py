# ENHANCED FIBONACCI TRADING BOT - Based on Professional Trading Principles
# Gold (XAU/USD) - Multi-Timeframe Fibonacci + Volume + MA Confluence Strategy
# 
# Key Improvements:
# 1. Multi-timeframe analysis (H1 for trend, M15 for entries)
# 2. Golden Zone focus (50-61.8% Fibonacci levels)
# 3. Volume confirmation
# 4. Candlestick pattern recognition (engulfing, long wicks)
# 5. Moving Average confluence (EMA 200)
# 6. Prevents duplicate/similar entries
# 7. Better risk/reward (minimum 2:1)
# 8. Trade quality over quantity

import MetaTrader5 as mt5
import time
import math
import statistics
import json
import os
from datetime import datetime, timedelta
from collections import deque

# ========== CONFIGURATION ==========
CONFIG = {
    # Trading Symbol
    "symbol": "XAUUSDm",
    
    # Multi-Timeframe Strategy
    "trend_timeframe": mt5.TIMEFRAME_H1,    # H1 for major trend and Fib levels
    "entry_timeframe": mt5.TIMEFRAME_M15,   # M15 for entry signals
    "confirm_timeframe": mt5.TIMEFRAME_M5,  # M5 for final confirmation
    
    # Lookback Periods
    "trend_lookback": 200,      # Bars for H1 trend analysis
    "entry_lookback": 100,      # Bars for M15 entry scan
    "confirm_lookback": 50,     # Bars for M5 confirmation
    
    # Fibonacci Strategy (Focus on Golden Zone)
    "swing_detection_width": 5,      # Wider pivot detection for H1
    "min_impulse_pips": 80,          # Minimum 80 pips for valid H1 impulse
    "fib_golden_zone_min": 0.50,    # 50% level
    "fib_golden_zone_max": 0.618,   # 61.8% level
    "fib_entry_tolerance_pips": 20,  # Must be near golden zone
    "pivot_detection_method": "zigzag",  # zigzag or static
    "zigzag_threshold_multiplier": 3.0,   # Multiplier for ATR-derived deviation
    "zigzag_atr_period": 10,              # ATR period for zigzag threshold
    "zigzag_depth": 10,                   # Minimum bars between pivots
    
    # Moving Average Confluence
    "use_ema_filter": True,
    "ema_period": 200,              # EMA 200 for trend + confluence
    "ema_confluence_pips": 30,      # EMA must be near Fib level
    
    # Volume Confirmation
    "use_volume_filter": True,
    "volume_ma_period": 20,
    "volume_surge_multiplier": 1.3,  # Volume must be 1.3x average
    
    # Candlestick Patterns
    "use_candle_patterns": True,
    "engulfing_min_body_ratio": 0.6,  # Body must be 60% of range
    "wick_rejection_ratio": 2.0,      # Wick must be 2x body for rejection
    
    # Entry Quality Filters
    "max_spread_pips": 18.0,           # Strict spread control
    "min_atr_pips": 15.0,             # Minimum volatility (H1 ATR)
    "atr_period": 14,
    
    # Risk Management
    "risk_per_trade_pct": 1.0,        # 1% risk per trade
    "min_rr_ratio": 2.0,              # Minimum 2:1 reward/risk
    "max_rr_ratio": 5.0,              # Cap at 5:1
    "stop_buffer_pips": 10.0,         # Buffer beyond swing
    "trailing_stop_activation_rr": 1.5,  # Activate trailing at 1.5:1
    
    # Position Management
    "max_open_positions": 1,          # ONE trade at a time (quality over quantity)
    "max_daily_trades": 3,            # Max 3 trades per day
    "duplicate_entry_prevention_pips": 50,  # Don't enter if recent trade within 50 pips
    "min_time_between_trades_min": 30,      # Minimum 30 minutes between trades
    "use_partial_tp": False,          # Enable partial take profit scaling
    "partial_tp_percent": 50,         # Percentage of position to close on partial TP
    "partial_tp_rr": 2.0,             # R:R ratio to trigger partial TP
    "use_trailing_stop": False,       # Enable trailing stop management
    "trailing_stop_distance_pips": 30,  # Trailing stop distance in pips
    
    # Daily Loss Protection
    "enforce_daily_loss_cap": False,  # Toggle daily loss protection
    "daily_loss_cap_pct": 3.0,        # Stop after -3% daily loss
    "daily_state_file": "daily_state_fib.json",
    
    # Execution
    "entry_mode": "market",           # Market execution
    "slippage_points": 50,
    "retry_on_requote": 2,
    "dry_run": False,                 # Set to True for testing
    
    # Logging
    "verbose": True,
    "log_file": "fib_bot_trades.log",
    "poll_interval_sec": 5.0,         # Check every 5 seconds (less aggressive)
}

# ========== UTILITY FUNCTIONS ==========

def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("No MT5 account info. Check MT5 terminal.")
    
    log(f"✓ Connected to MT5 | Account: {acc.login} | Equity: ${acc.equity:.2f} | Balance: ${acc.balance:.2f}")
    return acc

def select_symbol(symbol):
    """Select and validate symbol"""
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol}")
    
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        raise RuntimeError(f"Symbol {symbol} not available")
    
    return info

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    console_msg = msg.encode("ascii", errors="ignore").decode("ascii")
    print(console_msg)
    
    # Also log to file
    try:
        with open(CONFIG["log_file"], "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except:
        pass

def points_per_pip(symbol_info):
    """Calculate points per pip for Gold"""
    digits = symbol_info.digits
    if "XAU" in symbol_info.name.upper() or "GOLD" in symbol_info.name.upper():
        return 10  # Gold: 1 pip = 10 points = $0.10
    return 10 if digits in (5, 3) else 1

def pips_to_price(symbol_info, pips):
    """Convert pips to price difference"""
    return pips * points_per_pip(symbol_info) * symbol_info.point

def price_to_pips(symbol_info, price_diff):
    """Convert price difference to pips"""
    points = abs(price_diff) / symbol_info.point
    return points / points_per_pip(symbol_info)

def fetch_rates(symbol, timeframe, count):
    """Fetch historical rates"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates for {symbol} {timeframe}")
    return list(rates)

def get_tick(symbol):
    """Get current tick data"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("Failed to get tick")
    return tick

# ========== TECHNICAL INDICATORS ==========

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range"""
    trs = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        trs.append(max(hl, hc, lc))
    
    if len(trs) < period:
        return None
    return statistics.fmean(trs[-period:])

def calculate_ema(closes, period):
    """Calculate Exponential Moving Average"""
    if len(closes) < period:
        return None
    
    multiplier = 2.0 / (period + 1)
    ema = statistics.fmean(closes[:period])  # Start with SMA
    
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def detect_pivots(highs, lows, width):
    """Detect swing high/low pivots"""
    pivots = []
    n = len(highs)
    
    for i in range(width, n - width):
        # Check for swing high
        window_h = highs[i - width:i + width + 1]
        if highs[i] == max(window_h) and window_h.count(highs[i]) == 1:
            pivots.append((i, highs[i], "H"))
        
        # Check for swing low
        window_l = lows[i - width:i + width + 1]
        if lows[i] == min(window_l) and window_l.count(lows[i]) == 1:
            pivots.append((i, lows[i], "L"))
    
    return pivots

def detect_zigzag_pivots(rates, symbol_info):
    """Detect pivots using ATR-based zigzag logic"""
    n = len(rates)
    if n < 3:
        return []

    highs = [r['high'] for r in rates]
    lows = [r['low'] for r in rates]
    closes = [r['close'] for r in rates]

    atr_period = CONFIG.get("zigzag_atr_period", 10)
    atr_value = calculate_atr(highs, lows, closes, atr_period)

    if atr_value is None or atr_value <= 0:
        return []

    threshold_multiplier = CONFIG.get("zigzag_threshold_multiplier", 3.0)
    threshold = atr_value * threshold_multiplier
    threshold = max(threshold, symbol_info.point)

    depth = max(2, int(CONFIG.get("zigzag_depth", 10)))

    pivots = []
    direction = None  # None, "up", "down"

    candidate_high_idx = 0
    candidate_high = highs[0]
    candidate_low_idx = 0
    candidate_low = lows[0]

    for i in range(1, n):
        high = highs[i]
        low = lows[i]

        if direction in (None, "up"):
            if high >= candidate_high or candidate_high_idx < (pivots[-1][0] if pivots else -1):
                candidate_high = high
                candidate_high_idx = i

        if direction in (None, "down"):
            if low <= candidate_low or candidate_low_idx < (pivots[-1][0] if pivots else -1):
                candidate_low = low
                candidate_low_idx = i

        if direction in (None, "up"):
            if candidate_high - low >= threshold and i - candidate_high_idx >= depth:
                if not pivots or pivots[-1][2] != "H":
                    pivots.append((candidate_high_idx, candidate_high, "H"))
                direction = "down"
                candidate_low_idx = i
                candidate_low = low
                continue

        if direction in (None, "down"):
            if high - candidate_low >= threshold and i - candidate_low_idx >= depth:
                if not pivots or pivots[-1][2] != "L":
                    pivots.append((candidate_low_idx, candidate_low, "L"))
                direction = "up"
                candidate_high_idx = i
                candidate_high = high
                continue

    pivots.sort(key=lambda x: x[0])
    return pivots

def find_last_impulse(rates, min_pips, symbol_info):
    """Find the last significant impulse move using configured pivot detection"""
    method = CONFIG.get("pivot_detection_method", "zigzag").lower()

    if method == "zigzag":
        pivots = detect_zigzag_pivots(rates, symbol_info)
    else:
        width = CONFIG.get("swing_detection_width", 5)
        highs = [r['high'] for r in rates]
        lows = [r['low'] for r in rates]
        pivots = detect_pivots(highs, lows, width)

    if len(pivots) < 2:
        return None

    pivots.sort(key=lambda x: x[0])

    last = pivots[-1]
    prev = pivots[-2]

    if last[2] == "H" and prev[2] == "L":
        pips = price_to_pips(symbol_info, last[1] - prev[1])
        if pips >= min_pips:
            return {
                "direction": "up",
                "start": prev[1],
                "end": last[1],
                "start_idx": prev[0],
                "end_idx": last[0],
                "pips": pips
            }

    if last[2] == "L" and prev[2] == "H":
        pips = price_to_pips(symbol_info, prev[1] - last[1])
        if pips >= min_pips:
            return {
                "direction": "down",
                "start": prev[1],
                "end": last[1],
                "start_idx": prev[0],
                "end_idx": last[0],
                "pips": pips
            }

    return None

def calculate_fibonacci_levels(impulse):
    """Calculate Fibonacci retracement levels"""
    if impulse["direction"] == "up":
        # Uptrend: retracement from high
        high = impulse["end"]
        low = impulse["start"]
        diff = high - low
        
        return {
            "0.0": high,
            "23.6": high - 0.236 * diff,
            "38.2": high - 0.382 * diff,
            "50.0": high - 0.500 * diff,
            "61.8": high - 0.618 * diff,
            "78.6": high - 0.786 * diff,
            "100.0": low,
            "direction": "up"
        }
    else:
        # Downtrend: retracement from low
        high = impulse["start"]
        low = impulse["end"]
        diff = high - low
        
        return {
            "0.0": low,
            "23.6": low + 0.236 * diff,
            "38.2": low + 0.382 * diff,
            "50.0": low + 0.500 * diff,
            "61.8": low + 0.618 * diff,
            "78.6": low + 0.786 * diff,
            "100.0": high,
            "direction": "down"
        }

# ========== PATTERN RECOGNITION ==========

def is_bullish_engulfing(rates, idx):
    """Detect bullish engulfing pattern"""
    if idx < 1:
        return False
    
    current = rates[idx]
    previous = rates[idx - 1]
    
    # Current candle must be bullish
    if current['close'] <= current['open']:
        return False
    
    # Previous candle must be bearish
    if previous['close'] >= previous['open']:
        return False
    
    # Current body must engulf previous body
    current_body = current['close'] - current['open']
    prev_body = previous['open'] - previous['close']
    
    # Check body ratio
    current_range = current['high'] - current['low']
    if current_range == 0:
        return False
    
    body_ratio = current_body / current_range
    
    return (current['close'] > previous['open'] and 
            current['open'] < previous['close'] and
            body_ratio >= CONFIG["engulfing_min_body_ratio"])

def is_bearish_engulfing(rates, idx):
    """Detect bearish engulfing pattern"""
    if idx < 1:
        return False
    
    current = rates[idx]
    previous = rates[idx - 1]
    
    # Current candle must be bearish
    if current['close'] >= current['open']:
        return False
    
    # Previous candle must be bullish
    if previous['close'] <= previous['open']:
        return False
    
    # Current body must engulf previous body
    current_body = current['open'] - current['close']
    prev_body = previous['close'] - previous['open']
    
    # Check body ratio
    current_range = current['high'] - current['low']
    if current_range == 0:
        return False
    
    body_ratio = current_body / current_range
    
    return (current['close'] < previous['open'] and 
            current['open'] > previous['close'] and
            body_ratio >= CONFIG["engulfing_min_body_ratio"])

def has_long_lower_wick(rate):
    """Detect long lower wick (bullish rejection)"""
    candle_range = rate['high'] - rate['low']
    if candle_range == 0:
        return False
    
    body_top = max(rate['open'], rate['close'])
    body_bottom = min(rate['open'], rate['close'])
    body_size = abs(rate['close'] - rate['open'])
    
    lower_wick = body_bottom - rate['low']
    
    return lower_wick >= CONFIG["wick_rejection_ratio"] * body_size

def has_long_upper_wick(rate):
    """Detect long upper wick (bearish rejection)"""
    candle_range = rate['high'] - rate['low']
    if candle_range == 0:
        return False
    
    body_top = max(rate['open'], rate['close'])
    body_bottom = min(rate['open'], rate['close'])
    body_size = abs(rate['close'] - rate['open'])
    
    upper_wick = rate['high'] - body_top
    
    return upper_wick >= CONFIG["wick_rejection_ratio"] * body_size

# ========== VOLUME ANALYSIS ==========

def check_volume_surge(rates, period=20):
    """Check if current volume is surging"""
    if len(rates) < period + 1:
        return False
    
    volumes = [r['tick_volume'] for r in rates[-period-1:-1]]
    avg_volume = statistics.fmean(volumes)
    current_volume = rates[-1]['tick_volume']
    
    return current_volume >= avg_volume * CONFIG["volume_surge_multiplier"]

# ========== TRADE LOGIC ==========

def is_price_in_golden_zone(price, fib_levels, symbol_info):
    """Check if price is in the golden zone (50-61.8%)"""
    tolerance = pips_to_price(symbol_info, CONFIG["fib_entry_tolerance_pips"])
    
    fib_50 = fib_levels["50.0"]
    fib_618 = fib_levels["61.8"]
    
    min_level = min(fib_50, fib_618)
    max_level = max(fib_50, fib_618)
    
    return (min_level - tolerance) <= price <= (max_level + tolerance)

def check_ema_confluence(price, fib_level, ema_value, symbol_info):
    """Check if EMA confirms Fibonacci level"""
    tolerance = pips_to_price(symbol_info, CONFIG["ema_confluence_pips"])
    
    # Check if EMA is near the Fib level
    ema_near_fib = abs(ema_value - fib_level) <= tolerance
    
    # Check if price is near EMA
    price_near_ema = abs(price - ema_value) <= tolerance
    
    return ema_near_fib or price_near_ema

def get_recent_trade_prices():
    """Get prices of recent trades to prevent duplicates"""
    state = load_daily_state()
    today = today_key()
    
    if today not in state:
        return []
    
    trades = state[today].get("trades", [])
    
    # Get trades from last hour
    cutoff_time = datetime.now() - timedelta(minutes=CONFIG["min_time_between_trades_min"])
    recent_prices = []
    
    for trade in trades:
        trade_time = datetime.fromisoformat(trade.get("time", "2000-01-01"))
        if trade_time >= cutoff_time:
            recent_prices.append(trade.get("entry_price", 0))
    
    return recent_prices

def is_duplicate_entry(price, symbol_info):
    """Check if this entry is too close to a recent trade"""
    recent_prices = get_recent_trade_prices()
    tolerance = pips_to_price(symbol_info, CONFIG["duplicate_entry_prevention_pips"])
    
    for recent_price in recent_prices:
        if abs(price - recent_price) <= tolerance:
            return True
    
    return False

# ========== POSITION SIZING ==========

def calculate_position_size(symbol_info, equity, stop_distance_pips):
    """Calculate position size based on risk"""
    if stop_distance_pips <= 0:
        return 0.0
    
    risk_amount = equity * (CONFIG["risk_per_trade_pct"] / 100.0)
    
    # Gold: typically $1 per pip per 0.01 lot
    # Adjust based on contract size
    pip_value = symbol_info.trade_contract_size * 0.01 * pips_to_price(symbol_info, 1.0)
    
    if pip_value <= 0:
        pip_value = 1.0  # Fallback
    
    lots = risk_amount / (stop_distance_pips * pip_value)
    
    # Round to volume step
    step = symbol_info.volume_step
    lots = math.floor(lots / step) * step
    
    # Clamp to min/max
    lots = max(symbol_info.volume_min, min(symbol_info.volume_max, lots))
    
    return lots

# ========== STATE MANAGEMENT ==========

def load_daily_state():
    """Load daily state from file"""
    if os.path.exists(CONFIG["daily_state_file"]):
        try:
            with open(CONFIG["daily_state_file"], "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_daily_state(state):
    """Save daily state to file"""
    with open(CONFIG["daily_state_file"], "w") as f:
        json.dump(state, f, indent=2)

def today_key():
    """Get today's date key"""
    return datetime.now().strftime("%Y-%m-%d")

def record_trade(entry_price, direction, stop_loss, take_profit):
    """Record trade in daily state"""
    state = load_daily_state()
    today = today_key()
    
    if today not in state:
        state[today] = {"trades": [], "realized_pnl": 0.0}
    
    state[today]["trades"].append({
        "time": datetime.now().isoformat(),
        "entry_price": entry_price,
        "direction": direction,
        "sl": stop_loss,
        "tp": take_profit
    })
    
    save_daily_state(state)

def get_daily_trade_count():
    """Get number of trades today"""
    state = load_daily_state()
    today = today_key()
    
    if today not in state:
        return 0
    
    return len(state[today].get("trades", []))

def get_daily_pnl():
    """Get today's realized PnL from MT5 history"""
    start = datetime.combine(datetime.now().date(), datetime.min.time())
    deals = mt5.history_deals_get(start, datetime.now())
    
    pnl = 0.0
    if deals is not None:
        for deal in deals:
            pnl += deal.profit
    
    return pnl

def check_daily_loss_limit(equity):
    """Check if daily loss limit is reached"""
    if not CONFIG.get("enforce_daily_loss_cap", True):
        return False

    pnl = get_daily_pnl()
    state = load_daily_state()
    today = today_key()
    
    if today not in state:
        state[today] = {"start_equity": equity}
        save_daily_state(state)
        return False
    
    start_equity = state[today].get("start_equity", equity)
    max_loss = start_equity * (CONFIG["daily_loss_cap_pct"] / 100.0)
    
    if pnl <= -max_loss:
        log(f"⛔ DAILY LOSS LIMIT REACHED: ${pnl:.2f} / -${max_loss:.2f}")
        return True
    
    return False

# ========== ORDER EXECUTION ==========

def send_market_order(symbol_info, direction, entry_price, sl_price, tp_price, lots):
    """Send market order to MT5"""
    symbol = symbol_info.name
    
    if CONFIG["dry_run"]:
        log(f"🧪 DRY RUN: {direction.upper()} {symbol} {lots:.2f} lots @ {entry_price:.3f} | SL: {sl_price:.3f} | TP: {tp_price:.3f}")
        return True, "dry-run"
    
    # Determine order type and price
    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
    
    # Determine filling mode
    filling = symbol_info.filling_mode
    if filling & mt5.SYMBOL_FILLING_FOK:
        fill_mode = mt5.ORDER_FILLING_FOK
    elif filling & mt5.SYMBOL_FILLING_IOC:
        fill_mode = mt5.ORDER_FILLING_IOC
    else:
        fill_mode = mt5.ORDER_FILLING_RETURN
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": CONFIG["slippage_points"],
        "magic": 888777,
        "comment": "FibGoldenZone",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_mode,
    }
    
    # Send order with retries
    for attempt in range(1 + CONFIG["retry_on_requote"]):
        result = mt5.order_send(request)
        
        if result is None:
            return False, f"order_send failed: {mt5.last_error()}"
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log(f"✅ ORDER FILLED: {direction.upper()} {lots:.2f} lots @ {result.price:.3f}")
            return True, f"Order #{result.order}"
        
        if result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED):
            log(f"⚠️  Requote (attempt {attempt + 1})")
            time.sleep(0.3)
            continue
        
        return False, f"Order failed: {result.retcode} - {result.comment}"
    
    return False, "Max retries exceeded"

def get_open_positions(symbol):
    """Get open positions for symbol"""
    positions = mt5.positions_get(symbol=symbol)
    return list(positions) if positions is not None else []

# ========== POSITION MANAGEMENT ==========

def modify_position_sl(position, new_sl):
    """Modify stop loss of an open position"""
    symbol = position.symbol
    
    if CONFIG["dry_run"]:
        log(f"🧪 DRY RUN: Modify SL for {symbol} position #{position.ticket} to {new_sl:.3f}")
        return True
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        log(f"❌ Failed to modify SL: {mt5.last_error()}")
        return False
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"✅ Trailing SL updated: {new_sl:.3f}")
        return True
    else:
        log(f"⚠️ SL modify failed: {result.retcode} - {result.comment}")
        return False

def close_partial_position(position, percent):
    """Close partial position (e.g., 50%)"""
    symbol = position.symbol
    current_volume = position.volume
    close_volume = current_volume * (percent / 100.0)
    
    # Get symbol info for volume step
    symbol_info = mt5.symbol_info(position.symbol)
    if symbol_info is None:
        return False
    
    # Round to volume step
    step = symbol_info.volume_step
    close_volume = math.floor(close_volume / step) * step
    close_volume = max(symbol_info.volume_min, close_volume)
    
    # Don't close if volume too small
    if close_volume <= 0 or close_volume >= current_volume:
        return False
    
    if CONFIG["dry_run"]:
        log(f"🧪 DRY RUN: Close {percent}% ({close_volume:.2f} lots) of position #{position.ticket}")
        return True
    
    # Determine order type (opposite of position)
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    
    # Get filling mode
    filling = symbol_info.filling_mode
    if filling & mt5.SYMBOL_FILLING_FOK:
        fill_mode = mt5.ORDER_FILLING_FOK
    elif filling & mt5.SYMBOL_FILLING_IOC:
        fill_mode = mt5.ORDER_FILLING_IOC
    else:
        fill_mode = mt5.ORDER_FILLING_RETURN
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": close_volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": CONFIG["slippage_points"],
        "magic": 888777,
        "comment": "PartialTP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_mode,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        log(f"❌ Partial close failed: {mt5.last_error()}")
        return False
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"✅ PARTIAL TP: Closed {percent}% ({close_volume:.2f} lots) @ {price:.3f}")
        return True
    else:
        log(f"⚠️ Partial close failed: {result.retcode} - {result.comment}")
        return False

def manage_open_positions(symbol_info):
    """Manage trailing stops and partial TPs for open positions"""
    positions = get_open_positions(symbol_info.name)
    
    if not positions:
        return
    
    tick = get_tick(symbol_info.name)
    
    for pos in positions:
        # Skip if not our magic number
        if pos.magic != 888777:
            continue
        
        entry_price = pos.price_open
        current_sl = pos.sl
        current_tp = pos.tp
        current_volume = pos.volume
        
        # Calculate initial risk
        if pos.type == mt5.POSITION_TYPE_BUY:
            is_buy = True
            current_price = tick.bid
            initial_risk_pips = price_to_pips(symbol_info, entry_price - current_sl)
        else:
            is_buy = False
            current_price = tick.ask
            initial_risk_pips = price_to_pips(symbol_info, current_sl - entry_price)
        
        if initial_risk_pips <= 0:
            continue
        
        # Calculate current profit in pips
        if is_buy:
            profit_pips = price_to_pips(symbol_info, current_price - entry_price)
        else:
            profit_pips = price_to_pips(symbol_info, entry_price - current_price)
        
        # Calculate current R:R ratio
        current_rr = profit_pips / initial_risk_pips if initial_risk_pips > 0 else 0
        
        # === PARTIAL TAKE PROFIT ===
        if CONFIG["use_partial_tp"] and current_rr >= CONFIG["partial_tp_rr"]:
            # Check if we already took partial (by checking volume)
            # If position is still full size, take partial
            state = load_daily_state()
            pos_key = f"pos_{pos.ticket}_partial_taken"
            
            if not state.get(pos_key, False):
                log(f"💰 Profit at {current_rr:.2f}:1 R:R - Taking {CONFIG['partial_tp_percent']}% partial TP")
                
                if close_partial_position(pos, CONFIG["partial_tp_percent"]):
                    # Mark that we took partial for this position
                    state[pos_key] = True
                    save_daily_state(state)
                    
                    # Move SL to breakeven
                    if not CONFIG["dry_run"]:
                        modify_position_sl(pos, entry_price)
                        log(f"🔒 Stop Loss moved to BREAKEVEN: {entry_price:.3f}")
        
        # === TRAILING STOP ===
        if CONFIG["use_trailing_stop"] and current_rr >= CONFIG["trailing_stop_activation_rr"]:
            trail_distance = pips_to_price(symbol_info, CONFIG["trailing_stop_distance_pips"])
            
            if is_buy:
                # For buy: trail below current price
                new_sl = current_price - trail_distance
                
                # Only update if new SL is higher than current SL
                if new_sl > current_sl:
                    # Also ensure we don't set SL above breakeven before partial TP
                    state = load_daily_state()
                    pos_key = f"pos_{pos.ticket}_partial_taken"
                    partial_taken = state.get(pos_key, False)
                    
                    # Before partial TP, don't trail beyond breakeven
                    if not partial_taken and new_sl > entry_price:
                        new_sl = entry_price
                    
                    if new_sl > current_sl + pips_to_price(symbol_info, 2.0):  # Only update if 2+ pips improvement
                        modify_position_sl(pos, new_sl)
            else:
                # For sell: trail above current price
                new_sl = current_price + trail_distance
                
                # Only update if new SL is lower than current SL
                if new_sl < current_sl:
                    # Before partial TP, don't trail beyond breakeven
                    state = load_daily_state()
                    pos_key = f"pos_{pos.ticket}_partial_taken"
                    partial_taken = state.get(pos_key, False)
                    
                    if not partial_taken and new_sl < entry_price:
                        new_sl = entry_price
                    
                    if new_sl < current_sl - pips_to_price(symbol_info, 2.0):  # Only update if 2+ pips improvement
                        modify_position_sl(pos, new_sl)

# ========== MAIN TRADING LOGIC ==========

def analyze_market(symbol_info):
    """Main market analysis and signal generation"""
    
    symbol = symbol_info.name
    
    # === STEP 1: Fetch Multi-Timeframe Data ===
    try:
        trend_rates = fetch_rates(symbol, CONFIG["trend_timeframe"], CONFIG["trend_lookback"])
        entry_rates = fetch_rates(symbol, CONFIG["entry_timeframe"], CONFIG["entry_lookback"])
        confirm_rates = fetch_rates(symbol, CONFIG["confirm_timeframe"], CONFIG["confirm_lookback"])
    except Exception as e:
        if CONFIG["verbose"]:
            log(f"❌ Failed to fetch rates: {e}")
        return None
    
    # === STEP 2: Check Market Conditions ===
    
    # ATR filter (from H1)
    h1_highs = [r['high'] for r in trend_rates]
    h1_lows = [r['low'] for r in trend_rates]
    h1_closes = [r['close'] for r in trend_rates]
    
    atr = calculate_atr(h1_highs, h1_lows, h1_closes, CONFIG["atr_period"])
    if atr is None:
        return None
    
    atr_pips = price_to_pips(symbol_info, atr)
    
    if atr_pips < CONFIG["min_atr_pips"]:
        if CONFIG["verbose"]:
            log(f"📊 ATR too low: {atr_pips:.1f} pips (min: {CONFIG['min_atr_pips']})")
        return None
    
    # Spread filter
    tick = get_tick(symbol)
    spread_pips = price_to_pips(symbol_info, tick.ask - tick.bid)
    
    if spread_pips > CONFIG["max_spread_pips"]:
        if CONFIG["verbose"]:
            log(f"📊 Spread too wide: {spread_pips:.2f} pips (max: {CONFIG['max_spread_pips']})")
        return None
    
    # === STEP 3: Find Major Impulse Move (H1) ===
    impulse = find_last_impulse(
        trend_rates,
        CONFIG["min_impulse_pips"],
        symbol_info
    )
    
    if impulse is None:
        if CONFIG["verbose"]:
            log(f"📊 No valid H1 impulse found (min: {CONFIG['min_impulse_pips']} pips)")
        return None
    
    # === STEP 4: Calculate Fibonacci Levels ===
    fib_levels = calculate_fibonacci_levels(impulse)
    
    if CONFIG["verbose"]:
        log(f"📈 {impulse['direction'].upper()} Impulse: {impulse['pips']:.1f} pips | From {impulse['start']:.2f} to {impulse['end']:.2f}")
        log(f"   Fib 50.0%: {fib_levels['50.0']:.2f} | Fib 61.8%: {fib_levels['61.8']:.2f}")
    
    # === STEP 5: Calculate EMA 200 (M15 for entry timeframe) ===
    ema_200 = None
    if CONFIG["use_ema_filter"]:
        m15_closes = [r['close'] for r in entry_rates]
        ema_200 = calculate_ema(m15_closes, CONFIG["ema_period"])
        
        if ema_200 and CONFIG["verbose"]:
            log(f"   EMA 200: {ema_200:.2f}")
    
    # === STEP 6: Check if Price is in Golden Zone ===
    current_price = (tick.bid + tick.ask) / 2.0
    
    if not is_price_in_golden_zone(current_price, fib_levels, symbol_info):
        if CONFIG["verbose"]:
            log(f"📊 Price {current_price:.2f} not in Golden Zone (50-61.8%)")
        return None
    
    log(f"🎯 PRICE IN GOLDEN ZONE: {current_price:.2f}")
    
    # === STEP 7: Determine Trade Direction ===
    direction = None
    
    if impulse["direction"] == "up":
        # Uptrend: looking for buy on pullback
        if current_price <= fib_levels["61.8"]:
            direction = "buy"
    else:
        # Downtrend: looking for sell on bounce
        if current_price >= fib_levels["61.8"]:
            direction = "sell"
    
    if direction is None:
        return None
    
    # === STEP 8: Check EMA Confluence ===
    if CONFIG["use_ema_filter"] and ema_200 is not None:
        fib_golden = (fib_levels["50.0"] + fib_levels["61.8"]) / 2.0
        
        if not check_ema_confluence(current_price, fib_golden, ema_200, symbol_info):
            if CONFIG["verbose"]:
                log(f"📊 No EMA confluence | EMA: {ema_200:.2f} vs Fib: {fib_golden:.2f}")
            return None
        
        log(f"✅ EMA 200 confirms Fibonacci level")
    
    # === STEP 9: Check Volume Surge (M15) ===
    if CONFIG["use_volume_filter"]:
        if not check_volume_surge(entry_rates, CONFIG["volume_ma_period"]):
            if CONFIG["verbose"]:
                log(f"📊 No volume surge detected")
            return None
        
        log(f"✅ Volume surge confirmed")
    
    # === STEP 10: Check Candlestick Patterns (M5 for confirmation) ===
    if CONFIG["use_candle_patterns"]:
        last_candle_idx = len(confirm_rates) - 1
        
        pattern_confirmed = False
        
        if direction == "buy":
            if is_bullish_engulfing(confirm_rates, last_candle_idx):
                log(f"✅ Bullish engulfing pattern detected")
                pattern_confirmed = True
            elif has_long_lower_wick(confirm_rates[last_candle_idx]):
                log(f"✅ Long lower wick rejection detected")
                pattern_confirmed = True
        else:
            if is_bearish_engulfing(confirm_rates, last_candle_idx):
                log(f"✅ Bearish engulfing pattern detected")
                pattern_confirmed = True
            elif has_long_upper_wick(confirm_rates[last_candle_idx]):
                log(f"✅ Long upper wick rejection detected")
                pattern_confirmed = True
        
        if not pattern_confirmed:
            if CONFIG["verbose"]:
                log(f"📊 No candlestick pattern confirmation")
            return None
    
    # === STEP 11: Calculate Entry, Stop Loss, Take Profit ===
    if direction == "buy":
        entry_price = tick.ask
        stop_loss = impulse["start"] - pips_to_price(symbol_info, CONFIG["stop_buffer_pips"])
        
        # Calculate SL distance
        sl_pips = price_to_pips(symbol_info, entry_price - stop_loss)
        
        # Calculate TP based on R:R ratio
        tp_pips = sl_pips * CONFIG["min_rr_ratio"]
        tp_pips = min(tp_pips, sl_pips * CONFIG["max_rr_ratio"])  # Cap at max R:R
        
        take_profit = entry_price + pips_to_price(symbol_info, tp_pips)
    else:
        entry_price = tick.bid
        stop_loss = impulse["start"] + pips_to_price(symbol_info, CONFIG["stop_buffer_pips"])
        
        # Calculate SL distance
        sl_pips = price_to_pips(symbol_info, stop_loss - entry_price)
        
        # Calculate TP based on R:R ratio
        tp_pips = sl_pips * CONFIG["min_rr_ratio"]
        tp_pips = min(tp_pips, sl_pips * CONFIG["max_rr_ratio"])
        
        take_profit = entry_price - pips_to_price(symbol_info, tp_pips)
    
    # === STEP 12: Check for Duplicate Entry ===
    if is_duplicate_entry(entry_price, symbol_info):
        log(f"⚠️  Duplicate entry prevented (too close to recent trade)")
        return None
    
    # === STEP 13: Return Trade Signal ===
    return {
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "rr_ratio": tp_pips / sl_pips if sl_pips > 0 else 0,
        "impulse": impulse,
        "fib_levels": fib_levels
    }

def execute_trade(signal, symbol_info, equity):
    """Execute trade based on signal"""
    
    # Calculate position size
    lots = calculate_position_size(
        symbol_info,
        equity,
        signal["sl_pips"]
    )
    
    if lots <= 0:
        log(f"❌ Position size is zero (check risk settings)")
        return False
    
    log(f"\n{'='*60}")
    log(f"🚀 EXECUTING TRADE")
    log(f"{'='*60}")
    log(f"Direction:    {signal['direction'].upper()}")
    log(f"Entry:        {signal['entry_price']:.3f}")
    log(f"Stop Loss:    {signal['stop_loss']:.3f} ({signal['sl_pips']:.1f} pips)")
    log(f"Take Profit:  {signal['take_profit']:.3f} ({signal['tp_pips']:.1f} pips)")
    log(f"Risk/Reward:  1:{signal['rr_ratio']:.2f}")
    log(f"Position Size: {lots:.2f} lots")
    log(f"Risk Amount:  ${equity * CONFIG['risk_per_trade_pct'] / 100:.2f} ({CONFIG['risk_per_trade_pct']}%)")
    log(f"{'='*60}\n")
    
    # Send order
    success, message = send_market_order(
        symbol_info,
        signal["direction"],
        signal["entry_price"],
        signal["stop_loss"],
        signal["take_profit"],
        lots
    )
    
    if success:
        # Record trade
        record_trade(
            signal["entry_price"],
            signal["direction"],
            signal["stop_loss"],
            signal["take_profit"]
        )
        log(f"✅ Trade executed successfully: {message}")
    else:
        log(f"❌ Trade execution failed: {message}")
    
    return success

# ========== MAIN LOOP ==========

def run_bot():
    """Main bot loop"""
    
    log("\n" + "="*60)
    log("🚀 FIBONACCI GOLDEN ZONE TRADING BOT STARTED")
    log("="*60)
    log(f"Symbol: {CONFIG['symbol']}")
    log(f"Trend TF: H1 | Entry TF: M15 | Confirm TF: M5")
    log(f"Golden Zone: 50% - 61.8%")
    log(f"Min R:R: 1:{CONFIG['min_rr_ratio']}")
    log(f"Max Positions: {CONFIG['max_open_positions']}")
    log(f"Max Daily Trades: {CONFIG['max_daily_trades']}")
    log(f"Risk per Trade: {CONFIG['risk_per_trade_pct']}%")
    
    # Advanced features status
    if CONFIG["use_partial_tp"]:
        log(f"✅ Partial TP: {CONFIG['partial_tp_percent']}% at {CONFIG['partial_tp_rr']}:1 R:R")
    if CONFIG["use_trailing_stop"]:
        log(f"✅ Trailing Stop: {CONFIG['trailing_stop_distance_pips']} pips (activates at {CONFIG['trailing_stop_activation_rr']}:1)")

    pivot_method = CONFIG.get("pivot_detection_method", "zigzag").lower()
    if pivot_method == "zigzag":
        log(
            f"📐 Pivot Detection: ZigZag | ATR x{CONFIG.get('zigzag_threshold_multiplier', 3.0)} | Depth={CONFIG.get('zigzag_depth', 10)}"
        )
    else:
        log(
            f"📐 Pivot Detection: Static width={CONFIG.get('swing_detection_width', 5)} bars"
        )
    
    log("="*60 + "\n")
    
    # Initialize MT5
    acc = initialize_mt5()
    symbol_info = select_symbol(CONFIG["symbol"])
    
    log(f"Symbol Info: {symbol_info.name} | Spread: {symbol_info.spread} pts | Digits: {symbol_info.digits}")
    log(f"Volume: Min={symbol_info.volume_min} Step={symbol_info.volume_step} Max={symbol_info.volume_max}\n")
    
    loop_count = 0
    last_log_time = time.time()
    
    try:
        while True:
            loop_count += 1
            current_time = time.time()
            
            # Heartbeat every 60 seconds
            if current_time - last_log_time >= 60:
                log(f"💓 Bot alive | Loop #{loop_count} | Time: {datetime.now().strftime('%H:%M:%S')}")
                last_log_time = current_time
            
            try:
                # Get fresh account info
                acc = mt5.account_info()
                if acc is None:
                    log("❌ Lost MT5 connection")
                    time.sleep(5)
                    continue
                
                # Check daily loss limit
                if check_daily_loss_limit(acc.equity):
                    log("⛔ Trading halted for today (daily loss limit)")
                    time.sleep(60)
                    continue
                
                # Check max daily trades
                if get_daily_trade_count() >= CONFIG["max_daily_trades"]:
                    log("⛔ Max daily trades reached")
                    time.sleep(60)
                    continue
                
                # Check open positions
                open_positions = get_open_positions(CONFIG["symbol"])
                
                # Manage existing positions (trailing stops, partial TPs)
                if len(open_positions) > 0:
                    manage_open_positions(symbol_info)
                
                # Don't look for new trades if we have max positions
                if len(open_positions) >= CONFIG["max_open_positions"]:
                    time.sleep(CONFIG["poll_interval_sec"])
                    continue
                
                # Analyze market for new trades
                signal = analyze_market(symbol_info)
                
                if signal is not None:
                    # Execute trade
                    execute_trade(signal, symbol_info, acc.equity)
                    
                    # Wait before next scan
                    time.sleep(30)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
            
            # Sleep before next iteration
            time.sleep(CONFIG["poll_interval_sec"])
    
    except KeyboardInterrupt:
        log("\n⛔ Bot stopped by user")
    finally:
        mt5.shutdown()
        log("✅ MT5 connection closed")

# ========== ENTRY POINT ==========

if __name__ == "__main__":
    run_bot()

