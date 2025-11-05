import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np
from datetime import datetime
from scipy import stats  # For linear regression

# --- YOUR ACCOUNT DETAILS ---
ACCOUNT = 261525020           # <-- replace with your MT5 account number
PASSWORD = "Ae!8bfb666"      # <-- replace with your MT5 password
SERVER = "Exness-MT5Trial16"  # e.g., "Exness-MT5Real" or "ICMarketsSC-Demo"

# --- NEW TRADING PARAMETERS ---
SYMBOL = "XAUUSDm"
RISK_PERCENT = 1.0  # 1% risk per trade

# Dual ATR parameters (NEW)
A_BUY = 2.0    # Buy Sensitivity (Multiplier)
C_BUY = 1      # Buy ATR Period (0 = disable)
A_SELL = 2.0   # Sell Sensitivity (Multiplier)
C_SELL = 1     # Sell ATR Period (0 = disable)

# LinReg parameters (NEW)
SIGNAL_LENGTH = 7
SMA_SIGNAL = True
USE_LIN_REG = True
LINREG_LENGTH = 11

# Position state tracking (NEW)
posState = 0  # 0=flat, 1=long, -1=short

# --- CONNECT TO MT5 ---
def initialize_mt5():
    if not mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER):
        print("❌ MT5 connection failed:", mt5.last_error())
        return False
    else:
        print("✅ Connected to MetaTrader 5")
        return True

# --- GET ACCOUNT BALANCE ---
def get_account_balance():
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Failed to get account info")
        return 0
    return account_info.balance

# --- CALCULATE LOT SIZE BASED ON RISK ---
def calculate_lot_size(entry_price, stop_loss_price, risk_percent=1.0):
    balance = get_account_balance()
    risk_amount = balance * (risk_percent / 100)
    
    # Calculate stop distance in points
    point = mt5.symbol_info(SYMBOL).point
    stop_distance_points = abs(entry_price - stop_loss_price) / point
    
    # Calculate tick value
    tick_value = mt5.symbol_info(SYMBOL).trade_tick_value
    
    # Calculate lot size
    if stop_distance_points > 0 and tick_value > 0:
        lot_size = risk_amount / (stop_distance_points * tick_value)
        
        # Round to allowed lot steps
        step = mt5.symbol_info(SYMBOL).volume_step
        lot_size = round(lot_size / step) * step
        
        # Apply min/max limits
        min_lot = mt5.symbol_info(SYMBOL).volume_min
        max_lot = mt5.symbol_info(SYMBOL).volume_max
        lot_size = max(min(lot_size, max_lot), min_lot)
        
        return lot_size
    return 0.01  # Default minimum lot

# --- GET HISTORICAL DATA FOR CALCULATIONS ---
def get_historical_data(symbol, timeframe, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print(f"❌ Failed to get historical data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# --- CALCULATE ATR (STANDARD CALCULATION) ---
def calculate_atr(df, period=14):
    df = df.copy()
    
    # Standard ATR calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    return df

# --- CALCULATE EMA ---
def calculate_ema(df, period=1, source_col='close'):
    """Calculate EMA for given period and source column"""
    if len(df) < period:
        return pd.Series([np.nan] * len(df))
    
    # For period=1, EMA is just the source
    if period == 1:
        return df[source_col]
    
    # Standard EMA calculation
    alpha = 2.0 / (period + 1)
    ema_values = [df.iloc[0][source_col]]  # First value is the same as source
    
    for i in range(1, len(df)):
        ema = (df.iloc[i][source_col] * alpha) + (ema_values[i-1] * (1 - alpha))
        ema_values.append(ema)
    
    return pd.Series(ema_values)

# --- CALCULATE LINEAR REGRESSION ---
def calculate_linreg(df, length=11, source_col='close'):
    """Calculate linear regression values"""
    linreg_values = []
    
    for i in range(len(df)):
        if i < length - 1:
            linreg_values.append(np.nan)
            continue
            
        # Get the window of data
        window = df.iloc[i-length+1:i+1][source_col].values
        x = np.arange(length)
        
        # Calculate linear regression
        slope, intercept, _, _, _ = stats.linregress(x, window)
        linreg_value = intercept + slope * (length - 1)
        linreg_values.append(linreg_value)
    
    # Pad with NaN values at the beginning
    while len(linreg_values) < len(df):
        linreg_values.insert(0, np.nan)
    
    return pd.Series(linreg_values)

# --- NEW: DUAL ATR TRAILING STOP CALCULATION ---
def calculate_dual_atr_trailing(df, a_buy, c_buy, a_sell, c_sell):
    """Calculate dual ATR trailing stops (buy and sell sides)"""
    df = df.copy()
    
    # Calculate ATR for both sides
    if c_buy > 0:
        df = calculate_atr(df, c_buy)
        df['atr_buy'] = df['atr']
        df['nLoss_buy'] = a_buy * df['atr_buy']
    else:
        df['atr_buy'] = np.nan
        df['nLoss_buy'] = np.nan
    
    if c_sell > 0:
        if c_sell != c_buy:  # Only recalc if different period
            df_temp = calculate_atr(df, c_sell)
            df['atr_sell'] = df_temp['atr']
        else:
            df['atr_sell'] = df['atr']
        df['nLoss_sell'] = a_sell * df['atr_sell']
    else:
        df['atr_sell'] = np.nan
        df['nLoss_sell'] = np.nan
    
    # Initialize trailing stops
    trail_buy = []
    trail_sell = []
    
    # Calculate trailing stops (Pine Script logic)
    for i in range(len(df)):
        if i == 0 or np.isnan(df.iloc[i]['nLoss_buy']) or np.isnan(df.iloc[i]['nLoss_sell']):
            # First bar or disabled ATR
            trail_buy.append(df.iloc[i]['close'] - (df.iloc[i]['nLoss_buy'] if not np.isnan(df.iloc[i]['nLoss_buy']) else 0))
            trail_sell.append(df.iloc[i]['close'] + (df.iloc[i]['nLoss_sell'] if not np.isnan(df.iloc[i]['nLoss_sell']) else 0))
            continue
        
        # Buy trailing stop logic
        prev_trail_buy = trail_buy[i-1]
        current_close = df.iloc[i]['close']
        prev_close = df.iloc[i-1]['close']
        current_nLoss_buy = df.iloc[i]['nLoss_buy']
        
        if current_close > prev_trail_buy and prev_close > prev_trail_buy:
            trail_buy.append(max(prev_trail_buy, current_close - current_nLoss_buy))
        elif current_close < prev_trail_buy and prev_close < prev_trail_buy:
            trail_buy.append(min(prev_trail_buy, current_close + current_nLoss_buy))
        else:
            if current_close > prev_trail_buy:
                trail_buy.append(current_close - current_nLoss_buy)
            else:
                trail_buy.append(current_close + current_nLoss_buy)
        
        # Sell trailing stop logic
        prev_trail_sell = trail_sell[i-1]
        current_nLoss_sell = df.iloc[i]['nLoss_sell']
        
        if current_close > prev_trail_sell and prev_close > prev_trail_sell:
            trail_sell.append(max(prev_trail_sell, current_close - current_nLoss_sell))
        elif current_close < prev_trail_sell and prev_close < prev_trail_sell:
            trail_sell.append(min(prev_trail_sell, current_close + current_nLoss_sell))
        else:
            if current_close > prev_trail_sell:
                trail_sell.append(current_close - current_nLoss_sell)
            else:
                trail_sell.append(current_close + current_nLoss_sell)
    
    df['trail_buy'] = trail_buy
    df['trail_sell'] = trail_sell
    
    return df

# --- NEW: DETECT CROSSOVERS (PINE SCRIPT STYLE) ---
def detect_crossover(current_a, current_b, prev_a, prev_b):
    """Pine Script crossover: a[1] < b[1] and a > b"""
    return (prev_a < prev_b) and (current_a > current_b)

# --- NEW: GET TRADING SIGNAL WITH DUAL ATR ---
def get_trading_signal():
    global posState
    
    # Get historical data
    df = get_historical_data(SYMBOL, mt5.TIMEFRAME_M1, bars=100)
    if df is None or len(df) < 2:
        return "no_signal"
    
    # Calculate dual ATR trailing stops
    df = calculate_dual_atr_trailing(df, A_BUY, C_BUY, A_SELL, C_SELL)
    
    # Get current and previous values
    current_close = df.iloc[-1]['close']
    current_trail_buy = df.iloc[-1]['trail_buy']
    current_trail_sell = df.iloc[-1]['trail_sell']
    
    prev_close = df.iloc[-2]['close']
    prev_trail_buy = df.iloc[-2]['trail_buy']
    prev_trail_sell = df.iloc[-2]['trail_sell']
    
    # Calculate EMAs for crossover detection
    df['ema_buy'] = calculate_ema(df, 1, 'close')
    df['ema_sell'] = calculate_ema(df, 1, 'close')
    
    current_ema_buy = df.iloc[-1]['ema_buy']
    current_ema_sell = df.iloc[-1]['ema_sell']
    prev_ema_buy = df.iloc[-2]['ema_buy']
    prev_ema_sell = df.iloc[-2]['ema_sell']
    
    # Detect crossovers
    above_buy_cross = detect_crossover(current_ema_buy, current_trail_buy, prev_ema_buy, prev_trail_buy)
    below_sell_cross = detect_crossover(current_trail_sell, current_ema_sell, prev_trail_sell, prev_ema_sell)
    
    # Check if ATR systems are enabled
    buy_enabled = (C_BUY > 0) and (A_BUY > 0)
    sell_enabled = (C_SELL > 0) and (A_SELL > 0)
    
    # Generate signals with position state logic
    above_buy = buy_enabled and above_buy_cross
    buy_signal_raw = buy_enabled and (current_close > current_trail_buy) and above_buy
    
    below_sell = sell_enabled and below_sell_cross
    sell_signal_raw = sell_enabled and (current_close < current_trail_sell) and below_sell
    
    # Position state management (one-signal system)
    buy_signal_confirmed = buy_enabled and buy_signal_raw and posState <= 0
    sell_signal_confirmed = sell_enabled and sell_signal_raw and posState >= 0
    
    # Update position state
    if buy_signal_confirmed:
        posState = 1
        return "buy"
    elif sell_signal_confirmed:
        posState = -1
        return "sell"
    else:
        return "no_signal"

# --- CLOSE ALL POSITIONS FOR SYMBOL ---
def close_all_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return True
    
    closed_count = 0
    for position in positions:
        tick = mt5.symbol_info_tick(symbol)
        
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Closed by UT Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            closed_count += 1
            print(f"✅ Closed position {position.ticket} ({'BUY' if position.type == 0 else 'SELL'})")
        else:
            print(f"❌ Failed to close position {position.ticket}: {result}")
    
    return closed_count == len(positions)

# --- PLACE NEW TRADE ---
def place_trade(trade_type):
    global posState
    
    # Close any existing positions first
    print("🔄 Closing any existing positions...")
    if not close_all_positions(SYMBOL):
        print("❌ Failed to close existing positions")
        return False
    
    # Get current price
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("❌ Failed to get tick data")
        return False
    
    if trade_type == "buy":
        entry_price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        print(f"📈 Placing BUY order at {entry_price:.5f}")
    else:  # sell
        entry_price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        print(f"📈 Placing SELL order at {entry_price:.5f}")
    
    # Calculate lot size based on fixed risk
    balance = get_account_balance()
    base_lot = 0.01  # Minimum lot size
    risk_lot = balance * (RISK_PERCENT / 100) / 1000  # Simple calculation
    
    # Use the larger of base lot or risk-based lot
    lot_size = max(base_lot, round(risk_lot, 2))
    
    # Ensure lot size is within broker limits
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info:
        lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
    
    print(f"💰 Account Balance: ${balance:.2f}")
    print(f"📊 Lot Size: {lot_size} (Risk: {RISK_PERCENT}%)")
    
    # Place order (NO SL/TP - signal based only)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": order_type,
        "price": entry_price,
        "deviation": 20,
        "magic": 123456,
        "comment": "UT Bot Dual ATR Logic",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        print(f"❌ Order failed: No result from MT5")
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed: {result}")
        return False
    else:
        print(f"✅ {trade_type.upper()} order placed successfully!")
        print(f"   Entry: {entry_price:.5f}")
        print(f"   Lot Size: {lot_size}")
        print(f"   Ticket: {result.order}")
        
        # Update position state
        if trade_type == "buy":
            posState = 1
        else:
            posState = -1
            
        return True

def main_trading_loop():
    global posState
    
    print("🤖 UT Bot Auto Trading Started!")
    print(f"📈 Symbol: {SYMBOL}")
    print(f"⚡ Risk: {RISK_PERCENT}% per trade")
    print("=" * 50)
    print("🎯 USING DUAL ATR PINE SCRIPT LOGIC")
    print(f"   Buy: ATR({C_BUY}) * {A_BUY}")
    print(f"   Sell: ATR({C_SELL}) * {A_SELL}")
    print("   Position State Management: One-signal system")
    print("=" * 50)
    
    while True:
        try:
            # Get current signal
            signal = get_trading_signal()
            
            # Check if we have a new signal
            if signal != "no_signal":
                print(f"\n🎯 SIGNAL DETECTED: {signal.upper()}!")
                print(f"🔄 Auto-reversing position...")
                
                # Place the trade
                if place_trade(signal):
                    print(f"✅ Successfully executed {signal.upper()} position")
                    print(f"📊 Position State: {'LONG' if posState == 1 else 'SHORT' if posState == -1 else 'FLAT'}")
                else:
                    print("❌ Failed to execute trade")
            else:
                # Show current market state for monitoring
                df = get_historical_data(SYMBOL, mt5.TIMEFRAME_M1, bars=2)
                if df is not None and len(df) >= 2:
                    df = calculate_dual_atr_trailing(df, A_BUY, C_BUY, A_SELL, C_SELL)
                    current_close = df.iloc[-1]['close']
                    current_trail_buy = df.iloc[-1]['trail_buy']
                    current_trail_sell = df.iloc[-1]['trail_sell']
                    
                    buy_position = "ABOVE" if current_close > current_trail_buy else "BELOW"
                    sell_position = "ABOVE" if current_close > current_trail_sell else "BELOW"
                    
                    print(f"📊 Monitoring: Price {buy_position} Buy Trail, {sell_position} Sell Trail | State: {'LONG' if posState == 1 else 'SHORT' if posState == -1 else 'FLAT'}", end='\r')
            
            # Wait before next check
            time.sleep(2)  # Check every 2 seconds for faster response
            
        except KeyboardInterrupt:
            print("\n Trading stopped by user")
            break
        except Exception as e:
            print(f" Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

# --- START THE BOT ---
if __name__ == "__main__":
    if initialize_mt5():
        # Check symbol availability
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"❌ Symbol {SYMBOL} not found.")
            mt5.shutdown()
            quit()
        
        if not symbol_info.visible:
            mt5.symbol_select(SYMBOL, True)
        
        # Start trading
        main_trading_loop()
        
        # Shutdown
        mt5.shutdown()
        print("🔌 MT5 disconnected")