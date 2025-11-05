import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np
from datetime import datetime

# --- YOUR ACCOUNT DETAILS ---
ACCOUNT = 261525020           # <-- replace with your MT5 account number
PASSWORD = "Ae!8bfb666"  # <-- replace with your MT5 password
SERVER = "Exness-MT5Trial16"  # e.g., "Exness-MT5Real" or "ICMarketsSC-Demo"

# --- TRADING PARAMETERS ---
SYMBOL = "XAUUSDm"
RISK_PERCENT = 1.0  # 0.1% risk per trade
KEY_VALUE = 2.0     # ATR multiplier (sensitivity)
ATR_PERIOD = 14
TIMEFRAME = mt5.TIMEFRAME_M1  # You can change this (M1, M5, M15, etc.)
USE_HEIKIN_ASHI = False

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
def calculate_lot_size(entry_price, stop_loss_price, risk_percent=0.1):
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

# --- CALCULATE HEIKIN ASHI PRICES ---
def calculate_heikin_ashi(df):
    ha_df = df.copy()
    
    # First bar
    if len(ha_df) > 0:
        ha_df.loc[ha_df.index[0], 'ha_close'] = (ha_df.iloc[0]['open'] + ha_df.iloc[0]['high'] + 
                                               ha_df.iloc[0]['low'] + ha_df.iloc[0]['close']) / 4
        ha_df.loc[ha_df.index[0], 'ha_open'] = (ha_df.iloc[0]['open'] + ha_df.iloc[0]['close']) / 2
        ha_df.loc[ha_df.index[0], 'ha_high'] = ha_df.iloc[0]['high']
        ha_df.loc[ha_df.index[0], 'ha_low'] = ha_df.iloc[0]['low']
    
    # Subsequent bars
    for i in range(1, len(ha_df)):
        ha_df.loc[ha_df.index[i], 'ha_open'] = (ha_df.iloc[i-1]['ha_open'] + ha_df.iloc[i-1]['ha_close']) / 2
        ha_df.loc[ha_df.index[i], 'ha_close'] = (ha_df.iloc[i]['open'] + ha_df.iloc[i]['high'] + 
                                               ha_df.iloc[i]['low'] + ha_df.iloc[i]['close']) / 4
        ha_df.loc[ha_df.index[i], 'ha_high'] = max(ha_df.iloc[i]['high'], ha_df.iloc[i]['ha_open'], ha_df.iloc[i]['ha_close'])
        ha_df.loc[ha_df.index[i], 'ha_low'] = min(ha_df.iloc[i]['low'], ha_df.iloc[i]['ha_open'], ha_df.iloc[i]['ha_close'])
    
    return ha_df

# --- CALCULATE ATR (EXACT PINE SCRIPT LOGIC) ---
def calculate_atr(df, period=14):
    df = df.copy()
    
    # Pine Script ATR calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    return df

# --- CALCULATE EMA (EXACT PINE SCRIPT LOGIC) ---
def calculate_ema(df, period=1):
    """Calculate EMA exactly like Pine Script's ta.ema()"""
    if len(df) < period:
        return pd.Series([np.nan] * len(df))
    
    # For period=1, EMA is just the source (close price)
    if period == 1:
        return df['src']
    
    # Use the standard EMA formula (alpha = 2/(period+1))
    alpha = 2.0 / (period + 1)
    ema_values = [df.iloc[0]['src']]  # First value is the same as source
    
    for i in range(1, len(df)):
        ema = (df.iloc[i]['src'] * alpha) + (ema_values[i-1] * (1 - alpha))
        ema_values.append(ema)
    
    return pd.Series(ema_values)

def calculate_trailing_stop(df, key_value=1.0, atr_period=10, use_heikin_ashi=False):
    df = df.copy()
    
    # Calculate ATR (Pine Script logic)
    df = calculate_atr(df, atr_period)
    
    # Use Heikin Ashi or normal close (Pine Script logic)
    if use_heikin_ashi:
        df = calculate_heikin_ashi(df)
        df['src'] = df['ha_close']
    else:
        df['src'] = df['close']
    
    # Calculate nLoss (Pine Script: nLoss = keyValue * atrValue)
    df['nLoss'] = key_value * df['atr']
    
    # Initialize trailing stop with nz() equivalent (Pine Script: nz(xATRTrailingStop[1], 0))
    trailing_stop = []
    
    # First bar: src - nLoss (if src > 0) or src + nLoss (if src < 0)
    if len(df) > 0:
        first_ts = df.iloc[0]['src'] - df.iloc[0]['nLoss']  # Default to src - nLoss
        trailing_stop.append(first_ts)
    
    # Calculate trailing stop for each bar (EXACT PINE SCRIPT LOGIC)
    for i in range(1, len(df)):
        prev_ts = trailing_stop[i-1] if i > 0 else 0  # nz(xATRTrailingStop[1], 0)
        current_src = df.iloc[i]['src']
        prev_src = df.iloc[i-1]['src']
        current_nLoss = df.iloc[i]['nLoss']
        
        # EXACT PINE SCRIPT LOGIC:
        if current_src > prev_ts and prev_src > prev_ts:
            # Uptrend: trailing stop moves up
            trailing_stop.append(max(prev_ts, current_src - current_nLoss))
        elif current_src < prev_ts and prev_src < prev_ts:
            # Downtrend: trailing stop moves down
            trailing_stop.append(min(prev_ts, current_src + current_nLoss))
        else:
            # Trend change
            if current_src > prev_ts:
                trailing_stop.append(current_src - current_nLoss)
            else:
                trailing_stop.append(current_src + current_nLoss)
    
    df['trailing_stop'] = trailing_stop
    return df

def get_trading_signal():
    # Get historical data
    df = get_historical_data(SYMBOL, TIMEFRAME, bars=100)
    if df is None:
        return "no_signal"
    
    # Calculate trailing stop with indicators
    df = calculate_trailing_stop(df, KEY_VALUE, ATR_PERIOD, USE_HEIKIN_ASHI)
    
    if len(df) < 2:  # Need at least 2 bars for crossover calculation
        return "no_signal"
    
    # Get current and previous values
    current_src = df.iloc[-1]['src']
    current_ts = df.iloc[-1]['trailing_stop']
    prev_src = df.iloc[-2]['src']
    prev_ts = df.iloc[-2]['trailing_stop']
    
    # Calculate EMA1 (Pine Script: ema1 = ta.ema(src, 1))
    df['ema1'] = calculate_ema(df, 1)
    current_ema1 = df.iloc[-1]['ema1']
    prev_ema1 = df.iloc[-2]['ema1']
    
    # EXACT PINE SCRIPT CROSSOVER LOGIC:
    # crossover(a, b) => a[1] < b[1] and a > b (exactly as in Pine Script)
    ema_crosses_above_ts = (prev_ema1 < prev_ts) and (current_ema1 > current_ts)
    ts_crosses_above_ema = (prev_ts < prev_ema1) and (current_ts > current_ema1)
    
    # FINAL SIGNALS (EXACT PINE SCRIPT):
    buy_signal = (current_src > current_ts) and ema_crosses_above_ts
    sell_signal = (current_src < current_ts) and ts_crosses_above_ema
    
    # Debug information
    if buy_signal or sell_signal:
        print(f"🔍 DEBUG SIGNAL:")
        print(f"   Current: SRC={current_src:.5f}, TS={current_ts:.5f}, EMA1={current_ema1:.5f}")
        print(f"   Previous: SRC={prev_src:.5f}, TS={prev_ts:.5f}, EMA1={prev_ema1:.5f}")
        print(f"   EMA Crosses TS: {ema_crosses_above_ts}")
        print(f"   TS Crosses EMA: {ts_crosses_above_ema}")
    
    if buy_signal:
        print(f"🎯 EXACT PINE SCRIPT BUY SIGNAL")
        return "buy"
    elif sell_signal:
        print(f"🎯 EXACT PINE SCRIPT SELL SIGNAL")
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
        "comment": "UT Bot Pine Script Logic",
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
        return True

def main_trading_loop():
    print("🤖 UT Bot Auto Trading Started!")
    print(f"📈 Symbol: {SYMBOL}")
    print(f"⚡ Risk: {RISK_PERCENT}% per trade")
    print(f"🔧 ATR Period: {ATR_PERIOD}, Key Value: {KEY_VALUE}")
    print(f"📊 Timeframe: {TIMEFRAME}")
    print(f"🕯️ Heikin Ashi: {USE_HEIKIN_ASHI}")
    print("=" * 50)
    print("🎯 USING EXACT PINE SCRIPT LOGIC")
    print("   Buy: src > trailingStop AND crossover(ema1, trailingStop)")
    print("   Sell: src < trailingStop AND crossover(trailingStop, ema1)")
    print("   crossover(a, b) => a[1] < b[1] and a > b")  # Exact Pine Script definition
    print("=" * 50)
    
    last_signal = None
    
    while True:
        try:
            # Get current signal
            signal = get_trading_signal()
            
            # Check if we have a new signal (Pine Script generates alerts on each occurrence)
            if signal != "no_signal":
                print(f"\n🎯 SIGNAL DETECTED: {signal.upper()}!")
                print(f"🔄 Auto-reversing position...")
                
                # Place the trade
                if place_trade(signal):
                    last_signal = signal
                    print(f"✅ Successfully executed {signal.upper()} position")
                else:
                    print("❌ Failed to execute trade")
            else:
                # Show current market state for monitoring
                df = get_historical_data(SYMBOL, TIMEFRAME, bars=2)
                if df is not None and len(df) >= 2:
                    df = calculate_trailing_stop(df, KEY_VALUE, ATR_PERIOD, USE_HEIKIN_ASHI)
                    current_src = df.iloc[-1]['src']
                    current_ts = df.iloc[-1]['trailing_stop']
                    position = "ABOVE" if current_src > current_ts else "BELOW"
                    print(f"📊 Monitoring: SRC {position} TS ({current_src:.5f} vs {current_ts:.5f})", end='\r')
            
            # Wait before next check
            time.sleep(2)  # Check every 2 seconds for faster response
            
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
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