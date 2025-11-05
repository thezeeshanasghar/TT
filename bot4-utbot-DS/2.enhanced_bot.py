import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import talib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Initialize MT5 connection
def initialize_mt5():
    # Path to your MT5 terminal
    path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    
    if not mt5.initialize(path=path):
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False
    
    # Replace with your account credentials
    account = 261525131
    password = "Ae!8bfb666"
    server = "Exness-MT5Trial16"
    
    authorized = mt5.login(account, password=password, server=server)
    if not authorized:
        print("Failed to connect to account: ", mt5.last_error())
        return False
    
    print("Connected to MetaTrader 5")
    print(f"Account: {account}, Server: {server}")
    return True

# Load trade history for analysis
def load_trade_history():
    history_file = "trade_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return {"trades": [], "performance_metrics": {}}

# Save trade history
def save_trade_history(trade_data):
    history_file = "trade_history.json"
    with open(history_file, 'w') as f:
        json.dump(trade_data, f, indent=4)

# Get market data from MT5
def get_market_data(symbol, timeframe, num_bars=100):
    # Select the symbol
    mt5.symbol_select(symbol, True)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        print(f"Failed to get rates for {symbol}, error: {mt5.last_error()}")
        return pd.DataFrame()
    
    # Convert to DataFrame with correct column names
    df = pd.DataFrame(rates)
    # Convert time from seconds to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Improved ATR trailing stop calculation
def calculate_atr_trailing_stop(close_prices, high_prices, low_prices, key_value=3, atr_period=14):
    # Calculate ATR
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
    nLoss = key_value * atr
    
    # Initialize arrays
    xATRTrailingStop = np.zeros_like(close_prices)
    pos = np.zeros_like(close_prices)
    
    # Find the first non-NaN value in nLoss
    start_idx = 0
    for i in range(len(nLoss)):
        if not np.isnan(nLoss[i]):
            start_idx = i
            break
    
    # Initialize first value
    xATRTrailingStop[start_idx] = close_prices[start_idx] - nLoss[start_idx]
    pos[start_idx] = 1 if close_prices[start_idx] > xATRTrailingStop[start_idx] else -1
    
    # Calculate trailing stop for the rest of the values
    for i in range(start_idx + 1, len(close_prices)):
        if np.isnan(nLoss[i]):
            xATRTrailingStop[i] = xATRTrailingStop[i-1]
            pos[i] = pos[i-1]
            continue
            
        if close_prices[i] > xATRTrailingStop[i-1]:
            xATRTrailingStop[i] = max(xATRTrailingStop[i-1], close_prices[i] - nLoss[i])
        else:
            xATRTrailingStop[i] = min(xATRTrailingStop[i-1], close_prices[i] + nLoss[i])
        
        # Determine position
        if close_prices[i] > xATRTrailingStop[i]:
            pos[i] = 1
        elif close_prices[i] < xATRTrailingStop[i]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]
    
    return xATRTrailingStop, pos

# Calculate position size based on risk
def calculate_position_size(symbol, entry_price, stop_loss, risk_percent, account_balance):
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get info for {symbol}")
        return 0
    
    # Calculate risk amount
    risk_amount = account_balance * risk_percent / 100
    
    # Calculate stop distance in points
    if entry_price > stop_loss:  # Long position
        stop_distance = entry_price - stop_loss
    else:  # Short position
        stop_distance = stop_loss - entry_price
    
    # Convert to points
    point = symbol_info.point
    stop_distance_points = stop_distance / point
    
    # Calculate tick value
    tick_value = symbol_info.trade_tick_value
    
    # Calculate position size
    position_size = risk_amount / (stop_distance_points * tick_value)
    
    # Normalize to lot size
    lot_step = symbol_info.volume_step
    position_size = round(position_size / lot_step) * lot_step
    
    # Ensure it's within min/max limits
    position_size = max(symbol_info.volume_min, min(symbol_info.volume_max, position_size))
    
    return position_size

# Send order to MT5
def send_order(symbol, order_type, volume, sl, tp, comment=""):
    # Get current price
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get tick data for {symbol}")
        return None
    
    if order_type == mt5.ORDER_TYPE_BUY:
        price = symbol_info.ask
    else:
        price = symbol_info.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    return result

# Main trading function
def run_strategy():
    # Strategy parameters
    symbol = "XAUUSD"  # Try different symbol names
    timeframe = mt5.TIMEFRAME_M5  # Use M5 timeframe for better signals
    key_value = 2.5  # Reduced from 3 to make it more sensitive
    atr_period = 14  # ATR period
    risk_percent = 1.0  # Reduced risk to 1% for safety
    ema_period = 50  # Reduced from 200 to make it more responsive
    
    print(f"Starting strategy for {symbol} on {timeframe} timeframe")
    
    # Try different symbol variations
    symbol_variations = ["XAUUSD", "XAUUSDm", "GOLD", "XAU/USD"]
    selected_symbol = None
    
    for sym in symbol_variations:
        if mt5.symbol_select(sym, True):
            selected_symbol = sym
            print(f"Selected symbol: {sym}")
            break
    
    if selected_symbol is None:
        print("No valid symbol found. Please check symbol name.")
        return
    
    symbol = selected_symbol
    
    # Initialize variables for signal detection
    last_signal_time = None
    signal_cooldown = timedelta(minutes=5)  # 5-minute cooldown between signals
    
    while True:
        try:
            # Get market data
            df = get_market_data(symbol, timeframe, 200)  # Increased to ensure enough data for ATR
            if df.empty:
                print("No data received, trying again in 10 seconds...")
                time.sleep(10)
                continue
            
            # Print latest data for debugging
            print(f"Latest close price: {df['close'].iloc[-1]}, Time: {df.index[-1]}")
            
            # Calculate indicators
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # Calculate EMA
            ema = talib.EMA(close_prices, timeperiod=ema_period)
            
            # Calculate ATR trailing stop
            atr_trailing_stop, pos = calculate_atr_trailing_stop(
                close_prices, high_prices, low_prices, key_value, atr_period
            )
            
            # Get current values
            current_close = close_prices[-1]
            current_ema = ema[-1] if not np.isnan(ema[-1]) else 0
            current_atr_trailing = atr_trailing_stop[-1]
            current_pos = pos[-1]
            prev_pos = pos[-2] if len(pos) > 1 else 0
            
            # Check trend condition
            bullish = current_close > current_ema
            bearish = current_close < current_ema
            
            # Calculate ATR for stop loss
            atr_value = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)[-1]
            
            # Print signals for debugging
            print(f"Position: {current_pos}, Prev Position: {prev_pos}")
            print(f"Bullish: {bullish}, Bearish: {bearish}")
            print(f"Close: {current_close}, EMA: {current_ema}")
            print(f"ATR Trailing Stop: {current_atr_trailing}, ATR Value: {atr_value}")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                time.sleep(10)
                continue
            
            account_balance = account_info.balance
            positions = mt5.positions_get(symbol=symbol)
            
            # Check if we have valid ATR values
            if np.isnan(current_atr_trailing) or np.isnan(atr_value):
                print("ATR values not ready yet, waiting for more data...")
                time.sleep(10)
                continue
            
            # Check for buy signal (price above trailing stop and bullish trend)
            current_time = datetime.now()
            if (current_pos == 1 and prev_pos != 1 and bullish and 
                not any(p.type == mt5.ORDER_TYPE_BUY for p in positions) and
                (last_signal_time is None or current_time - last_signal_time > signal_cooldown)):
                
                print("BUY SIGNAL DETECTED")
                
                # Calculate stop loss and take profit
                sl = current_close - atr_value * 2.0  # Reduced from 1.5
                tp = current_close + atr_value * 4.0  # Increased from 3.0
                
                # Calculate position size
                volume = calculate_position_size(symbol, current_close, sl, risk_percent, account_balance)
                
                if volume > 0:
                    # Send buy order
                    result = send_order(symbol, mt5.ORDER_TYPE_BUY, volume, sl, tp, "UT Bot Buy")
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Buy order failed: {result.retcode}")
                    else:
                        print(f"Buy order executed: {volume} lots")
                        last_signal_time = current_time
                else:
                    print("Position size calculation failed")
            
            # Check for sell signal (price below trailing stop and bearish trend)
            elif (current_pos == -1 and prev_pos != -1 and bearish and 
                  not any(p.type == mt5.ORDER_TYPE_SELL for p in positions) and
                  (last_signal_time is None or current_time - last_signal_time > signal_cooldown)):
                
                print("SELL SIGNAL DETECTED")
                
                # Calculate stop loss and take profit
                sl = current_close + atr_value * 2.0  # Reduced from 1.5
                tp = current_close - atr_value * 4.0  # Increased from 3.0
                
                # Calculate position size
                volume = calculate_position_size(symbol, current_close, sl, risk_percent, account_balance)
                
                if volume > 0:
                    # Send sell order
                    result = send_order(symbol, mt5.ORDER_TYPE_SELL, volume, sl, tp, "UT Bot Sell")
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Sell order failed: {result.retcode}")
                    else:
                        print(f"Sell order executed: {volume} lots")
                        last_signal_time = current_time
                else:
                    print("Position size calculation failed")
            
            # Check for exit signals
            for position in positions:
                if (position.type == mt5.ORDER_TYPE_BUY and current_pos == -1 and 
                    position.profit > 0):
                    # Close profitable long position
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": position.ticket,
                        "symbol": symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": mt5.symbol_info_tick(symbol).bid,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "UT Bot Close Long",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }
                    result = mt5.order_send(close_request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print("Closed profitable long position")
                
                elif (position.type == mt5.ORDER_TYPE_SELL and current_pos == 1 and 
                      position.profit > 0):
                    # Close profitable short position
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": position.ticket,
                        "symbol": symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": mt5.symbol_info_tick(symbol).ask,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "UT Bot Close Short",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }
                    result = mt5.order_send(close_request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print("Closed profitable short position")
            
            # Wait for next bar
            print("Waiting for next bar...")
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in strategy: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

# Performance reporting function
def generate_performance_report():
    trade_history = load_trade_history()
    
    print("\n" + "="*50)
    print("PERFORMANCE REPORT")
    print("="*50)
    
    # Show recent trades
    print(f"Last 5 Trades:")
    recent_trades = trade_history.get('trades', [])[-5:]
    for trade in recent_trades:
        status = "OPEN" if trade['close_time'] is None else "CLOSED"
        profit = trade['profit']
        profit_str = f"+{profit:.2f}" if profit > 0 else f"{profit:.2f}"
        print(f"{trade['symbol']} {trade['type']} {status} P&L: {profit_str}")
    
    print("="*50)

# Main execution
if __name__ == "__main__":
    if initialize_mt5():
        try:
            # Generate initial performance report
            generate_performance_report()
            
            # Run the strategy
            run_strategy()
            
        except KeyboardInterrupt:
            print("Strategy stopped by user")
            # Generate final performance report
            generate_performance_report()
        finally:
            mt5.shutdown()
    else:
        print("Failed to initialize MT5 connection")