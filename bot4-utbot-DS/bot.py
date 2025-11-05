import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import talib

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    
    # Replace with your account credentials
    account = 261525131  # Your account number
    password = "Ae!8bfb666"
    server = "Exness-MT5Trial16"
    
    authorized = mt5.login(account, password=password, server=server)
    if not authorized:
        print("Failed to connect to account: ", mt5.last_error())
        return False
    
    print("Connected to MetaTrader 5")
    return True

# Calculate ATR trailing stop (UT Bot Alerts logic)
def calculate_atr_trailing_stop(close_prices, high_prices, low_prices, key_value=3, atr_period=14, use_heikin_ashi=False):
    if use_heikin_ashi:
        # Calculate Heikin Ashi candles
        ha_close = (close_prices + high_prices + low_prices + close_prices) / 4
        src = ha_close
    else:
        src = close_prices
    
    # Calculate ATR
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
    nLoss = key_value * atr
    
    # Initialize arrays
    xATRTrailingStop = np.zeros_like(src)
    pos = np.zeros_like(src)
    
    # Calculate trailing stop
    for i in range(1, len(src)):
        if i == 1:
            xATRTrailingStop_prev = src[i-1] - nLoss[i-1]
        else:
            xATRTrailingStop_prev = xATRTrailingStop[i-1]
        
        if src[i] > xATRTrailingStop_prev:
            xATRTrailingStop[i] = max(xATRTrailingStop_prev, src[i] - nLoss[i])
        else:
            xATRTrailingStop[i] = min(xATRTrailingStop_prev, src[i] + nLoss[i])
        
        # Determine position
        if src[i-1] > xATRTrailingStop[i-1] and src[i] < xATRTrailingStop[i]:
            pos[i] = -1
        elif src[i-1] < xATRTrailingStop[i-1] and src[i] > xATRTrailingStop[i]:
            pos[i] = 1
        else:
            pos[i] = pos[i-1]
    
    return xATRTrailingStop, pos

# Get market data from MT5
def get_market_data(symbol, timeframe, num_bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    if 'time' not in df.columns:
        return pd.DataFrame()
    
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

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

# Check if we're in a trading session
def in_trading_session(time_session="0000-2400"):
    now = datetime.now()
    current_time = now.strftime("%H%M")
    
    if time_session == "0000-2400":
        return True
    
    try:
        start, end = time_session.split('-')
        start_time = int(start)
        end_time = int(end)
        current_time_int = int(current_time)
        
        if start_time <= end_time:
            return start_time <= current_time_int <= end_time
        else:  # Session spans midnight
            return current_time_int >= start_time or current_time_int <= end_time
    except:
        return True

# Send order to MT5
def send_order(symbol, order_type, volume, sl, tp, comment=""):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
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
    symbol = "EURUSDm"
    timeframe = mt5.TIMEFRAME_M1
    key_value = 3
    atr_period = 14
    use_heikin_ashi = False
    risk_percent = 2.5
    ema_period = 200
    time_session = "0000-2400"  # 24-hour trading
    
    print("Starting UT Bot Alerts strategy")
    
    while True:
        try:
            # Check if we're in trading session
            if not in_trading_session(time_session):
                print("Outside trading session, waiting...")
                time.sleep(60)
                continue
            
            # Get market data
            df = get_market_data(symbol, timeframe, 300)
            if df.empty:
                print("No data received, waiting...")
                time.sleep(60)
                continue
            
            # Calculate indicators
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # Calculate EMA
            ema = talib.EMA(close_prices, timeperiod=ema_period)
            
            # Calculate ATR trailing stop
            atr_trailing_stop, pos = calculate_atr_trailing_stop(
                close_prices, high_prices, low_prices, 
                key_value, atr_period, use_heikin_ashi
            )
            
            # Get current values
            current_close = close_prices[-1]
            current_ema = ema[-1]
            current_atr_trailing = atr_trailing_stop[-1]
            current_pos = pos[-1]
            prev_pos = pos[-2]
            
            # Check trend condition
            bullish = current_close > current_ema
            bearish = current_close < current_ema
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                time.sleep(60)
                continue
            
            account_balance = account_info.balance
            positions = mt5.positions_get(symbol=symbol)
            
            # Check for buy signal
            if (current_pos == 1 and prev_pos != 1 and bullish and 
                not any(p.type == mt5.ORDER_TYPE_BUY for p in positions)):
                
                # Calculate stop loss and take profit
                atr_value = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)[-1]
                sl = current_close - atr_value * 1.5
                tp = current_close + atr_value * 3.0
                
                # Calculate position size
                volume = calculate_position_size(symbol, current_close, sl, risk_percent, account_balance)
                
                # Send buy order
                result = send_order(symbol, mt5.ORDER_TYPE_BUY, volume, sl, tp, "UT Bot Buy")
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Buy order failed: {result.retcode}")
                else:
                    print(f"Buy order executed: {volume} lots")
            
            # Check for sell signal
            elif (current_pos == -1 and prev_pos != -1 and bearish and 
                  not any(p.type == mt5.ORDER_TYPE_SELL for p in positions)):
                
                # Calculate stop loss and take profit
                atr_value = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)[-1]
                sl = current_close + atr_value * 1.5
                tp = current_close - atr_value * 3.0
                
                # Calculate position size
                volume = calculate_position_size(symbol, current_close, sl, risk_percent, account_balance)
                
                # Send sell order
                result = send_order(symbol, mt5.ORDER_TYPE_SELL, volume, sl, tp, "UT Bot Sell")
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Sell order failed: {result.retcode}")
                else:
                    print(f"Sell order executed: {volume} lots")
            
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
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in strategy: {e}")
            time.sleep(60)

# Main execution
if __name__ == "__main__":
    if initialize_mt5():
        run_strategy()
    else:
        print("Failed to initialize MT5 connection")
    
    mt5.shutdown()