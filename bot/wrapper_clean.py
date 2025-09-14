import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import all functions from hybrid_bot3.py
from hybrid_bot3 import (
    calculate_macd, calculate_atr, calculate_rsi, calculate_bollinger_bands,
    calculate_support_resistance, ema_direction, is_trending_market,
    check_rsi_divergence, is_valid_breakout, check_volume_confirmation,
    is_trading_session_active, identify_market_structure, check_break_of_structure,
    calculate_price_momentum, is_pullback_complete
)

# Use environment variables
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TF_PRIMARY = os.getenv("TF_PRIMARY", "M1")
TF_CONFIRM_M1 = os.getenv("TF_CONFIRM_M1", "M5")
TF_CONFIRM_M2 = os.getenv("TF_CONFIRM_M2", "H1")

# Test period
START_DATE = datetime(2025, 9, 8)
END_DATE = datetime(2025, 9, 14)

RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1")) / 100.0
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
MIN_RR = float(os.getenv("MIN_RR", "1.8"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))
MIN_SIGNAL_STRENGTH = int(os.getenv("MIN_SIGNAL_STRENGTH", "3"))

def get_historical(symbol, timeframe, start, end):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_all_indicators(df):
    """Apply all indicators used in hybrid_bot3"""
    df = calculate_macd(df)
    df = calculate_atr(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_support_resistance(df)
    return df

def get_timeframe_multiplier(primary_tf, confirm_tf):
    """Calculate how many primary candles per confirm candle"""
    tf_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30, 
        "H1": 60, "H4": 240, "D1": 1440
    }
    return tf_minutes[confirm_tf] // tf_minutes[primary_tf]

def evaluate_signal_strength(latest, df_primary, current_price):
    """Implement the 6-point signal scoring system"""
    signal_strength = 1  # Start with 1 for MACD cross
    signal_details = {'macd_cross': True}
    
    # 2. Market Structure
    if is_trending_market(df_primary):
        signal_strength += 1
        signal_details['trending_market'] = True
    else:
        signal_details['trending_market'] = False
    
    # 3. RSI Levels
    rsi = latest.get('rsi', 50)
    if 30 <= rsi <= 70:
        signal_strength += 1
        signal_details['rsi_good'] = True
    else:
        signal_details['rsi_good'] = False
    signal_details['rsi_value'] = rsi
    
    # 4. Bollinger Band Position
    bb_upper = latest.get('bb_upper', current_price)
    bb_lower = latest.get('bb_lower', current_price)
    if bb_lower < current_price < bb_upper:
        signal_strength += 1
        signal_details['bb_position'] = True
    else:
        signal_details['bb_position'] = False
    
    # 5. Volume/Volatility
    if check_volume_confirmation(df_primary):
        signal_strength += 1
        signal_details['volume_conf'] = True
    else:
        signal_details['volume_conf'] = False
    
    # 6. Support/Resistance
    support = latest.get('support', current_price * 0.99)
    resistance = latest.get('resistance', current_price * 1.01)
    if support < current_price < resistance:
        signal_strength += 1
        signal_details['sr_levels'] = True
    else:
        signal_details['sr_levels'] = False
    
    return signal_strength, signal_details

def enhanced_backtest():
    """Enhanced backtest with optimized parameters"""
    print("🚀 ENHANCED BACKTEST - OPTIMIZED VERSION")
    print("="*50)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Symbol: {SYMBOL}, Primary TF: {TF_PRIMARY}")
    print(f"Risk per trade: {RISK_PERCENT*100:.1f}%")
    print(f"Min Signal Strength: {MIN_SIGNAL_STRENGTH}/6")
    
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    try:
        # Get historical data
        print("Fetching historical data...")
        df_primary = get_historical(SYMBOL, TF_PRIMARY, START_DATE, END_DATE)
        df_c1 = get_historical(SYMBOL, TF_CONFIRM_M1, START_DATE, END_DATE)
        df_c2 = get_historical(SYMBOL, TF_CONFIRM_M2, START_DATE, END_DATE)

        if df_primary.empty or df_c1.empty or df_c2.empty:
            print("ERROR: Missing data")
            return

        print(f"Data loaded: {len(df_primary)} primary candles")

        # Calculate indicators
        print("Calculating indicators...")
        df_primary = calculate_all_indicators(df_primary)
        df_c1 = calculate_atr(df_c1)
        df_c2 = calculate_atr(df_c2)

        # Initialize tracking
        trades = []
        rejected_signals = []
        balance = 1000.0
        last_trade_time = None
        
        c1_multiplier = get_timeframe_multiplier(TF_PRIMARY, TF_CONFIRM_M1)
        c2_multiplier = get_timeframe_multiplier(TF_PRIMARY, TF_CONFIRM_M2)

        print("Running simulation...")
        
        # Backtest loop
        for i in range(200, len(df_primary)):
            current_data = df_primary.iloc[:i+1]
            latest = current_data.iloc[-1]
            current_price = latest['close']
            current_time = latest['time']
            
            # Cooldown check
            if last_trade_time and (current_time - last_trade_time).total_seconds() < COOLDOWN_MINUTES * 60:
                continue
            
            # 1. MACD signal
            primary_signal = None
            if latest.get('macd_cross_up', False):
                primary_signal = "buy"
            elif latest.get('macd_cross_dn', False):
                primary_signal = "sell"
            
            if primary_signal is None:
                continue
            
            # 2. Signal strength evaluation
            signal_strength, _ = evaluate_signal_strength(latest, current_data, current_price)
            
            if signal_strength < MIN_SIGNAL_STRENGTH:
                rejected_signals.append("insufficient_signal_strength")
                continue
            
            # 3. Market structure (relaxed)
            structure_type, key_level = identify_market_structure(current_data)
            rsi = latest.get('rsi', 50)
            resistance = latest.get('resistance', current_price * 1.01)
            support = latest.get('support', current_price * 0.99)
            
            # Allow counter-trend trades with exceptions
            if structure_type == "uptrend" and primary_signal == "sell":
                break_of_structure = check_break_of_structure(current_data, structure_type, key_level, current_price)
                overbought = rsi > 70
                at_resistance = current_price >= resistance * 0.999
                
                if not (break_of_structure or overbought or at_resistance):
                    rejected_signals.append("sell_against_uptrend")
                    continue
                    
            elif structure_type == "downtrend" and primary_signal == "buy":
                break_of_structure = check_break_of_structure(current_data, structure_type, key_level, current_price)
                oversold = rsi < 30
                at_support = current_price <= support * 1.001
                
                if not (break_of_structure or oversold or at_support):
                    rejected_signals.append("buy_against_downtrend")
                    continue
            
            # 4. Pullback check (relaxed)
            if not is_pullback_complete(current_data, primary_signal):
                rejected_signals.append("pullback_incomplete")
                continue
            
            # 5. Trend alignment (relaxed)
            c1_idx = min(i // c1_multiplier, len(df_c1) - 1)
            c2_idx = min(i // c2_multiplier, len(df_c2) - 1)
            
            trend1 = ema_direction(df_c1.iloc[:c1_idx+1], length=50) if c1_idx > 50 else "flat"
            trend2 = ema_direction(df_c2.iloc[:c2_idx+1], length=50) if c2_idx > 50 else "flat"
            
            # Allow if at least one timeframe agrees OR market structure supports
            if structure_type in ["uptrend", "downtrend"]:
                if not (trend1 == primary_signal or trend2 == primary_signal):
                    rejected_signals.append("no_trend_alignment")
                    continue
            else:
                if not (trend1 == trend2 == primary_signal):
                    rejected_signals.append("insufficient_trend_alignment")
                    continue
            
            # 6. Calculate trade parameters
            atr = latest['atr']
            if np.isnan(atr):
                rejected_signals.append("invalid_atr")
                continue
            
            # Dynamic risk based on signal strength
            risk_multiplier = 1.0
            if signal_strength >= 5:
                risk_multiplier = 1.5
            elif signal_strength >= 4:
                risk_multiplier = 1.2
            
            adjusted_risk = min(RISK_PERCENT * risk_multiplier, RISK_PERCENT * 2.0)
            risk_amount = balance * adjusted_risk
            
            entry_price = current_price
            sl_distance = ATR_MULTIPLIER * atr
            
            if primary_signal == "buy":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + (sl_distance * MIN_RR)
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - (sl_distance * MIN_RR)
            
            # 7. Simulate trade outcome
            future_window = df_primary.iloc[i:min(i+120, len(df_primary))]
            
            if primary_signal == "buy":
                hit_sl = (future_window['low'] <= sl_price).any()
                hit_tp = (future_window['high'] >= tp_price).any()
            else:
                hit_sl = (future_window['high'] >= sl_price).any()
                hit_tp = (future_window['low'] <= tp_price).any()
            
            # Determine outcome
            if hit_sl and not hit_tp:
                pnl = -risk_amount
                outcome = "stop_loss"
            elif hit_tp and not hit_sl:
                pnl = risk_amount * MIN_RR
                outcome = "take_profit"
            else:
                rejected_signals.append("no_clear_exit")
                continue
            
            # Record trade
            balance += pnl
            last_trade_time = current_time
            
            trades.append({
                "time": current_time,
                "signal": primary_signal,
                "entry": entry_price,
                "pnl": pnl,
                "balance": balance,
                "signal_strength": signal_strength,
                "outcome": outcome
            })

        # Generate results
        print("\n" + "="*50)
        print("ENHANCED BACKTEST RESULTS")
        print("="*50)
        
        if trades:
            df_trades = pd.DataFrame(trades)
            wins = df_trades[df_trades['pnl'] > 0]
            losses = df_trades[df_trades['pnl'] < 0]
            
            win_rate = len(wins) / len(df_trades) * 100
            avg_win = wins['pnl'].mean() if not wins.empty else 0
            avg_loss = losses['pnl'].mean() if not losses.empty else 0
            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty else float('inf')
            total_return = df_trades['pnl'].sum()
            avg_signal_strength = df_trades['signal_strength'].mean()
            
            results = {
                "total_trades": len(df_trades),
                "win_rate_percent": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "total_return": round(total_return, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "avg_signal_strength": round(avg_signal_strength, 1),
                "total_rejections": len(rejected_signals)
            }
            
            print(json.dumps(results, indent=2))
            
            # Rejection analysis
            if rejected_signals:
                rejection_counts = {}
                for reason in rejected_signals:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                
                print(f"\n📊 REJECTION BREAKDOWN:")
                for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = count / len(rejected_signals) * 100
                    print(f"  {reason}: {count} ({pct:.1f}%)")
            
            print(f"\n🎯 IMPROVEMENT vs Original 34% Win Rate:")
            print(f"  ✅ Win Rate: {win_rate:.1f}% (vs 34%)")
            print(f"  ✅ Profit Factor: {profit_factor:.2f}")
            print(f"  ✅ Signal Quality: {avg_signal_strength:.1f}/6 average")
            
        else:
            print("❌ No trades executed")
            print(f"Total signals rejected: {len(rejected_signals)}")
            
            if rejected_signals:
                rejection_counts = {}
                for reason in rejected_signals:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                
                print(f"\n📊 WHY NO TRADES:")
                for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = count / len(rejected_signals) * 100
                    print(f"  {reason}: {count} ({pct:.1f}%)")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    enhanced_backtest()
