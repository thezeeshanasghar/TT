import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import json
from dotenv import load_dotenv
import os
from itertools import product

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

# Multiple test periods for robust optimization
TEST_PERIODS = [
    {"name": "Week1", "start": datetime(2025, 9, 8), "end": datetime(2025, 9, 14)},
    {"name": "Week2", "start": datetime(2025, 9, 1), "end": datetime(2025, 9, 7)},
    {"name": "Week3", "start": datetime(2025, 8, 25), "end": datetime(2025, 8, 31)},
    {"name": "Week4", "start": datetime(2025, 8, 18), "end": datetime(2025, 8, 24)},
    {"name": "Month", "start": datetime(2025, 8, 15), "end": datetime(2025, 9, 14)},
]

# Parameter ranges for optimization
OPTIMIZATION_GRID = {
    "MIN_SIGNAL_STRENGTH": [2, 3, 4],
    "COOLDOWN_MINUTES": [15, 20, 30, 45],
    "ATR_MULTIPLIER": [1.2, 1.5, 1.8, 2.0],
    "MIN_RR": [1.5, 1.8, 2.0, 2.5],
    "REQUIRE_STRICT_TREND": [True, False],  # New parameter for trend alignment
}

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

def run_backtest_with_params(start_date, end_date, params):
    """Run backtest with specific parameter set"""
    RISK_PERCENT = 0.01  # Fixed 1%
    
    if not mt5.initialize():
        return None

    try:
        # Get historical data
        df_primary = get_historical(SYMBOL, TF_PRIMARY, start_date, end_date)
        df_c1 = get_historical(SYMBOL, TF_CONFIRM_M1, start_date, end_date)
        df_c2 = get_historical(SYMBOL, TF_CONFIRM_M2, start_date, end_date)

        if df_primary.empty or df_c1.empty or df_c2.empty:
            return None

        # Calculate indicators
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
        
        # Backtest loop
        for i in range(200, len(df_primary)):
            current_data = df_primary.iloc[:i+1]
            latest = current_data.iloc[-1]
            current_price = latest['close']
            current_time = latest['time']
            
            # Cooldown check
            if last_trade_time and (current_time - last_trade_time).total_seconds() < params['COOLDOWN_MINUTES'] * 60:
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
            
            if signal_strength < params['MIN_SIGNAL_STRENGTH']:
                rejected_signals.append("insufficient_signal_strength")
                continue
            
            # 3. Market structure (relaxed)
            structure_type, key_level = identify_market_structure(current_data)
            rsi = latest.get('rsi', 50)
            resistance = latest.get('resistance', current_price * 1.01)
            support = latest.get('support', current_price * 0.99)
            
            # Counter-trend filtering
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
            
            # 5. Trend alignment (configurable strictness)
            c1_idx = min(i // c1_multiplier, len(df_c1) - 1)
            c2_idx = min(i // c2_multiplier, len(df_c2) - 1)
            
            trend1 = ema_direction(df_c1.iloc[:c1_idx+1], length=50) if c1_idx > 50 else "flat"
            trend2 = ema_direction(df_c2.iloc[:c2_idx+1], length=50) if c2_idx > 50 else "flat"
            
            if params['REQUIRE_STRICT_TREND']:
                # Strict: require both timeframes to agree
                if not (trend1 == trend2 == primary_signal):
                    rejected_signals.append("strict_trend_alignment_failed")
                    continue
            else:
                # Relaxed: allow if at least one agrees OR market structure supports
                if structure_type in ["uptrend", "downtrend"]:
                    if not (trend1 == primary_signal or trend2 == primary_signal):
                        rejected_signals.append("no_trend_alignment")
                        continue
                else:
                    if not (trend1 == primary_signal or trend2 == primary_signal):
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
            sl_distance = params['ATR_MULTIPLIER'] * atr
            
            if primary_signal == "buy":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + (sl_distance * params['MIN_RR'])
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - (sl_distance * params['MIN_RR'])
            
            # 7. Simulate trade outcome
            future_window = df_primary.iloc[i:min(i+120, len(df_primary))]
            
            if primary_signal == "buy":
                hit_sl = (future_window['low'] <= sl_price).any()
                hit_tp = (future_window['high'] >= tp_price).any()
                
                # Check which is hit first
                if hit_sl and hit_tp:
                    sl_idx = future_window[future_window['low'] <= sl_price].index[0]
                    tp_idx = future_window[future_window['high'] >= tp_price].index[0]
                    hit_sl_first = sl_idx < tp_idx
                else:
                    hit_sl_first = hit_sl
            else:
                hit_sl = (future_window['high'] >= sl_price).any()
                hit_tp = (future_window['low'] <= tp_price).any()
                
                # Check which is hit first
                if hit_sl and hit_tp:
                    sl_idx = future_window[future_window['high'] >= sl_price].index[0]
                    tp_idx = future_window[future_window['low'] <= tp_price].index[0]
                    hit_sl_first = sl_idx < tp_idx
                else:
                    hit_sl_first = hit_sl
            
            # Determine outcome
            if hit_sl_first:
                pnl = -risk_amount
                outcome = "stop_loss"
            elif hit_tp:
                pnl = risk_amount * params['MIN_RR']
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

        # Calculate performance metrics
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_return": 0,
                "max_dd": 0,
                "sharpe": 0,
                "avg_signal_strength": 0,
                "score": 0
            }

        df_trades = pd.DataFrame(trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]
        
        win_rate = len(wins) / len(df_trades) * 100
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
        total_return = df_trades['pnl'].sum()
        
        # Calculate max drawdown
        equity_curve = df_trades['balance']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_dd = abs(drawdown.min())
        
        # Calculate Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
        
        avg_signal_strength = df_trades['signal_strength'].mean()
        
        # Composite score for optimization
        # Prioritize: win_rate, profit_factor, total_return, minimize max_dd
        if profit_factor == float('inf'):
            profit_factor = 10  # Cap for scoring
        
        score = (win_rate * 0.3 + 
                min(profit_factor, 5) * 20 * 0.25 + 
                max(total_return, 0) * 0.02 * 0.25 + 
                np.log(len(df_trades) + 1) * 10 * 0.1 +
                avg_signal_strength * 3 * 0.1) / max(max_dd / 10 + 1, 1)
        
        return {
            "total_trades": len(df_trades),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "total_return": round(total_return, 2),
            "max_dd": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "avg_signal_strength": round(avg_signal_strength, 1),
            "score": round(score, 2)
        }

    except Exception as e:
        print(f"Error in backtest: {e}")
        return None
    finally:
        mt5.shutdown()

def optimize_parameters():
    """Comprehensive parameter optimization"""
    print("🔧 COMPREHENSIVE PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Testing {len(list(product(*OPTIMIZATION_GRID.values())))} parameter combinations")
    print(f"Across {len(TEST_PERIODS)} time periods")
    print("This may take a few minutes...\n")
    
    best_configs = []
    all_results = {}
    
    # Generate all parameter combinations
    param_names = list(OPTIMIZATION_GRID.keys())
    param_values = list(OPTIMIZATION_GRID.values())
    
    total_combinations = len(list(product(*param_values)))
    current_combo = 0
    
    for param_combo in product(*param_values):
        current_combo += 1
        params = dict(zip(param_names, param_combo))
        
        if current_combo % 10 == 0:
            print(f"Progress: {current_combo}/{total_combinations} ({current_combo/total_combinations*100:.1f}%)")
        
        config_name = f"SS{params['MIN_SIGNAL_STRENGTH']}_CD{params['COOLDOWN_MINUTES']}_ATR{params['ATR_MULTIPLIER']}_RR{params['MIN_RR']}_ST{params['REQUIRE_STRICT_TREND']}"
        
        period_scores = []
        period_results = {}
        
        for period in TEST_PERIODS:
            result = run_backtest_with_params(period['start'], period['end'], params)
            
            if result:
                period_results[period['name']] = result
                # Only include periods with at least some trades for scoring
                if result['total_trades'] > 0:
                    period_scores.append(result['score'])
        
        # Calculate average score across periods
        avg_score = np.mean(period_scores) if period_scores else 0
        
        config_summary = {
            'params': params,
            'avg_score': round(avg_score, 2),
            'period_results': period_results,
            'periods_with_trades': len(period_scores)
        }
        
        all_results[config_name] = config_summary
        
        # Keep track of best configurations
        if avg_score > 0:
            best_configs.append((config_name, avg_score, params))
    
    # Sort by score
    best_configs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 TOP 10 PARAMETER COMBINATIONS")
    print("="*60)
    
    for i, (config_name, score, params) in enumerate(best_configs[:10], 1):
        print(f"\n{i}. Score: {score:.2f}")
        print(f"   Signal Strength: {params['MIN_SIGNAL_STRENGTH']}/6")
        print(f"   Cooldown: {params['COOLDOWN_MINUTES']} min")
        print(f"   ATR Multiplier: {params['ATR_MULTIPLIER']}")
        print(f"   Min R:R: {params['MIN_RR']}")
        print(f"   Strict Trend: {params['REQUIRE_STRICT_TREND']}")
        
        # Show performance across periods
        period_data = all_results[config_name]['period_results']
        total_trades = sum(p.get('total_trades', 0) for p in period_data.values())
        avg_win_rate = np.mean([p.get('win_rate', 0) for p in period_data.values() if p.get('total_trades', 0) > 0])
        avg_profit_factor = np.mean([p.get('profit_factor', 0) for p in period_data.values() if p.get('total_trades', 0) > 0 and p.get('profit_factor', 0) != float('inf')])
        
        print(f"   Avg Win Rate: {avg_win_rate:.1f}%")
        print(f"   Avg Profit Factor: {avg_profit_factor:.2f}")
        print(f"   Total Trades: {total_trades}")
    
    # Get the best configuration
    if best_configs:
        best_config_name, best_score, best_params = best_configs[0]
        
        print(f"\n🎯 OPTIMAL .ENV CONFIGURATION:")
        print("="*40)
        print(f"MIN_SIGNAL_STRENGTH={best_params['MIN_SIGNAL_STRENGTH']}")
        print(f"COOLDOWN_MINUTES={best_params['COOLDOWN_MINUTES']}")
        print(f"ATR_MULTIPLIER={best_params['ATR_MULTIPLIER']}")
        print(f"MIN_RR={best_params['MIN_RR']}")
        print(f"# REQUIRE_STRICT_TREND={best_params['REQUIRE_STRICT_TREND']} (code parameter)")
        
        # Detailed results for best config
        print(f"\n📊 DETAILED RESULTS FOR BEST CONFIG:")
        print("="*40)
        best_period_results = all_results[best_config_name]['period_results']
        
        for period_name, results in best_period_results.items():
            if results['total_trades'] > 0:
                print(f"\n{period_name}:")
                print(f"  Trades: {results['total_trades']}")
                print(f"  Win Rate: {results['win_rate']}%")
                print(f"  Profit Factor: {results['profit_factor']}")
                print(f"  Return: ${results['total_return']}")
                print(f"  Max DD: {results['max_dd']}%")
                print(f"  Signal Quality: {results['avg_signal_strength']}/6")
    
    # Save full results to file
    with open('optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to 'optimization_results.json'")
    return all_results, best_configs

if __name__ == "__main__":
    optimize_parameters()
