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
    calculate_support_resistance, calculate_supertrend, ema_direction, is_trending_market,
    check_rsi_divergence, is_valid_breakout, check_volume_confirmation,
    is_trading_session_active, identify_market_structure, check_break_of_structure,
    calculate_price_momentum, is_pullback_complete,
    detect_order_blocks, detect_fair_value_gaps, detect_break_of_structure,
    detect_equal_highs_lows, detect_premium_discount_zones
)

# Use environment variables
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TF_PRIMARY = os.getenv("TF_PRIMARY", "M1")
TF_CONFIRM_M1 = os.getenv("TF_CONFIRM_M1", "M5")
TF_CONFIRM_M2 = os.getenv("TF_CONFIRM_M2", "H1")

# Test period
START_DATE = datetime(2025, 9, 15)
END_DATE = datetime(2025, 9, 19)

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
    df = calculate_supertrend(df)  # Add SuperTrend
    return df

def get_timeframe_multiplier(primary_tf, confirm_tf):
    """Calculate how many primary candles per confirm candle"""
    tf_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30, 
        "H1": 60, "H4": 240, "D1": 1440
    }
    return tf_minutes[confirm_tf] // tf_minutes[primary_tf]

def evaluate_signal_strength_smc(latest, df_primary, current_price, primary_signal, 
                                order_blocks, fvgs, bos_choch, equal_highs, equal_lows,
                                premium_zone, discount_zone, equilibrium_zone):
    """Enhanced signal strength evaluation with SMC concepts (12+ point system)"""
    signal_strength = 0
    signal_reasons = []
    signal_details = {}
    
    # 1. MACD Cross (already detected, so we count it)
    signal_strength += 1
    signal_reasons.append("MACD cross")
    signal_details['macd_cross'] = True
    
    # 2. SuperTrend Confirmation (Strong signal)
    if latest.get('supertrend_buy', False) and primary_signal == "buy":
        signal_strength += 2
        signal_reasons.append("SuperTrend buy signal")
        signal_details['supertrend_aligned'] = True
    elif latest.get('supertrend_sell', False) and primary_signal == "sell":
        signal_strength += 2
        signal_reasons.append("SuperTrend sell signal")
        signal_details['supertrend_aligned'] = True
    elif latest.get('supertrend_signal', 0) == 1 and primary_signal == "buy":
        signal_strength += 1
        signal_reasons.append("SuperTrend bullish")
        signal_details['supertrend_aligned'] = True
    elif latest.get('supertrend_signal', 0) == -1 and primary_signal == "sell":
        signal_strength += 1
        signal_reasons.append("SuperTrend bearish")
        signal_details['supertrend_aligned'] = True
    else:
        signal_details['supertrend_aligned'] = False
    
    # 3. Break of Structure / Change of Character
    if bos_choch:
        if bos_choch == 'bullish_bos' and primary_signal == "buy":
            signal_strength += 2
            signal_reasons.append("Bullish BOS")
            signal_details['bos_choch'] = 'bullish_bos'
        elif bos_choch == 'bearish_bos' and primary_signal == "sell":
            signal_strength += 2
            signal_reasons.append("Bearish BOS")
            signal_details['bos_choch'] = 'bearish_bos'
        elif bos_choch == 'bullish_choch' and primary_signal == "buy":
            signal_strength += 3
            signal_reasons.append("Bullish CHoCH")
            signal_details['bos_choch'] = 'bullish_choch'
        elif bos_choch == 'bearish_choch' and primary_signal == "sell":
            signal_strength += 3
            signal_reasons.append("Bearish CHoCH")
            signal_details['bos_choch'] = 'bearish_choch'
    
    # 4. Market Structure
    if is_trending_market(df_primary):
        signal_strength += 1
        signal_reasons.append("Trending market")
        signal_details['trending_market'] = True
    else:
        signal_details['trending_market'] = False
    
    # 5. RSI Levels
    rsi = latest.get('rsi', 50)
    if primary_signal == "buy" and 30 <= rsi <= 70:
        signal_strength += 1
        signal_reasons.append(f"RSI good ({rsi:.1f})")
        signal_details['rsi_good'] = True
    elif primary_signal == "sell" and 30 <= rsi <= 70:
        signal_strength += 1
        signal_reasons.append(f"RSI good ({rsi:.1f})")
        signal_details['rsi_good'] = True
    else:
        signal_details['rsi_good'] = False
    signal_details['rsi_value'] = rsi
    
    # 6. Bollinger Band Position
    bb_upper = latest.get('bb_upper', current_price)
    bb_lower = latest.get('bb_lower', current_price)
    if primary_signal == "buy" and current_price < bb_upper:
        signal_strength += 1
        signal_reasons.append("BB position good")
        signal_details['bb_position'] = True
    elif primary_signal == "sell" and current_price > bb_lower:
        signal_strength += 1
        signal_reasons.append("BB position good")
        signal_details['bb_position'] = True
    else:
        signal_details['bb_position'] = False
    
    # 7. Volume/Volatility
    if check_volume_confirmation(df_primary):
        signal_strength += 1
        signal_reasons.append("Volume confirmed")
        signal_details['volume_conf'] = True
    else:
        signal_details['volume_conf'] = False
    
    # 8. Support/Resistance
    support = latest.get('support', current_price * 0.99)
    resistance = latest.get('resistance', current_price * 1.01)
    if primary_signal == "buy" and current_price > support * 1.001:
        signal_strength += 1
        signal_reasons.append("Above support")
        signal_details['sr_levels'] = True
    elif primary_signal == "sell" and current_price < resistance * 0.999:
        signal_strength += 1
        signal_reasons.append("Below resistance")
        signal_details['sr_levels'] = True
    else:
        signal_details['sr_levels'] = False
    
    # 9. Order Block Confirmation
    signal_details['order_block_aligned'] = False
    for ob in order_blocks:
        if primary_signal == "buy" and ob['type'] == 'bullish':
            if ob['bottom'] <= current_price <= ob['top'] * 1.01:
                signal_strength += 2
                signal_reasons.append("At bullish order block")
                signal_details['order_block_aligned'] = True
                break
        elif primary_signal == "sell" and ob['type'] == 'bearish':
            if ob['bottom'] * 0.99 <= current_price <= ob['top']:
                signal_strength += 2
                signal_reasons.append("At bearish order block")
                signal_details['order_block_aligned'] = True
                break
    
    # 10. Fair Value Gap Support
    signal_details['fvg_support'] = False
    for fvg in fvgs[-5:]:  # Check last 5 FVGs
        if primary_signal == "buy" and fvg['type'] == 'bullish':
            if fvg['bottom'] <= current_price <= fvg['top'] * 1.02:
                signal_strength += 1
                signal_reasons.append("Bullish FVG support")
                signal_details['fvg_support'] = True
                break
        elif primary_signal == "sell" and fvg['type'] == 'bearish':
            if fvg['bottom'] * 0.98 <= current_price <= fvg['top']:
                signal_strength += 1
                signal_reasons.append("Bearish FVG resistance")
                signal_details['fvg_support'] = True
                break
    
    # 11. Equal Highs/Lows
    signal_details['equal_levels'] = False
    if equal_highs and primary_signal == "sell":
        for eq_high in equal_highs[-3:]:
            if abs(current_price - eq_high[0]) / current_price < 0.002:
                signal_strength += 1
                signal_reasons.append("At equal highs")
                signal_details['equal_levels'] = True
                break
    elif equal_lows and primary_signal == "buy":
        for eq_low in equal_lows[-3:]:
            if abs(current_price - eq_low[0]) / current_price < 0.002:
                signal_strength += 1
                signal_reasons.append("At equal lows")
                signal_details['equal_levels'] = True
                break
    
    # 12. Premium/Discount Zone
    if premium_zone and discount_zone and equilibrium_zone:
        if primary_signal == "buy":
            if discount_zone[0] <= current_price <= discount_zone[1]:
                signal_strength += 2
                signal_reasons.append("In discount zone")
                signal_details['zone_optimal'] = True
            elif equilibrium_zone[0] <= current_price <= equilibrium_zone[1]:
                signal_strength += 1
                signal_reasons.append("At equilibrium")
                signal_details['zone_neutral'] = True
            elif premium_zone[0] <= current_price <= premium_zone[1]:
                signal_strength -= 1
                signal_reasons.append("In premium zone (risky)")
                signal_details['zone_risky'] = True
        elif primary_signal == "sell":
            if premium_zone[0] <= current_price <= premium_zone[1]:
                signal_strength += 2
                signal_reasons.append("In premium zone")
                signal_details['zone_optimal'] = True
            elif equilibrium_zone[0] <= current_price <= equilibrium_zone[1]:
                signal_strength += 1
                signal_reasons.append("At equilibrium")
                signal_details['zone_neutral'] = True
            elif discount_zone[0] <= current_price <= discount_zone[1]:
                signal_strength -= 1
                signal_reasons.append("In discount zone (risky)")
                signal_details['zone_risky'] = True
    
    signal_details['total_strength'] = signal_strength
    signal_details['reasons'] = signal_reasons
    
    return signal_strength, signal_details

def enhanced_backtest():
    """Enhanced backtest with SMC indicators"""
    print("🚀 ENHANCED BACKTEST - WITH SMART MONEY CONCEPTS")
    print("="*50)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Symbol: {SYMBOL}, Primary TF: {TF_PRIMARY}")
    print(f"Risk per trade: {RISK_PERCENT*100:.1f}%")
    print(f"Min Signal Strength: {MIN_SIGNAL_STRENGTH}/12+ (with SMC)")
    
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

        # Initialize SMC variables
        order_blocks = []
        fvgs = []
        equal_highs = []
        equal_lows = []
        premium_zone = None
        discount_zone = None
        equilibrium_zone = None

        print("Running simulation...")
        total_candles = len(df_primary) - 200
        
        # Backtest loop with progress tracking
        for iteration, i in enumerate(range(200, len(df_primary))):
            current_data = df_primary.iloc[:i+1]
            latest = current_data.iloc[-1]
            current_price = latest['close']
            current_time = latest['time']
            
            # Progress indicator every 10% of the way
            if iteration % (total_candles // 10) == 0:
                progress_pct = (iteration / total_candles) * 100
                print(f"Progress: {progress_pct:.0f}% ({iteration}/{total_candles} candles processed)")
            
            # Cooldown check
            if last_trade_time and (current_time - last_trade_time).total_seconds() < COOLDOWN_MINUTES * 60:
                continue
            
            # Calculate SMC indicators (optimized - every 5th candle for non-critical checks)
            if iteration % 5 == 0:  # Recalculate every 5 candles for performance
                order_blocks = detect_order_blocks(current_data)
                fvgs = detect_fair_value_gaps(current_data)
                equal_highs, equal_lows = detect_equal_highs_lows(current_data)
                premium_zone, discount_zone, equilibrium_zone = detect_premium_discount_zones(current_data)
            
            # Always check BOS/CHoCH as it's critical for signals
            bos_choch, bos_level = detect_break_of_structure(current_data)
            
            # 1. Primary signal detection (MACD + SuperTrend)
            primary_signal = None
            
            # MACD signal
            if latest.get('macd_cross_up', False):
                primary_signal = "buy"
            elif latest.get('macd_cross_dn', False):
                primary_signal = "sell"
            
            # SuperTrend can also initiate signals
            if primary_signal is None:
                if latest.get('supertrend_buy', False):
                    primary_signal = "buy"
                elif latest.get('supertrend_sell', False):
                    primary_signal = "sell"
            
            # CHoCH is a very strong signal
            if bos_choch == 'bullish_choch':
                primary_signal = "buy"
            elif bos_choch == 'bearish_choch':
                primary_signal = "sell"
            
            if primary_signal is None:
                continue
            
            # 2. Signal strength evaluation with SMC
            signal_strength, signal_details = evaluate_signal_strength_smc(
                latest, current_data, current_price, primary_signal,
                order_blocks, fvgs, bos_choch, equal_highs, equal_lows,
                premium_zone, discount_zone, equilibrium_zone
            )
            
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
            
            # Dynamic risk based on signal strength (adjusted for SMC)
            risk_multiplier = 1.0
            if signal_strength >= 8:  # Very strong signal with SMC confluence
                risk_multiplier = 1.8
            elif signal_strength >= 6:  # Strong signal
                risk_multiplier = 1.5
            elif signal_strength >= 4:  # Good signal
                risk_multiplier = 1.2
            
            # Extra boost for specific high-confidence setups
            if bos_choch and ('choch' in bos_choch.lower()):
                risk_multiplier *= 1.2  # CHoCH is very reliable
            if signal_details.get('order_block_aligned') and signal_details.get('fvg_support'):
                risk_multiplier *= 1.1  # Both SMC concepts align
            
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
                "outcome": outcome,
                "reasons": signal_details.get('reasons', []),
                "supertrend_aligned": signal_details.get('supertrend_aligned', False),
                "bos_choch": signal_details.get('bos_choch', None),
                "order_block": signal_details.get('order_block_aligned', False),
                "fvg": signal_details.get('fvg_support', False),
                "zone": 'premium' if signal_details.get('zone_optimal') and primary_signal == 'sell' 
                       else 'discount' if signal_details.get('zone_optimal') and primary_signal == 'buy'
                       else 'neutral',
                "risk_multiplier": risk_multiplier
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
            
            # SMC Analysis
            smc_trades = [t for t in trades if t.get('bos_choch') or t.get('order_block') or t.get('fvg')]
            supertrend_trades = [t for t in trades if t.get('supertrend_aligned')]
            choch_trades = [t for t in trades if t.get('bos_choch') and 'choch' in str(t.get('bos_choch')).lower()]
            
            print(f"\n📊 SMC ANALYSIS:")
            if smc_trades:
                smc_wins = [t for t in smc_trades if t['pnl'] > 0]
                smc_win_rate = len(smc_wins) / len(smc_trades) * 100
                print(f"  SMC-confirmed trades: {len(smc_trades)} ({smc_win_rate:.1f}% win rate)")
            
            if supertrend_trades:
                st_wins = [t for t in supertrend_trades if t['pnl'] > 0]
                st_win_rate = len(st_wins) / len(supertrend_trades) * 100
                print(f"  SuperTrend-aligned: {len(supertrend_trades)} ({st_win_rate:.1f}% win rate)")
            
            if choch_trades:
                choch_wins = [t for t in choch_trades if t['pnl'] > 0]
                choch_win_rate = len(choch_wins) / len(choch_trades) * 100 if choch_trades else 0
                print(f"  CHoCH trades: {len(choch_trades)} ({choch_win_rate:.1f}% win rate)")
            
            # Zone analysis
            premium_trades = [t for t in trades if t.get('zone') == 'premium']
            discount_trades = [t for t in trades if t.get('zone') == 'discount']
            
            if premium_trades or discount_trades:
                print(f"\n📍 ZONE ANALYSIS:")
                if premium_trades:
                    p_wins = [t for t in premium_trades if t['pnl'] > 0]
                    print(f"  Premium zone sells: {len(premium_trades)} ({len(p_wins)/len(premium_trades)*100:.1f}% win rate)")
                if discount_trades:
                    d_wins = [t for t in discount_trades if t['pnl'] > 0]
                    print(f"  Discount zone buys: {len(discount_trades)} ({len(d_wins)/len(discount_trades)*100:.1f}% win rate)")
            
            # Best performing signal combinations
            print(f"\n🏆 TOP SIGNAL COMBINATIONS:")
            signal_combos = {}
            for trade in trades:
                key = f"Strength {trade['signal_strength']}"
                if key not in signal_combos:
                    signal_combos[key] = {'count': 0, 'wins': 0, 'total_pnl': 0}
                signal_combos[key]['count'] += 1
                signal_combos[key]['total_pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    signal_combos[key]['wins'] += 1
            
            for combo, stats in sorted(signal_combos.items(), key=lambda x: x[1]['total_pnl'], reverse=True)[:5]:
                combo_wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
                print(f"  {combo}: {stats['count']} trades, {combo_wr:.1f}% WR, ${stats['total_pnl']:.2f} profit")
            
            print(f"\n🎯 IMPROVEMENT vs Original:")
            print(f"  ✅ Win Rate: {win_rate:.1f}% (vs 34% baseline)")
            print(f"  ✅ Profit Factor: {profit_factor:.2f}")
            print(f"  ✅ Signal Quality: {avg_signal_strength:.1f}/12+ average")
            print(f"  ✅ Risk-adjusted returns: {total_return/1000*100:.1f}% with dynamic sizing")
            
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
