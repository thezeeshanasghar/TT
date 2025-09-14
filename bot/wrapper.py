import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import MetaTrader5 as mt5

# --- Import your HybridBot3 logic here ---
from hybrid_bot3 import calculate_macd, calculate_atr, ema_direction

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

SYMBOL = "XAUUSD"
TF_PRIMARY = "M1"
TF_CONFIRM_M1 = "M5"
TF_CONFIRM_M2 = "H1"

START_DATE = datetime(2025, 9, 8)
END_DATE   = datetime(2025, 9, 12)

RISK_PERCENT = 0.008
ATR_MULTIPLIER = 1.5
MIN_RR = 1.8

def get_historical(symbol, timeframe, start, end):
    """Fetch historical data between two dates."""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }
    rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def backtest():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    df_primary = get_historical(SYMBOL, TF_PRIMARY, START_DATE, END_DATE)
    df_c1 = get_historical(SYMBOL, TF_CONFIRM_M1, START_DATE, END_DATE)
    df_c2 = get_historical(SYMBOL, TF_CONFIRM_M2, START_DATE, END_DATE)

    if df_primary.empty or df_c1.empty or df_c2.empty:
        print("Missing historical data.")
        return

    # Pre-calc indicators
    df_primary = calculate_macd(df_primary)
    df_primary = calculate_atr(df_primary)

    trades = []
    balance = 1000.0  # start equity

    for i in range(200, len(df_primary)):  # start after warmup
        window = df_primary.iloc[:i+1]
        latest = window.iloc[-1]

        # Primary MACD signal
        signal = None
        if latest['macd_cross_up']:
            signal = "buy"
        elif latest['macd_cross_dn']:
            signal = "sell"

        if signal is None:
            continue

        # Trend confirmation
        trend1 = ema_direction(df_c1.iloc[:i//5+1], length=50)
        trend2 = ema_direction(df_c2.iloc[:i//60+1], length=50)

        if not (signal == trend1 == trend2):
            continue

        atr = latest['atr']
        if np.isnan(atr):
            continue

        entry_price = latest['close']
        sl_dist = atr * ATR_MULTIPLIER
        if signal == "buy":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * MIN_RR
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * MIN_RR

        rr_est = abs((tp - entry_price) / (entry_price - sl))
        if rr_est < MIN_RR:
            continue

        # Risk sizing (0.8% of balance)
        risk_amt = balance * RISK_PERCENT
        lot = risk_amt / (sl_dist * 10)  # simplified contract calc
        lot = max(lot, 0.01)

        # Simulate outcome
        future_window = df_primary.iloc[i:i+60]  # next 60 bars (~1h for M1)
        hit_sl = (future_window['low'] <= sl).any() if signal == "buy" else (future_window['high'] >= sl).any()
        hit_tp = (future_window['high'] >= tp).any() if signal == "buy" else (future_window['low'] <= tp).any()

        pnl = 0
        if hit_sl and not hit_tp:
            pnl = -risk_amt
        elif hit_tp and not hit_sl:
            pnl = risk_amt * MIN_RR
        elif hit_tp and hit_sl:
            # whichever came first
            first_hit = future_window.apply(lambda r: sl <= r['low'] if signal=="buy" else sl >= r['high'], axis=1).idxmax()
            pnl = -risk_amt if first_hit else risk_amt * MIN_RR
        else:
            continue  # no resolution

        balance += pnl
        trades.append({
            "time": latest['time'],
            "signal": signal,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "pnl": pnl,
            "balance": balance
        })

    mt5.shutdown()

    results = pd.DataFrame(trades)
    print("\n=== Backtest Results ===")
    print(results.tail())
    print(f"\nFinal Balance: {balance:.2f}")
    print(f"Total Trades: {len(results)}")
    print(f"Win Rate: {(results['pnl']>0).mean()*100:.1f}%")

if __name__ == "__main__":
    backtest()
