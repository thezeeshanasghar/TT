#!/usr/bin/env python3
"""
Hybrid Bot 3
Combines rule-based MACD + S/R with optional LLM confirmation.
Test on demo first.
"""

import os
import sys
import time
import logging
from time import sleep
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Try to import MetaTrader5, fall back to mock if not available (e.g., on macOS)
try:
    import MetaTrader5 as mt5
    print("Using real MetaTrader5")
except ImportError:
    print("MetaTrader5 not available, using mock version for testing")
    import mock_mt5 as mt5

# Optional LLM libs (same style as Bot1). If LLM_PROVIDER != 'groq' we skip LLM.
try:
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
except Exception:
    ChatGroq = None
    PromptTemplate = None

# Load env
load_dotenv()

# Config
ACCOUNT = int(os.getenv("MT5_ACCOUNT", "0"))
PASSWORD = os.getenv("MT5_PASSWORD", "")
SERVER = os.getenv("MT5_SERVER", "")
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TF_PRIMARY = os.getenv("TF_PRIMARY", "M1")       # M1, M5, M15, M30, H1, etc.
TF_CONFIRM_M1 = os.getenv("TF_CONFIRM_M1", "M5")
TF_CONFIRM_M2 = os.getenv("TF_CONFIRM_M2", "H1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none") # 'groq' or 'none'
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.8")) / 100.0  # e.g. 0.008 for 0.8%
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "35"))  # points (broker-specific)
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
MIN_RR = float(os.getenv("MIN_RR", "1.8"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))  # Minimum time between trades
MIN_SIGNAL_STRENGTH = int(os.getenv("MIN_SIGNAL_STRENGTH", "3"))  # Minimum signal strength (1-6)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Map TF names to MT5 constants
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

# LLM prompt templates (lightweight) - for confirmation only
LLM_CONFIRM_PROMPT = """
You are an institutional XAUUSD assistant using Smart Money Concepts. Given the following data, reply ONLY with CONFIRM or REJECT and a short reason (<=30 words).

Primary Signal: {primary_signal} (Strength: {signal_strength})
Signal Reasons: {signal_reasons}
SuperTrend: {supertrend} (1=bullish, -1=bearish)
Market Structure: {market_structure}
BOS/CHoCH: {bos_choch}
Order Blocks nearby: {order_blocks}
Primary TF candles (last 8): {primary_data}
Confirm TF1 (recent direction): {confirm1}
Confirm TF2 (recent direction): {confirm2}
Current Spread: {spread}
Risk Params: SL distance {sl_dist:.4f}, ATR {atr:.4f}, RR target {min_rr}

Rules: Confirm only if signal has strong SMC confluence, trend alignment, and acceptable risk parameters.
"""

class LLMConfirmer:
    def __init__(self, provider="none", api_key=None):
        self.provider = provider
        self.api_key = api_key
        self.client = None
        if provider == "groq" and ChatGroq is not None:
            self.client = ChatGroq(temperature=0.0, model_name="deepseek-r1-distill-llama-70b", api_key=api_key)
        else:
            self.client = None
            if provider != "none":
                logging.warning("LLM provider specified but client library not available. LLM disabled.")

    def confirm(self, context: dict) -> dict:
        """Return {'decision': 'CONFIRM'|'REJECT', 'reason': str}"""
        if self.provider == "none" or self.client is None:
            # No LLM: fallback to deterministic quick check
            # Simple rule: primary_signal must be 'buy' or 'sell' and both confirm directions same
            if context['primary_signal'] in ("buy", "sell") and context['confirm1'] == context['confirm2'] == context['primary_signal']:
                return {"decision": "CONFIRM", "reason": "Heuristic: trend alignment"}
            return {"decision": "REJECT", "reason": "Heuristic: no alignment"}
        # Use LLM
        prompt = LLM_CONFIRM_PROMPT.format(
            primary_signal=context['primary_signal'],
            primary_data=context['primary_data'],
            confirm1=context['confirm1'],
            confirm2=context['confirm2'],
            spread=context['spread'],
            sl_dist=context['sl_dist'],
            atr=context['atr'],
            min_rr=context['min_rr']
        )
        try:
            template = PromptTemplate(template=prompt, input_variables=[])
            # calling client directly. exact invocation depends on client lib; .invoke as in Bot1
            response = self.client(template.template) if hasattr(self.client, '__call__') else self.client.invoke({"input": prompt})
            # parse: expect short "CONFIRM - reason" or "REJECT - reason"
            text = str(response)
            # keep parsing simple:
            if "CONFIRM" in text.upper():
                reason = text.split("\n", 1)[1].strip() if "\n" in text else ""
                return {"decision": "CONFIRM", "reason": reason[:120]}
            else:
                reason = text.split("\n", 1)[1].strip() if "\n" in text else ""
                return {"decision": "REJECT", "reason": reason[:120]}
        except Exception as e:
            logging.error(f"LLM error: {e}")
            return {"decision": "REJECT", "reason": "LLM error"}

# Utility indicators
def calculate_macd(df, fast=12, slow=26, sig=9):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=sig, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_dn'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    return df

def calculate_atr(df, period=14):
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(period).mean()
    return df

def ema_direction(df, length=50):
    # returns "buy" if ema slope is up, "sell" if down, "flat" otherwise
    ema = df['close'].ewm(span=length, adjust=False).mean()
    slope = ema.iloc[-1] - ema.iloc[-5] if len(ema) >= 6 else ema.iloc[-1] - ema.iloc[0]
    if slope > 0:
        return "buy"
    elif slope < 0:
        return "sell"
    return "flat"

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands for volatility assessment"""
    df['bb_middle'] = df['close'].rolling(period).mean()
    bb_std = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    return df

def calculate_support_resistance(df, lookback=20):
    """Identify key support and resistance levels"""
    df['support'] = df['low'].rolling(lookback, center=True).min()
    df['resistance'] = df['high'].rolling(lookback, center=True).max()
    return df

def calculate_supertrend(df, period=10, multiplier=3.0):
    """Calculate SuperTrend indicator - a powerful trend following indicator"""
    df = df.copy()
    
    # Calculate ATR
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(period).mean()
    
    # HL2 (High + Low)/2
    hl2 = (df['high'] + df['low']) / 2
    
    # Calculate SuperTrend bands
    df['up_band'] = hl2 - (multiplier * df['atr'])
    df['down_band'] = hl2 + (multiplier * df['atr'])
    
    # Initialize SuperTrend values
    df['supertrend'] = 0.0
    df['supertrend_signal'] = 0
    
    for i in range(period, len(df)):
        # Update bands based on previous values
        if df['close'].iloc[i-1] > df['up_band'].iloc[i-1]:
            df.loc[df.index[i], 'up_band'] = max(df['up_band'].iloc[i], df['up_band'].iloc[i-1])
        
        if df['close'].iloc[i-1] < df['down_band'].iloc[i-1]:
            df.loc[df.index[i], 'down_band'] = min(df['down_band'].iloc[i], df['down_band'].iloc[i-1])
        
        # Determine trend
        if i == period:
            if df['close'].iloc[i] <= df['up_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['up_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_signal'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['down_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_signal'] = 1
        else:
            # Previous trend was bullish
            if df['supertrend_signal'].iloc[i-1] == 1:
                if df['close'].iloc[i] <= df['down_band'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['down_band'].iloc[i]
                    df.loc[df.index[i], 'supertrend_signal'] = -1
                else:
                    df.loc[df.index[i], 'supertrend'] = df['up_band'].iloc[i]
                    df.loc[df.index[i], 'supertrend_signal'] = 1
            # Previous trend was bearish
            else:
                if df['close'].iloc[i] >= df['up_band'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['up_band'].iloc[i]
                    df.loc[df.index[i], 'supertrend_signal'] = 1
                else:
                    df.loc[df.index[i], 'supertrend'] = df['down_band'].iloc[i]
                    df.loc[df.index[i], 'supertrend_signal'] = -1
    
    # Detect SuperTrend signals
    df['supertrend_buy'] = (df['supertrend_signal'] == 1) & (df['supertrend_signal'].shift(1) == -1)
    df['supertrend_sell'] = (df['supertrend_signal'] == -1) & (df['supertrend_signal'].shift(1) == 1)
    
    return df

def is_trending_market(df, ema_fast=20, ema_slow=50):
    """Determine if market is trending or ranging"""
    ema_fast_val = df['close'].ewm(span=ema_fast).mean().iloc[-1]
    ema_slow_val = df['close'].ewm(span=ema_slow).mean().iloc[-1]
    
    # Calculate trend strength
    trend_strength = abs(ema_fast_val - ema_slow_val) / ema_slow_val
    return trend_strength > 0.002  # 0.2% minimum trend strength

def check_rsi_divergence(df, period=14, lookback=5):
    """Check for RSI divergence with price action"""
    if len(df) < lookback + period:
        return False
    
    recent_prices = df['close'].tail(lookback)
    recent_rsi = df['rsi'].tail(lookback)
    
    # Check for bullish divergence (price makes lower lows, RSI makes higher lows)
    price_trend = recent_prices.iloc[-1] < recent_prices.iloc[0]
    rsi_trend = recent_rsi.iloc[-1] > recent_rsi.iloc[0]
    
    return price_trend != rsi_trend

def is_valid_breakout(df, current_price, lookback=20):
    """Check if current price represents a valid breakout"""
    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    
    # Check if price is breaking above recent resistance or below recent support
    breakout_up = current_price > recent_high * 1.001  # 0.1% buffer
    breakout_down = current_price < recent_low * 0.999  # 0.1% buffer
    
    return breakout_up or breakout_down

def check_volume_confirmation(df):
    """Check if recent volume supports the signal (placeholder for volume data)"""
    # Since MT5 tick volume might not be reliable, we use ATR as proxy for volatility/activity
    if 'atr' not in df.columns:
        return True
    
    recent_atr = df['atr'].tail(3).mean()
    avg_atr = df['atr'].tail(20).mean()
    
    return recent_atr > avg_atr * 0.8  # Recent activity should be at least 80% of average

def is_trading_session_active():
    """Check if we're in an active trading session for XAUUSD"""
    now = datetime.now()
    current_hour = now.hour
    
    # XAUUSD is active during major forex sessions
    # London: 8-17 GMT, New York: 13-22 GMT, Asian: 22-8 GMT
    # Convert to approximate local times - this should be adjusted for your timezone
    active_hours = list(range(1, 9)) + list(range(13, 23))  # Active sessions
    
    return current_hour in active_hours

def identify_market_structure(df, lookback=20):
    """Identify market structure: higher highs/lower lows, break of structure"""
    if len(df) < lookback + 5:
        return "unknown", None
    
    # Get recent highs and lows
    recent_data = df.tail(lookback).copy()
    highs = recent_data['high']
    lows = recent_data['low']
    
    # Find swing highs and lows (peaks and troughs)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent_data) - 2):
        # Swing high: higher than 2 candles before and after
        if (recent_data.iloc[i]['high'] > recent_data.iloc[i-1]['high'] and 
            recent_data.iloc[i]['high'] > recent_data.iloc[i-2]['high'] and
            recent_data.iloc[i]['high'] > recent_data.iloc[i+1]['high'] and 
            recent_data.iloc[i]['high'] > recent_data.iloc[i+2]['high']):
            swing_highs.append((i, recent_data.iloc[i]['high']))
        
        # Swing low: lower than 2 candles before and after  
        if (recent_data.iloc[i]['low'] < recent_data.iloc[i-1]['low'] and 
            recent_data.iloc[i]['low'] < recent_data.iloc[i-2]['low'] and
            recent_data.iloc[i]['low'] < recent_data.iloc[i+1]['low'] and 
            recent_data.iloc[i]['low'] < recent_data.iloc[i+2]['low']):
            swing_lows.append((i, recent_data.iloc[i]['low']))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "consolidation", None
    
    # Analyze trend structure
    recent_highs = [h[1] for h in swing_highs[-3:]]
    recent_lows = [l[1] for l in swing_lows[-3:]]
    
    # Check for higher highs and higher lows (uptrend)
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        higher_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
        higher_lows = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        # Check for lower highs and lower lows (downtrend)
        lower_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
        lower_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        if higher_highs and higher_lows:
            return "uptrend", swing_highs[-1][1]  # Return last swing high as key level
        elif lower_highs and lower_lows:
            return "downtrend", swing_lows[-1][1]  # Return last swing low as key level
    
    return "consolidation", None

def check_break_of_structure(df, structure_type, key_level, current_price):
    """Check if current price represents a break of market structure"""
    if structure_type == "unknown" or key_level is None:
        return False
    
    # For uptrend, look for break below recent swing low
    if structure_type == "uptrend":
        return current_price < key_level * 0.999  # 0.1% buffer
    
    # For downtrend, look for break above recent swing high  
    elif structure_type == "downtrend":
        return current_price > key_level * 1.001  # 0.1% buffer
    
    return False

def calculate_price_momentum(df, period=10):
    """Calculate price momentum over specified period"""
    if len(df) < period + 1:
        return 0
    
    current_price = df['close'].iloc[-1]
    past_price = df['close'].iloc[-(period+1)]
    
    momentum = (current_price - past_price) / past_price
    return momentum

def detect_order_blocks(df, lookback=50, internal=False):
    """Detect order blocks based on Smart Money Concepts"""
    order_blocks = []
    
    # Identify swing highs and lows
    for i in range(2, len(df) - 2):
        # Swing high: higher than 2 candles before and after
        if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
            df.iloc[i]['high'] > df.iloc[i-2]['high'] and
            df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
            df.iloc[i]['high'] > df.iloc[i+2]['high']):
            
            # Look for the last down candle before the swing high (bearish OB)
            for j in range(i-1, max(0, i-10), -1):
                if df.iloc[j]['close'] < df.iloc[j]['open']:
                    order_blocks.append({
                        'type': 'bearish',
                        'top': df.iloc[j]['high'],
                        'bottom': df.iloc[j]['low'],
                        'time': df.iloc[j]['time'],
                        'index': j,
                        'internal': internal
                    })
                    break
        
        # Swing low: lower than 2 candles before and after  
        if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
            df.iloc[i]['low'] < df.iloc[i-2]['low'] and
            df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
            df.iloc[i]['low'] < df.iloc[i+2]['low']):
            
            # Look for the last up candle before the swing low (bullish OB)
            for j in range(i-1, max(0, i-10), -1):
                if df.iloc[j]['close'] > df.iloc[j]['open']:
                    order_blocks.append({
                        'type': 'bullish',
                        'top': df.iloc[j]['high'],
                        'bottom': df.iloc[j]['low'],
                        'time': df.iloc[j]['time'],
                        'index': j,
                        'internal': internal
                    })
                    break
    
    # Keep only the most recent order blocks
    order_blocks = sorted(order_blocks, key=lambda x: x['index'], reverse=True)[:5]
    return order_blocks

def detect_fair_value_gaps(df, threshold_multiplier=1.5):
    """Detect Fair Value Gaps (FVG) - price inefficiencies"""
    fvgs = []
    
    if len(df) < 3:
        return fvgs
    
    # Calculate average candle range for threshold
    avg_range = df['high'].rolling(20).mean() - df['low'].rolling(20).mean()
    
    for i in range(2, len(df)):
        # Bullish FVG: gap between candle[i-2] high and candle[i] low
        gap_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
        if gap_size > 0 and gap_size > avg_range.iloc[i] * 0.1:  # Minimum 10% of average range
            fvgs.append({
                'type': 'bullish',
                'top': df.iloc[i]['low'],
                'bottom': df.iloc[i-2]['high'],
                'index': i,
                'size': gap_size
            })
        
        # Bearish FVG: gap between candle[i-2] low and candle[i] high
        gap_size = df.iloc[i-2]['low'] - df.iloc[i]['high']
        if gap_size > 0 and gap_size > avg_range.iloc[i] * 0.1:
            fvgs.append({
                'type': 'bearish',
                'top': df.iloc[i-2]['low'],
                'bottom': df.iloc[i]['high'],
                'index': i,
                'size': gap_size
            })
    
    return fvgs

def detect_break_of_structure(df, lookback=20):
    """Detect Break of Structure (BOS) and Change of Character (CHoCH)"""
    if len(df) < lookback + 5:
        return None, None
    
    # Find recent swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(df) - 2):
        # Swing high
        if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
            df.iloc[i]['high'] > df.iloc[i-2]['high'] and
            df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
            df.iloc[i]['high'] > df.iloc[i+2]['high']):
            swing_highs.append((i, df.iloc[i]['high']))
        
        # Swing low
        if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
            df.iloc[i]['low'] < df.iloc[i-2]['low'] and
            df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
            df.iloc[i]['low'] < df.iloc[i+2]['low']):
            swing_lows.append((i, df.iloc[i]['low']))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None, None
    
    # Check for BOS/CHoCH
    last_high = swing_highs[-1][1]
    prev_high = swing_highs[-2][1]
    last_low = swing_lows[-1][1]
    prev_low = swing_lows[-2][1]
    current_close = df.iloc[-1]['close']
    
    # Bullish BOS: Higher high and higher low
    if last_high > prev_high and last_low > prev_low:
        if current_close > last_high:
            return 'bullish_bos', last_high
    
    # Bearish BOS: Lower high and lower low
    if last_high < prev_high and last_low < prev_low:
        if current_close < last_low:
            return 'bearish_bos', last_low
    
    # Bullish CHoCH: Was in downtrend, now breaking above
    if last_high < prev_high and current_close > last_high:
        return 'bullish_choch', last_high
    
    # Bearish CHoCH: Was in uptrend, now breaking below
    if last_low > prev_low and current_close < last_low:
        return 'bearish_choch', last_low
    
    return None, None

def detect_equal_highs_lows(df, lookback=20, threshold=0.001):
    """Detect equal highs and equal lows for better S/R detection"""
    if len(df) < lookback:
        return [], []
    
    recent_data = df.tail(lookback)
    highs = recent_data['high'].values
    lows = recent_data['low'].values
    
    equal_highs = []
    equal_lows = []
    
    # Find equal highs
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] < threshold:
                equal_highs.append((highs[i], i, j))
    
    # Find equal lows
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / lows[i] < threshold:
                equal_lows.append((lows[i], i, j))
    
    return equal_highs, equal_lows

def detect_premium_discount_zones(df, lookback=50):
    """Detect premium, discount, and equilibrium zones based on recent price range"""
    if len(df) < lookback:
        return None, None, None
    
    # Get recent high and low
    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    
    # Calculate zones
    range_size = recent_high - recent_low
    equilibrium = (recent_high + recent_low) / 2
    
    # Premium zone: Top 25% of range
    premium_zone = (recent_high - range_size * 0.25, recent_high)
    
    # Discount zone: Bottom 25% of range  
    discount_zone = (recent_low, recent_low + range_size * 0.25)
    
    # Equilibrium zone: Middle 10% of range
    equilibrium_zone = (equilibrium - range_size * 0.05, equilibrium + range_size * 0.05)
    
    return premium_zone, discount_zone, equilibrium_zone

def is_pullback_complete(df, signal_type, lookback=3):
    """Check if price has completed a healthy pullback for entry - RELAXED VERSION"""
    if len(df) < lookback + 5:
        return True  # Allow entry if insufficient data
    
    recent_data = df.tail(lookback)
    older_data = df.tail(lookback + 5).head(5)  # Shorter lookback period
    
    # Get current price momentum to determine if pullback is reasonable
    current_price = df['close'].iloc[-1]
    ema_short = df['close'].ewm(span=5).mean().iloc[-1]
    
    if signal_type == "buy":
        # For buy signal, allow entry if:
        # 1. Price is above short EMA (still bullish), OR
        # 2. Had a minor pullback (0.2% instead of 0.5%), OR
        # 3. Price action shows buying pressure (recent low > older low)
        recent_low = recent_data['low'].min()
        older_low = older_data['low'].min()
        
        # More permissive conditions
        above_ema = current_price >= ema_short * 0.999  # Allow slight below EMA
        minor_pullback = recent_low < df['high'].tail(10).max() * 0.998  # Only 0.2% pullback required
        buying_pressure = recent_low >= older_low * 0.998  # Higher lows pattern
        
        return above_ema or minor_pullback or buying_pressure
    
    elif signal_type == "sell":
        # For sell signal, allow entry if:
        # 1. Price is below short EMA (still bearish), OR
        # 2. Had a minor pullback (0.2% instead of 0.5%), OR
        # 3. Price action shows selling pressure (recent high < older high)
        recent_high = recent_data['high'].max()
        older_high = older_data['high'].max()
        
        # More permissive conditions
        below_ema = current_price <= ema_short * 1.001  # Allow slight above EMA
        minor_pullback = recent_high > df['low'].tail(10).min() * 1.002  # Only 0.2% pullback required
        selling_pressure = recent_high <= older_high * 1.002  # Lower highs pattern
        
        return below_ema or minor_pullback or selling_pressure
    
    return True  # Default to allow if uncertain

# This orphaned backtest code has been removed to fix the main trading logic

# MT5 helpers
def initialize_mt5():
    if not mt5.initialize():
        logging.error("MT5 initialize failed")
        sys.exit(1)
    if ACCOUNT != 0:
        if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
            logging.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            sys.exit(1)
    logging.info("MT5 initialized")

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass

def get_rates(symbol, timeframe, n=200):
    tf_const = TF_MAP.get(timeframe, mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_account_info():
    account_info = mt5.account_info()
    if account_info is None:
        return None
    return account_info

def current_spread_points(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return float('inf')
    # for many brokers price diff * 100 = points — this is broker-specific
    spread = (tick.ask - tick.bid)
    # convert to "points" relative to symbol digit: multiply by 100 for XAUUSD approx. We'll return raw price difference too.
    return spread

def compute_lot_size(symbol, entry_price, sl_price, risk_amount):
    """
    Compute lot size so that loss if SL hit ≈ risk_amount.
    Use symbol tick info to convert price distance to money per lot.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            logging.warning("Symbol info unavailable; using fallback lot 0.01")
            return 0.01
        tick_size = info.trade_tick_size if hasattr(info, 'trade_tick_size') else info.point
        tick_value = info.trade_tick_value if hasattr(info, 'trade_tick_value') else None
        contract_size = info.trade_contract_size if hasattr(info, 'trade_contract_size') else 1.0

        price_dist = abs(entry_price - sl_price)
        if price_dist <= 0 or tick_value is None or tick_size is None:
            logging.warning("Insufficient tick info or zero SL distance; using fallback lot 0.01")
            return 0.01

        # loss per 1 lot = (price_dist / tick_size) * tick_value
        loss_per_lot = (price_dist / tick_size) * tick_value
        if loss_per_lot <= 0:
            logging.warning("Calculated loss per lot non-positive; fallback 0.01")
            return 0.01
        lot = risk_amount / loss_per_lot
        # enforce broker min/max
        lot = max(lot, info.volume_min if hasattr(info, 'volume_min') else 0.01)
        lot = min(lot, info.volume_max if hasattr(info, 'volume_max') else lot)
        # round to broker step
        step = info.volume_step if hasattr(info, 'volume_step') else 0.01
        lot = (int(lot / step)) * step
        if lot <= 0:
            lot = step
        return float(round(lot, 2))
    except Exception as e:
        logging.error(f"Error computing lot size: {e}")
        return 0.01

def place_market_order(symbol, direction, volume, sl_price, tp_price, deviation=20, magic=987654):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error("Cannot fetch tick for order")
        return None
    price = tick.ask if direction == "buy" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": int(deviation),
        "magic": magic,
        "comment": "HybridBot3",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    return result

def main_loop():
    initialize_mt5()
    confirmer = LLMConfirmer(provider=LLM_PROVIDER, api_key=GROQ_API_KEY if LLM_PROVIDER == "groq" else None)
    last_trade_time = None  # Track last trade time for cooldown
    trade_count = 0  # Track number of trades

    try:
        while True:
            try:
                # Basic sanity & market open
                info = mt5.symbol_info(SYMBOL)
                if info is None:
                    logging.error(f"{SYMBOL} not available on server.")
                    sleep(SLEEP_SECONDS)
                    continue
                if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                    logging.info("Symbol trading disabled. Sleeping.")
                    sleep(300)
                    continue

                spread = current_spread_points(SYMBOL)
                logging.debug(f"Current spread: {spread}")

                if spread > (MAX_SPREAD * info.point):
                    logging.info(f"Spread {spread} too wide vs threshold {MAX_SPREAD * info.point}. Skipping.")
                    sleep(SLEEP_SECONDS)
                    continue

                # get data
                df_primary = get_rates(SYMBOL, TF_PRIMARY, n=200)
                df_c1 = get_rates(SYMBOL, TF_CONFIRM_M1, n=200)
                df_c2 = get_rates(SYMBOL, TF_CONFIRM_M2, n=200)

                if df_primary.empty or df_c1.empty or df_c2.empty:
                    logging.warning("Missing data for one or more timeframes. Skipping.")
                    sleep(SLEEP_SECONDS)
                    continue

                # indicators
                df_primary = calculate_macd(df_primary)
                df_primary = calculate_atr(df_primary)
                df_primary = calculate_rsi(df_primary)
                df_primary = calculate_bollinger_bands(df_primary)
                df_primary = calculate_support_resistance(df_primary)
                df_primary = calculate_supertrend(df_primary)  # Add SuperTrend
                df_c1 = calculate_atr(df_c1)
                df_c2 = calculate_atr(df_c2)
                
                # Smart Money Concepts
                order_blocks = detect_order_blocks(df_primary)
                fvgs = detect_fair_value_gaps(df_primary)
                bos_choch, bos_level = detect_break_of_structure(df_primary)
                equal_highs, equal_lows = detect_equal_highs_lows(df_primary)
                premium_zone, discount_zone, equilibrium_zone = detect_premium_discount_zones(df_primary)

                # Check if we're in an active trading session
                if not is_trading_session_active():
                    logging.debug("Outside active trading hours.")
                    sleep(SLEEP_SECONDS)
                    continue

                # Check cooldown period
                current_time = datetime.now()
                if last_trade_time and (current_time - last_trade_time).total_seconds() < COOLDOWN_MINUTES * 60:
                    remaining_cooldown = COOLDOWN_MINUTES * 60 - (current_time - last_trade_time).total_seconds()
                    logging.debug(f"Cooldown active: {remaining_cooldown/60:.1f} minutes remaining.")
                    sleep(SLEEP_SECONDS)
                    continue

                # Primary signal from MACD on latest candle
                latest = df_primary.iloc[-1]
                current_price = latest['close']
                
                # Enhanced signal detection with multiple confirmations
                primary_signal = None
                signal_strength = 0
                signal_reasons = []
                
                # 1. MACD Cross Signal
                if latest.get('macd_cross_up', False):
                    primary_signal = "buy"
                    signal_strength += 1
                    signal_reasons.append("MACD bullish cross")
                elif latest.get('macd_cross_dn', False):
                    primary_signal = "sell"
                    signal_strength += 1
                    signal_reasons.append("MACD bearish cross")
                
                # 2. SuperTrend Signal (Strong confirmation)
                if latest.get('supertrend_buy', False):
                    if primary_signal == "buy":
                        signal_strength += 2  # Strong confirmation
                        signal_reasons.append("SuperTrend buy signal")
                    elif primary_signal is None:
                        primary_signal = "buy"
                        signal_strength += 1
                        signal_reasons.append("SuperTrend buy signal")
                elif latest.get('supertrend_sell', False):
                    if primary_signal == "sell":
                        signal_strength += 2  # Strong confirmation
                        signal_reasons.append("SuperTrend sell signal")
                    elif primary_signal is None:
                        primary_signal = "sell"
                        signal_strength += 1
                        signal_reasons.append("SuperTrend sell signal")
                
                # 3. Break of Structure / Change of Character (Very strong signal)
                if bos_choch:
                    if bos_choch == 'bullish_bos' and primary_signal == "buy":
                        signal_strength += 2
                        signal_reasons.append("Bullish Break of Structure")
                    elif bos_choch == 'bearish_bos' and primary_signal == "sell":
                        signal_strength += 2
                        signal_reasons.append("Bearish Break of Structure")
                    elif bos_choch == 'bullish_choch':
                        if primary_signal != "sell":  # Don't fight CHoCH
                            primary_signal = "buy"
                            signal_strength += 3  # Very strong signal
                            signal_reasons.append("Bullish Change of Character")
                    elif bos_choch == 'bearish_choch':
                        if primary_signal != "buy":  # Don't fight CHoCH
                            primary_signal = "sell"
                            signal_strength += 3  # Very strong signal
                            signal_reasons.append("Bearish Change of Character")

                if primary_signal is None:
                    logging.debug("No primary signal detected.")
                    sleep(SLEEP_SECONDS)
                    continue

                # 2. Market Structure Confirmation
                if not is_trending_market(df_primary):
                    logging.debug("Market is ranging, skipping signal.")
                    sleep(SLEEP_SECONDS)
                    continue

                # 3. RSI Oversold/Overbought Levels (avoid extreme levels)
                rsi = latest.get('rsi', 50)
                if primary_signal == "buy" and (rsi < 30 or rsi > 75):
                    logging.debug(f"RSI {rsi:.2f} not suitable for buy signal.")
                    sleep(SLEEP_SECONDS)
                    continue
                elif primary_signal == "sell" and (rsi > 70 or rsi < 25):
                    logging.debug(f"RSI {rsi:.2f} not suitable for sell signal.")
                    sleep(SLEEP_SECONDS)
                    continue
                else:
                    signal_strength += 1

                # 4. Bollinger Band Position
                bb_upper = latest.get('bb_upper', current_price)
                bb_lower = latest.get('bb_lower', current_price)
                bb_middle = latest.get('bb_middle', current_price)
                
                if primary_signal == "buy" and current_price > bb_upper:
                    logging.debug("Price above upper Bollinger Band, avoiding buy.")
                    sleep(SLEEP_SECONDS)
                    continue
                elif primary_signal == "sell" and current_price < bb_lower:
                    logging.debug("Price below lower Bollinger Band, avoiding sell.")
                    sleep(SLEEP_SECONDS)
                    continue
                else:
                    signal_strength += 1

                # 5. Volume/Volatility Confirmation
                if not check_volume_confirmation(df_primary):
                    logging.debug("Insufficient volume/volatility for signal.")
                    sleep(SLEEP_SECONDS)
                    continue
                else:
                    signal_strength += 1

                # 6. Support/Resistance Levels
                support = latest.get('support', current_price * 0.99)
                resistance = latest.get('resistance', current_price * 1.01)
                
                if primary_signal == "buy" and current_price < support * 1.001:
                    logging.debug("Price too close to support for buy signal.")
                    sleep(SLEEP_SECONDS)
                    continue
                elif primary_signal == "sell" and current_price > resistance * 0.999:
                    logging.debug("Price too close to resistance for sell signal.")
                    sleep(SLEEP_SECONDS)
                    continue
                else:
                    signal_strength += 1
                    signal_reasons.append("Good S/R positioning")
                
                # 7. Order Block Confirmation (Smart Money Concepts)
                valid_order_block = False
                for ob in order_blocks:
                    if primary_signal == "buy" and ob['type'] == 'bullish':
                        # Check if price is near bullish order block
                        if ob['bottom'] <= current_price <= ob['top'] * 1.01:
                            signal_strength += 2
                            signal_reasons.append("At bullish order block")
                            valid_order_block = True
                            break
                    elif primary_signal == "sell" and ob['type'] == 'bearish':
                        # Check if price is near bearish order block
                        if ob['bottom'] * 0.99 <= current_price <= ob['top']:
                            signal_strength += 2
                            signal_reasons.append("At bearish order block")
                            valid_order_block = True
                            break
                
                # 8. Fair Value Gap Confirmation
                fvg_support = False
                for fvg in fvgs[-5:]:  # Check last 5 FVGs
                    if primary_signal == "buy" and fvg['type'] == 'bullish':
                        # Check if price is in or near bullish FVG
                        if fvg['bottom'] <= current_price <= fvg['top'] * 1.02:
                            signal_strength += 1
                            signal_reasons.append("Bullish FVG support")
                            fvg_support = True
                            break
                    elif primary_signal == "sell" and fvg['type'] == 'bearish':
                        # Check if price is in or near bearish FVG
                        if fvg['bottom'] * 0.98 <= current_price <= fvg['top']:
                            signal_strength += 1
                            signal_reasons.append("Bearish FVG resistance")
                            fvg_support = True
                            break
                
                # 9. Equal Highs/Lows Confirmation
                if equal_highs and primary_signal == "sell":
                    # Check if we're near equal highs (resistance)
                    for eq_high in equal_highs[-3:]:
                        if abs(current_price - eq_high[0]) / current_price < 0.002:
                            signal_strength += 1
                            signal_reasons.append("At equal highs resistance")
                            break
                elif equal_lows and primary_signal == "buy":
                    # Check if we're near equal lows (support)
                    for eq_low in equal_lows[-3:]:
                        if abs(current_price - eq_low[0]) / current_price < 0.002:
                            signal_strength += 1
                            signal_reasons.append("At equal lows support")
                            break
                
                # 10. Premium/Discount Zone Confirmation (Smart Money Concepts)
                if premium_zone and discount_zone and equilibrium_zone:
                    # Buy signals are stronger in discount zone
                    if primary_signal == "buy":
                        if discount_zone[0] <= current_price <= discount_zone[1]:
                            signal_strength += 2
                            signal_reasons.append("In discount zone (optimal buy)")
                        elif equilibrium_zone[0] <= current_price <= equilibrium_zone[1]:
                            signal_strength += 1
                            signal_reasons.append("At equilibrium (fair value)")
                        elif premium_zone[0] <= current_price <= premium_zone[1]:
                            # Buying in premium zone is risky
                            signal_strength -= 1
                            signal_reasons.append("In premium zone (risky buy)")
                    
                    # Sell signals are stronger in premium zone
                    elif primary_signal == "sell":
                        if premium_zone[0] <= current_price <= premium_zone[1]:
                            signal_strength += 2
                            signal_reasons.append("In premium zone (optimal sell)")
                        elif equilibrium_zone[0] <= current_price <= equilibrium_zone[1]:
                            signal_strength += 1
                            signal_reasons.append("At equilibrium (fair value)")
                        elif discount_zone[0] <= current_price <= discount_zone[1]:
                            # Selling in discount zone is risky
                            signal_strength -= 1
                            signal_reasons.append("In discount zone (risky sell)")

                # Require minimum signal strength (now configurable)
                max_possible_strength = 12  # Updated max strength with new indicators
                if signal_strength < MIN_SIGNAL_STRENGTH:
                    logging.debug(f"Signal strength {signal_strength}/{max_possible_strength} insufficient (min: {MIN_SIGNAL_STRENGTH}). Reasons: {', '.join(signal_reasons)}")
                    sleep(SLEEP_SECONDS)
                    continue

                logging.info(f"Strong {primary_signal.upper()} signal detected with strength {signal_strength}/{max_possible_strength}")
                logging.info(f"Signal reasons: {', '.join(signal_reasons)}")

                # 7. Market Structure Analysis
                structure_type, key_level = identify_market_structure(df_primary)
                logging.info(f"Market structure: {structure_type}, key level: {key_level}")
                
                # Check if signal aligns with market structure (RELAXED)
                if structure_type == "uptrend" and primary_signal == "sell":
                    # Allow sell signals in uptrend if:
                    # 1. There's a break of structure, OR
                    # 2. RSI shows overbought (>70), OR  
                    # 3. Price is at resistance level
                    rsi = latest.get('rsi', 50)
                    resistance = latest.get('resistance', current_price * 1.01)
                    
                    break_of_structure = check_break_of_structure(df_primary, structure_type, key_level, current_price)
                    overbought = rsi > 70
                    at_resistance = current_price >= resistance * 0.999
                    
                    if not (break_of_structure or overbought or at_resistance):
                        logging.debug("Sell signal against uptrend rejected (no BOS, not overbought, not at resistance).")
                        sleep(SLEEP_SECONDS)
                        continue
                        
                elif structure_type == "downtrend" and primary_signal == "buy":
                    # Allow buy signals in downtrend if:
                    # 1. There's a break of structure, OR
                    # 2. RSI shows oversold (<30), OR
                    # 3. Price is at support level
                    rsi = latest.get('rsi', 50)
                    support = latest.get('support', current_price * 0.99)
                    
                    break_of_structure = check_break_of_structure(df_primary, structure_type, key_level, current_price)
                    oversold = rsi < 30
                    at_support = current_price <= support * 1.001
                    
                    if not (break_of_structure or oversold or at_support):
                        logging.debug("Buy signal against downtrend rejected (no BOS, not oversold, not at support).")
                        sleep(SLEEP_SECONDS)
                        continue
                
                # 8. Check for healthy pullback completion
                if not is_pullback_complete(df_primary, primary_signal):
                    logging.debug("Pullback not complete, waiting for better entry.")
                    sleep(SLEEP_SECONDS)
                    continue

                # 9. Price momentum confirmation
                momentum = calculate_price_momentum(df_primary)
                if primary_signal == "buy" and momentum < -0.001:  # Negative momentum for buy
                    logging.debug(f"Negative momentum {momentum:.4f} for buy signal.")
                    sleep(SLEEP_SECONDS)
                    continue
                elif primary_signal == "sell" and momentum > 0.001:  # Positive momentum for sell
                    logging.debug(f"Positive momentum {momentum:.4f} for sell signal.")
                    sleep(SLEEP_SECONDS)
                    continue

                # trend alignment using EMA50/200 slope heuristic from confirm TFs
                trend1 = ema_direction(df_c1, length=50)
                trend2 = ema_direction(df_c2, length=50)
                logging.info(f"Primary signal: {primary_signal}. Confirm1: {trend1}, Confirm2: {trend2}")
                
                # Enhanced trend confirmation with market structure
                if structure_type in ["uptrend", "downtrend"]:
                    # If we have clear market structure, require at least one timeframe alignment
                    if not (trend1 == primary_signal or trend2 == primary_signal):
                        logging.debug("No trend alignment with higher timeframes.")
                        sleep(SLEEP_SECONDS)
                        continue
                else:
                    # For consolidation, require both timeframes to agree
                    if not (trend1 == trend2 == primary_signal):
                        logging.debug("Insufficient trend alignment in consolidating market.")
                        sleep(SLEEP_SECONDS)
                        continue

                # compute SL using ATR on primary TF
                atr = df_primary['atr'].iloc[-1] if 'atr' in df_primary.columns else None
                if atr is None or np.isnan(atr):
                    logging.warning("ATR invalid. Skipping.")
                    sleep(SLEEP_SECONDS)
                    continue
                sl_distance = ATR_MULTIPLIER * atr
                entry_price = mt5.symbol_info_tick(SYMBOL).ask if primary_signal == "buy" else mt5.symbol_info_tick(SYMBOL).bid
                if entry_price is None:
                    logging.warning("Tick unavailable, skipping.")
                    sleep(SLEEP_SECONDS)
                    continue

                # compute SL and TP prices
                if primary_signal == "buy":
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + (sl_distance * MIN_RR)
                else:
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - (sl_distance * MIN_RR)

                # sanity RR check (approx)
                rr_est = abs((tp_price - entry_price) / (entry_price - sl_price)) if sl_price != entry_price else 0
                if rr_est < MIN_RR:
                    logging.info(f"Estimated RR {rr_est:.2f} < min {MIN_RR}. Skipping.")
                    sleep(SLEEP_SECONDS)
                    continue

                # LLM confirmation context
                # Prepare compact primary_data
                primary_rows = df_primary.tail(8)[['time','open','high','low','close']].to_dict('records')
                context = {
                    "primary_signal": primary_signal,
                    "primary_data": primary_rows,
                    "confirm1": trend1,
                    "confirm2": trend2,
                    "spread": spread,
                    "sl_dist": sl_distance,
                    "atr": atr,
                    "min_rr": MIN_RR,
                    "signal_strength": signal_strength,
                    "signal_reasons": signal_reasons,
                    "supertrend": latest.get('supertrend_signal', 0),
                    "bos_choch": bos_choch if bos_choch else "none",
                    "order_blocks": len([ob for ob in order_blocks if ob['type'] == 'bullish' if primary_signal == 'buy' else ob['type'] == 'bearish']),
                    "market_structure": structure_type
                }

                llm_result = confirmer.confirm(context)
                logging.info(f"LLM decision: {llm_result}")

                if llm_result['decision'] != "CONFIRM":
                    logging.info("LLM rejected the signal. Reason: %s", llm_result.get('reason'))
                    sleep(SLEEP_SECONDS)
                    continue

                # Enhanced risk sizing with dynamic position sizing
                account = get_account_info()
                if account is None:
                    logging.error("Account info unavailable.")
                    sleep(SLEEP_SECONDS)
                    continue
                balance = float(account.balance)
                
                # Dynamic risk adjustment based on signal strength and market conditions
                base_risk = RISK_PERCENT
                risk_multiplier = 1.0
                
                # Increase risk for high-quality signals (adjusted for new max strength)
                if signal_strength >= 8:  # Very strong signal with SMC confluence
                    risk_multiplier *= 1.8
                elif signal_strength >= 6:  # Strong signal
                    risk_multiplier *= 1.5
                elif signal_strength >= 4:  # Good signal
                    risk_multiplier *= 1.2
                
                # Extra boost for specific high-confidence setups
                if bos_choch and ('CHoCH' in bos_choch):
                    risk_multiplier *= 1.2  # Change of Character is very strong
                if valid_order_block and fvg_support:
                    risk_multiplier *= 1.1  # Both SMC concepts align
                
                # Adjust risk based on volatility
                atr_ratio = atr / df_primary['atr'].tail(50).mean() if not df_primary['atr'].tail(50).empty else 1.0
                if atr_ratio > 1.5:  # High volatility
                    risk_multiplier *= 0.7  # Reduce risk
                elif atr_ratio < 0.7:  # Low volatility
                    risk_multiplier *= 1.3  # Increase risk slightly
                
                # Adjust risk based on market structure confidence
                if structure_type in ["uptrend", "downtrend"]:
                    risk_multiplier *= 1.1  # Slightly more confident in trending markets
                
                # Apply maximum risk cap
                adjusted_risk = min(base_risk * risk_multiplier, base_risk * 2.0)  # Max 2x base risk
                risk_amount = balance * adjusted_risk
                
                logging.info(f"Risk adjustment: base={base_risk:.3f}, multiplier={risk_multiplier:.2f}, final={adjusted_risk:.3f}")
                
                lot = compute_lot_size(SYMBOL, entry_price, sl_price, risk_amount)
                if lot <= 0:
                    logging.error("Calculated lot size invalid. Skipping.")
                    sleep(SLEEP_SECONDS)
                    continue

                logging.info(f"Placing {primary_signal.upper()} order: entry={entry_price:.3f}, SL={sl_price:.3f}, TP={tp_price:.3f}, LOT={lot}")
                logging.info(f"Trade #{trade_count + 1} - Signal strength: {signal_strength}/{max_possible_strength}, Market structure: {structure_type}")
                logging.info(f"SMC Analysis: BOS/CHoCH={bos_choch}, Order Blocks={len(order_blocks)}, FVGs={len(fvgs)}, SuperTrend={latest.get('supertrend_signal', 'N/A')}")

                result = place_market_order(SYMBOL, primary_signal, lot, sl_price, tp_price)
                if result is None:
                    logging.error("Order send returned None.")
                else:
                    logging.info("Order sent result: %s", result)
                    # Update trade tracking on successful order
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        last_trade_time = current_time
                        trade_count += 1
                        logging.info(f"Trade executed successfully. Total trades today: {trade_count}")

                # Sleep short to avoid duplicate orders
                sleep(5)
            except Exception as inner:
                logging.exception("Main loop exception: %s", inner)
                sleep(10)
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main_loop()
