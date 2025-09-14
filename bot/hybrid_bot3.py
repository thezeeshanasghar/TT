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
You are an institutional XAUUSD assistant. Given the following structured facts, reply ONLY with CONFIRM or REJECT and a short reason (<=30 words).

Primary Signal: {primary_signal}
Primary TF candles (last 8): {primary_data}
Confirm TF1 (recent direction): {confirm1}
Confirm TF2 (recent direction): {confirm2}
Current Spread: {spread}
Risk Params: SL distance {sl_dist:.4f}, ATR {atr:.4f}, RR target {min_rr}

Rules: Confirm only if trend alignment is present and price respects support/resistance and spread acceptable.
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
                df_c1 = calculate_atr(df_c1)
                df_c2 = calculate_atr(df_c2)

                # Primary signal from MACD on latest candle
                latest = df_primary.iloc[-1]
                primary_signal = None
                if latest.get('macd_cross_up', False):
                    primary_signal = "buy"
                elif latest.get('macd_cross_dn', False):
                    primary_signal = "sell"
                else:
                    primary_signal = None

                if primary_signal is None:
                    logging.debug("No primary MACD signal.")
                    sleep(SLEEP_SECONDS)
                    continue

                # trend alignment using EMA50/200 slope heuristic from confirm TFs
                trend1 = ema_direction(df_c1, length=50)
                trend2 = ema_direction(df_c2, length=50)
                logging.info(f"Primary signal: {primary_signal}. Confirm1: {trend1}, Confirm2: {trend2}")

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
                    "min_rr": MIN_RR
                }

                llm_result = confirmer.confirm(context)
                logging.info(f"LLM decision: {llm_result}")

                if llm_result['decision'] != "CONFIRM":
                    logging.info("LLM rejected the signal. Reason: %s", llm_result.get('reason'))
                    sleep(SLEEP_SECONDS)
                    continue

                # risk sizing
                account = get_account_info()
                if account is None:
                    logging.error("Account info unavailable.")
                    sleep(SLEEP_SECONDS)
                    continue
                balance = float(account.balance)
                risk_amount = balance * RISK_PERCENT
                lot = compute_lot_size(SYMBOL, entry_price, sl_price, risk_amount)
                if lot <= 0:
                    logging.error("Calculated lot size invalid. Skipping.")
                    sleep(SLEEP_SECONDS)
                    continue

                logging.info(f"Placing {primary_signal.upper()} order: entry={entry_price:.3f}, SL={sl_price:.3f}, TP={tp_price:.3f}, LOT={lot}")

                result = place_market_order(SYMBOL, primary_signal, lot, sl_price, tp_price)
                if result is None:
                    logging.error("Order send returned None.")
                else:
                    logging.info("Order sent result: %s", result)

                # Sleep short to avoid duplicate orders
                sleep(5)
            except Exception as inner:
                logging.exception("Main loop exception: %s", inner)
                sleep(10)
    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main_loop()
