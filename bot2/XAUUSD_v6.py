###############################################################################
# Block 1 : Imports
###############################################################################
import sys, time, os, json, joblib
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
from dotenv import load_dotenv

if sys.version_info >= (3, 12):
    print(f"{datetime.now()}: WARNING – MetaTrader5 wheels exist only for "
          f"Python ≤ 3.11; you are on {sys.version.split()[0]}", flush=True)
    
# Load environment variables from default location and project root fallback
load_dotenv()
_ROOT_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if os.path.isfile(_ROOT_ENV):
    load_dotenv(_ROOT_ENV, override=False)
###############################################################################
# Block 2 : User settings
###############################################################################
def _get_env_str(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        print(f"{datetime.now()}: Missing required environment variable: {name}", flush=True)
        sys.exit(1)
    return value

def _get_env_int(name: str, default: Optional[int] = None, required: bool = False) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        if required and default is None:
            print(f"{datetime.now()}: Missing required environment variable: {name}", flush=True)
            sys.exit(1)
        return int(default) if default is not None else 0
    try:
        return int(value)
    except ValueError:
        print(f"{datetime.now()}: Invalid int for {name}: {value}", flush=True)
        sys.exit(1)

def _get_env_float(name: str, default: Optional[float] = None, required: bool = False) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        if required and default is None:
            print(f"{datetime.now()}: Missing required environment variable: {name}", flush=True)
            sys.exit(1)
        return float(default) if default is not None else 0.0
    try:
        return float(value)
    except ValueError:
        print(f"{datetime.now()}: Invalid float for {name}: {value}", flush=True)
        sys.exit(1)

def _parse_timeframe(tf_str: str, default_tf: int) -> int:
    if not tf_str:
        return default_tf
    tf_str = tf_str.upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    return mapping.get(tf_str, default_tf)

# Required MT5 credentials
LOGIN = _get_env_int("MT5_LOGIN", required=True)
PASSWORD = _get_env_str("MT5_PASSWORD", required=True)
SERVER = _get_env_str("MT5_SERVER", required=True)

# Trading instrument and timeframes
SYMBOL = _get_env_str("SYMBOL", default="XAUUSD")
TF_ENTRY = _parse_timeframe(os.getenv("TF_ENTRY", "M5"), mt5.TIMEFRAME_M5)
TF_TREND = _parse_timeframe(os.getenv("TF_TREND", "H1"), mt5.TIMEFRAME_H1)

# Strategy parameters
EMA_PERIOD   = _get_env_int("EMA_PERIOD", default=22)
RSI_PERIOD   = _get_env_int("RSI_PERIOD", default=14)
LOT_SIZE     = _get_env_float("LOT_SIZE", default=0.1)
RR_RATIO     = _get_env_float("RR_RATIO", default=2.0)
SL_ATR_MULTI = _get_env_float("SL_ATR_MULTI", default=2.0)
THRESHOLD    = _get_env_float("THRESHOLD", default=0.60)
WARMUP_TRADES = _get_env_int("WARMUP_TRADES", default=30)
# Risk per trade (percent). Example: 0.8 => 0.8% risk
RISK_PERCENT = _get_env_float("RISK_PERCENT", default=0.8) / 100.0

# File system
BASE_DIR   = _get_env_str("BASE_DIR", default=os.path.expanduser("~/Ex-machina"))
SYMBOL_DIR = os.path.join(BASE_DIR, SYMBOL)
os.makedirs(SYMBOL_DIR, exist_ok=True)
FEATURE_FILE = os.path.join(SYMBOL_DIR, "features.csv") # Switched to CSV for easier handling
MODEL_FILE   = os.path.join(SYMBOL_DIR, "lr_model.joblib")

# Learning cadence and loop interval
RETRAIN_EVERY = _get_env_int("RETRAIN_EVERY", default=8)
LOOP_SECONDS  = _get_env_int("LOOP_SECONDS", default=300)  # 5-minute loop cadence

# ========================
# Betting hours (local time)
# ========================
BETTING_START = _get_env_int("BETTING_START", default=6)    # Start trading at HH local time (inclusive)
BETTING_END   = _get_env_int("BETTING_END", default=19)     # Stop trading at HH local time (exclusive)

# Magic number to tag MT5 trades from this bot
MT5_MAGIC = _get_env_int("MT5_MAGIC", default=12345)

###############################################################################
# Block 2.5 : Setting change tracking for audit trail
###############################################################################
SETTING_FILE = os.path.join(BASE_DIR, "setting.csv")

def save_settings(old_settings: dict, new_settings: dict, trade_count: int, model: Optional[LogisticRegression]):
    """Appends old/new settings and model params to setting.csv for audit trail."""
    import csv

    # Prepare model coefficients and intercept for logging
    if model is not None:
        coef = model.coef_[0] if hasattr(model, "coef_") else []
        intercept = model.intercept_[0] if hasattr(model, "intercept_") else 0
    else:
        coef = []
        intercept = 0
    row = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "trade_count": trade_count,
        "old_settings": json.dumps(old_settings),
        "new_settings": json.dumps(new_settings),
        "model_coef": json.dumps(coef.tolist() if hasattr(coef, 'tolist') else list(coef)),
        "model_intercept": intercept
    }
    write_header = not os.path.isfile(SETTING_FILE)
    with open(SETTING_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def current_settings_dict():
    """Return a dict of all current key settings for tracking."""
    return {
        "LOGIN": LOGIN,
        "SERVER": SERVER,
        "SYMBOL": SYMBOL,
        "TF_ENTRY": TF_ENTRY,
        "TF_TREND": TF_TREND,
        "EMA_PERIOD": EMA_PERIOD,
        "RSI_PERIOD": RSI_PERIOD,
        "LOT_SIZE": LOT_SIZE,
        "RISK_PERCENT": RISK_PERCENT,
        "RR_RATIO": RR_RATIO,
        "SL_ATR_MULTI": SL_ATR_MULTI,
        "THRESHOLD": THRESHOLD,
        "WARMUP_TRADES": WARMUP_TRADES,
        "BASE_DIR": BASE_DIR,
        "MT5_MAGIC": MT5_MAGIC,
        "RETRAIN_EVERY": RETRAIN_EVERY,
        "BETTING_START": BETTING_START,
        "BETTING_END": BETTING_END,
        "LOOP_SECONDS": LOOP_SECONDS,
    }

###############################################################################
# Block 3 : Strategy & feature engineering
###############################################################################
def fetch_rates(timeframe: int, bars: int = 500) -> pd.DataFrame:
    """Fetches historical data from MetaTrader 5."""
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
    return pd.DataFrame(rates) if rates is not None else pd.DataFrame()
def indicator_pack(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds technical indicators to the dataframe."""
    if df.empty:
        return df
    
    # Exponential Moving Average (EMA)
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    # Average True Range (ATR)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    # Average Directional Index (ADX)
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.)
    tr14     = tr.rolling(14).sum()
    plus_di  = 100 * pd.Series(plus_dm).rolling(14).sum() / tr14
    minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / tr14
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    df["adx"] = dx.rolling(14).mean()
    # Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df
def get_trade_signal() -> Tuple[Optional[str], Dict[str, bool], Optional[pd.Series], Optional[pd.Series], Optional[float]]:
    """
    Checks trading conditions and returns a signal along with filter progress.
    Now includes RSI as a filter!
    """
    entry_df = indicator_pack(fetch_rates(TF_ENTRY, 300))
    trend_df = indicator_pack(fetch_rates(TF_TREND, 300))
    if entry_df.empty or trend_df.empty or len(entry_df) < 2:
        return None, {}, None, None, None  # Return if data feed has issues
    last, prev = entry_df.iloc[-1], entry_df.iloc[-2]
    trend_last = trend_df.iloc[-1]
    atr_median = entry_df["atr"].median()
    # --- Filter Conditions ---
    rsi_min, rsi_max = 30, 70
    buy_filters = {
        "crossed_up": prev.close < prev.ema and last.close > last.ema,
        "trend_up": trend_last.close > trend_last.ema,
        "atr_ok": last.atr > atr_median,
        "adx_ok": last.adx > 20,
        "rsi_ok": rsi_min < last.rsi < rsi_max
    }
    sell_filters = {
        "crossed_down": prev.close > prev.ema and last.close < last.ema,
        "trend_down": trend_last.close < trend_last.ema,
        "atr_ok": last.atr > atr_median,
        "adx_ok": last.adx > 20,
        "rsi_ok": rsi_min < last.rsi < rsi_max
    }
    # --- Logging Filter Progress ---
    print(f"{datetime.now()}: BUY filter progress:")
    for key, value in buy_filters.items():
        print(f"  {'✓' if value else '✗'} {key}: {value}")
    print(f"{sum(buy_filters.values())}/{len(buy_filters)} filters passed for BUY")
    print(f"{datetime.now()}: SELL filter progress:")
    for key, value in sell_filters.items():
        print(f"  {'✓' if value else '✗'} {key}: {value}")
    print(f"{sum(sell_filters.values())}/{len(sell_filters)} filters passed for SELL")
    # --- Determine Signal ---
    if all(buy_filters.values()):
        return "BUY", buy_filters, last, trend_last, atr_median
    if all(sell_filters.values()):
        return "SELL", sell_filters, last, trend_last, atr_median
    
    # Return BUY or SELL filter progress based on which had more signals, for context
    filters = buy_filters if sum(buy_filters.values()) > sum(sell_filters.values()) else sell_filters
    return None, filters, last, trend_last, atr_median
def build_features(candle: pd.Series, trend_candle: pd.Series, atr_median: float) -> Dict[str, Any]:
    """Constructs a feature dictionary for a given candle."""
    return {
        "timestamp": int(candle.time),
        "hour": datetime.fromtimestamp(candle.time).hour,
        "candle_size": candle.high - candle.low,
        "ema_distance": abs(candle.close - candle.ema),
        "atr": candle.atr,
        "adx": candle.adx,
        "rsi": candle.rsi,
        "volume": candle.tick_volume,
        "trend_above_ema": int(trend_candle.close > trend_candle.ema),
        "range_status": int(candle.adx < 20),
        "volatility_level": int(candle.atr > atr_median),
        "outcome": -1,  # Default: -1=pending, 0=loss, 1=win
        "entered": 0,
        "had_signal": 0,
    }
###############################################################################
# Block 4 : Learning engine
###############################################################################
def load_dataset() -> pd.DataFrame:
    """Loads the feature dataset from a CSV file and ensures required columns exist."""
    columns = [
        "timestamp", "hour", "candle_size", "ema_distance", "atr", "adx",
        "rsi", "volume", "trend_above_ema", "range_status",
        "volatility_level", "outcome", "entered", "had_signal"
    ]
    if not os.path.isfile(FEATURE_FILE):
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(FEATURE_FILE)
    # Add any missing columns (for backward compatibility)
    for col in columns:
        if col not in df.columns:
            df[col] = -1 if col == "outcome" else 0
    return df
def save_dataset(df: pd.DataFrame):
    """Saves the entire dataset back to the CSV file."""
    df.to_csv(FEATURE_FILE, index=False)
def train_model(df: pd.DataFrame) -> Optional[LogisticRegression]:
    """Trains and saves the logistic regression model."""
    feature_cols = ["hour", "candle_size", "ema_distance", "atr", "adx", "rsi",
                    "volume", "trend_above_ema", "range_status", "volatility_level"]
    
    # Train only on trades that have concluded (win or loss)
    trades = df[df.outcome.isin([0, 1])]
    if len(trades) < WARMUP_TRADES:
        print(f"{datetime.now()}: Not enough completed trades ({len(trades)}) to train. Need {WARMUP_TRADES}.")
        return None
        
    X, y = trades[feature_cols], trades["outcome"]
    model = LogisticRegression(max_iter=500, class_weight='balanced').fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model
def get_model() -> Optional[LogisticRegression]:
    """Loads a pre-trained model from disk."""
    return joblib.load(MODEL_FILE) if os.path.isfile(MODEL_FILE) else None

###############################################################################
# Block 4.5 : Robust Open Position Check
###############################################################################
def has_open_position(symbol, magic):
    """Check with MT5 if there's any open position for this symbol and magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False  # Treat as no position in case of error
    for pos in positions:
        if pos.magic == magic:
            return True
    return False

###############################################################################
# Block 4.6 : Risk-based position sizing
###############################################################################
def compute_lot_size(symbol: str, entry_price: float, sl_price: float, risk_amount: float) -> float:
    """Compute lot size so that loss at SL approximates risk_amount.

    Falls back to 0.01 if symbol info is unavailable or inputs are invalid.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.01
        tick_size = getattr(info, 'trade_tick_size', getattr(info, 'point', None))
        tick_value = getattr(info, 'trade_tick_value', None)
        if tick_value is None or tick_size is None or tick_size <= 0:
            return 0.01

        price_dist = abs(float(entry_price) - float(sl_price))
        if price_dist <= 0:
            return 0.01

        loss_per_lot = (price_dist / float(tick_size)) * float(tick_value)
        if loss_per_lot <= 0:
            return 0.01

        lot = float(risk_amount) / loss_per_lot
        # Enforce broker constraints
        vol_min = getattr(info, 'volume_min', 0.01)
        vol_max = getattr(info, 'volume_max', lot)
        vol_step = getattr(info, 'volume_step', 0.01)
        lot = max(lot, vol_min)
        lot = min(lot, vol_max)
        # Round down to step
        lot = (int(lot / vol_step)) * vol_step if vol_step > 0 else lot
        if lot <= 0:
            lot = vol_step if vol_step > 0 else 0.01
        return float(round(lot, 2))
    except Exception:
        return 0.01

###############################################################################
# Block 5 : Main loop
###############################################################################
if not mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD):
    print(f"{datetime.now()}: MT5 initialize failed – {mt5.last_error()}", flush=True)
    sys.exit()
print(f"{datetime.now()}: Connected – {SYMBOL}", flush=True)
model = get_model()
df = load_dataset()
open_trades: Dict[int, int] = {}

# Suppress pandas FutureWarning for clean logs
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

while True:
    print(f"{datetime.now()}: Loop tick", flush=True)
    try:
        now = datetime.now()
        current_hour = now.hour

        # === Only trade during betting hours ===
        if not (BETTING_START <= current_hour < BETTING_END):
            print(f"{now}: Outside betting hours ({BETTING_START}-{BETTING_END}), skipping trade logic.", flush=True)
            time.sleep(60)
            continue

        signal, filters, candle, trend_candle, atr_median = get_trade_signal()

        if candle is None:  # Data feed issue
            time.sleep(30) # Wait before retrying
            continue
        feat = build_features(candle, trend_candle, atr_median)
        feat["had_signal"] = int(signal is not None)

        df_new_row = pd.DataFrame([feat])
        # Only concatenate if df_new_row is not empty and not all-NA
        if not df_new_row.empty and df_new_row.notna().any().any():
            df = pd.concat([df, df_new_row], ignore_index=True)

        # ----- Model-based trade filtering -----
        prob = 0.5
        if model is not None:
            feature_cols = ["hour", "candle_size", "ema_distance", "atr", "adx", "rsi",
                            "volume", "trend_above_ema", "range_status", "volatility_level"]
            prob = model.predict_proba(df_new_row[feature_cols])[0, 1]
        total_trades_seen = (df["outcome"] != -1).sum()
        use_filter = (model is not None) and (total_trades_seen >= WARMUP_TRADES)
        accept_trade = (prob >= THRESHOLD) if use_filter else True

        # ========== Only one trade open at a time (robust, using MT5) ==========
        if signal and accept_trade:
            if has_open_position(SYMBOL, MT5_MAGIC):   # NEW: direct MT5 open position check
                print(f"{now}: Open trade detected by MT5, skipping new trade.", flush=True)
            else:
                tick = mt5.symbol_info_tick(SYMBOL)
                price = tick.ask if signal == "BUY" else tick.bid

                # IMPROVEMENT: Using ATR for a more dynamic Stop Loss and risk-based lot sizing
                sl_points = candle.atr * SL_ATR_MULTI
                tp_points = sl_points * RR_RATIO
                sl_price = price - sl_points if signal == "BUY" else price + sl_points
                tp_price = price + tp_points if signal == "BUY" else price - tp_points

                # Compute dynamic lot size based on account balance and SL distance
                account = mt5.account_info()
                if account is None:
                    print(f"{now}: Account info unavailable, using fixed LOT_SIZE.", flush=True)
                    lot = LOT_SIZE
                else:
                    risk_amount = float(account.balance) * RISK_PERCENT
                    lot = compute_lot_size(SYMBOL, price, sl_price, risk_amount)
                    if lot <= 0:
                        lot = LOT_SIZE

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": float(sl_price),
                    "tp": float(tp_price),
                    "deviation": 20,
                    "magic": MT5_MAGIC, # Use a magic number to identify trades from this bot
                }

                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    feat["entered"] = 1
                    open_trades[result.order] = feat["timestamp"]
                    print(f"{now}: {signal} ticket={result.order} prob={prob:.2%}", flush=True)
                else:
                    print(f"{now}: Order send failed, retcode={result.retcode}", flush=True)

        # ----- Monitor and Record Closed Trades -----
        for ticket in list(open_trades):
            deals = mt5.history_deals_get(ticket=ticket)
            if deals:
                # Assuming the first deal corresponds to the closing of the position
                profit = deals[0].profit
                signal_timestamp = open_trades.pop(ticket)

                # Update the original entry in the dataframe with the outcome
                df.loc[df['timestamp'] == signal_timestamp, 'outcome'] = int(profit > 0)

                print(f"{datetime.now()}: Ticket {ticket} closed. P/L: {profit:.2f}. Updating dataset.", flush=True)
                # ----- Retrain Model Periodically -----
                closed_trade_count = (df['outcome'] != -1).sum()
                if closed_trade_count > 0 and closed_trade_count % RETRAIN_EVERY == 0:
                    print(f"{datetime.now()}: Reached {closed_trade_count} closed trades. Retraining model...")

                    # Save OLD settings before learning
                    old_settings = current_settings_dict()
                    model = train_model(df)
                    if model:
                        print(f"{datetime.now()}: Model successfully retrained.", flush=True)
                        # Save NEW settings after learning
                        new_settings = current_settings_dict()
                        save_settings(old_settings, new_settings, closed_trade_count, model)
                        print(f"{datetime.now()}: Settings audit logged to {SETTING_FILE}", flush=True)

        # Persist all updates to disk
        save_dataset(df)
    except Exception as e:
        print(f"{datetime.now()}: Runtime error – {e}", flush=True)
    time.sleep(max(0, LOOP_SECONDS - (time.time() % LOOP_SECONDS)))
