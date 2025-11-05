"""
MT5 Fibonacci Auto-Trader (M5 primary, M15/M30 trend)
Requirements:
  - MetaTrader5 installed and logged-in
  - pip install MetaTrader5 pandas numpy
  - Algorithmic trading allowed on MT5 terminal
Test on demo first.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import math
import logging
from datetime import datetime, timedelta, timezone

# ---------- CONFIG ----------
SYMBOL = "XAUUSDm"           # symbol to trade
TIMEFRAME_PRIMARY = mt5.TIMEFRAME_M5
TIMEFRAME_TREND = mt5.TIMEFRAME_M15  # can swap to M30 if you prefer
LOOKBACK_BARS = 200         # bars to load for swing detection on M5
SWING_LOOKBACK = 50         # window to search swing high/low on M5
RISK_PERCENT = 0.5          # percent of equity to risk per trade (0.5 = 0.5%)
RISK_REWARD = 2.0           # desired RR ratio
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
RETRIEVE_SECONDS = 60       # how often to check (in seconds). M5 bars close every 5 minutes.
MAX_SPREAD_PIPS = 18.0       # max accepted spread in pips (symbol dependent)
MIN_LOT = None              # if left None, read from symbol_info
MAX_LOT = None
VOLUME_STEP = None
TRADE_COMMENT = "FiboAutoM5"
LOGFILE = "fibo_bot.log"
SWING_DEPTH = 8        # bars to each side for pivot detection
SWING_DEVIATION = 0.15 # minimum % move between swings

# ----------------------------

# Setup logger
logging.basicConfig(filename=LOGFILE, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


class MT5FiboTrader:
    def __init__(self, symbol):
        self.symbol = symbol
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed, code={mt5.last_error()}")
        logging.info("MT5 initialized")
        self._fetch_symbol_info()

    def _fetch_symbol_info(self):
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol {self.symbol} not found in Market Watch")
        logging.info(f"Symbol info: {self.symbol} tick_size={info.trade_tick_size} tick_value={info.trade_tick_value}"
                     f" contract={info.trade_contract_size} digits={info.digits}")
        global MIN_LOT, MAX_LOT, VOLUME_STEP
        MIN_LOT = info.volume_min if info.volume_min is not None else 0.01
        MAX_LOT = info.volume_max if info.volume_max is not None else 100
        VOLUME_STEP = info.volume_step if info.volume_step is not None else 0.01
        self.digits = info.digits
        self.tick_size = info.trade_tick_size if info.trade_tick_size else info.mt5_tick_size if hasattr(info,'mt5_tick_size') else 0.00001
        # In many brokers tick_value can be 0 for some symbols; fallback to calculation later
        self.tick_value = info.trade_tick_value if info.trade_tick_value else None

    def shutdown(self):
        mt5.shutdown()
        logging.info("MT5 shutdown")

    def get_bars(self, timeframe, count=LOOKBACK_BARS):
        utc_to = datetime.now(timezone.utc)
        rates = mt5.copy_rates_from(self.symbol, timeframe, utc_to, count)
        if rates is None:
            logging.error(f"No rates returned for {self.symbol} timeframe {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def detect_recent_swings(self, df, depth=10, deviation=0.2):
        """
        Detects dynamic swing highs/lows using simple ZigZag-like logic.
        - depth: number of bars on each side required to confirm a swing.
        - deviation: minimum price change (%) from last swing to accept a new one.
        Returns: dict with most recent swing high & low.
        """
        if df is None or len(df) < depth * 2 + 3:
            return None

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        idxs = np.arange(len(df))

        swing_highs = []
        swing_lows = []

        # Find local highs/lows
        for i in range(depth, len(df) - depth):
            if highs[i] == max(highs[i - depth:i + depth + 1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i - depth:i + depth + 1]):
                swing_lows.append((i, lows[i]))

        if not swing_highs or not swing_lows:
            return None

        # Keep only last significant swing high/low
        last_high_idx, last_high_price = swing_highs[-1]
        last_low_idx, last_low_price = swing_lows[-1]

        # Ensure order (last formed comes after previous opposite swing)
        if last_high_idx > last_low_idx:
            last_swing_type = "high"
        else:
            last_swing_type = "low"

        # Apply deviation filter (ignore minor zigzags)
        if last_swing_type == "high":
            diff_pct = abs((last_high_price - last_low_price) / last_low_price) * 100
        else:
            diff_pct = abs((last_high_price - last_low_price) / last_high_price) * 100

        if diff_pct < deviation:
            return None  # no strong enough swing movement

        logging.info(f"Dynamic swing: low={last_low_price:.3f} (idx={last_low_idx}), "
                     f"high={last_high_price:.3f} (idx={last_high_idx}), "
                     f"move={diff_pct:.2f}%")

        return {
            'low_price': float(last_low_price),
            'low_idx': int(last_low_idx),
            'high_price': float(last_high_price),
            'high_idx': int(last_high_idx)
    }

    def fib_levels(self, high, low):
        """
        Compute fib retracement levels from swing high/low.
        If high>low: use descending levels typical for pullbacks (retracement from high down to low)
        Return dict level->price
        """
        levels = {}
        for f in FIB_LEVELS:
            # Retracement price: high - (high - low) * f
            price = high - (high - low) * f
            levels[f] = float(round(price, self.digits))
        logging.info(f"Fibonacci levels computed between high={high} and low={low}: {levels}")
        return levels

    def get_trend(self, timeframe=TIMEFRAME_TREND, sma_period=50):
        df = self.get_bars(timeframe, count=sma_period + 10)
        if df is None or len(df) < sma_period + 2:
            logging.warning("Not enough bars to determine trend")
            return None
        df['sma'] = df['close'].rolling(window=sma_period).mean()
        # slope of SMA: last - previous
        slope = df['sma'].iloc[-1] - df['sma'].iloc[-3]
        trend = 'bull' if slope > 0 else 'bear' if slope < 0 else 'flat'
        logging.info(f"Higher timeframe ({timeframe}) SMA slope={slope:.8f} -> trend={trend}")
        return trend

    def price_near_level(self, price, level_price, tolerance_pips=5.0):
        """
        Check if price is within tolerance pips of level_price.
        tolerance_pips: number of pips (not points). For 5-digit EURUSD, pip = 0.0001
        """
        pip_value = 0.0001 if self.digits >= 5 else 0.01
        tolerance = tolerance_pips * pip_value
        return abs(price - level_price) <= tolerance

    def account_info(self):
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("Failed to get account info")
        return info

    def calc_lot_by_risk(self, stop_price, entry_price, risk_percent=RISK_PERCENT):
        """
        Calculate volume (lots) so that risk = risk_percent of equity.
        Use symbol tick_value where possible.
        """
        acc = mt5.account_info()
        equity = acc.equity
        risk_amount = equity * (risk_percent / 100.0)
        # Stop in pips:
        pip = 0.0001 if self.digits >= 5 else 0.01
        stop_pips = abs(entry_price - stop_price) / pip
        if stop_pips <= 0:
            logging.error("Invalid stop pips, can't calculate lot")
            return None

        # Derive pip value per lot:
        # Try to use tick_value/tick_size/trade_contract_size to compute pip value
        symbol = mt5.symbol_info(self.symbol)
        tick_value = getattr(symbol, 'trade_tick_value', None) or getattr(symbol, 'tick_value', None)
        tick_size = getattr(symbol, 'trade_tick_size', None) or getattr(symbol, 'tick_size', None)
        contract = getattr(symbol, 'trade_contract_size', 1.0) or 1.0

        if tick_value and tick_size:
            pip_value_per_lot = (tick_value / tick_size) * pip  # approximate
        else:
            # fallback approximate formula for FX: pip_value_per_lot ≈ (contract_size * pip) * lot / quote_currency_value
            # This is rough; better to use real tick_value from broker
            pip_value_per_lot = contract * pip

        if pip_value_per_lot == 0:
            pip_value_per_lot = 1.0  # safeguard

        lots = risk_amount / (stop_pips * pip_value_per_lot)
        # adjust to broker volume step
        volume = self.round_volume(lots)
        # enforce min/max
        volume = max(volume, MIN_LOT)
        volume = min(volume, MAX_LOT)
        logging.info(f"Equity={equity:.2f}, risk_amount={risk_amount:.2f}, stop_pips={stop_pips:.2f},"
                     f" pip_value_per_lot={pip_value_per_lot:.4f} -> volume={volume}")
        return volume

    def round_volume(self, volume):
        # round to nearest volume step
        step = VOLUME_STEP or 0.01
        volume = math.floor(volume / step) * step
        # ensure not zero
        if volume < step:
            volume = 0.0
        # round to step decimals
        decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        return round(volume, decimals)

    def check_spread_ok(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        spread = (tick.ask - tick.bid)
        pip = 0.0001 if self.digits >= 5 else 0.01
        spread_pips = spread / pip
        logging.info(f"Spread for {self.symbol}: {spread_pips:.2f} pips")
        return spread_pips <= MAX_SPREAD_PIPS

    def place_market_order(self, buy, volume, sl_price, tp_price):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logging.error("No tick data, can't place order")
            return None

        price = tick.ask if buy else tick.bid
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": deviation,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "comment": TRADE_COMMENT
        }

        result = mt5.order_send(request)
        if result is None:
            logging.error("order_send returned None")
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed, retcode={result.retcode}, result={result}")
            return result
        logging.info(f"Order placed successfully: ticket={result.order}")
        return result

    def run_once(self):
        # 1. Load data
        df_m5 = self.get_bars(TIMEFRAME_PRIMARY, count=LOOKBACK_BARS)
        if df_m5 is None:
            return
        df_latest = df_m5.iloc[-1]
        last_close = float(df_latest['close'])
        # 2. detect swings
        swings = self.detect_recent_swings(df_m5, depth=SWING_DEPTH, deviation=SWING_DEVIATION)
        if swings is None:
            return
        high = swings['high_price']
        low = swings['low_price']
        # Ensure we have correct orientation: prefer high > low
        if not (high > low):
            logging.warning("Detected high is not greater than low; skipping")
            return

        # 3. fib levels
        fibs = self.fib_levels(high, low)

        # 4. trend
        trend = self.get_trend(TIMEFRAME_TREND)
        if trend is None:
            return

        # 5. Check spread
        if not self.check_spread_ok():
            logging.warning("Spread too high; skipping this cycle")
            return

        # 6. Strategy conditions (example):
        # - If higher-timeframe trend is bullish, we look for long entries on retracement to 0.382-0.618 zone
        # - If bearish trend, look for shorts in same zone
        zone_low = fibs[0.382]
        zone_high = fibs[0.618]
        # allow some tolerance in pips
        tolerance_pips = 4.0

        # bullish
        if trend == 'bull':
            # price should be between zone_high and zone_low (remember: for fib computed as high - (high-low)*f,
            # 0.382 is higher price than 0.618 for this setting)
            if zone_high <= last_close <= zone_low or self.price_near_level(last_close, zone_low, tolerance_pips):
                logging.info("Bullish setup detected")
                # place buy: SL = low - small buffer, TP = entry + risk*RR
                sl = low - (2 * (0.0001 if self.digits >= 5 else 0.01))
                entry = last_close
                tp = entry + (entry - sl) * RISK_REWARD
                vol = self.calc_lot_by_risk(stop_price=sl, entry_price=entry)
                if vol and vol >= MIN_LOT:
                    self.place_market_order(buy=True, volume=vol, sl_price=sl, tp_price=tp)
                else:
                    logging.warning(f"Calculated volume {vol} not usable")
            else:
                logging.info("No bullish retracement -> price not in fib zone")
        elif trend == 'bear':
            # for shorts: price should be retracing up to 0.382-0.618 (but values inverted)
            zone_low_s = fibs[0.618]
            zone_high_s = fibs[0.382]
            if zone_high_s >= last_close >= zone_low_s or self.price_near_level(last_close, zone_low_s, tolerance_pips):
                logging.info("Bearish setup detected")
                sl = high + (2 * (0.0001 if self.digits >= 5 else 0.01))
                entry = last_close
                tp = entry - (sl - entry) * RISK_REWARD
                vol = self.calc_lot_by_risk(stop_price=sl, entry_price=entry)
                if vol and vol >= MIN_LOT:
                    self.place_market_order(buy=False, volume=vol, sl_price=sl, tp_price=tp)
                else:
                    logging.warning(f"Calculated volume {vol} not usable")
            else:
                logging.info("No bearish retracement -> price not in fib zone")
        else:
            logging.info("Trend flat; skipping trades")
        logging.info(f"SWING check: low={swings['low_price']:.3f}, high={swings['high_price']:.3f}, "
             f"bars={len(df_m5)} last_close={last_close:.3f}")


    def loop(self, interval_seconds=RETRIEVE_SECONDS):
        logging.info("Starting main loop. Press Ctrl+C to stop.")
        try:
            while True:
                start = time.time()
                try:
                    self.run_once()
                except Exception as e:
                    logging.exception(f"Error during run_once: {e}")
                elapsed = time.time() - start
                sleep_for = max(1, interval_seconds - elapsed)
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received, shutting down.")
            self.shutdown()


if __name__ == "__main__":
    trader = MT5FiboTrader(SYMBOL)
    trader.loop()
