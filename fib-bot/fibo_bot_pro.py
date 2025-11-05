"""
MT5 Fibonacci Auto-Trader PRO (M1 execution, M5 trend)
Logs Fibonacci levels clearly for visual reference.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import math
import logging
from datetime import datetime, timezone

# ---------- CONFIG ----------
SYMBOL = "XAUUSDm"
TIMEFRAME_PRIMARY = mt5.TIMEFRAME_M1
TIMEFRAME_TREND = mt5.TIMEFRAME_M5
LOOKBACK_BARS = 300
SWING_DEPTH = 10
SWING_DEVIATION = 0.15
RISK_PERCENT = 0.5
RISK_REWARD = 2.0
MAX_SPREAD_USD = 2.5
RETRIEVE_SECONDS = 30
TRADE_COMMENT = "FiboAutoM1"
LOGFILE = "fibo_bot_visual.log"
# ----------------------------

logging.basicConfig(filename=LOGFILE, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.getLogger().addHandler(console)


class MT5FiboTrader:
    def __init__(self, symbol):
        self.symbol = symbol
        self.last_bar_time = None
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed.")
        logging.info("MT5 initialized")
        self._fetch_symbol_info()

    def _fetch_symbol_info(self):
        info = mt5.symbol_info(self.symbol)
        self.digits = info.digits
        self.contract = info.trade_contract_size or 100
        self.volume_min = info.volume_min or 0.01
        self.volume_step = info.volume_step or 0.01
        logging.info(f"{self.symbol} info loaded: digits={self.digits}, contract={self.contract}")

    def shutdown(self):
        mt5.shutdown()
        logging.info("MT5 shutdown")

    def get_bars(self, timeframe, count):
        utc_to = datetime.now(timezone.utc)
        rates = mt5.copy_rates_from(self.symbol, timeframe, utc_to, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def detect_recent_swings(self, df, depth=SWING_DEPTH, deviation=SWING_DEVIATION):
        highs, lows = df['high'].values, df['low'].values
        swing_highs, swing_lows = [], []

        for i in range(depth, len(df) - depth):
            if highs[i] == max(highs[i - depth:i + depth + 1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i - depth:i + depth + 1]):
                swing_lows.append((i, lows[i]))

        if not swing_highs or not swing_lows:
            return None
        last_high_idx, last_high = swing_highs[-1]
        last_low_idx, last_low = swing_lows[-1]

        if abs((last_high - last_low) / last_low) * 100 < deviation:
            return None

        return {'high_price': last_high, 'low_price': last_low, 'high_idx': last_high_idx, 'low_idx': last_low_idx}

    def fib_levels(self, high, low):
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        fibs = {f: round(high - (high - low) * f, self.digits) for f in fib_ratios}
        fibs['high'] = high
        fibs['low'] = low
        logging.info("Fibonacci levels:")
        for k, v in fibs.items():
            if isinstance(k, float):
                logging.info(f"  {k:<5}: {v}")
        return fibs

    def get_trend(self, timeframe=TIMEFRAME_TREND, sma_period=50):
        df = self.get_bars(timeframe, sma_period + 10)
        if df is None or len(df) < sma_period:
            return None
        df['sma'] = df['close'].rolling(sma_period).mean()
        slope = df['sma'].iloc[-1] - df['sma'].iloc[-3]
        trend = 'bull' if slope > 0 else 'bear'
        logging.info(f"Trend({timeframe}) slope={slope:.4f} => {trend}")
        return trend

    def calc_atr(self, df, period=14):
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def spread_ok(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False
        spread_usd = tick.ask - tick.bid
        logging.info(f"Spread = {spread_usd:.2f} USD")
        return spread_usd <= MAX_SPREAD_USD

    def calc_lot_by_risk(self, sl, entry, risk_percent=RISK_PERCENT):
        acc = mt5.account_info()
        equity = acc.equity
        risk_amount = equity * (risk_percent / 100)
        distance = abs(entry - sl)
        lot = risk_amount / (distance * self.contract / 100.0)
        lot = max(self.volume_min, round(lot / self.volume_step) * self.volume_step)
        return lot

    def candle_pattern(self, df):
        o1, c1 = df['open'].iloc[-2], df['close'].iloc[-2]
        o2, c2 = df['open'].iloc[-1], df['close'].iloc[-1]
        bullish = c2 > o2 and o2 < c1 and c2 > o1
        bearish = c2 < o2 and o2 > c1 and c2 < o1
        return bullish, bearish

    def place_order(self, buy, vol, sl, tp):
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if buy else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": vol,
            "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 50,
            "magic": 99999,
            "comment": TRADE_COMMENT
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Trade failed: {result}")
        else:
            logging.info(f"{'BUY' if buy else 'SELL'} placed @ {price:.3f} vol={vol}")
        return result

    def run_once(self):
        df = self.get_bars(TIMEFRAME_PRIMARY, LOOKBACK_BARS)
        if df is None or len(df) < 100:
            return
        last_time = df['time'].iloc[-1]
        if self.last_bar_time == last_time:
            return
        self.last_bar_time = last_time

        swings = self.detect_recent_swings(df)
        if not swings:
            return
        fibs = self.fib_levels(swings['high_price'], swings['low_price'])

        trend = self.get_trend()
        if not trend or not self.spread_ok():
            return

        last_close = df['close'].iloc[-1]
        atr = self.calc_atr(df)
        bull, bear = self.candle_pattern(df)
        zone_low, zone_high = fibs[0.618], fibs[0.382]
        in_zone = zone_low <= last_close <= zone_high

        if trend == 'bull' and bull and in_zone:
            sl = fibs['low'] - 0.5 * atr
            tp = last_close + (last_close - sl) * RISK_REWARD
            vol = self.calc_lot_by_risk(sl, last_close)
            self.place_order(True, vol, sl, tp)

        elif trend == 'bear' and bear and in_zone:
            sl = fibs['high'] + 0.5 * atr
            tp = last_close - (sl - last_close) * RISK_REWARD
            vol = self.calc_lot_by_risk(sl, last_close)
            self.place_order(False, vol, sl, tp)

        logging.info(f"[CHECK] close={last_close:.2f}, trend={trend}, "
                     f"zone={zone_low:.2f}-{zone_high:.2f}, ATR={atr:.2f}")

    def loop(self):
        logging.info("Starting loop on M1...")
        try:
            while True:
                self.run_once()
                time.sleep(RETRIEVE_SECONDS)
        except KeyboardInterrupt:
            self.shutdown()


if __name__ == "__main__":
    bot = MT5FiboTrader(SYMBOL)
    bot.loop()
