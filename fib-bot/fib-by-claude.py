"""
MT5 Fibonacci Auto-Trader PRO v2.0 (M1 execution, M5 trend)
Production-ready with comprehensive risk management and error handling.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# ==================== CONFIGURATION ====================
@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    SYMBOL: str = "XAUUSDm"
    TIMEFRAME_PRIMARY: int = mt5.TIMEFRAME_M1
    TIMEFRAME_TREND: int = mt5.TIMEFRAME_M5
    LOOKBACK_BARS: int = 300
    
    # Swing detection
    SWING_DEPTH: int = 10
    SWING_DEVIATION: float = 0.15
    MAX_SWING_AGE_BARS: int = 50
    
    # Risk management
    RISK_PERCENT: float = 0.5
    RISK_REWARD: float = 2.0
    MAX_DAILY_LOSS_PERCENT: float = 2.0
    MAX_SPREAD_USD: float = 18.0
    
    # Execution
    RETRIEVE_SECONDS: int = 30
    SLIPPAGE_DEVIATION: int = 50
    MAGIC_NUMBER: int = 99999
    TRADE_COMMENT: str = "FiboAutoM1_v2"
    
    # Indicators
    SMA_PERIOD: int = 50
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 0.5
    
    # Logging
    LOGFILE: str = "fibo_bot_v2.log"
    LOG_LEVEL: int = logging.INFO

config = TradingConfig()

# ==================== LOGGING SETUP ====================
def setup_logging():
    """Configure logging with file and console handlers"""
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(config.LOGFILE)
    file_handler.setLevel(config.LOG_LEVEL)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

setup_logging()

# ==================== DATA CLASSES ====================
@dataclass
class SwingPoints:
    """Container for swing high/low data"""
    high_price: float
    low_price: float
    high_idx: int
    low_idx: int
    high_age: int
    low_age: int

@dataclass
class FibonacciLevels:
    """Container for Fibonacci retracement levels"""
    high: float
    low: float
    level_236: float
    level_382: float
    level_500: float
    level_618: float
    level_786: float
    
    def get_golden_zone(self) -> Tuple[float, float]:
        """Return the golden zone (0.618-0.382 range)"""
        return (min(self.level_618, self.level_382), 
                max(self.level_618, self.level_382))
    
    def log_levels(self):
        """Log all Fibonacci levels"""
        logging.info("=" * 50)
        logging.info("FIBONACCI LEVELS:")
        logging.info(f"  High:  {self.high:.5f}")
        logging.info(f"  0.236: {self.level_236:.5f}")
        logging.info(f"  0.382: {self.level_382:.5f} <-- Golden Zone")  # Changed arrow
        logging.info(f"  0.500: {self.level_500:.5f}")
        logging.info(f"  0.618: {self.level_618:.5f} <-- Golden Zone")  # Changed arrow
        logging.info(f"  0.786: {self.level_786:.5f}")
        logging.info(f"  Low:   {self.low:.5f}")
        logging.info("=" * 50)

@dataclass
class TradeSignal:
    """Container for trade signal data"""
    direction: str  # 'BUY' or 'SELL'
    entry: float
    stop_loss: float
    take_profit: float
    lot_size: float
    reason: str

# ==================== MAIN TRADER CLASS ====================
class MT5FiboTrader:
    """
    Advanced MT5 Fibonacci trading bot with comprehensive risk management
    """
    
    def __init__(self, symbol: str = config.SYMBOL):
        self.symbol = symbol
        self.last_bar_time = None
        self.daily_loss = 0.0
        self.today = datetime.now().date()
        self.trades_today = 0
        self.consecutive_losses = 0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Initialize MT5
        if not self._initialize_mt5():
            raise RuntimeError("MT5 initialization failed")
        
        self._fetch_symbol_info()
        logging.info("=" * 60)
        logging.info("MT5 FIBONACCI AUTO-TRADER v2.0 INITIALIZED")
        logging.info("=" * 60)

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection with retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            if mt5.initialize():
                logging.info("MT5 initialized successfully")
                return True
            logging.warning(f"MT5 init attempt {attempt + 1}/{max_attempts} failed")
            time.sleep(2)
        return False

    def _fetch_symbol_info(self):
        """Fetch and store symbol specifications"""
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise ValueError(f"Symbol {self.symbol} not found")
        
        # Make symbol visible if needed
        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                raise ValueError(f"Failed to select symbol {self.symbol}")
            info = mt5.symbol_info(self.symbol)
        
        self.digits = info.digits
        self.point = info.point
        self.contract_size = info.trade_contract_size
        self.volume_min = info.volume_min
        self.volume_max = info.volume_max
        self.volume_step = info.volume_step
        self.stops_level = info.trade_stops_level
        self.tick_value = info.trade_tick_value
        self.tick_size = info.trade_tick_size
        
        logging.info(f"Symbol Info: {self.symbol}")
        logging.info(f"  Digits: {self.digits}")
        logging.info(f"  Point: {self.point}")
        logging.info(f"  Contract Size: {self.contract_size}")
        logging.info(f"  Volume: {self.volume_min} - {self.volume_max} (step: {self.volume_step})")
        logging.info(f"  Stops Level: {self.stops_level}")

    def check_connection(self) -> bool:
        """Check MT5 connection health"""
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logging.error("MT5 terminal not connected!")
            return False
        
        if not terminal_info.connected:
            logging.error("MT5 terminal lost connection!")
            return False
        
        # Check symbol availability
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logging.error(f"Symbol {self.symbol} not available!")
            return False
        
        return True

    def shutdown(self):
        """Safely shutdown MT5 connection"""
        logging.info("Shutting down MT5 connection...")
        self._log_session_summary()
        mt5.shutdown()
        logging.info("MT5 shutdown complete")

    def _log_session_summary(self):
        """Log trading session summary"""
        logging.info("=" * 60)
        logging.info("SESSION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total Trades: {self.total_trades}")
        logging.info(f"Winning Trades: {self.winning_trades}")
        logging.info(f"Losing Trades: {self.losing_trades}")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            logging.info(f"Win Rate: {win_rate:.1f}%")
        logging.info(f"Daily Loss: ${self.daily_loss:.2f}")
        logging.info("=" * 60)

    # ==================== DATA RETRIEVAL ====================
    
    def get_bars(self, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """
        Retrieve historical bar data from MT5
        
        Args:
            timeframe: MT5 timeframe constant
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLC data or None on error
        """
        try:
            utc_to = datetime.now(timezone.utc)
            rates = mt5.copy_rates_from(self.symbol, timeframe, utc_to, count)
            
            if rates is None or len(rates) == 0:
                logging.warning(f"No data received for {self.symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logging.error(f"Error retrieving bars: {e}", exc_info=True)
            return None

    # ==================== TECHNICAL ANALYSIS ====================
    
    def detect_swing_points(self, df: pd.DataFrame) -> Optional[SwingPoints]:
        """
        Detect recent swing high and low points
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            SwingPoints object or None if no valid swings found
        """
        try:
            highs = df['high'].values
            lows = df['low'].values
            swing_highs = []
            swing_lows = []
            
            depth = config.SWING_DEPTH
            
            # Detect swing highs and lows
            for i in range(depth, len(df) - depth):
                # Swing high: highest point in the window
                if highs[i] == max(highs[i - depth:i + depth + 1]):
                    swing_highs.append((i, highs[i]))
                
                # Swing low: lowest point in the window
                if lows[i] == min(lows[i - depth:i + depth + 1]):
                    swing_lows.append((i, lows[i]))
            
            if not swing_highs or not swing_lows:
                logging.debug("No swing points detected")
                return None
            
            # Get most recent swings
            last_high_idx, last_high = swing_highs[-1]
            last_low_idx, last_low = swing_lows[-1]
            
            # Calculate ages
            last_idx = len(df) - 1
            high_age = last_idx - last_high_idx
            low_age = last_idx - last_low_idx
            
            # Check if swings are recent enough
            if high_age > config.MAX_SWING_AGE_BARS or low_age > config.MAX_SWING_AGE_BARS:
                logging.debug(f"Swings too old: high={high_age}, low={low_age} bars")
                return None
            
            # Check minimum swing range
            swing_range = abs((last_high - last_low) / last_low) * 100
            if swing_range < config.SWING_DEVIATION:
                logging.debug(f"Swing range too small: {swing_range:.2f}%")
                return None
            
            swing_points = SwingPoints(
                high_price=last_high,
                low_price=last_low,
                high_idx=last_high_idx,
                low_idx=last_low_idx,
                high_age=high_age,
                low_age=low_age
            )
            
            logging.info(f"Swing detected: High={last_high:.5f} ({high_age} bars), "
                        f"Low={last_low:.5f} ({low_age} bars), Range={swing_range:.2f}%")
            
            return swing_points
            
        except Exception as e:
            logging.error(f"Error detecting swings: {e}", exc_info=True)
            return None

    def calculate_fibonacci_levels(self, high: float, low: float) -> FibonacciLevels:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: Swing high price
            low: Swing low price
            
        Returns:
            FibonacciLevels object
        """
        diff = high - low
        
        levels = FibonacciLevels(
            high=round(high, self.digits),
            low=round(low, self.digits),
            level_236=round(high - diff * 0.236, self.digits),
            level_382=round(high - diff * 0.382, self.digits),
            level_500=round(high - diff * 0.500, self.digits),
            level_618=round(high - diff * 0.618, self.digits),
            level_786=round(high - diff * 0.786, self.digits)
        )
        
        levels.log_levels()
        return levels

    def get_trend(self) -> Optional[str]:
        """
        Determine market trend using SMA on higher timeframe
        
        Returns:
            'bull', 'bear', or None
        """
        try:
            df = self.get_bars(config.TIMEFRAME_TREND, config.SMA_PERIOD + 20)
            
            if df is None or len(df) < config.SMA_PERIOD:
                logging.warning("Insufficient data for trend calculation")
                return None
            
            # Calculate SMA
            df['sma'] = df['close'].rolling(config.SMA_PERIOD).mean()
            
            # Check trend slope
            sma_current = df['sma'].iloc[-1]
            sma_prev = df['sma'].iloc[-3]
            slope = sma_current - sma_prev
            
            # Current price vs SMA
            current_price = df['close'].iloc[-1]
            price_above_sma = current_price > sma_current
            
            # Determine trend
            if slope > 0 and price_above_sma:
                trend = 'bull'
            elif slope < 0 and not price_above_sma:
                trend = 'bear'
            else:
                trend = 'neutral'
            
            logging.info(f"Trend Analysis: {trend.upper()} (slope={slope:.5f}, "
                        f"price={'above' if price_above_sma else 'below'} SMA)")
            
            return trend if trend != 'neutral' else None
            
        except Exception as e:
            logging.error(f"Error calculating trend: {e}", exc_info=True)
            return None

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> float:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period (default from config)
            
        Returns:
            ATR value
        """
        if period is None:
            period = config.ATR_PERIOD
        
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            # Fallback if ATR is invalid
            if pd.isna(atr) or atr == 0:
                logging.warning("Invalid ATR, using high-low range")
                atr = df['high'].iloc[-1] - df['low'].iloc[-1]
            
            logging.info(f"ATR({period}): {atr:.5f}")
            return atr
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}", exc_info=True)
            # Fallback to simple range
            return df['high'].iloc[-1] - df['low'].iloc[-1]

    def detect_candlestick_pattern(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Detect bullish/bearish engulfing patterns
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (is_bullish, is_bearish)
        """
        try:
            if len(df) < 2:
                return False, False
            
            # Previous candle
            o1, c1, h1, l1 = df['open'].iloc[-2], df['close'].iloc[-2], \
                             df['high'].iloc[-2], df['low'].iloc[-2]
            
            # Current candle
            o2, c2, h2, l2 = df['open'].iloc[-1], df['close'].iloc[-1], \
                             df['high'].iloc[-1], df['low'].iloc[-1]
            
            # Bullish engulfing
            bullish = (c2 > o2 and  # Current is bullish
                      c1 < o1 and   # Previous was bearish
                      o2 <= c1 and  # Current opens at/below previous close
                      c2 >= o1)     # Current closes at/above previous open
            
            # Bearish engulfing
            bearish = (c2 < o2 and  # Current is bearish
                      c1 > o1 and   # Previous was bullish
                      o2 >= c1 and  # Current opens at/above previous close
                      c2 <= o1)     # Current closes at/below previous open
            
            if bullish:
                logging.info("📈 Bullish engulfing pattern detected")
            if bearish:
                logging.info("📉 Bearish engulfing pattern detected")
            
            return bullish, bearish
            
        except Exception as e:
            logging.error(f"Error detecting patterns: {e}", exc_info=True)
            return False, False

    # ==================== RISK MANAGEMENT ====================
    
    def check_spread(self) -> bool:
        """
        Check if spread is within acceptable limits
        
        Returns:
            True if spread is acceptable
        """
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logging.warning("Unable to get tick data")
                return False
            
            spread_points = tick.ask - tick.bid
            
            # Convert to USD
            spread_usd = (spread_points / self.tick_size) * self.tick_value
            
            is_ok = spread_usd <= config.MAX_SPREAD_USD
            
            if is_ok:
                logging.info(f"[OK] Spread OK: ${spread_usd:.2f}")  # Changed symbol
            else:
                logging.warning(f"[X] Spread too high: ${spread_usd:.2f} (max: ${config.MAX_SPREAD_USD})")  # Changed symbol
            
            return is_ok
            
        except Exception as e:
            logging.error(f"Error checking spread: {e}", exc_info=True)
            return False

    def calculate_lot_size(self, entry: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk percentage
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Lot size
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Unable to get account info")
                return self.volume_min
            
            equity = account_info.equity
            risk_amount = equity * (config.RISK_PERCENT / 100)
            
            # Calculate distance in points
            distance_points = abs(entry - stop_loss)
            
            # Calculate lot size
            # For gold: 1 lot = 100 oz, each point = tick_value
            lot = risk_amount / (distance_points / self.tick_size * self.tick_value)
            
            # Round to valid lot size
            lot = round(lot / self.volume_step) * self.volume_step
            
            # Apply limits
            lot = max(self.volume_min, min(lot, self.volume_max))
            
            risk_actual = (distance_points / self.tick_size * self.tick_value) * lot
            risk_pct = (risk_actual / equity) * 100
            
            logging.info(f"Position Sizing:")
            logging.info(f"  Equity: ${equity:.2f}")
            logging.info(f"  Risk Amount: ${risk_amount:.2f} ({config.RISK_PERCENT}%)")
            logging.info(f"  Distance: {distance_points:.5f}")
            logging.info(f"  Lot Size: {lot:.2f}")
            logging.info(f"  Actual Risk: ${risk_actual:.2f} ({risk_pct:.2f}%)")
            
            return lot
            
        except Exception as e:
            logging.error(f"Error calculating lot size: {e}", exc_info=True)
            return self.volume_min

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been reached
        
        Returns:
            True if trading is allowed
        """
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if current_date != self.today:
            logging.info(f"New trading day: {current_date}")
            self.daily_loss = 0.0
            self.today = current_date
            self.trades_today = 0
            self.consecutive_losses = 0
        
        account_info = mt5.account_info()
        if account_info is None:
            return False
        
        max_daily_loss = account_info.equity * (config.MAX_DAILY_LOSS_PERCENT / 100)
        
        if self.daily_loss >= max_daily_loss:
            logging.warning(f"⛔ Daily loss limit reached: ${self.daily_loss:.2f} / ${max_daily_loss:.2f}")
            return False
        
        # Additional safety: stop after 3 consecutive losses
        if self.consecutive_losses >= 3:
            logging.warning(f"⛔ Too many consecutive losses: {self.consecutive_losses}")
            return False
        
        logging.info(f"Daily Loss: ${self.daily_loss:.2f} / ${max_daily_loss:.2f} "
                    f"(Consecutive losses: {self.consecutive_losses})")
        
        return True

    def has_open_position(self) -> bool:
        """
        Check if there's an open position for this symbol
        
        Returns:
            True if position exists
        """
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            
            if positions is None:
                return False
            
            open_positions = len(positions)
            
            if open_positions > 0:
                logging.info(f"Open positions: {open_positions}")
                for pos in positions:
                    logging.info(f"  Position: {pos.type} | Volume: {pos.volume} | "
                               f"Price: {pos.price_open} | Profit: ${pos.profit:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking positions: {e}", exc_info=True)
            return True  # Err on the side of caution

    def validate_trade_parameters(self, is_buy: bool, entry: float, 
                              stop_loss: float, take_profit: float) -> bool:
        """
        Validate trade parameters before execution
        
        Args:
            is_buy: True for buy, False for sell
            entry: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if parameters are valid
        """
        try:
            # Check minimum distance from stops level
            min_distance = self.stops_level * self.point
            
            sl_distance = abs(entry - stop_loss)
            tp_distance = abs(entry - take_profit)
            
            if min_distance > 0:
                if sl_distance < min_distance:
                    logging.warning(f"[X] SL too close: {sl_distance:.5f} < {min_distance:.5f}")
                    return False
                
                if tp_distance < min_distance:
                    logging.warning(f"[X] TP too close: {tp_distance:.5f} < {min_distance:.5f}")
                    return False
            
            # Validate direction logic
            if is_buy:
                if stop_loss >= entry:
                    logging.error(f"[X] Invalid BUY: SL ({stop_loss}) >= Entry ({entry})")
                    return False
                if take_profit <= entry:
                    logging.error(f"[X] Invalid BUY: TP ({take_profit}) <= Entry ({entry})")
                    return False
            else:
                if stop_loss <= entry:
                    logging.error(f"[X] Invalid SELL: SL ({stop_loss}) <= Entry ({entry})")
                    return False
                if take_profit >= entry:
                    logging.error(f"[X] Invalid SELL: TP ({take_profit}) >= Entry ({entry})")
                    return False
            
            # Check risk:reward ratio
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            actual_rr = reward / risk if risk > 0 else 0
            
            if actual_rr < config.RISK_REWARD * 0.9:  # Allow 10% tolerance
                logging.warning(f"[X] Risk:Reward too low: {actual_rr:.2f} < {config.RISK_REWARD}")
                return False
            
            logging.info(f"[OK] Trade validation passed (R:R = 1:{actual_rr:.2f})")
            return True
            
        except Exception as e:
            logging.error(f"Error validating trade: {e}", exc_info=True)
            return False

    # ==================== ORDER EXECUTION ====================
    
    def place_order(self, signal: TradeSignal) -> bool:
        """
        Execute a trade order
        
        Args:
            signal: TradeSignal object with order details
            
        Returns:
            True if order was successful
        """
        try:
            is_buy = signal.direction == 'BUY'
            
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logging.error("Unable to get current price")
                return False
            
            price = tick.ask if is_buy else tick.bid
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": signal.lot_size,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": config.SLIPPAGE_DEVIATION,
                "magic": config.MAGIC_NUMBER,
                "comment": config.TRADE_COMMENT,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Log order details
            logging.info("=" * 60)
            logging.info(f"🎯 PLACING {signal.direction} ORDER")
            logging.info("=" * 60)
            logging.info(f"Reason: {signal.reason}")
            logging.info(f"Entry: {price:.5f}")
            logging.info(f"Stop Loss: {signal.stop_loss:.5f}")
            logging.info(f"Take Profit: {signal.take_profit:.5f}")
            logging.info(f"Lot Size: {signal.lot_size}")
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logging.error("Order send returned None")
                return False
            
            # Check result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"❌ Order failed: {result.retcode} - {result.comment}")
                return False
            
            # Success!
            slippage = abs(result.price - price)
            logging.info("=" * 60)
            logging.info(f"✅ ORDER EXECUTED SUCCESSFULLY")
            logging.info("=" * 60)
            logging.info(f"Order: #{result.order}")
            logging.info(f"Deal: #{result.deal}")
            logging.info(f"Filled Price: {result.price:.5f}")
            logging.info(f"Slippage: {slippage:.5f}")
            logging.info(f"Volume: {result.volume}")
            logging.info("=" * 60)
            
            # Update statistics
            self.total_trades += 1
            self.trades_today += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error placing order: {e}", exc_info=True)
            return False

    # ==================== SIGNAL GENERATION ====================
    
    def generate_signal(self, df: pd.DataFrame, fib_levels: FibonacciLevels, 
                       trend: str, atr: float) -> Optional[TradeSignal]:
        """
        Generate trading signal based on conditions
        
        Args:
            df: DataFrame with price data
            fib_levels: Fibonacci levels
            trend: Market trend
            atr: Average True Range
            
        Returns:
            TradeSignal object or None
        """
        try:
            current_price = df['close'].iloc[-1]
            
            # Get golden zone (0.618 - 0.382)
            zone_low, zone_high = fib_levels.get_golden_zone()
            
            # Check if price is in golden zone
            in_golden_zone = zone_low <= current_price <= zone_high
            
            # Get candlestick patterns
            is_bullish, is_bearish = self.detect_candlestick_pattern(df)
            
            logging.info(f"Signal Analysis:")
            logging.info(f"  Current Price: {current_price:.5f}")
            logging.info(f"  Golden Zone: {zone_low:.5f} - {zone_high:.5f}")
            logging.info(f"  In Zone: {in_golden_zone}")
            logging.info(f"  Trend: {trend}")
            logging.info(f"  Bullish Pattern: {is_bullish}")
            logging.info(f"  Bearish Pattern: {is_bearish}")
            
            # BUY Signal
            if trend == 'bull' and is_bullish and in_golden_zone:
                stop_loss = fib_levels.low - (config.ATR_MULTIPLIER * atr)
                risk = current_price - stop_loss
                take_profit = current_price + (risk * config.RISK_REWARD)
                lot_size = self.calculate_lot_size(current_price, stop_loss)
                
                if not self.validate_trade_parameters(True, current_price, stop_loss, take_profit):
                    return None
                
                return TradeSignal(
                    direction='BUY',
                    entry=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=lot_size,
                    reason="Bullish trend + Bullish pattern + Golden zone"
                )
            
            # SELL Signal
            elif trend == 'bear' and is_bearish and in_golden_zone:
                stop_loss = fib_levels.high + (config.ATR_MULTIPLIER * atr)
                risk = stop_loss - current_price
                take_profit = current_price - (risk * config.RISK_REWARD)
                lot_size = self.calculate_lot_size(current_price, stop_loss)
                
                if not self.validate_trade_parameters(False, current_price, stop_loss, take_profit):
                    return None
                
                return TradeSignal(
                    direction='SELL',
                    entry=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=lot_size,
                    reason="Bearish trend + Bearish pattern + Golden zone"
                )
            
            return None
            
        except Exception as e:
            logging.error(f"Error generating signal: {e}", exc_info=True)
            return None

    # ==================== MAIN LOGIC ====================
    
    def run_once(self):
        """Execute one iteration of the trading logic"""
        try:
            logging.info("\n" + "=" * 60)
            logging.info(f"SCAN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info("=" * 60)
            
            # Check connection
            if not self.check_connection():
                logging.error("Connection check failed")
                return
            
            # Check daily loss limit
            if not self.check_daily_loss_limit():
                logging.warning("Daily loss limit reached, skipping")
                return
            
            # Check for existing positions
            if self.has_open_position():
                logging.info("Position already open, skipping scan")
                return
            
            # Get price data
            df = self.get_bars(config.TIMEFRAME_PRIMARY, config.LOOKBACK_BARS)
            if df is None or len(df) < 100:
                logging.warning("Insufficient bar data")
                return
            
            # Check if new bar
            last_time = df['time'].iloc[-1]
            if self.last_bar_time == last_time:
                logging.debug("Waiting for new bar...")
                return
            
            self.last_bar_time = last_time
            logging.info(f"New bar detected: {last_time}")
            
            # Detect swing points
            swing_points = self.detect_swing_points(df)
            if swing_points is None:
                logging.info("No valid swing points detected")
                return
            
            # Calculate Fibonacci levels
            fib_levels = self.calculate_fibonacci_levels(
                swing_points.high_price,
                swing_points.low_price
            )
            
            # Get market trend
            trend = self.get_trend()
            if trend is None:
                logging.info("No clear trend, skipping")
                return
            
            # Check spread
            if not self.check_spread():
                logging.info("Spread too high, skipping")
                return
            
            # Calculate ATR
            atr = self.calculate_atr(df)
            
            # Generate trading signal
            signal = self.generate_signal(df, fib_levels, trend, atr)
            
            if signal is None:
                logging.info("No trading signal generated")
                return
            
            # Execute trade
            success = self.place_order(signal)
            
            if success:
                logging.info("✅ Trade executed successfully!")
            else:
                logging.error("❌ Trade execution failed")
            
        except Exception as e:
            logging.error(f"Error in run_once: {e}", exc_info=True)

    def loop(self):
        """Main trading loop"""
        logging.info("\n" + "=" * 70)
        logging.info("STARTING MT5 FIBONACCI AUTO-TRADER v2.0")
        logging.info("=" * 70)
        logging.info(f"Symbol: {config.SYMBOL}")
        logging.info(f"Primary Timeframe: M{config.TIMEFRAME_PRIMARY}")
        logging.info(f"Trend Timeframe: M{config.TIMEFRAME_TREND}")
        logging.info(f"Risk per Trade: {config.RISK_PERCENT}%")
        logging.info(f"Risk:Reward Ratio: 1:{config.RISK_REWARD}")
        logging.info(f"Max Daily Loss: {config.MAX_DAILY_LOSS_PERCENT}%")
        logging.info(f"Scan Interval: {config.RETRIEVE_SECONDS}s")
        logging.info("=" * 70)
        logging.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                time.sleep(config.RETRIEVE_SECONDS)
                
        except KeyboardInterrupt:
            logging.info("\n" + "=" * 60)
            logging.info("SHUTDOWN REQUESTED BY USER")
            logging.info("=" * 60)
            self.shutdown()
        except Exception as e:
            logging.error(f"Fatal error in main loop: {e}", exc_info=True)
            self.shutdown()


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        bot = MT5FiboTrader(config.SYMBOL)
        bot.loop()
    except Exception as e:
        logging.error(f"Failed to start bot: {e}", exc_info=True)