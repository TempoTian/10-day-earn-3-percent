#!/usr/bin/env python3
"""
Enhanced Chinese Stock Analyzer with ML Predictions
Analyzes Chinese A-shares and H-shares with technical indicators and ML predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
import pickle
import os
from datetime import datetime, timedelta
from chinese_stock_downloader import ChineseStockDownloader
import time

warnings.filterwarnings('ignore')

class ChineseStockAnalyzer:
    def __init__(self, data_source='yfinance'):
        self.data = None
        self.symbol = None
        self.market_type = None
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.ml_model_used = False
        self.model_dir = "chinese_models"
        self.model_info = {}
        self.best_params = {}
        self.downloader = ChineseStockDownloader(data_source)
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def get_model_filename(self, symbol, market='A'):
        """
        Generate model filename for a specific stock
        """
        return f"{self.model_dir}/{symbol}_{market}_model.pkl"
    
    def save_model(self, symbol, market='A'):
        """
        Save trained model to file
        """
        if self.model is None:
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'symbol': symbol,
                'market': market,
                'last_trained': datetime.now(),
                'data_points': len(self.data) if self.data is not None else 0,
                'features': self.create_ml_features(),
                'model_info': self.model_info
            }
            
            filename = self.get_model_filename(symbol, market)
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, symbol, market='A'):
        """
        Load trained model from file
        """
        try:
            filename = self.get_model_filename(symbol, market)
            
            if not os.path.exists(filename):
                print(f"Model file not found: {filename}")
                return False
            
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if model is still valid (not too old)
            last_trained = model_data['last_trained']
            days_old = (datetime.now() - last_trained).days
            
            if days_old > 30:  # Model older than 30 days
                print(f"Model is {days_old} days old, will retrain")
                return False
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_info = model_data.get('model_info', {})
            
            print(f"✅ Model loaded from {filename} (trained {days_old} days ago)")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def update_model(self, symbol, market='A'):
        """
        Update existing model with new data
        """
        if self.model is None:
            print("No model to update")
            return False
        
        try:
            # Download new data since last training
            if 'last_trained' in self.model_info:
                last_trained_date = self.model_info['last_trained']
                if isinstance(last_trained_date, str):
                    last_trained_date = datetime.fromisoformat(last_trained_date)
                
                # Download data from last training date
                ticker = yf.Ticker(self.get_chinese_stock_symbol(symbol, market))
                new_data = ticker.history(start=last_trained_date.date(), end=None)
                
                if len(new_data) < 10:  # Need at least 10 new data points
                    print("Insufficient new data for model update")
                    return False
                
                # Combine with existing data
                if self.data is not None:
                    self.data = pd.concat([self.data, new_data]).drop_duplicates()
                else:
                    self.data = new_data
                
                # Retrain model
                success = self.train_ml_model(holding_period=10, profit_threshold=0.03)
                
                if success:
                    self.save_model(symbol, market)
                    print(f"✅ Model updated with {len(new_data)} new data points")
                
                return success
            
        except Exception as e:
            print(f"Error updating model: {str(e)}")
            return False
    
    def get_chinese_stock_symbol(self, symbol, market='A'):
        """
        Convert Chinese stock symbols to proper format
        """
        symbol = symbol.upper().strip()
        
        # A-shares (Shanghai and Shenzhen)
        if market.upper() == 'A':
            if symbol.startswith('6'):
                return f"{symbol}.SS"  # Shanghai
            elif symbol.startswith(('0', '3')):
                return f"{symbol}.SZ"  # Shenzhen
            else:
                if len(symbol) == 6:
                    if symbol.startswith('6'):
                        return f"{symbol}.SS"
                    else:
                        return f"{symbol}.SZ"
        
        # H-shares (Hong Kong)
        elif market.upper() == 'H':
            if not symbol.endswith('.HK'):
                return f"{symbol}.HK"
            return symbol
        
        return symbol
    
    def download_chinese_stock_data(self, symbol, market='A', period="2y"):
        """Download Chinese stock data using the new downloader"""
        # Add delay to avoid server resistance
        time.sleep(0.5)
        
        self.symbol = symbol
        self.market_type = market
        
        # Get stock name
        stock_name = self.downloader.get_stock_name(symbol, market)
        
        # Download data
        data = self.downloader.download_stock_data(symbol, market, period)
        
        if data is not None:
            self.data = data
            print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"Stock: {symbol} - {stock_name}")
            return True, stock_name
        else:
            return False, symbol
    
    def calculate_chinese_indicators(self):
        """Calculate technical indicators for Chinese stocks"""
        if self.data is None or self.data.empty:
            print("❌ No data available for indicator calculation")
            return False
        
        try:
            # Ensure we have the required columns (handle both yfinance and akshare formats)
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col.lower() for col in self.data.columns]
            
            # Check if we have the required columns
            missing_columns = [col for col in required_columns if col not in available_columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                print(f"Available columns: {list(self.data.columns)}")
                return False
            
            # Use lowercase column names for consistency
            data = self.data.copy()
            
            # Calculate basic indicators
            data['Returns'] = data['close'].pct_change()
            data['Volume_MA_20'] = data['volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['volume'] / data['Volume_MA_20']
            
            # Moving averages
            data['SMA_20'] = data['close'].rolling(window=20).mean()
            data['SMA_50'] = data['close'].rolling(window=50).mean()
            data['EMA_12'] = data['close'].ewm(span=12).mean()
            data['EMA_26'] = data['close'].ewm(span=26).mean()
            
            # Momentum indicators
            data['Price_Momentum_5'] = data['close'] / data['close'].shift(5) - 1
            data['Price_Momentum_10'] = data['close'] / data['close'].shift(10) - 1
            data['Price_Momentum_20'] = data['close'] / data['close'].shift(20) - 1
            data['Price_Momentum_3'] = data['close'] / data['close'].shift(3) - 1
            
            # Volatility
            data['Volatility_10'] = data['Returns'].rolling(window=10).std()
            data['Volatility_20'] = data['Returns'].rolling(window=20).std()
            data['Volatility_50'] = data['Returns'].rolling(window=50).std()
            
            # Bollinger Bands
            data['BB_Upper'] = data['SMA_20'] + (data['close'].rolling(window=20).std() * 2)
            data['BB_Lower'] = data['SMA_20'] - (data['close'].rolling(window=20).std() * 2)
            data['BB_Position'] = (data['close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Support and resistance
            data['Support_20'] = data['low'].rolling(window=20).min()
            data['Resistance_20'] = data['high'].rolling(window=20).max()
            
            # RSI (multiple timeframes)
            data['RSI'] = self.calculate_rsi(data['close'])
            data['RSI_7'] = self.calculate_rsi(data['close'], window=7)
            data['RSI_21'] = self.calculate_rsi(data['close'], window=21)
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Normalized MACD (relative to price for cross-stock comparability)
            data['MACD_Norm'] = data['MACD'] / data['close']
            data['MACD_Signal_Norm'] = data['MACD_Signal'] / data['close']
            data['MACD_Hist_Norm'] = data['MACD_Histogram'] / data['close']
            
            # ATR (Average True Range) for volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR'] = true_range.rolling(window=14).mean()
            data['ATR_Ratio'] = data['ATR'] / data['close']
            
            # Stochastic Oscillator
            low_14 = data['low'].rolling(window=14).min()
            high_14 = data['high'].rolling(window=14).max()
            data['Stoch_K'] = ((data['close'] - low_14) / (high_14 - low_14)) * 100
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
            
            # ADX (Trend Strength)
            plus_dm = data['high'].diff().clip(lower=0)
            minus_dm = (-data['low'].diff()).clip(lower=0)
            atr_14 = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
            data['ADX'] = dx.rolling(window=14).mean()
            data['DI_Plus'] = plus_di
            data['DI_Minus'] = minus_di
            
            # RSI slope (direction of RSI movement)
            data['RSI_Slope'] = data['RSI'] - data['RSI'].shift(3)
            
            # OBV and its rate of change
            obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
            data['OBV_ROC_5'] = obv.pct_change(periods=5)
            data['OBV_ROC_10'] = obv.pct_change(periods=10)
            
            # Volume-price divergence
            data['Vol_Price_Divergence'] = (
                data['close'].pct_change(5).fillna(0) * -1 +
                data['volume'].pct_change(5).fillna(0)
            )
            
            # Price acceleration
            data['Price_Acceleration'] = data['Price_Momentum_5'] - data['Price_Momentum_5'].shift(5)
            
            # Candle body ratio
            price_range = data['high'] - data['low']
            data['Body_Ratio'] = (data['close'] - data['open']) / price_range.replace(0, np.nan)
            
            # ===== 筹码集中度 (Chip Concentration) proxy features =====
            # These approximate accumulation/distribution patterns from OHLCV data.
            # True chip data (股东户数) requires East Money / akshare special APIs.
            
            # 1. Price Consolidation Index — tight range = chips being collected
            #    (low ATR relative to recent average means price is consolidating)
            atr_mean_50 = data['ATR'].rolling(window=50).mean()
            data['Chip_Consolidation'] = 1 - (data['ATR'] / atr_mean_50).clip(upper=2)
            
            # 2. Volume Shrink During Consolidation — declining volume + tight range
            #    is the classic accumulation (吸筹) pattern
            vol_trend_20 = data['volume'].rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / (x.mean() + 1e-10) if len(x) == 20 else 0,
                raw=False
            )
            price_range_norm = (data['high'].rolling(20).max() - data['low'].rolling(20).min()) / data['close']
            data['Chip_Accumulation'] = (-vol_trend_20) * (1 - price_range_norm.clip(upper=1))
            
            # 3. Chaikin Money Flow (CMF) — measures buying/selling pressure
            #    Positive CMF = money flowing in = accumulation
            mfv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / \
                  (data['high'] - data['low']).replace(0, np.nan) * data['volume']
            data['CMF_20'] = mfv.rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
            
            # 4. Volume Concentration Ratio — ratio of volume on narrow-range days
            #    vs wide-range days. High ratio = stealth accumulation
            median_range = price_range.rolling(window=20).median()
            narrow_mask = (price_range < median_range).astype(float)
            wide_mask = (price_range >= median_range).astype(float)
            narrow_vol = (data['volume'] * narrow_mask).rolling(window=20).sum()
            wide_vol = (data['volume'] * wide_mask).rolling(window=20).sum()
            data['Chip_Vol_Concentration'] = narrow_vol / wide_vol.replace(0, np.nan)
            
            # 5. Smart Money Indicator — large-volume days with small price change
            #    suggest institutional accumulation (big volume, small move = absorbing supply)
            abs_return = data['Returns'].abs()
            median_return = abs_return.rolling(window=20).median()
            median_vol = data['volume'].rolling(window=20).median()
            smart_money_day = ((data['volume'] > median_vol * 1.5) & (abs_return < median_return)).astype(float)
            data['Chip_Smart_Money'] = smart_money_day.rolling(window=20).mean()
            
            # 6. Turnover Decline Trend — steadily declining turnover during
            #    sideways price = chips moving from weak hands to strong hands
            if 'turnover_rate' in data.columns:
                turnover = pd.to_numeric(data['turnover_rate'], errors='coerce')
            else:
                turnover = data['volume'] / data['volume'].rolling(window=250).mean()
            turnover_slope = turnover.rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0,
                raw=False
            )
            data['Chip_Turnover_Decline'] = -turnover_slope
            
            # 7. Price-Volume Divergence Score — price flat/up but volume declining
            #    is a strong accumulation signal
            price_slope = data['close'].rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x / x.iloc[0], 1)[0] if len(x) == 20 else 0,
                raw=False
            )
            data['Chip_PV_Divergence'] = price_slope * data['Chip_Turnover_Decline']
            
            self.data = data
            print("Chinese market indicators calculated successfully!")
            return True
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return False
    
    def create_ml_features(self):
        """Create expanded feature set for ML model using only normalized/relative features"""
        if self.data is None or len(self.data) < 50:
            return None
        
        try:
            # Price vs moving averages
            self.data['Price_vs_SMA20'] = (self.data['close'] - self.data['SMA_20']) / self.data['SMA_20']
            self.data['Price_vs_SMA50'] = (self.data['close'] - self.data['SMA_50']) / self.data['SMA_50']
            self.data['SMA_20_vs_50'] = (self.data['SMA_20'] - self.data['SMA_50']) / self.data['SMA_50']
            
            # Market regime features
            self.data['Bull_Market'] = ((self.data['close'] > self.data['SMA_20']) & (self.data['SMA_20'] > self.data['SMA_50'])).astype(int)
            self.data['Bear_Market'] = ((self.data['close'] < self.data['SMA_20']) & (self.data['SMA_20'] < self.data['SMA_50'])).astype(int)
            
            # Core feature list (only normalized/relative features)
            features = [
                # Momentum (relative)
                'Price_Momentum_3', 'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20',
                'Price_Acceleration',
                # Volatility (relative)
                'Volatility_10', 'Volatility_20', 'Volatility_50', 'ATR_Ratio',
                # Volume (ratios)
                'Volume_Ratio', 'OBV_ROC_5', 'OBV_ROC_10', 'Vol_Price_Divergence',
                # RSI (already normalized 0-100)
                'RSI', 'RSI_7', 'RSI_21', 'RSI_Slope',
                # Normalized MACD
                'MACD_Norm', 'MACD_Signal_Norm', 'MACD_Hist_Norm',
                # Bollinger (position is 0-1)
                'BB_Position',
                # Stochastic (0-100)
                'Stoch_K', 'Stoch_D',
                # ADX (0-100)
                'ADX', 'DI_Plus', 'DI_Minus',
                # Price vs MA (relative)
                'Price_vs_SMA20', 'Price_vs_SMA50', 'SMA_20_vs_50',
                # Market regime
                'Bull_Market', 'Bear_Market',
                # Candle pattern
                'Body_Ratio',
                # 筹码集中度 (Chip Concentration) proxy features
                'Chip_Consolidation', 'Chip_Accumulation', 'CMF_20',
                'Chip_Vol_Concentration', 'Chip_Smart_Money',
                'Chip_Turnover_Decline', 'Chip_PV_Divergence',
            ]
            
            # Only keep features that exist in the data
            available_features = [f for f in features if f in self.data.columns]
            missing_features = [f for f in features if f not in self.data.columns]
            if missing_features:
                print(f"⚠️  Skipping missing features: {missing_features}")
            
            if len(available_features) < 10:
                print(f"❌ Too few features available: {len(available_features)}")
                return None
            
            return available_features
            
        except Exception as e:
            print(f"Error creating ML features: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_chinese_technical_score(self):
        """Calculate technical score for Chinese stocks"""
        if self.data is None or len(self.data) < 20:
            return 50
        
        current = self.data.iloc[-1]
        score = 50
        
        # Momentum scoring
        if current['Price_Momentum_5'] > 0.05:
            score += 15
        elif current['Price_Momentum_5'] > 0.02:
            score += 10
        elif current['Price_Momentum_5'] < -0.05:
            score -= 15
        
        # Volume scoring
        if current['Volume_Ratio'] > 1.5:
            score += 10
        elif current['Volume_Ratio'] < 0.5:
            score -= 5
        
        # Moving average scoring
        if current['close'] > current['SMA_20'] > current['SMA_50']:
            score += 15
        elif current['close'] < current['SMA_20'] < current['SMA_50']:
            score -= 15
        
        # Volatility scoring
        avg_volatility = self.data['Volatility_20'].mean()
        if current['Volatility_20'] > avg_volatility * 1.5:
            score -= 10
        elif current['Volatility_20'] < avg_volatility * 0.5:
            score += 5
        
        return max(0, min(100, score))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def calculate_bollinger_position(self, prices, window=20, std_dev=2):
        """Calculate Bollinger Bands position"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return (prices - lower_band) / (upper_band - lower_band)
    
    def create_target_variable(self, holding_period=10, profit_threshold=0.03):
        """
        Create target variable for ML model (3% profit threshold - more realistic)
        """
        future_returns = self.data['close'].shift(-holding_period) / self.data['close'] - 1
        self.data['Target'] = np.where(future_returns > profit_threshold, 1, 0)
        return 'Target'
    
    def prepare_ml_data(self, holding_period=10, profit_threshold=0.03):
        """
        Prepare data for machine learning model, handling inf/nan values
        """
        features = self.create_ml_features()
        target = self.create_target_variable(holding_period, profit_threshold)
        
        ml_data = self.data[features + [target]].copy()
        ml_data = ml_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        X = ml_data[features]
        y = ml_data[target]
        
        return X, y
    
    def walk_forward_validate(self, holding_period=10, profit_threshold=0.03, step=30):
        """
        Walk-forward validation: train on expanding window, predict next `step` days,
        roll forward. Returns a dict of honest reliability metrics.
        """
        try:
            X, y = self.prepare_ml_data(holding_period, profit_threshold)
            if len(X) < 150:
                return None

            window_size = int(len(X) * 0.6)
            all_true, all_pred, all_prob = [], [], []

            for start in range(window_size, len(X) - step, step):
                X_train = X.iloc[:start]
                y_train = y.iloc[:start]
                X_test = X.iloc[start:start + step]
                y_test = y.iloc[start:start + step]

                if len(X_test) == 0 or len(y_train.unique()) < 2:
                    continue

                model = self.create_advanced_pipeline()
                try:
                    model.fit(X_train, y_train)
                    probs = model.predict_proba(X_test)[:, 1]
                    preds = (probs > 0.30).astype(int)
                    all_true.extend(y_test.values)
                    all_pred.extend(preds)
                    all_prob.extend(probs)
                except Exception:
                    continue

            if len(all_true) == 0:
                return None

            all_true = np.array(all_true)
            all_pred = np.array(all_pred)
            all_prob = np.array(all_prob)

            accuracy = (all_true == all_pred).mean()
            naive_acc = (all_true == 0).mean()
            edge = accuracy - naive_acc

            buy_mask = all_pred == 1
            buy_count = int(buy_mask.sum())
            buy_precision = float(all_true[buy_mask].mean()) if buy_count > 0 else 0.0

            gain_mask = all_true == 1
            buy_recall = float(all_pred[gain_mask].mean()) if gain_mask.sum() > 0 else 0.0

            # Calibration for high-confidence bucket (>= 60%)
            high_conf_mask = all_prob >= 0.6
            high_conf_actual = float(all_true[high_conf_mask].mean()) if high_conf_mask.sum() > 2 else None

            if edge > 0.03 and buy_precision > 0.35:
                reliability = 'GOOD'
            elif edge > 0 and buy_precision > 0.25:
                reliability = 'MODERATE'
            elif edge > -0.03:
                reliability = 'WEAK'
            else:
                reliability = 'POOR'

            return {
                'wf_samples': len(all_true),
                'wf_accuracy': round(accuracy, 3),
                'wf_baseline': round(naive_acc, 3),
                'wf_edge': round(edge, 3),
                'wf_buy_count': buy_count,
                'wf_buy_precision': round(buy_precision, 3),
                'wf_buy_recall': round(buy_recall, 3),
                'wf_high_conf_actual': round(high_conf_actual, 3) if high_conf_actual is not None else None,
                'wf_reliability': reliability,
            }
        except Exception as e:
            print(f"⚠️  Walk-forward validation error: {e}")
            return None

    def train_ml_model(self, holding_period=10, profit_threshold=0.03):
        """
        Train advanced machine learning model for Chinese stocks (ENHANCED)
        """
        try:
            X, y = self.prepare_ml_data(holding_period, profit_threshold)
            
            if len(X) < 100:
                print("Insufficient data for ML model training")
                return False
            
            # Check class balance
            class_counts = y.value_counts()
            print(f"📊 Class Balance: {class_counts.to_dict()}")
            print(f"📊 Positive Class Ratio: {class_counts.get(1, 0) / len(y):.3f}")
            
            # Use stratified split to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create advanced pipeline with feature selection
            self.model = self.create_advanced_pipeline()
            
            # Train the pipeline
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Time-series aware cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Get predictions for debugging
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            print(f"📊 Training Predictions: {np.bincount(train_pred)}")
            print(f"📊 Test Predictions: {np.bincount(test_pred)}")
            print(f"📊 Cross-Validation: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
            
            # Store model information
            self.model_info = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'features_count': len(X.columns),
                'data_points': len(X),
                'last_trained': datetime.now().isoformat(),
                'holding_period': holding_period,
                'profit_threshold': profit_threshold,
                'class_balance': class_counts.to_dict(),
                'model_type': 'Advanced Pipeline (Feature Selection + Ensemble)'
            }
            
            print(f"🚀 ADVANCED ML Model Training Results:")
            print(f"Training Accuracy: {train_score:.3f}")
            print(f"Test Accuracy: {test_score:.3f}")
            print(f"Cross-Validation: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
            print(f"Features Used: {len(X.columns)}")
            print(f"Data Points: {len(X)}")
            print(f"Model Type: Advanced Pipeline (Feature Selection + Ensemble)")
            
            # Use more realistic threshold for model acceptance
            self.ml_model_used = test_score > 0.65 and cv_mean > 0.55
            return self.ml_model_used
            
        except Exception as e:
            print(f"Error in ML model training: {str(e)}")
            return False
    
    def create_advanced_pipeline(self):
        """
        Build an ensemble pipeline: RobustScaler -> SelectKBest -> VotingClassifier.
        Tuned for Chinese market characteristics (higher volatility, class imbalance).
        """
        n_features = len([c for c in self.data.columns]) if self.data is not None else 30
        k = min(20, n_features)
        feature_selector = SelectKBest(score_func=f_classif, k=k)
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=25,
            min_samples_leaf=12,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=30,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=42,
        )
        
        lr = LogisticRegression(
            C=0.5,
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
        )
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft',
            weights=[0.45, 0.35, 0.20],
        )
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selector', feature_selector),
            ('ensemble', ensemble),
        ])
        
        return pipeline
    
    def get_ml_prediction(self):
        """
        Get ML prediction for current data (PIPELINE VERSION)
        """
        if self.model is None:
            return None, None
        
        try:
            features = self.create_ml_features()
            X = self.data[features].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(X) == 0:
                return None, None
            
            current_features = X.iloc[-1:].values
            
            try:
                current_features_numeric = current_features.astype(float)
                if np.isnan(current_features_numeric).any():
                    return None, None
            except (ValueError, TypeError):
                pass
            
            probability = self.model.predict_proba(current_features)[0][1]
            prediction = 1 if probability > 0.30 else 0
            return prediction, probability
            
        except Exception as e:
            return None, None
    
    def estimate_lowest_price_10_days(self, symbol, current_price, market='A'):
        """
        Estimate the lowest price in the next 10 days for Chinese stocks
        """
        if self.data is None or len(self.data) < 20:
            return current_price * 0.95  # Default 5% decline estimate
        
        # Calculate historical volatility
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods
        recent_lows = []
        for i in range(max(0, len(self.data) - 15), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_low = self.data['low'].iloc[i:i+10].min()
                period_start = self.data['close'].iloc[i]
                recent_lows.append(period_low / period_start - 1)
        
        avg_low_return = np.mean(recent_lows) if recent_lows else -0.05  # More realistic for Chinese markets
        
        # Calculate technical indicators for price direction
        current = self.data.iloc[-1]
        
        # Momentum-based adjustment
        momentum_adjustment = 1.0
        if current['Price_Momentum_5'] < -0.05:
            momentum_adjustment = 0.9  # Strong downward momentum
        elif current['Price_Momentum_5'] < -0.02:
            momentum_adjustment = 0.95  # Moderate downward momentum
        elif current['Price_Momentum_5'] > 0.05:
            momentum_adjustment = 1.05  # Strong upward momentum
        elif current['Price_Momentum_5'] > 0.02:
            momentum_adjustment = 1.02  # Moderate upward momentum
        
        # Moving average-based adjustment
        ma_adjustment = 1.0
        if current['close'] < current['SMA_20'] < current['SMA_50']:
            ma_adjustment = 0.95  # Strong downtrend
        elif current['close'] < current['SMA_20']:
            ma_adjustment = 0.98  # Moderate downtrend
        elif current['close'] > current['SMA_20'] > current['SMA_50']:
            ma_adjustment = 1.05  # Strong uptrend
        elif current['close'] > current['SMA_20']:
            ma_adjustment = 1.02  # Moderate uptrend
        
        # Volume-based adjustment
        volume_adjustment = 1.0
        if current['Volume_Ratio'] > 2.0:
            volume_adjustment = 0.95  # High volume often precedes decline
        elif current['Volume_Ratio'] < 0.5:
            volume_adjustment = 1.02  # Low volume may indicate stability
        
        # Chinese market specific adjustments
        chinese_adjustment = 1.0
        if market.upper() == 'A':
            # A-shares often have higher volatility
            chinese_adjustment = 0.98
        elif market.upper() == 'H':
            # H-shares more stable
            chinese_adjustment = 0.99
        
        # Calculate estimated lowest price
        base_estimate = current_price * (1 + avg_low_return)
        
        # Apply adjustments
        adjusted_estimate = base_estimate * momentum_adjustment * ma_adjustment * volume_adjustment * chinese_adjustment
        
        # Ensure reasonable bounds for Chinese markets (between -10% and +2%)
        min_decline = current_price * 0.90
        max_decline = current_price * 1.02
        
        estimated_low = max(min_decline, min(adjusted_estimate, max_decline))
        
        return estimated_low
    
    def calculate_ml_price_confidence(self, estimated_price, current_price, direction='high'):
        """
        Calculate ML-based confidence for price estimates
        Returns confidence level (0-100) and reasoning
        """
        if self.model is None or self.data is None:
            return 50, "No ML model available"
        
        try:
            # Get ML prediction and probability
            ml_prediction, ml_probability = self.get_ml_prediction()
            
            if ml_probability is None:
                return 50, "ML prediction not available"
            
            # Calculate price change percentage
            price_change = (estimated_price - current_price) / current_price
            
            # Base confidence on ML probability and historical accuracy
            base_confidence = 50
            
            # Adjust confidence based on ML prediction alignment
            if direction == 'high':
                # For high price estimate
                if ml_prediction == 1:  # ML predicts rise
                    if ml_probability > 0.7:
                        base_confidence += 30  # Strong rise prediction
                    elif ml_probability > 0.6:
                        base_confidence += 20  # Moderate rise prediction
                    elif ml_probability > 0.5:
                        base_confidence += 10  # Weak rise prediction
                else:  # ML predicts decline
                    if ml_probability < 0.3:
                        base_confidence -= 30  # Strong decline prediction
                    elif ml_probability < 0.4:
                        base_confidence -= 20  # Moderate decline prediction
                    elif ml_probability < 0.5:
                        base_confidence -= 10  # Weak decline prediction
            else:
                # For low price estimate
                if ml_prediction == 0:  # ML predicts decline
                    if ml_probability < 0.3:
                        base_confidence += 30  # Strong decline prediction
                    elif ml_probability < 0.4:
                        base_confidence += 20  # Moderate decline prediction
                    elif ml_probability < 0.5:
                        base_confidence += 10  # Weak decline prediction
                else:  # ML predicts rise
                    if ml_probability > 0.7:
                        base_confidence -= 30  # Strong rise prediction
                    elif ml_probability > 0.6:
                        base_confidence -= 20  # Moderate rise prediction
                    elif ml_probability > 0.5:
                        base_confidence -= 10  # Weak rise prediction
            
            # Adjust confidence based on historical model accuracy
            if 'test_score' in self.model_info:
                model_accuracy = self.model_info['test_score']
                accuracy_bonus = int((model_accuracy - 0.5) * 20)  # -10 to +10 based on accuracy
                base_confidence += accuracy_bonus
            
            # Adjust confidence based on price change magnitude
            if abs(price_change) > 0.1:  # >10% change
                base_confidence -= 10  # Less confident for large changes
            elif abs(price_change) < 0.02:  # <2% change
                base_confidence += 10  # More confident for small changes
            
            # Adjust confidence based on volatility
            current_volatility = self.data['Volatility_20'].iloc[-1]
            avg_volatility = self.data['Volatility_20'].mean()
            
            if current_volatility > avg_volatility * 1.5:
                base_confidence -= 15  # Less confident in high volatility
            elif current_volatility < avg_volatility * 0.5:
                base_confidence += 10  # More confident in low volatility
            
            # Ensure confidence is within bounds
            confidence = max(0, min(100, base_confidence))
            
            # Generate reasoning
            reasoning = []
            if ml_prediction == 1 and direction == 'high':
                reasoning.append(f"ML predicts rise ({ml_probability:.1%} probability)")
            elif ml_prediction == 0 and direction == 'low':
                reasoning.append(f"ML predicts decline ({ml_probability:.1%} probability)")
            else:
                reasoning.append(f"ML prediction conflicts with {direction} estimate")
            
            if 'test_score' in self.model_info:
                reasoning.append(f"Model accuracy: {self.model_info['test_score']:.1%}")
            
            if abs(price_change) > 0.1:
                reasoning.append("Large price change reduces confidence")
            elif abs(price_change) < 0.02:
                reasoning.append("Small price change increases confidence")
            
            if current_volatility > avg_volatility * 1.5:
                reasoning.append("High volatility reduces confidence")
            elif current_volatility < avg_volatility * 0.5:
                reasoning.append("Low volatility increases confidence")
            
            return confidence, " | ".join(reasoning)
            
        except Exception as e:
            return 50, f"Error calculating confidence: {str(e)}"
    
    def estimate_highest_price_10_days(self, symbol, current_price, market='A'):
        """
        Estimate the highest price in the next 10 days for Chinese stocks (IMPROVED ACCURACY)
        """
        if self.data is None or len(self.data) < 20:
            return current_price * 1.03  # More realistic default
        
        # Calculate historical volatility
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods (more realistic approach)
        recent_highs = []
        for i in range(max(0, len(self.data) - 30), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_high = self.data['high'].iloc[i:i+10].max()
                period_start = self.data['close'].iloc[i]
                recent_highs.append(period_high / period_start - 1)
        
        # Use median instead of mean for more realistic estimates
        avg_high_return = np.median(recent_highs) if recent_highs else 0.04  # More realistic 4%
        
        # Calculate technical indicators for price direction
        current = self.data.iloc[-1]
        
        # Momentum-based adjustment (more conservative)
        momentum_adjustment = 1.0
        if current['Price_Momentum_5'] > 0.05:
            momentum_adjustment = 1.03  # Reduced from 1.1
        elif current['Price_Momentum_5'] > 0.02:
            momentum_adjustment = 1.02  # Reduced from 1.05
        elif current['Price_Momentum_5'] < -0.05:
            momentum_adjustment = 0.97  # Reduced decline
        elif current['Price_Momentum_5'] < -0.02:
            momentum_adjustment = 0.98  # Reduced decline
        
        # Moving average-based adjustment
        ma_adjustment = 1.0
        if current['close'] > current['SMA_20'] > current['SMA_50']:
            ma_adjustment = 1.02  # Reduced from 1.05
        elif current['close'] > current['SMA_20']:
            ma_adjustment = 1.01  # Reduced from 1.02
        elif current['close'] < current['SMA_20'] < current['SMA_50']:
            ma_adjustment = 0.98  # Reduced from 0.95
        elif current['close'] < current['SMA_20']:
            ma_adjustment = 0.99  # Reduced from 0.98
        
        # Volume-based adjustment (more conservative)
        volume_adjustment = 1.0
        if current['Volume_Ratio'] > 2.0:
            volume_adjustment = 1.02  # Reduced from 1.08
        elif current['Volume_Ratio'] > 1.5:
            volume_adjustment = 1.01  # Reduced from 1.05
        elif current['Volume_Ratio'] < 0.5:
            volume_adjustment = 0.99  # Reduced from 0.95
        
        # Chinese market specific adjustments (more realistic)
        chinese_adjustment = 1.0
        if market.upper() == 'A':
            # A-shares - more conservative
            chinese_adjustment = 1.01  # Reduced from 1.05
        elif market.upper() == 'H':
            # H-shares - more stable
            chinese_adjustment = 1.005  # Reduced from 1.02
        
        # Calculate estimated highest price
        base_estimate = current_price * (1 + avg_high_return)
        
        # Apply adjustments
        adjusted_estimate = base_estimate * momentum_adjustment * ma_adjustment * volume_adjustment * chinese_adjustment
        
        # Ensure reasonable bounds (between 1% and 8% increase - more realistic)
        min_increase = current_price * 1.01
        max_increase = current_price * 1.08
        
        estimated_high = max(min_increase, min(adjusted_estimate, max_increase))
        
        return estimated_high
    
    def analyze_chinese_stock(self, symbol, market='A'):
        """
        Analyze Chinese stock with ML model integration and persistence
        """
        print(f"\n{'='*50}")
        print(f"CHINESE STOCK ANALYSIS: {symbol} ({market}-shares)")
        print(f"{'='*50}")
        
        # Try to load existing model first
        model_loaded = self.load_model(symbol, market)
        
        if not model_loaded:
            # Download data and train new model
            download_success, stock_name = self.download_chinese_stock_data(symbol, market)
            if not download_success:
                print(f"Failed to download data for {symbol}. Cannot analyze.")
                return None
            
            self.calculate_chinese_indicators()
            
            # Train ML model
            ml_success = self.train_ml_model(holding_period=10, profit_threshold=0.03)
            
            if ml_success:
                # Save the newly trained model
                self.save_model(symbol, market)
        else:
            # Model loaded successfully, just download latest data for analysis
            download_success, stock_name = self.download_chinese_stock_data(symbol, market)
            if not download_success:
                print(f"Failed to download data for {symbol}. Cannot analyze.")
                return None
            
            self.calculate_chinese_indicators()
            ml_success = True
        
        # Get current values
        current = self.data.iloc[-1]
        current_price = current['close']
        
        # Calculate technical score
        technical_score = self.calculate_chinese_technical_score()
        
        # Get ML prediction
        ml_prediction, ml_probability = self.get_ml_prediction()
        
        # Calculate final score (60% technical + 40% ML)
        if ml_probability is not None:
            ml_score = int(ml_probability * 100)
            final_score = int(technical_score * 0.6 + ml_score * 0.4)
        else:
            final_score = int(technical_score)
        
        # Determine recommendation
        if final_score >= 80:
            recommendation = "STRONG BUY"
            confidence = "Very High"
        elif final_score >= 70:
            recommendation = "BUY"
            confidence = "High"
        elif final_score >= 60:
            recommendation = "HOLD"
            confidence = "Moderate"
        else:
            recommendation = "HOLD"
            confidence = "Low"
        
        # Data-driven price estimates (no longer hardcoded)
        estimated_high_10d = self.estimate_highest_price_10_days(symbol, current_price, market)
        estimated_low_10d = self.estimate_lowest_price_10_days(symbol, current_price, market)
        potential_gain_10d = (estimated_high_10d - current_price) / current_price
        potential_loss_10d = (estimated_low_10d - current_price) / current_price
        
        # Calculate confidence for price estimates
        high_confidence, high_reasoning = self.calculate_ml_price_confidence(estimated_high_10d, current_price, 'high')
        low_confidence, low_reasoning = self.calculate_ml_price_confidence(estimated_low_10d, current_price, 'low')
        
        return {
            'symbol': self.symbol,
            'market': self.market_type,
            'score': final_score,
            'technical_score': technical_score,
            'ml_probability': ml_probability,
            'ml_prediction': ml_prediction,
            'ml_model_used': self.model_info.get('model_type', 'Unknown'),
            'recommendation': recommendation,
            'confidence': confidence,
            'current_price': current_price,
            'estimated_high_10d': estimated_high_10d,
            'estimated_low_10d': estimated_low_10d,
            'potential_gain_10d': potential_gain_10d,
            'potential_loss_10d': potential_loss_10d,
            'high_confidence': high_confidence,
            'high_reasoning': high_reasoning,
            'low_confidence': low_confidence,
            'low_reasoning': low_reasoning,
            'momentum_5d': current['Price_Momentum_5'],
            'volume_ratio': current['Volume_Ratio'],
            'volatility': current['Volatility_20'],
            'stock_name': stock_name # Add stock_name to the result
        }
    
    def compare_chinese_stocks(self, stocks_list):
        """
        Compare multiple Chinese stocks with ML model integration
        """
        print(f"\n{'='*60}")
        print("CHINESE STOCKS COMPARISON WITH ML MODEL")
        print(f"{'='*60}")
        
        results = []
        
        for stock_info in stocks_list:
            symbol = stock_info['symbol']
            market = stock_info.get('market', 'A')
            
            try:
                result = self.analyze_chinese_stock(symbol, market)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        if not results:
            print("No valid results generated")
            return None
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Display results
        print(f"\n{'='*60}")
        print("CHINESE STOCKS RECOMMENDATION SUMMARY")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['symbol']} ({result['market']}-shares)")
            print(f"   Current Price: {result['current_price']:.2f}")
            print(f"   Estimated High (10d): {result['estimated_high_10d']:.2f}")
            print(f"   Estimated Low (10d): {result['estimated_low_10d']:.2f}")
            print(f"   Potential Gain: {result['potential_gain_10d']:.2%}")
            print(f"   Potential Loss: {result['potential_loss_10d']:.2%}")
            print(f"   Score: {result['score']:.2f}/100")
            print(f"   Technical Score: {result['technical_score']:.2f}/100")
            if result['ml_probability'] is not None:
                print(f"   ML Probability: {result['ml_probability']:.3f}")
                print(f"   ML Prediction: {result['ml_prediction']}")
            print(f"   ML Model Used: {result['ml_model_used']}")
            print(f"   Recommendation: {result['recommendation']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   5-day Momentum: {result['momentum_5d']:.2%}")
            print(f"   Volume Ratio: {result['volume_ratio']:.2f}")
        
        # Best recommendation
        best = results[0]
        print(f"\n{'='*60}")
        print(f"🎯 BEST CHINESE STOCK: {best['symbol']} ({best['market']}-shares)")
        print(f"{'='*60}")
        print(f"Current Price: {best['current_price']:.2f}")
        print(f"Estimated High (10d): {best['estimated_high_10d']:.2f}")
        print(f"Estimated Low (10d): {best['estimated_low_10d']:.2f}")
        print(f"Potential Gain: {best['potential_gain_10d']:.2%}")
        print(f"Potential Loss: {best['potential_loss_10d']:.2%}")
        print(f"Score: {best['score']:.2f}/100")
        print(f"Technical Score: {best['technical_score']:.2f}/100")
        if best['ml_probability'] is not None:
            print(f"ML Probability: {best['ml_probability']:.3f}")
            print(f"ML Prediction: {best['ml_prediction']}")
        print(f"ML Model Used: {best['ml_model_used']}")
        print(f"Recommendation: {best['recommendation']}")
        print(f"Confidence: {best['confidence']}")
        
        return best
    
    def estimate_sell_point(self, symbol, buy_price, market='A', buy_date=None, holding_period=10):
        """
        Estimate optimal sell point and price for Chinese stocks
        """
        print(f"\n{'='*60}")
        print(f"CHINESE STOCK SELL POINT ESTIMATION: {symbol} ({market}-shares)")
        print(f"{'='*60}")
        print(f"Buy Price: {buy_price:.2f}")
        if buy_date:
            print(f"Buy Date: {buy_date}")
        print(f"Target Holding Period: {holding_period} days")
        
        # Download latest data
        download_success, stock_name = self.download_chinese_stock_data(symbol, market)
        if not download_success:
            print(f"Failed to download data for {symbol}. Cannot estimate sell point.")
            return None
        
        # Calculate indicators
        self.calculate_chinese_indicators()
        
        # Get current price
        current_price = self.data.iloc[-1]['close']
        current_return = (current_price - buy_price) / buy_price
        
        print(f"Current Price: {current_price:.2f}")
        print(f"Current Return: {current_return:.2%}")
        
        # Analyze sell signals for Chinese markets
        sell_analysis = self.analyze_chinese_sell_signals(buy_price, market)
        
        # Generate sell recommendation
        recommendation = self.generate_chinese_sell_recommendation(sell_analysis, current_return, buy_price, market)
        
        # Walk-forward validation for sell signals
        sell_wf = self.walk_forward_validate_sell(buy_price)
        if sell_wf:
            recommendation.update(sell_wf)
            print(f"📉 Sell WF: edge={sell_wf['sell_wf_edge']:+.1%}, "
                  f"sell_precision={sell_wf['sell_wf_sell_precision']:.0%}, "
                  f"rating={sell_wf['sell_wf_rating']}")
        
        return {
            'symbol': symbol,
            'market': market,
            'buy_price': buy_price,
            'current_price': current_price,
            'current_return': current_return,
            'sell_analysis': sell_analysis,
            'recommendation': recommendation,
            'stock_name': stock_name
        }
    
    def analyze_chinese_sell_signals(self, buy_price, market='A'):
        """
        Enhanced sell signal analysis combining profit management rules,
        distribution detection (chip features), and ML downside prediction.
        """
        if self.data is None or len(self.data) < 20:
            return None
        
        current = self.data.iloc[-1]
        current_price = current['close']
        current_return = (current_price - buy_price) / buy_price
        
        analysis = {
            'technical_score': 0,
            'ml_score': 0,
            'combined_score': 0,
            'ml_prediction': None,
            'ml_probability': None,
            'sell_signals': [],
            'hold_signals': [],
            'risk_factors': [],
            'profit_potential': 0,
            'stop_loss_triggered': False,
            'limit_up_near': False,
            'limit_down_near': False
        }
        
        # ===== 1. PROFIT / LOSS MANAGEMENT (most reliable) =====
        # Trailing stop: check if price has dropped from recent peak
        if len(self.data) >= 10:
            recent_peak = self.data['close'].iloc[-10:].max()
            drawdown_from_peak = (current_price - recent_peak) / recent_peak
            if drawdown_from_peak < -0.05:
                analysis['sell_signals'].append(f"Trailing stop: -{abs(drawdown_from_peak):.1%} from 10d peak (¥{recent_peak:.2f})")
                analysis['risk_factors'].append("Price falling from recent peak")
            elif drawdown_from_peak < -0.03:
                analysis['sell_signals'].append(f"Approaching trailing stop: {drawdown_from_peak:.1%} from 10d peak")
        
        if current_return >= 0.08:
            analysis['sell_signals'].append(f"Strong profit target reached ({current_return:.1%}) — consider taking profit")
            analysis['profit_potential'] = current_return
        elif current_return >= 0.05:
            analysis['sell_signals'].append(f"Profit target reached ({current_return:.1%})")
            analysis['profit_potential'] = current_return
        elif current_return >= 0.03:
            analysis['hold_signals'].append(f"Moderate profit ({current_return:.1%}) — watch for exit signals")
            analysis['profit_potential'] = current_return
        
        if current_return <= -0.05:
            analysis['sell_signals'].append(f"Stop loss triggered ({current_return:.1%})")
            analysis['stop_loss_triggered'] = True
            analysis['risk_factors'].append("Significant loss — cut losses")
        elif current_return <= -0.03:
            analysis['sell_signals'].append(f"Approaching stop loss ({current_return:.1%})")
            analysis['risk_factors'].append("Loss position")
        
        # ===== 2. DISTRIBUTION DETECTION (chip features) =====
        cmf = current.get('CMF_20')
        if cmf is not None and not np.isnan(cmf):
            if cmf < -0.15:
                analysis['sell_signals'].append(f"Strong money outflow (CMF={cmf:.2f}) — distribution")
            elif cmf < -0.05:
                analysis['sell_signals'].append(f"Money outflow detected (CMF={cmf:.2f})")
            elif cmf > 0.15:
                analysis['hold_signals'].append(f"Strong money inflow (CMF={cmf:.2f}) — accumulation")
            elif cmf > 0.05:
                analysis['hold_signals'].append(f"Money inflow detected (CMF={cmf:.2f})")
        
        smart_money = current.get('Chip_Smart_Money')
        if smart_money is not None and not np.isnan(smart_money):
            if smart_money > 0.3:
                analysis['hold_signals'].append("Smart money accumulation detected")
            elif smart_money < 0.05:
                analysis['sell_signals'].append("No smart money activity — weak support")
        
        chip_accum = current.get('Chip_Accumulation')
        if chip_accum is not None and not np.isnan(chip_accum):
            if chip_accum < -0.5:
                analysis['sell_signals'].append("Volume rising during consolidation — potential distribution")
        
        # ===== 3. ML PREDICTION =====
        ml_prediction, ml_probability = None, None
        try:
            if self.model is None:
                if not self.load_model(self.symbol, market):
                    print(f"Training new ML model for {self.symbol}...")
                    self.train_ml_model(holding_period=10, profit_threshold=0.03)
            
            ml_prediction, ml_probability = self.get_ml_prediction()
            analysis['ml_prediction'] = ml_prediction
            analysis['ml_probability'] = ml_probability
            
            if ml_probability is not None:
                ml_sell_score = int((1 - ml_probability) * 100)
                analysis['ml_score'] = ml_sell_score
                if ml_probability < 0.25:
                    analysis['sell_signals'].append(f"ML predicts strong decline ({ml_probability:.0%} rise prob)")
                elif ml_probability < 0.4:
                    analysis['sell_signals'].append(f"ML predicts decline ({ml_probability:.0%} rise prob)")
                elif ml_probability > 0.7:
                    analysis['hold_signals'].append(f"ML predicts strong rise ({ml_probability:.0%} rise prob)")
                elif ml_probability > 0.55:
                    analysis['hold_signals'].append(f"ML predicts rise ({ml_probability:.0%} rise prob)")
            else:
                print(f"⚠️  ML prediction not available for {self.symbol}")
        except Exception as e:
            print(f"⚠️  ML analysis failed for {self.symbol}: {e}")
        
        # ===== 4. TECHNICAL SIGNALS (supplementary) =====
        tech_score = self.calculate_chinese_sell_technical_score(current_return)
        analysis['technical_score'] = tech_score
        
        # Momentum
        mom5 = current.get('Price_Momentum_5', 0)
        mom3 = current.get('Price_Momentum_3', 0)
        if not np.isnan(mom5) and not np.isnan(mom3):
            if mom5 < -0.05 and mom3 < -0.03:
                analysis['sell_signals'].append("Multi-timeframe momentum negative")
            elif mom5 > 0.05 and mom3 > 0.02:
                analysis['hold_signals'].append("Multi-timeframe momentum positive")
        
        # RSI divergence: price making new high but RSI is lower
        if len(self.data) >= 20:
            rsi_now = current.get('RSI', 50)
            price_20d_high = self.data['close'].iloc[-20:].max()
            rsi_at_high = self.data['RSI'].iloc[-20:].max() if 'RSI' in self.data else 50
            if not np.isnan(rsi_now) and not np.isnan(rsi_at_high):
                if current_price >= price_20d_high * 0.99 and rsi_now < rsi_at_high - 10:
                    analysis['sell_signals'].append(f"Bearish RSI divergence (price near high but RSI weaker)")
        
        # MA crossover: SMA_20 crossing below SMA_50 (death cross)
        if len(self.data) >= 2:
            sma20_now = current.get('SMA_20')
            sma50_now = current.get('SMA_50')
            prev = self.data.iloc[-2]
            sma20_prev = prev.get('SMA_20')
            sma50_prev = prev.get('SMA_50')
            if all(v is not None and not np.isnan(v) for v in [sma20_now, sma50_now, sma20_prev, sma50_prev]):
                if sma20_prev >= sma50_prev and sma20_now < sma50_now:
                    analysis['sell_signals'].append("Death cross: SMA_20 crossed below SMA_50")
                elif sma20_prev <= sma50_prev and sma20_now > sma50_now:
                    analysis['hold_signals'].append("Golden cross: SMA_20 crossed above SMA_50")
        
        # Volatility spike
        avg_vol = self.data['Volatility_20'].mean() if 'Volatility_20' in self.data else 0
        cur_vol = current.get('Volatility_20', 0)
        if not np.isnan(cur_vol) and not np.isnan(avg_vol) and avg_vol > 0:
            if cur_vol > avg_vol * 2:
                analysis['risk_factors'].append(f"Volatility spike ({cur_vol/avg_vol:.1f}x average)")
        
        # ===== 5. COMBINED SCORE =====
        # Weight: 40% profit management, 30% ML, 20% technical, 10% distribution
        profit_score = 50
        if current_return >= 0.08: profit_score = 85
        elif current_return >= 0.05: profit_score = 75
        elif current_return >= 0.03: profit_score = 60
        elif current_return <= -0.05: profit_score = 95
        elif current_return <= -0.03: profit_score = 80
        elif current_return <= -0.01: profit_score = 60
        
        ml_sell_score = analysis.get('ml_score', 50)
        
        dist_score = 50
        if cmf is not None and not np.isnan(cmf):
            dist_score = max(0, min(100, 50 - int(cmf * 200)))
        
        combined = int(0.40 * profit_score + 0.30 * ml_sell_score + 0.20 * tech_score + 0.10 * dist_score)
        analysis['combined_score'] = max(0, min(100, combined))
        analysis['profit_score'] = profit_score
        analysis['distribution_score'] = dist_score
        
        return analysis
    
    def calculate_chinese_sell_technical_score(self, current_return=0):
        """
        Enhanced technical sell score (0-100). Higher = stronger sell signal.
        Incorporates multi-timeframe analysis and chip distribution cues.
        """
        if self.data is None or len(self.data) < 20:
            return 50
        
        current = self.data.iloc[-1]
        score = 50
        
        # Multi-timeframe momentum (more reliable than single timeframe)
        for col, weight in [('Price_Momentum_3', 8), ('Price_Momentum_5', 10), ('Price_Momentum_10', 7)]:
            val = current.get(col, 0)
            if val is not None and not np.isnan(val):
                if val < -0.05: score += weight
                elif val < -0.02: score += weight // 2
                elif val > 0.05: score -= weight
                elif val > 0.02: score -= weight // 2
        
        # Price vs MAs
        price = current['close']
        for ma_col, weight in [('SMA_20', 8), ('SMA_50', 8)]:
            ma_val = current.get(ma_col)
            if ma_val is not None and not np.isnan(ma_val):
                if price < ma_val: score += weight
                else: score -= weight // 2
        
        # RSI with overbought emphasis
        rsi = current.get('RSI', 50)
        if not np.isnan(rsi):
            if rsi > 80: score += 15
            elif rsi > 70: score += 10
            elif rsi < 25: score -= 12
            elif rsi < 35: score -= 8
        
        # Stochastic overbought
        stoch_k = current.get('Stoch_K', 50)
        if not np.isnan(stoch_k):
            if stoch_k > 85: score += 8
            elif stoch_k < 15: score -= 8
        
        # ADX: strong trend + DI_Minus > DI_Plus = bearish trend
        adx = current.get('ADX', 0)
        di_plus = current.get('DI_Plus', 0)
        di_minus = current.get('DI_Minus', 0)
        if all(not np.isnan(v) for v in [adx, di_plus, di_minus] if v is not None):
            if adx > 25 and di_minus > di_plus:
                score += 10
            elif adx > 25 and di_plus > di_minus:
                score -= 8
        
        # Volume-price divergence (price up + volume down = distribution)
        vpd = current.get('Vol_Price_Divergence', 0)
        if vpd is not None and not np.isnan(vpd):
            if vpd > 0.5: score += 8
            elif vpd < -0.5: score -= 5
        
        # Chip distribution signals
        cmf = current.get('CMF_20', 0)
        if cmf is not None and not np.isnan(cmf):
            if cmf < -0.1: score += 10
            elif cmf > 0.1: score -= 8
        
        return max(0, min(100, score))
    
    def walk_forward_validate_sell(self, buy_price=None, step=10):
        """
        Walk-forward backtest of sell signals: at each point check if the
        sell/hold decision was correct based on the next 10 days' price action.
        """
        if self.data is None or len(self.data) < 100:
            return None
        try:
            data = self.data
            sell_correct, sell_wrong, hold_correct, hold_wrong = 0, 0, 0, 0
            total = 0

            for i in range(60, len(data) - step, step):
                current_price = data['close'].iloc[i]
                future_price = data['close'].iloc[i + step]
                future_return = (future_price - current_price) / current_price

                sim_buy = buy_price if buy_price else current_price * 0.95
                sim_return = (current_price - sim_buy) / sim_buy

                current = data.iloc[i]

                # Simplified combined sell score (same logic as the enhanced version)
                profit_score = 50
                if sim_return >= 0.08: profit_score = 85
                elif sim_return >= 0.05: profit_score = 75
                elif sim_return >= 0.03: profit_score = 60
                elif sim_return <= -0.05: profit_score = 95
                elif sim_return <= -0.03: profit_score = 80

                tech_score = 50
                for col, w in [('Price_Momentum_3', 8), ('Price_Momentum_5', 10), ('Price_Momentum_10', 7)]:
                    v = current.get(col, 0)
                    if v is not None and not np.isnan(v):
                        if v < -0.05: tech_score += w
                        elif v < -0.02: tech_score += w // 2
                        elif v > 0.05: tech_score -= w
                        elif v > 0.02: tech_score -= w // 2
                rsi = current.get('RSI', 50)
                if not np.isnan(rsi):
                    if rsi > 80: tech_score += 15
                    elif rsi > 70: tech_score += 10
                    elif rsi < 25: tech_score -= 12
                cmf = current.get('CMF_20', 0)
                dist_score = 50
                if cmf is not None and not np.isnan(cmf):
                    dist_score = max(0, min(100, 50 - int(cmf * 200)))
                    if cmf < -0.1: tech_score += 10
                    elif cmf > 0.1: tech_score -= 8

                combined = int(0.40 * profit_score + 0.30 * 50 + 0.20 * tech_score + 0.10 * dist_score)
                decision = 'SELL' if combined >= 60 else 'HOLD'

                total += 1
                if decision == 'SELL':
                    if future_return < 0: sell_correct += 1
                    else: sell_wrong += 1
                else:
                    if future_return >= 0: hold_correct += 1
                    else: hold_wrong += 1

            if total == 0:
                return None

            total_sell = sell_correct + sell_wrong
            sell_precision = sell_correct / total_sell if total_sell > 0 else 0
            hold_precision = hold_correct / (hold_correct + hold_wrong) if (hold_correct + hold_wrong) > 0 else 0
            overall_acc = (sell_correct + hold_correct) / total
            naive_acc = (hold_correct + sell_wrong) / total
            edge = overall_acc - naive_acc

            if edge > 0.05 and sell_precision > 0.55:
                rating = 'GOOD'
            elif edge > 0 and sell_precision > 0.45:
                rating = 'MODERATE'
            elif edge > -0.05:
                rating = 'WEAK'
            else:
                rating = 'POOR'

            return {
                'sell_wf_total': total,
                'sell_wf_sell_count': total_sell,
                'sell_wf_sell_precision': round(sell_precision, 3),
                'sell_wf_hold_precision': round(hold_precision, 3),
                'sell_wf_accuracy': round(overall_acc, 3),
                'sell_wf_baseline': round(naive_acc, 3),
                'sell_wf_edge': round(edge, 3),
                'sell_wf_rating': rating,
            }
        except Exception as e:
            print(f"⚠️  Sell walk-forward error: {e}")
            return None

    def generate_chinese_sell_recommendation(self, sell_analysis, current_return, buy_price, market='A'):
        """
        Generate sell recommendation for Chinese stocks with ML integration
        """
        if not sell_analysis:
            return None
        
        # Use combined score (technical + ML) for recommendation
        combined_score = sell_analysis['combined_score']
        tech_score = sell_analysis['technical_score']
        ml_score = sell_analysis['ml_score']
        ml_probability = sell_analysis['ml_probability']
        
        # Get current price for price estimates
        current_price = self.data.iloc[-1]['close']
        
        # Calculate estimated high and low prices for next 10 days
        estimated_high_10d = self.estimate_highest_price_10_days(self.symbol, current_price, market)
        estimated_low_10d = self.estimate_lowest_price_10_days(self.symbol, current_price, market)
        
        # Calculate potential gains and losses
        potential_gain_10d = (estimated_high_10d - current_price) / current_price
        potential_loss_10d = (estimated_low_10d - current_price) / current_price
        
        # Calculate ML-based confidence for price estimates
        high_confidence, high_reasoning = self.calculate_ml_price_confidence(estimated_high_10d, current_price, 'high')
        low_confidence, low_reasoning = self.calculate_ml_price_confidence(estimated_low_10d, current_price, 'low')
        
        # Determine sell action based on combined score
        if combined_score >= 80:
            action = "SELL NOW"
            urgency = "VERY HIGH"
            reasoning = "Very strong technical and ML signals indicate immediate selling"
        elif combined_score >= 70:
            action = "SELL SOON"
            urgency = "HIGH"
            reasoning = "Strong technical and ML signals indicate selling within 1-2 days"
        elif combined_score >= 60:
            action = "SELL"
            urgency = "MEDIUM"
            reasoning = "Moderate sell signals detected - consider selling this week"
        elif combined_score >= 45:
            action = "HOLD"
            urgency = "LOW"
            reasoning = "Mixed signals - monitor closely but no immediate action needed"
        elif combined_score >= 30:
            action = "HOLD/ADD"
            urgency = "LOW"
            reasoning = "Weak buy signals - consider holding or adding small positions"
        else:
            action = "BUY/ADD"
            urgency = "LOW"
            reasoning = "Strong buy signals - consider adding to position"
        
        # Calculate target sell price for Chinese markets
        if current_return >= 0.05:
            # Already at profit target, suggest selling at current price
            target_price = buy_price * 1.05  # 5% profit
        elif current_return >= 0.03:
            # Moderate profit, suggest waiting for 5% target
            target_price = buy_price * 1.05
        elif current_return <= -0.03:
            # At stop loss, suggest immediate selling
            target_price = current_return * buy_price
        else:
            # Calculate based on combined analysis
            if combined_score >= 60:
                target_price = buy_price * (1 + max(current_return, 0.01))
            else:
                target_price = buy_price * 1.03
        
        # Risk assessment for Chinese markets
        risk_level = "LOW"
        if sell_analysis['stop_loss_triggered']:
            risk_level = "VERY HIGH"
        elif sell_analysis['limit_up_near']:
            risk_level = "HIGH"
        elif len(sell_analysis['risk_factors']) > 2:
            risk_level = "MEDIUM"
        elif combined_score >= 70:
            risk_level = "HIGH"
        elif combined_score >= 60:
            risk_level = "MEDIUM"
        
        # Add score interpretation with ML insights
        score_interpretation = ""
        if combined_score >= 80:
            score_interpretation = "Very Strong Sell Signal"
        elif combined_score >= 70:
            score_interpretation = "Strong Sell Signal"
        elif combined_score >= 60:
            score_interpretation = "Moderate Sell Signal"
        elif combined_score >= 45:
            score_interpretation = "Neutral Signal"
        elif combined_score >= 30:
            score_interpretation = "Weak Buy Signal"
        else:
            score_interpretation = "Strong Buy Signal"
        
        # Add ML insights
        ml_insights = ""
        if ml_probability is not None:
            if ml_probability < 0.3:
                ml_insights = f"ML strongly predicts decline ({ml_probability:.1%} probability of rise)"
            elif ml_probability < 0.4:
                ml_insights = f"ML moderately predicts decline ({ml_probability:.1%} probability of rise)"
            elif ml_probability > 0.7:
                ml_insights = f"ML strongly predicts rise ({ml_probability:.1%} probability of rise)"
            elif ml_probability > 0.6:
                ml_insights = f"ML moderately predicts rise ({ml_probability:.1%} probability of rise)"
            else:
                ml_insights = f"ML neutral ({ml_probability:.1%} probability of rise)"
        else:
            ml_insights = "ML prediction not available"
        
        # Ensure ml fields are never None (safe defaults for display)
        safe_ml_prob = ml_probability if ml_probability is not None else 0.0
        safe_ml_pred = sell_analysis['ml_prediction'] if sell_analysis['ml_prediction'] is not None else 0
        safe_ml_score = ml_score if ml_score is not None else 0
        
        return {
            'action': action,
            'urgency': urgency,
            'reasoning': reasoning,
            'technical_score': tech_score,
            'ml_score': safe_ml_score,
            'combined_score': combined_score,
            'ml_probability': safe_ml_prob,
            'ml_prediction': safe_ml_pred,
            'ml_available': ml_probability is not None,
            'score_interpretation': score_interpretation,
            'ml_insights': ml_insights,
            'target_price': target_price,
            'risk_level': risk_level,
            'sell_signals': sell_analysis['sell_signals'],
            'hold_signals': sell_analysis['hold_signals'],
            'risk_factors': sell_analysis['risk_factors'],
            'profit_potential': sell_analysis['profit_potential'],
            'limit_up_near': sell_analysis['limit_up_near'],
            'limit_down_near': sell_analysis['limit_down_near'],
            'estimated_high_10d': estimated_high_10d,
            'estimated_low_10d': estimated_low_10d,
            'potential_gain_10d': potential_gain_10d,
            'potential_loss_10d': potential_loss_10d,
            'high_confidence': high_confidence,
            'high_reasoning': high_reasoning,
            'low_confidence': low_confidence,
            'low_reasoning': low_reasoning
        } 