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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
import pickle
import os
from datetime import datetime, timedelta
from chinese_stock_downloader import ChineseStockDownloader
from chinese_stock_cache import ChineseStockCache
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
        self.cache = ChineseStockCache()  # Add cache access
        
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
        """Download Chinese stock data using the new downloader with cache checking"""
        # Add delay to avoid server resistance
        time.sleep(0.5)
        
        self.symbol = symbol
        self.market_type = market
        
        # Get stock name
        stock_name = self.downloader.get_stock_name(symbol, market)
        
        # Check if we already have data for this symbol and period in memory
        if (self.data is not None and 
            hasattr(self, 'symbol') and 
            self.symbol == symbol and 
            len(self.data) > 50):
            print(f"✅ Using existing data in memory for {symbol} ({len(self.data)} days)")
            return True, stock_name
        
        # Check cache first
        print(f"🔍 Checking cache for {symbol}...")
        cached_data = self.cache.get_cached_stock_data(symbol, market, period)
        
        if cached_data is not None and len(cached_data) > 50:
            print(f"✅ Using cached data for {symbol} ({len(cached_data)} days)")
            self.data = cached_data
            return True, stock_name
        
        # Download data only if not in cache
        print(f"📥 Downloading fresh data for {symbol}...")
        data = self.downloader.download_stock_data(symbol, market, period)
        
        if data is not None:
            self.data = data
            # Cache the downloaded data
            self.cache.cache_stock_data(symbol, market, period, data)
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
            
            # RSI
            data['RSI'] = self.calculate_rsi(data['close'])
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            self.data = data
            print("Chinese market indicators calculated successfully!")
            return True
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return False
    
    def create_ml_features(self):
        """Create features for ML model"""
        if self.data is None or len(self.data) < 50:
            return None
        
        try:
            # Price vs moving averages
            self.data['Price_vs_SMA20'] = (self.data['close'] - self.data['SMA_20']) / self.data['SMA_20']
            self.data['Price_vs_SMA50'] = (self.data['close'] - self.data['SMA_50']) / self.data['SMA_50']
            
            # Market regime features
            self.data['Bull_Market'] = (self.data['close'] > self.data['SMA_20']) & (self.data['SMA_20'] > self.data['SMA_50']).astype(int)
            self.data['Bear_Market'] = (self.data['close'] < self.data['SMA_20']) & (self.data['SMA_20'] < self.data['SMA_50']).astype(int)
            
            # Feature list
            features = [
                'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20', 'Price_Momentum_3',
                'Volatility_10', 'Volatility_20', 'Volatility_50',
                'Volume_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Position', 'Price_vs_SMA20', 'Price_vs_SMA50',
                'Bull_Market', 'Bear_Market'
            ]
            
            # Check if all features exist
            missing_features = [f for f in features if f not in self.data.columns]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                return None
            
            return features
            
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
        Prepare data for machine learning model
        """
        features = self.create_ml_features()
        target = self.create_target_variable(holding_period, profit_threshold)
        
        # Remove rows with NaN values
        ml_data = self.data[features + [target]].dropna()
        
        X = ml_data[features]
        y = ml_data[target]
        
        return X, y
    
    def train_ml_model(self, holding_period=10, profit_threshold=0.03):
        """
        Train advanced machine learning model for Chinese stocks (ENHANCED)
        """
        try:
            print(f"🔄 Preparing ML training data...")
            X, y = self.prepare_ml_data(holding_period, profit_threshold)
            
            if len(X) < 100:
                print("❌ Insufficient data for ML model training")
                return False
            
            # Check class balance
            class_counts = y.value_counts()
            print(f"📊 Class Balance: {class_counts.to_dict()}")
            print(f"📊 Positive Class Ratio: {class_counts.get(1, 0) / len(y):.3f}")
            
            print(f"🔄 Splitting data into training and test sets...")
            # Use stratified split to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Training set: {len(X_train)} samples")
            print(f"📊 Test set: {len(X_test)} samples")
            
            # Create advanced pipeline with feature selection
            print(f"🤖 Creating advanced ML pipeline...")
            self.model = self.create_advanced_pipeline()
            
            # Train the pipeline
            print(f"🚀 Training advanced ML model...")
            print(f"   📈 This may take a few minutes for {len(X_train)} samples")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            print(f"📊 Evaluating model performance...")
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation score with stratified folds
            print(f"🔄 Running 5-fold cross-validation...")
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
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
            
            print(f"✅ ML Model Training Results:")
            print(f"   Training Accuracy: {train_score:.3f}")
            print(f"   Test Accuracy: {test_score:.3f}")
            print(f"   Cross-Validation: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
            print(f"   Features Used: {len(X.columns)}")
            print(f"   Data Points: {len(X)}")
            print(f"   Model Type: Advanced Pipeline (Feature Selection + Ensemble)")
            
            # Use more realistic threshold for model acceptance
            self.ml_model_used = test_score > 0.65 and cv_mean > 0.55
            return self.ml_model_used
            
        except Exception as e:
            print(f"❌ Error in ML model training: {str(e)}")
            return False
    
    def create_advanced_pipeline(self):
        """
        Create advanced pipeline with feature selection and ensemble
        """
        # Feature selection step
        feature_selector = SelectKBest(score_func=f_classif, k=15)  # Select top 15 features
        
        # Create base models
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=25,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=30,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=42
        )
        
        lr = LogisticRegression(
            C=0.5,
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft',
            weights=[0.5, 0.3, 0.2]
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_selector', feature_selector),
            ('ensemble', ensemble)
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
            X = self.data[features].dropna()
            
            if len(X) == 0:
                return None, None
            
            # Get the most recent data point
            current_features = X.iloc[-1:].values
            
            # Check if we have valid features (handle non-numeric data)
            try:
                current_features_numeric = current_features.astype(float)
                if np.isnan(current_features_numeric).any():
                    return None, None
            except (ValueError, TypeError):
                # If conversion fails, assume data is valid (contains boolean features)
                pass
            
            # Use pipeline for prediction (includes scaling and feature selection)
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
            
            # Adjust confidence based on ML prediction alignment (reduced adjustments)
            if direction == 'high':
                # For high price estimate
                if ml_prediction == 1:  # ML predicts rise
                    if ml_probability > 0.7:
                        base_confidence += 15  # Reduced from 30
                    elif ml_probability > 0.6:
                        base_confidence += 10  # Reduced from 20
                    elif ml_probability > 0.5:
                        base_confidence += 5   # Reduced from 10
                else:  # ML predicts decline
                    if ml_probability < 0.3:
                        base_confidence -= 15  # Reduced from 30
                    elif ml_probability < 0.4:
                        base_confidence -= 10  # Reduced from 20
                    elif ml_probability < 0.5:
                        base_confidence -= 5   # Reduced from 10
            else:
                # For low price estimate
                if ml_prediction == 0:  # ML predicts decline
                    if ml_probability < 0.3:
                        base_confidence += 15  # Reduced from 30
                    elif ml_probability < 0.4:
                        base_confidence += 10  # Reduced from 20
                    elif ml_probability < 0.5:
                        base_confidence += 5   # Reduced from 10
                else:  # ML predicts rise
                    if ml_probability > 0.7:
                        base_confidence -= 15  # Reduced from 30
                    elif ml_probability > 0.6:
                        base_confidence -= 10  # Reduced from 20
                    elif ml_probability > 0.5:
                        base_confidence -= 5   # Reduced from 10
            
            # Adjust confidence based on historical model accuracy (reduced)
            if 'test_score' in self.model_info:
                model_accuracy = self.model_info['test_score']
                accuracy_bonus = int((model_accuracy - 0.5) * 10)  # Reduced from 20 to 10
                base_confidence += accuracy_bonus
            
            # Adjust confidence based on price change magnitude (reduced)
            if abs(price_change) > 0.1:  # >10% change
                base_confidence -= 5   # Reduced from 10
            elif abs(price_change) < 0.02:  # <2% change
                base_confidence += 5   # Reduced from 10
            
            # Adjust confidence based on volatility (reduced)
            current_volatility = self.data['Volatility_20'].iloc[-1]
            avg_volatility = self.data['Volatility_20'].mean()
            
            if current_volatility > avg_volatility * 1.5:
                base_confidence -= 8   # Reduced from 15
            elif current_volatility < avg_volatility * 0.5:
                base_confidence += 5   # Reduced from 10
            
            # Ensure confidence is within bounds (0-100)
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
        Two-stage approach: pre-output with existing model, then final output after retraining
        """
        print(f"\n{'='*50}")
        print(f"CHINESE STOCK ANALYSIS: {symbol} ({market}-shares)")
        print(f"{'='*50}")
        
        # Clear data if switching to a different stock
        if hasattr(self, 'symbol') and self.symbol != symbol:
            print(f"🔄 Switching from {self.symbol} to {symbol}, clearing previous data...")
            self.clear_data()
        
        # Set symbol and market
        self.symbol = symbol
        self.market_type = market
        
        # Try to load existing model first
        model_loaded = self.load_model(symbol, market)
        
        # Check if we already have data for this symbol
        data_already_available = (self.data is not None and 
                                hasattr(self, 'symbol') and 
                                self.symbol == symbol and 
                                len(self.data) > 50)
        
        if data_already_available:
            print(f"✅ Using existing data for {symbol} ({len(self.data)} days)")
            # Get stock name from cache or downloader
            stock_name = self.downloader.get_stock_name(symbol, market)
        else:
            # Download data only if not already available
            print(f"📥 Downloading data for {symbol}...")
            download_success, stock_name = self.download_chinese_stock_data(symbol, market)
            if not download_success:
                print(f"Failed to download data for {symbol}. Cannot analyze.")
                return None
        
        # Calculate indicators if not already calculated
        if not data_already_available or 'Price_Momentum_5' not in self.data.columns:
            self.calculate_chinese_indicators()
        
        # PHASE 1: INITIAL ANALYSIS with existing model or fallback
        print(f"\n🔍 PHASE 1: INITIAL ANALYSIS")
        pre_result = self._generate_pre_output(stock_name, model_loaded)
        
        if pre_result:
            print(f"✅ Initial analysis completed!")
            print(f"   Score: {pre_result['score']:.1f}/100")
            print(f"   Recommendation: {pre_result['recommendation']}")
            if pre_result['ml_probability'] is not None:
                print(f"   ML Probability: {pre_result['ml_probability']:.1%}")
        
        # PHASE 2: MODEL PREPARATION AND RETRAINING
        print(f"\n🤖 PHASE 2: MODEL PREPARATION")
        if not model_loaded:
            # Train ML model only if not already trained
            print(f"🤖 Training new ML model for {symbol}...")
            ml_success = self.train_ml_model(holding_period=10, profit_threshold=0.03)
            
            if ml_success:
                # Save the newly trained model
                self.save_model(symbol, market)
                print(f"✅ New model trained and saved!")
            else:
                print(f"⚠️  Model training failed, using fallback")
        else:
            print(f"✅ Using existing ML model for {symbol}")
            ml_success = True
        
        # PHASE 3: FINAL ANALYSIS WITH OPTIMIZED MODEL
        print(f"\n🎯 PHASE 3: FINAL ANALYSIS")
        final_result = self._generate_final_output(stock_name, ml_success)
        
        if final_result:
            print(f"✅ Final analysis completed!")
            print(f"   Score: {final_result['score']:.1f}/100")
            print(f"   Recommendation: {final_result['recommendation']}")
            if final_result['ml_probability'] is not None:
                print(f"   ML Probability: {final_result['ml_probability']:.1%}")
            
            # Show improvement if available
            if pre_result and final_result:
                score_improvement = final_result['score'] - pre_result['score']
                if abs(score_improvement) > 1:
                    print(f"   Score Change: {score_improvement:+.1f} points")
        
        # Save the final result to a file
        self.save_analysis_to_file(final_result, symbol, market)
        
        return final_result
    
    def _generate_pre_output(self, stock_name, model_loaded):
        """
        Generate pre-output using existing model or fallback
        """
        try:
            # Get current values
            current = self.data.iloc[-1]
            current_price = current['close']
            
            # Calculate technical score
            technical_score = self.calculate_chinese_technical_score()
            
            # Get ML prediction using existing model or fallback
            if model_loaded and self.model is not None:
                ml_prediction, ml_probability = self.get_ml_prediction()
                ml_model_used = self.model_info.get('model_type', 'Existing Model')
            else:
                # Use fallback ML calculation
                ml_prediction, ml_probability = self._calculate_fallback_ml()
                ml_model_used = 'Fallback ML (Technical + Market Conditions)'
            
            # Calculate final score (60% technical + 40% ML)
            if ml_probability is not None:
                ml_score = int(ml_probability * 100)
                final_score = int(technical_score * 0.6 + ml_score * 0.4)
            else:
                ml_score = 0
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
            
            # Calculate price estimates
            estimated_high_10d = current_price * 1.08  # 8% potential gain
            estimated_low_10d = current_price * 0.98   # 2% potential loss
            potential_gain_10d = 0.08
            potential_loss_10d = -0.02
            
            # Calculate confidence for price estimates
            high_confidence, high_reasoning = self.calculate_ml_price_confidence(estimated_high_10d, current_price, 'high')
            low_confidence, low_reasoning = self.calculate_ml_price_confidence(estimated_low_10d, current_price, 'low')
            
            return {
                'symbol': self.symbol,
                'market': self.market_type,
                'score': final_score,
                'technical_score': technical_score,
                'ml_score': ml_score,  # Add ML score to the result
                'ml_probability': ml_probability,
                'ml_prediction': ml_prediction,
                'ml_model_used': ml_model_used,
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
                'stock_name': stock_name,
                'stage': 'pre_output'
            }
            
        except Exception as e:
            print(f"❌ Error generating pre-output: {str(e)}")
            return None
    
    def _generate_final_output(self, stock_name, ml_success):
        """
        Generate final output after model retraining
        """
        try:
            # Get current values
            current = self.data.iloc[-1]
            current_price = current['close']
            
            # Calculate technical score
            technical_score = self.calculate_chinese_technical_score()
            
            # Get ML prediction using retrained model
            if ml_success and self.model is not None:
                ml_prediction, ml_probability = self.get_ml_prediction()
                ml_model_used = self.model_info.get('model_type', 'Retrained Model')
            else:
                # Use fallback if retraining failed
                ml_prediction, ml_probability = self._calculate_fallback_ml()
                ml_model_used = 'Fallback ML (Technical + Market Conditions)'
            
            # Calculate final score (60% technical + 40% ML)
            if ml_probability is not None:
                ml_score = int(ml_probability * 100)
                final_score = int(technical_score * 0.6 + ml_score * 0.4)
            else:
                ml_score = 0
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
            
            # Calculate price estimates
            estimated_high_10d = current_price * 1.08  # 8% potential gain
            estimated_low_10d = current_price * 0.98   # 2% potential loss
            potential_gain_10d = 0.08
            potential_loss_10d = -0.02
            
            # Calculate confidence for price estimates
            high_confidence, high_reasoning = self.calculate_ml_price_confidence(estimated_high_10d, current_price, 'high')
            low_confidence, low_reasoning = self.calculate_ml_price_confidence(estimated_low_10d, current_price, 'low')
            
            return {
                'symbol': self.symbol,
                'market': self.market_type,
                'score': final_score,
                'technical_score': technical_score,
                'ml_score': ml_score,  # Add ML score to the result
                'ml_probability': ml_probability,
                'ml_prediction': ml_prediction,
                'ml_model_used': ml_model_used,
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
                'stock_name': stock_name,
                'stage': 'final_output'
            }
            
        except Exception as e:
            print(f"❌ Error generating final output: {str(e)}")
            return None
    
    def _calculate_fallback_ml(self):
        """
        Calculate fallback ML prediction using technical indicators and market conditions
        """
        try:
            if self.data is None or len(self.data) < 20:
                return None, None
            
            current = self.data.iloc[-1]
            
            # Base probability
            base_prob = 0.5
            
            # RSI-based adjustment
            if 'RSI' in current and not pd.isna(current['RSI']):
                rsi = current['RSI']
                if rsi > 70:
                    base_prob -= 0.2  # Overbought - lower probability of rise
                elif rsi > 60:
                    base_prob -= 0.1
                elif rsi < 30:
                    base_prob += 0.2  # Oversold - higher probability of rise
                elif rsi < 40:
                    base_prob += 0.1
            
            # Momentum-based adjustment
            momentum_5 = current['Price_Momentum_5']
            if momentum_5 > 0.05:
                base_prob += 0.15  # Strong positive momentum
            elif momentum_5 > 0.02:
                base_prob += 0.1   # Moderate positive momentum
            elif momentum_5 < -0.05:
                base_prob -= 0.15  # Strong negative momentum
            elif momentum_5 < -0.02:
                base_prob -= 0.1   # Moderate negative momentum
            
            # Volume-based adjustment
            volume_ratio = current['Volume_Ratio']
            if volume_ratio > 1.5:
                base_prob += 0.1   # High volume - potential accumulation
            elif volume_ratio < 0.5:
                base_prob -= 0.05  # Low volume - lack of interest
            
            # Moving average-based adjustment
            price = current['close']
            sma_20 = current['SMA_20']
            sma_50 = current['SMA_50']
            
            if price > sma_20 > sma_50:
                base_prob += 0.1   # Strong uptrend
            elif price > sma_20:
                base_prob += 0.05  # Moderate uptrend
            elif price < sma_20 < sma_50:
                base_prob -= 0.1   # Strong downtrend
            elif price < sma_20:
                base_prob -= 0.05  # Moderate downtrend
            
            # MACD-based adjustment
            if 'MACD' in current and not pd.isna(current['MACD']):
                macd = current['MACD']
                if macd > 0:
                    base_prob += 0.05  # Positive MACD
                else:
                    base_prob -= 0.05  # Negative MACD
            
            # Ensure probability is within bounds
            probability = max(0.0, min(1.0, base_prob))
            
            # Determine prediction
            prediction = 1 if probability > 0.5 else 0
            
            return prediction, probability
            
        except Exception as e:
            print(f"⚠️  Fallback ML calculation failed: {str(e)}")
            return None, None
    
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
        
        return {
            'symbol': symbol,
            'market': market,
            'buy_price': buy_price,
            'current_price': current_price,
            'current_return': current_return,
            'sell_analysis': sell_analysis,
            'recommendation': recommendation,
            'stock_name': stock_name # Add stock_name to the result
        }
    
    def analyze_chinese_sell_signals(self, buy_price, market='A'):
        """
        Analyze sell signals specific to Chinese markets with ML integration
        """
        if self.data is None or len(self.data) < 20:
            return None
        
        current = self.data.iloc[-1]
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
        
        # Calculate technical sell score
        tech_score = self.calculate_chinese_sell_technical_score()
        analysis['technical_score'] = tech_score
        
        # Try to get ML prediction
        try:
            # Ensure we have a model for this stock
            if self.model is None:
                # Try to load existing model
                if not self.load_model(self.symbol, market):
                    # Train new model if loading fails
                    print(f"Training new ML model for {self.symbol}...")
                    self.train_ml_model(holding_period=10, profit_threshold=0.03)
            
            # Get ML prediction
            ml_prediction, ml_probability = self.get_ml_prediction()
            analysis['ml_prediction'] = ml_prediction
            analysis['ml_probability'] = ml_probability
            
            # Calculate ML score for sell analysis
            if ml_probability is not None:
                # Convert ML probability to sell signal strength (0-100)
                # For sell analysis: Higher probability of decline = higher sell signal
                # ML Probability 0.271 means 27.1% chance of rise, so 72.9% chance of decline
                sell_signal_strength = int((1 - ml_probability) * 100)
                analysis['ml_score'] = sell_signal_strength  # Keep for compatibility
                analysis['sell_signal_strength'] = sell_signal_strength  # More descriptive name
                
                # Combine technical and ML scores (70% technical, 30% ML)
                combined_score = int(0.7 * tech_score + 0.3 * sell_signal_strength)
                analysis['combined_score'] = combined_score
            else:
                # If ML not available, use technical score only
                analysis['combined_score'] = tech_score
                print(f"⚠️  ML prediction not available for {self.symbol}, using technical score only")
                
        except Exception as e:
            print(f"⚠️  ML analysis failed for {self.symbol}: {str(e)}")
            # Use technical score only if ML fails
            analysis['combined_score'] = tech_score
        
        # Current return analysis
        current_return = (current['close'] - buy_price) / buy_price
        
        # Profit target analysis (Chinese markets often have different targets)
        if current_return >= 0.05:  # 5% profit target for Chinese markets
            analysis['sell_signals'].append("Profit target reached (5%+)")
            analysis['profit_potential'] = current_return
        elif current_return >= 0.03:  # 3% profit target
            analysis['sell_signals'].append("Moderate profit target reached (3%+)")
            analysis['profit_potential'] = current_return
        
        # Stop loss analysis
        if current_return <= -0.03:  # 3% stop loss for Chinese markets
            analysis['sell_signals'].append("Stop loss triggered (-3%+)")
            analysis['stop_loss_triggered'] = True
            analysis['risk_factors'].append("Significant loss position")
        elif current_return <= -0.02:  # 2% stop loss
            analysis['sell_signals'].append("Approaching stop loss (-2%+)")
            analysis['risk_factors'].append("Loss position")
        
        # ML-based signals
        if ml_probability is not None:
            if ml_probability < 0.3:
                analysis['sell_signals'].append(f"ML predicts strong decline (prob: {ml_probability:.1%})")
            elif ml_probability < 0.4:
                analysis['sell_signals'].append(f"ML predicts moderate decline (prob: {ml_probability:.1%})")
            elif ml_probability > 0.7:
                analysis['hold_signals'].append(f"ML predicts strong rise (prob: {ml_probability:.1%})")
            elif ml_probability > 0.6:
                analysis['hold_signals'].append(f"ML predicts moderate rise (prob: {ml_probability:.1%})")
        
        # Chinese market specific analysis
        if market.upper() == 'A':
            # A-shares specific signals
            if current['Volume_Ratio'] > 3.0:
                analysis['sell_signals'].append("Extremely high volume - potential distribution")
            elif current['Volume_Ratio'] > 2.0:
                analysis['sell_signals'].append("High volume - monitor closely")
            
            # Limit up/down analysis for A-shares
            if hasattr(current, 'Near_Limit_Up') and current['Near_Limit_Up'] > 95:
                analysis['limit_up_near'] = True
                analysis['sell_signals'].append("Near limit up - high risk of reversal")
            elif hasattr(current, 'Near_Limit_Down') and current['Near_Limit_Down'] < -95:
                analysis['limit_down_near'] = True
                analysis['hold_signals'].append("Near limit down - potential bounce")
        
        # Technical indicator analysis
        if current['Price_Momentum_5'] < -0.05:
            analysis['sell_signals'].append("Strong negative 5-day momentum")
        elif current['Price_Momentum_5'] < -0.02:
            analysis['sell_signals'].append("Negative 5-day momentum")
        elif current['Price_Momentum_5'] > 0.05:
            analysis['hold_signals'].append("Strong positive 5-day momentum")
        elif current['Price_Momentum_5'] > 0.02:
            analysis['hold_signals'].append("Positive 5-day momentum")
        
        # Moving average analysis
        if current['close'] < current['SMA_20']:
            analysis['sell_signals'].append("Price below 20-day SMA")
        else:
            analysis['hold_signals'].append("Price above 20-day SMA")
        
        if current['close'] < current['SMA_50']:
            analysis['sell_signals'].append("Price below 50-day SMA")
        else:
            analysis['hold_signals'].append("Price above 50-day SMA")
        
        # Volume analysis
        if current['Volume_Ratio'] > 1.5:
            analysis['sell_signals'].append("High volume - potential distribution")
        elif current['Volume_Ratio'] < 0.5:
            analysis['hold_signals'].append("Low volume - accumulation possible")
        
        # Volatility analysis
        avg_volatility = self.data['Volatility_20'].mean()
        if current['Volatility_20'] > avg_volatility * 1.5:
            analysis['risk_factors'].append("High volatility - increased risk")
        elif current['Volatility_20'] < avg_volatility * 0.5:
            analysis['hold_signals'].append("Low volatility - stable conditions")
        
        return analysis
    
    def calculate_chinese_sell_technical_score(self):
        """
        Calculate technical score for Chinese stock sell decision (0-100)
        Higher score = stronger sell signal
        """
        if self.data is None or len(self.data) < 20:
            return 50
        
        current = self.data.iloc[-1]
        score = 50  # Neutral base score
        
        # Momentum analysis (negative momentum = sell signal)
        momentum_5 = current['Price_Momentum_5']
        if momentum_5 < -0.05:
            score += 25  # Strong sell signal
        elif momentum_5 < -0.02:
            score += 15  # Moderate sell signal
        elif momentum_5 > 0.05:
            score -= 20  # Strong buy signal (hold) - but don't go below 0
        elif momentum_5 > 0.02:
            score -= 10  # Moderate buy signal (hold)
        
        # Moving average analysis (below MA = sell signal)
        price = current['close']
        sma_20 = current['SMA_20']
        sma_50 = current['SMA_50']
        
        if price < sma_20:
            score += 15  # Below short-term MA
        else:
            score -= 10  # Above short-term MA
        
        if price < sma_50:
            score += 15  # Below long-term MA
        else:
            score -= 10  # Above long-term MA
        
        # Volume analysis (high volume often precedes decline)
        volume_ratio = current['Volume_Ratio']
        if volume_ratio > 2.0:
            score += 15  # High volume - potential distribution
        elif volume_ratio > 1.5:
            score += 10  # Moderate high volume
        elif volume_ratio < 0.5:
            score -= 5   # Low volume - accumulation possible
        
        # Volatility analysis (high volatility = increased risk)
        avg_volatility = self.data['Volatility_20'].mean()
        current_volatility = current['Volatility_20']
        if current_volatility > avg_volatility * 1.5:
            score += 15  # High volatility - increased risk
        elif current_volatility > avg_volatility * 1.2:
            score += 10  # Moderate high volatility
        elif current_volatility < avg_volatility * 0.5:
            score -= 5   # Low volatility - stable conditions
        
        # RSI analysis (if available)
        if 'RSI' in current and not pd.isna(current['RSI']):
            rsi = current['RSI']
            if rsi > 80:
                score += 20  # Very overbought - strong sell signal
            elif rsi > 70:
                score += 15  # Overbought - sell signal
            elif rsi < 20:
                score -= 20  # Very oversold - strong buy signal
            elif rsi < 30:
                score -= 15  # Oversold - buy signal
        
        # MACD analysis (if available)
        if 'MACD' in current and not pd.isna(current['MACD']):
            macd = current['MACD']
            if macd < 0:
                score += 10  # Negative MACD - bearish
            else:
                score -= 5   # Positive MACD - bullish
        
        # Bollinger Band analysis (if available)
        if 'BB_Position' in current and not pd.isna(current['BB_Position']):
            bb_pos = current['BB_Position']
            if bb_pos > 0.8:
                score += 10  # Near upper band - potential reversal
            elif bb_pos < 0.2:
                score -= 10  # Near lower band - potential bounce
        
        # Ensure score is within bounds (0-100)
        score = max(0, min(100, score))
        
        return score
    
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
        
        # Add ML insights with clearer explanation for sell analysis
        ml_insights = ""
        if ml_probability is not None:
            decline_probability = 1 - ml_probability  # Convert to decline probability
            if ml_probability < 0.3:
                ml_insights = f"ML strongly predicts decline ({decline_probability:.1%} chance of decline, {ml_probability:.1%} chance of rise)"
            elif ml_probability < 0.4:
                ml_insights = f"ML moderately predicts decline ({decline_probability:.1%} chance of decline, {ml_probability:.1%} chance of rise)"
            elif ml_probability > 0.7:
                ml_insights = f"ML strongly predicts rise ({ml_probability:.1%} chance of rise, {decline_probability:.1%} chance of decline)"
            elif ml_probability > 0.6:
                ml_insights = f"ML moderately predicts rise ({ml_probability:.1%} chance of rise, {decline_probability:.1%} chance of decline)"
            else:
                ml_insights = f"ML neutral ({ml_probability:.1%} chance of rise, {decline_probability:.1%} chance of decline)"
        else:
            ml_insights = "ML prediction not available"
        
        return {
            'action': action,
            'urgency': urgency,
            'reasoning': reasoning,
            'technical_score': tech_score,
            'ml_score': ml_score,
            'combined_score': combined_score,
            'ml_probability': ml_probability,
            'ml_prediction': sell_analysis['ml_prediction'],
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

    def clear_data(self):
        """Clear current data when switching to a different stock"""
        self.data = None
        self.symbol = None
        self.market_type = None 

    def save_analysis_to_file(self, result, symbol, market='A'):
        """
        Save single stock analysis results to a single file with timestamp
        """
        if not result:
            return False
        
        try:
            # Use a single file for all analyses
            filename = "chinese_stock_analyses.txt"
            
            with open(filename, 'a', encoding='utf-8') as f:
                # Add separator and timestamp
                f.write(f"\n{'='*80}\n")
                f.write(f"🔍 CHINESE STOCK ANALYSIS REPORT\n")
                f.write(f"Symbol: {symbol} ({market}-shares)\n")
                f.write(f"Stock Name: {result.get('stock_name', 'Unknown')}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
                
                # Analysis results
                f.write(f"📊 ANALYSIS RESULTS:\n")
                f.write(f"   Final Score: {result.get('score', 0):.1f}/100\n")
                f.write(f"   Recommendation: {result.get('recommendation', 'UNKNOWN')}\n")
                f.write(f"   Confidence: {result.get('confidence', 'UNKNOWN')}\n")
                f.write(f"   Current Price: ¥{result.get('current_price', 0):.2f}\n\n")
                
                # Scoring breakdown
                f.write(f"📈 SCORING BREAKDOWN:\n")
                f.write(f"   Technical Score: {result.get('technical_score', 0):.1f}/100\n")
                f.write(f"   ML Score: {result.get('ml_score', 0):.1f}/100\n")
                f.write(f"   ML Probability: {result.get('ml_probability', 0):.1%}\n")
                f.write(f"   ML Model Used: {result.get('ml_model_used', 'Unknown')}\n\n")
                
                # Price estimates
                if result.get('estimated_high_10d', 0) > 0 and result.get('estimated_low_10d', 0) > 0:
                    f.write(f"📈 10-DAY PRICE ESTIMATES:\n")
                    f.write(f"   High: ¥{result.get('estimated_high_10d', 0):.2f} (Confidence: {result.get('high_confidence', 0):.1f}%)\n")
                    f.write(f"   Low: ¥{result.get('estimated_low_10d', 0):.2f} (Confidence: {result.get('low_confidence', 0):.1f}%)\n")
                    f.write(f"   Potential Gain: +{result.get('potential_gain_10d', 0):.1%}\n")
                    f.write(f"   Potential Loss: {result.get('potential_loss_10d', 0):.1%}\n\n")
                
                # Technical indicators
                f.write(f"📊 TECHNICAL INDICATORS:\n")
                f.write(f"   5-Day Momentum: {result.get('momentum_5d', 0):.2%}\n")
                f.write(f"   Volume Ratio: {result.get('volume_ratio', 0):.2f}\n")
                f.write(f"   Volatility (20d): {result.get('volatility', 0):.2%}\n\n")
                
                # ML insights
                if result.get('ml_probability') is not None:
                    f.write(f"🤖 ML INSIGHTS:\n")
                    f.write(f"   ML Prediction: {result.get('ml_prediction', 'Unknown')}\n")
                    f.write(f"   ML Probability: {result.get('ml_probability', 0):.1%}\n")
                    f.write(f"   Model Type: {result.get('ml_model_used', 'Unknown')}\n\n")
                
                # Analysis stage
                f.write(f"🔄 ANALYSIS STAGE: {result.get('stage', 'Unknown')}\n\n")
                
                f.write(f"{'='*80}\n")
                f.write(f"💡 DISCLAIMER: This analysis is for educational purposes only.\n")
                f.write(f"   Always conduct your own research before making investment decisions.\n")
                f.write(f"{'='*80}\n")
            
            print(f"💾 Analysis saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving analysis to file: {str(e)}")
            return False