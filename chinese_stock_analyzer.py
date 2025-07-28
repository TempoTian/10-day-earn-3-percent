import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import warnings
import pickle
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class ChineseStockAnalyzer:
    def __init__(self):
        self.data = None
        self.symbol = None
        self.market_type = None
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.ml_model_used = False
        self.model_dir = "chinese_models"
        self.model_info = {}
        self.best_params = {}
        
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
        """
        Download Chinese stock data
        """
        try:
            self.symbol = symbol.upper()
            self.market_type = market.upper()
            
            formatted_symbol = self.get_chinese_stock_symbol(symbol, market)
            print(f"Downloading data for {symbol} ({market}-shares) as {formatted_symbol}")
            
            ticker = yf.Ticker(formatted_symbol)
            self.data = ticker.history(period=period)
            
            if self.data.empty:
                print(f"No data found for {formatted_symbol}")
                return False
            
            print(f"Successfully downloaded {len(self.data)} days of data for {symbol}")
            print(f"Data range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            
            return True
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            return False
    
    def calculate_chinese_indicators(self):
        """
        Calculate indicators for Chinese markets
        """
        if self.data is None:
            return
        
        # Basic indicators
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA_20']
        
        # Moving averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # Chinese market specific
        self.data['Price_Momentum_5'] = self.data['Close'] / self.data['Close'].shift(5) - 1
        self.data['Price_Momentum_10'] = self.data['Close'] / self.data['Close'].shift(10) - 1
        
        # Volatility
        self.data['Volatility_20'] = self.data['Returns'].rolling(window=20).std()
        
        # Additional indicators for ML model
        self.data['Price_Momentum_3'] = self.data['Close'] / self.data['Close'].shift(3) - 1
        self.data['Price_Momentum_20'] = self.data['Close'] / self.data['Close'].shift(20) - 1
        self.data['Volatility_10'] = self.data['Returns'].rolling(window=10).std()
        self.data['Volatility_50'] = self.data['Returns'].rolling(window=50).std()
        
        print("Chinese market indicators calculated successfully!")
    
    def create_ml_features(self):
        """
        Create advanced features for ML model (ENHANCED)
        """
        features = [
            'Returns', 'Price_Momentum_3', 'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20',
            'SMA_20', 'SMA_50', 'Volume_Ratio', 'Volatility_10', 'Volatility_20', 'Volatility_50'
        ]
        
        # Add more sophisticated features
        if len(self.data) > 50:
            # Price position relative to moving averages
            self.data['Price_vs_SMA20'] = (self.data['Close'] - self.data['SMA_20']) / self.data['SMA_20']
            self.data['Price_vs_SMA50'] = (self.data['Close'] - self.data['SMA_50']) / self.data['SMA_50']
            
            # Trend strength
            self.data['Trend_Strength'] = abs(self.data['SMA_20'] - self.data['SMA_50']) / self.data['SMA_50']
            
            # Volume trend
            self.data['Volume_Trend'] = self.data['Volume'].rolling(window=5).mean() / self.data['Volume'].rolling(window=20).mean()
            
            # Momentum consistency
            self.data['Momentum_Consistency'] = self.data['Price_Momentum_5'].rolling(window=5).std()
            
            # Advanced technical indicators
            self.data['RSI'] = self.calculate_rsi(self.data['Close'], window=14)
            self.data['MACD'] = self.calculate_macd(self.data['Close'])
            self.data['BB_Position'] = self.calculate_bollinger_position(self.data['Close'])
            
            # Price patterns
            self.data['Price_Range'] = (self.data['High'] - self.data['Low']) / self.data['Close']
            self.data['Gap_Up'] = (self.data['Open'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1)
            self.data['Gap_Down'] = (self.data['Close'].shift(1) - self.data['Open']) / self.data['Close'].shift(1)
            
            # Volume analysis
            self.data['Volume_Price_Trend'] = self.data['Volume'] * self.data['Returns']
            self.data['Volume_SMA_Ratio'] = self.data['Volume'] / self.data['Volume'].rolling(window=10).mean()
            
            # Volatility features
            self.data['Volatility_Ratio'] = self.data['Volatility_10'] / self.data['Volatility_20']
            self.data['Volatility_Change'] = self.data['Volatility_20'].pct_change()
            
            # Momentum features
            self.data['Momentum_Acceleration'] = self.data['Price_Momentum_5'].diff()
            self.data['Momentum_Reversal'] = np.where(self.data['Price_Momentum_5'] > 0, 
                                                     self.data['Price_Momentum_5'].shift(1) < 0, 
                                                     self.data['Price_Momentum_5'].shift(1) > 0).astype(int)
            
            # Market regime features
            self.data['Bull_Market'] = (self.data['Close'] > self.data['SMA_20']) & (self.data['SMA_20'] > self.data['SMA_50']).astype(int)
            self.data['Bear_Market'] = (self.data['Close'] < self.data['SMA_20']) & (self.data['SMA_20'] < self.data['SMA_50']).astype(int)
            
            # Additional features
            features.extend([
                'Price_vs_SMA20', 'Price_vs_SMA50', 'Trend_Strength', 'Volume_Trend', 'Momentum_Consistency',
                'RSI', 'MACD', 'BB_Position', 'Price_Range', 'Gap_Up', 'Gap_Down',
                'Volume_Price_Trend', 'Volume_SMA_Ratio', 'Volatility_Ratio', 'Volatility_Change',
                'Momentum_Acceleration', 'Momentum_Reversal', 'Bull_Market', 'Bear_Market'
            ])
        
        return features
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
        future_returns = self.data['Close'].shift(-holding_period) / self.data['Close'] - 1
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
            
            # Cross-validation score with stratified folds
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
            
            # Check if we have valid features
            if np.isnan(current_features).any():
                return None, None
            
            # Use pipeline for prediction (includes scaling and feature selection)
            try:
                probability = self.model.predict_proba(current_features)[0][1]
                prediction = 1 if probability > 0.30 else 0
                return prediction, probability
            except Exception as e:
                # Fallback to direct prediction
                prediction = self.model.predict(current_features)[0]
                probability = 0.5 if prediction == 1 else 0.5
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
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['Close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods
        recent_lows = []
        for i in range(max(0, len(self.data) - 15), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_low = self.data['Low'].iloc[i:i+10].min()
                period_start = self.data['Close'].iloc[i]
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
        if current['Close'] < current['SMA_20'] < current['SMA_50']:
            ma_adjustment = 0.95  # Strong downtrend
        elif current['Close'] < current['SMA_20']:
            ma_adjustment = 0.98  # Moderate downtrend
        elif current['Close'] > current['SMA_20'] > current['SMA_50']:
            ma_adjustment = 1.05  # Strong uptrend
        elif current['Close'] > current['SMA_20']:
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
    
    def estimate_highest_price_10_days(self, symbol, current_price, market='A'):
        """
        Estimate the highest price in the next 10 days for Chinese stocks (IMPROVED ACCURACY)
        """
        if self.data is None or len(self.data) < 20:
            return current_price * 1.03  # More realistic default
        
        # Calculate historical volatility
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['Close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods (more realistic approach)
        recent_highs = []
        for i in range(max(0, len(self.data) - 30), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_high = self.data['High'].iloc[i:i+10].max()
                period_start = self.data['Close'].iloc[i]
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
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            ma_adjustment = 1.02  # Reduced from 1.05
        elif current['Close'] > current['SMA_20']:
            ma_adjustment = 1.01  # Reduced from 1.02
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            ma_adjustment = 0.98  # Reduced from 0.95
        elif current['Close'] < current['SMA_20']:
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
            if not self.download_chinese_stock_data(symbol, market):
                return None
            
            self.calculate_chinese_indicators()
            
            # Train ML model
            ml_success = self.train_ml_model(holding_period=10, profit_threshold=0.03)
            
            if ml_success:
                # Save the newly trained model
                self.save_model(symbol, market)
        else:
            # Model loaded successfully, just download latest data for analysis
            if not self.download_chinese_stock_data(symbol, market):
                return None
            
            self.calculate_chinese_indicators()
            ml_success = True
        
        current = self.data.iloc[-1]
        
        # Calculate technical score
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
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            score += 15
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            score -= 15
        
        # Volatility scoring
        avg_volatility = self.data['Volatility_20'].mean()
        if current['Volatility_20'] > avg_volatility * 1.5:
            score -= 10
        elif current['Volatility_20'] < avg_volatility * 0.5:
            score += 5
        
        score = max(0, min(100, score))
        
        # Get ML prediction
        ml_prediction, ml_probability = self.get_ml_prediction()
        
        # Calculate combined score (60% Technical, 40% ML if available)
        if ml_probability is not None:
            combined_score = 0.6 * (score / 100) + 0.4 * ml_probability
            final_score = int(combined_score * 100)
        else:
            final_score = score
        
        # Generate recommendation
        if final_score >= 80:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif final_score >= 65:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif final_score >= 45:
            recommendation = "HOLD"
            confidence = "LOW"
        else:
            recommendation = "AVOID"
            confidence = "HIGH"
        
        # Calculate estimated highest and lowest prices
        current_price = current['Close']
        estimated_high = self.estimate_highest_price_10_days(symbol, current_price, market)
        estimated_low = self.estimate_lowest_price_10_days(symbol, current_price, market)
        potential_gain = (estimated_high - current_price) / current_price
        potential_loss = (estimated_low - current_price) / current_price
        
        return {
            'symbol': symbol,
            'market': market,
            'current_price': current_price,
            'estimated_high_10d': estimated_high,
            'estimated_low_10d': estimated_low,
            'potential_gain_10d': potential_gain,
            'potential_loss_10d': potential_loss,
            'score': final_score,
            'technical_score': score,
            'ml_prediction': ml_prediction,
            'ml_probability': ml_probability,
            'ml_model_used': ml_success,
            'model_loaded': model_loaded,
            'recommendation': recommendation,
            'confidence': confidence,
            'momentum_5d': current['Price_Momentum_5'],
            'volume_ratio': current['Volume_Ratio'],
            'volatility': current['Volatility_20']
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
            print(f"   Score: {result['score']}/100")
            print(f"   Technical Score: {result['technical_score']}/100")
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
        print(f"Score: {best['score']}/100")
        print(f"Technical Score: {best['technical_score']}/100")
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
        if not self.download_chinese_stock_data(symbol, market):
            return None
        
        # Calculate indicators
        self.calculate_chinese_indicators()
        
        # Get current market position
        current_price = self.data.iloc[-1]['Close']
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
            'recommendation': recommendation
        }
    
    def analyze_chinese_sell_signals(self, buy_price, market='A'):
        """
        Analyze sell signals specific to Chinese markets
        """
        if self.data is None or len(self.data) < 20:
            return None
        
        current = self.data.iloc[-1]
        analysis = {
            'technical_score': 0,
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
        
        # Current return analysis
        current_return = (current['Close'] - buy_price) / buy_price
        
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
        if current['Close'] < current['SMA_20']:
            analysis['sell_signals'].append("Price below 20-day SMA")
        else:
            analysis['hold_signals'].append("Price above 20-day SMA")
        
        if current['Close'] < current['SMA_50']:
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
        """
        if self.data is None or len(self.data) < 20:
            return 50
        
        current = self.data.iloc[-1]
        score = 50  # Neutral base score
        
        # Momentum analysis
        if current['Price_Momentum_5'] < -0.05:
            score += 20  # Strong sell signal
        elif current['Price_Momentum_5'] < -0.02:
            score += 10  # Moderate sell signal
        elif current['Price_Momentum_5'] > 0.05:
            score -= 20  # Strong buy signal (hold)
        elif current['Price_Momentum_5'] > 0.02:
            score -= 10  # Moderate buy signal (hold)
        
        # Moving average analysis
        if current['Close'] < current['SMA_20']:
            score += 15  # Below short-term MA
        else:
            score -= 15  # Above short-term MA
        
        if current['Close'] < current['SMA_50']:
            score += 15  # Below long-term MA
        else:
            score -= 15  # Above long-term MA
        
        # Volume analysis
        if current['Volume_Ratio'] > 2.0:
            score += 10  # High volume
        elif current['Volume_Ratio'] < 0.5:
            score -= 5  # Low volume
        
        # Volatility analysis
        avg_volatility = self.data['Volatility_20'].mean()
        if current['Volatility_20'] > avg_volatility * 1.5:
            score += 10  # High volatility
        elif current['Volatility_20'] < avg_volatility * 0.5:
            score -= 5  # Low volatility
        
        return max(0, min(100, score))
    
    def generate_chinese_sell_recommendation(self, sell_analysis, current_return, buy_price, market='A'):
        """
        Generate sell recommendation for Chinese stocks
        """
        if not sell_analysis:
            return None
        
        tech_score = sell_analysis['technical_score']
        
        # Determine sell action based on Chinese market characteristics
        if tech_score >= 75:
            action = "SELL NOW"
            urgency = "HIGH"
            reasoning = "Strong technical signals indicate selling"
        elif tech_score >= 60:
            action = "SELL SOON"
            urgency = "MEDIUM"
            reasoning = "Moderate sell signals detected"
        elif tech_score >= 40:
            action = "HOLD"
            urgency = "LOW"
            reasoning = "Mixed signals - consider holding"
        else:
            action = "HOLD/ADD"
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
            # Calculate based on technical analysis
            if tech_score >= 60:
                target_price = buy_price * (1 + max(current_return, 0.01))
            else:
                target_price = buy_price * 1.03
        
        # Risk assessment for Chinese markets
        risk_level = "LOW"
        if sell_analysis['stop_loss_triggered']:
            risk_level = "HIGH"
        elif sell_analysis['limit_up_near']:
            risk_level = "HIGH"
        elif len(sell_analysis['risk_factors']) > 2:
            risk_level = "MEDIUM"
        
        return {
            'action': action,
            'urgency': urgency,
            'reasoning': reasoning,
            'technical_score': tech_score,
            'target_price': target_price,
            'risk_level': risk_level,
            'sell_signals': sell_analysis['sell_signals'],
            'hold_signals': sell_analysis['hold_signals'],
            'risk_factors': sell_analysis['risk_factors'],
            'profit_potential': sell_analysis['profit_potential'],
            'limit_up_near': sell_analysis['limit_up_near'],
            'limit_down_near': sell_analysis['limit_down_near']
        } 