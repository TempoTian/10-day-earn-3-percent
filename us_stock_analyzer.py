import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class EnhancedStockAnalyzer:
    def __init__(self):
        self.data = None
        self.symbol = None
        self.backtest_results = None
        self.model = None
        self.scaler = StandardScaler()
        self.model_history = []
        self.model_performance = []
        self.last_update_date = None
        self.model_version = 0
        self.initial_training_complete = False
        self.model_dir = "us_models"
        self.model_info = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def get_model_filename(self, symbol):
        """
        Generate model filename for a specific stock
        """
        return f"{self.model_dir}/{symbol}_model.pkl"
    
    def save_model(self, symbol):
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
                'last_trained': datetime.now(),
                'data_points': len(self.data) if self.data is not None else 0,
                'features': self.create_features(),
                'model_info': self.model_info,
                'backtest_results': self.backtest_results
            }
            
            filename = self.get_model_filename(symbol)
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, symbol):
        """
        Load trained model from file
        """
        try:
            filename = self.get_model_filename(symbol)
            
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
            self.backtest_results = model_data.get('backtest_results', None)
            
            print(f"✅ Model loaded from {filename} (trained {days_old} days ago)")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def initialize_dynamic_model(self, symbol, initial_period="2y"):
        """
        Initialize dynamic model building system
        """
        print(f"\n{'='*60}")
        print(f"DYNAMIC MODEL INITIALIZATION: {symbol}")
        print(f"{'='*60}")
        
        self.symbol = symbol.upper()
        
        # Download initial data
        if not self.download_data(symbol, period=initial_period):
            return False
        
        # Calculate indicators
        self.calculate_advanced_indicators()
        
        # Initial model training
        print("Training initial model on 2-year historical data...")
        success = self.train_initial_model()
        
        if success:
            self.initial_training_complete = True
            self.last_update_date = self.data.index[-1].date()
            print(f"✅ Initial model trained successfully!")
            print(f"Model Version: {self.model_version}")
            print(f"Last Update: {self.last_update_date}")
            print(f"Training Data Points: {len(self.data)}")
        else:
            print("❌ Initial model training failed")
        
        return success
    
    def train_initial_model(self):
        """
        Train the initial model on historical data
        """
        try:
            X, y = self.prepare_ml_data(holding_period=10, profit_threshold=0.03)
            
            if len(X) < 100:
                print("Insufficient data for initial model training")
                return False
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Store model performance
            self.model_performance.append({
                'version': self.model_version,
                'train_score': train_score,
                'test_score': test_score,
                'data_points': len(X),
                'date': self.data.index[-1].date()
            })
            
            print(f"Initial Model Performance:")
            print(f"Training Accuracy: {train_score:.3f}")
            print(f"Test Accuracy: {test_score:.3f}")
            print(f"Data Points Used: {len(X)}")
            
            return test_score > 0.6
            
        except Exception as e:
            print(f"Error in initial model training: {str(e)}")
            return False
    
    def update_dynamic_model(self, new_data_period="3m"):
        """
        Update the model with new data
        """
        if not self.initial_training_complete:
            print("❌ Initial model not trained. Run initialize_dynamic_model first.")
            return False
        
        print(f"\n{'='*60}")
        print(f"DYNAMIC MODEL UPDATE: {self.symbol}")
        print(f"{'='*60}")
        
        # Download new data
        new_data = self.download_new_data(new_data_period)
        if new_data is None or len(new_data) < 20:
            print("❌ Insufficient new data for model update")
            return False
        
        # Combine with existing data
        self.data = pd.concat([self.data, new_data]).drop_duplicates()
        
        # Recalculate indicators
        self.calculate_advanced_indicators()
        
        # Update model
        success = self.update_model_weights()
        
        if success:
            self.model_version += 1
            self.last_update_date = self.data.index[-1].date()
            print(f"✅ Model updated successfully!")
            print(f"New Model Version: {self.model_version}")
            print(f"Last Update: {self.last_update_date}")
            print(f"Total Data Points: {len(self.data)}")
            print(f"New Data Points: {len(new_data)}")
        else:
            print("❌ Model update failed")
        
        return success
    
    def download_new_data(self, period="3m"):
        """
        Download new data since last update
        """
        try:
            ticker = yf.Ticker(self.symbol)
            
            if self.last_update_date:
                # Download data from last update date
                new_data = ticker.history(start=self.last_update_date, end=None)
            else:
                # Download recent data
                new_data = ticker.history(period=period)
            
            if new_data.empty:
                return None
            
            # Remove any overlap with existing data
            if self.data is not None:
                new_data = new_data[new_data.index > self.data.index[-1]]
            
            print(f"Downloaded {len(new_data)} new data points")
            return new_data
            
        except Exception as e:
            print(f"Error downloading new data: {str(e)}")
            return None
    
    def update_model_weights(self):
        """
        Update model weights with new data
        """
        try:
            X, y = self.prepare_ml_data(holding_period=10, profit_threshold=0.03)
            
            if len(X) < 150:  # Need more data for update
                print("Insufficient data for model update")
                return False
            
            # Split data (85% train, 15% test for updates)
            split_idx = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Retrain model with updated data
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate updated model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Store model performance
            self.model_performance.append({
                'version': self.model_version + 1,
                'train_score': train_score,
                'test_score': test_score,
                'data_points': len(X),
                'date': self.data.index[-1].date()
            })
            
            print(f"Updated Model Performance:")
            print(f"Training Accuracy: {train_score:.3f}")
            print(f"Test Accuracy: {test_score:.3f}")
            print(f"Total Data Points: {len(X)}")
            
            # Check if model improved
            if len(self.model_performance) > 1:
                prev_score = self.model_performance[-2]['test_score']
                improvement = test_score - prev_score
                print(f"Model Improvement: {improvement:+.3f}")
                
                if improvement > 0:
                    print("✅ Model accuracy improved!")
                elif improvement < -0.05:
                    print("⚠️  Model accuracy decreased significantly")
                else:
                    print("➡️  Model accuracy stable")
            
            return test_score > 0.6
            
        except Exception as e:
            print(f"Error in model update: {str(e)}")
            return False
    
    def get_model_status(self):
        """
        Get current model status and performance
        """
        if not self.initial_training_complete:
            return {
                'status': 'NOT_INITIALIZED',
                'message': 'Model not initialized. Run initialize_dynamic_model first.'
            }
        
        status = {
            'status': 'ACTIVE',
            'symbol': self.symbol,
            'model_version': self.model_version,
            'last_update': self.last_update_date,
            'total_data_points': len(self.data) if self.data is not None else 0,
            'initial_training_complete': self.initial_training_complete,
            'performance_history': self.model_performance
        }
        
        if self.model_performance:
            latest = self.model_performance[-1]
            status.update({
                'current_train_score': latest['train_score'],
                'current_test_score': latest['test_score'],
                'model_accuracy': latest['test_score']
            })
        
        return status
    
    def predict_with_dynamic_model(self, features):
        """
        Make prediction using the dynamic model
        """
        if self.model is None:
            return None, "Model not available"
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return prediction, probability
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def analyze_stock_with_dynamic_model(self, symbol, buy_price=None, holding_period=10, profit_threshold=0.03):
        """
        Analyze stock using dynamic model
        """
        print(f"\n{'='*60}")
        print(f"DYNAMIC MODEL ANALYSIS: {symbol}")
        print(f"{'='*60}")
        
        # Check if model is initialized
        if not self.initial_training_complete or self.symbol != symbol.upper():
            print("Initializing dynamic model...")
            if not self.initialize_dynamic_model(symbol):
                return None
        
        # Update model if needed (check if more than 1 month since last update)
        if self.last_update_date:
            days_since_update = (pd.Timestamp.now().date() - self.last_update_date).days
            if days_since_update > 30:
                print(f"Model last updated {days_since_update} days ago. Updating...")
                self.update_dynamic_model()
        
        # Get current prediction
        if self.model is None:
            print("❌ Model not available")
            return None
        
        # Download latest data for analysis
        if not self.download_data(symbol, period="1y"):
            return None
        
        self.calculate_advanced_indicators()
        
        # Get current features
        features = self.create_features()
        X = self.data[features].dropna()
        
        if len(X) == 0:
            print("❌ No features available for prediction")
            return None
        
        current_features = X.iloc[-1:].values
        prediction, probability = self.predict_with_dynamic_model(current_features)
        
        if prediction is None:
            print(f"❌ Prediction failed: {probability}")
            return None
        
        # Calculate technical score
        tech_score = self.calculate_technical_score(len(self.data) - 1)
        
        # Combined score (60% ML, 40% Technical)
        combined_score = 0.6 * probability + 0.4 * tech_score
        
        # Generate recommendation
        if combined_score >= 0.8:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif combined_score >= 0.6:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif combined_score >= 0.4:
            recommendation = "HOLD"
            confidence = "LOW"
        else:
            recommendation = "AVOID"
            confidence = "HIGH"
        
        # Get model status
        model_status = self.get_model_status()
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'probability': probability,
            'technical_score': tech_score,
            'combined_score': combined_score,
            'recommendation': recommendation,
            'confidence': confidence,
            'current_price': self.data.iloc[-1]['Close'],
            'model_status': model_status,
            'model_version': self.model_version
        }
    
    def download_data(self, symbol, period="2y"):
        """
        Download 2 years of stock data for comprehensive analysis
        """
        try:
            self.symbol = symbol.upper()
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for {symbol}")
                
            print(f"Successfully downloaded {len(self.data)} days of data for {symbol}")
            print(f"Data range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            
            return True
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            return False
    
    def calculate_advanced_indicators(self):
        """
        Calculate comprehensive technical indicators
        """
        if self.data is None:
            print("No data available. Please download data first.")
            return
            
        # Basic price data
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Moving Averages (Multiple timeframes)
        for period in [5, 10, 20, 50, 100, 200]:
            self.data[f'SMA_{period}'] = SMAIndicator(close=self.data['Close'], window=period).sma_indicator()
            self.data[f'EMA_{period}'] = EMAIndicator(close=self.data['Close'], window=period).ema_indicator()
        
        # MACD with multiple signals
        macd = MACD(close=self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Histogram'] = macd.macd_diff()
        self.data['MACD_Zero_Cross'] = np.where(self.data['MACD'] > 0, 1, 0)
        self.data['MACD_Signal_Cross'] = np.where(self.data['MACD'] > self.data['MACD_Signal'], 1, 0)
        
        # RSI with multiple timeframes
        for period in [7, 14, 21]:
            self.data[f'RSI_{period}'] = RSIIndicator(close=self.data['Close'], window=period).rsi()
        
        # Bollinger Bands with multiple periods
        for period in [10, 20, 50]:
            bb = BollingerBands(close=self.data['Close'], window=period)
            self.data[f'BB_Upper_{period}'] = bb.bollinger_hband()
            self.data[f'BB_Lower_{period}'] = bb.bollinger_lband()
            self.data[f'BB_Middle_{period}'] = bb.bollinger_mavg()
            self.data[f'BB_Width_{period}'] = (self.data[f'BB_Upper_{period}'] - self.data[f'BB_Lower_{period}']) / self.data[f'BB_Middle_{period}']
            self.data[f'BB_Position_{period}'] = (self.data['Close'] - self.data[f'BB_Lower_{period}']) / (self.data[f'BB_Upper_{period}'] - self.data[f'BB_Lower_{period}'])
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'])
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()
        self.data['Stoch_Overbought'] = np.where(self.data['Stoch_K'] > 80, 1, 0)
        self.data['Stoch_Oversold'] = np.where(self.data['Stoch_K'] < 20, 1, 0)
        
        # Volume indicators
        self.data['Volume_SMA_20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA_20']
        self.data['OBV'] = OnBalanceVolumeIndicator(close=self.data['Close'], volume=self.data['Volume']).on_balance_volume()
        self.data['ADI'] = AccDistIndexIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], volume=self.data['Volume']).acc_dist_index()
        
        # Advanced momentum indicators
        self.data['Williams_R'] = WilliamsRIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close']).williams_r()
        self.data['ROC'] = ROCIndicator(close=self.data['Close']).roc()
        
        # Volatility indicators
        self.data['ATR'] = AverageTrueRange(high=self.data['High'], low=self.data['Low'], close=self.data['Close']).average_true_range()
        self.data['ATR_Ratio'] = self.data['ATR'] / self.data['Close']
        
        # ADX (Trend Strength)
        adx = ADXIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'])
        self.data['ADX'] = adx.adx()
        self.data['DI_Plus'] = adx.adx_pos()
        self.data['DI_Minus'] = adx.adx_neg()
        
        # Price patterns and support/resistance
        self.data['Price_Range'] = self.data['High'] - self.data['Low']
        self.data['Price_Range_Ratio'] = self.data['Price_Range'] / self.data['Close']
        self.data['Gap_Up'] = np.where(self.data['Open'] > self.data['Close'].shift(1), 1, 0)
        self.data['Gap_Down'] = np.where(self.data['Open'] < self.data['Close'].shift(1), 1, 0)
        
        # Multiple timeframe momentum
        for period in [3, 5, 10, 20]:
            self.data[f'Momentum_{period}'] = self.data['Close'] / self.data['Close'].shift(period) - 1
            self.data[f'Volatility_{period}'] = self.data['Returns'].rolling(window=period).std()
        
        # Add Price_Momentum_5 for sell point estimation
        self.data['Price_Momentum_5'] = self.data['Close'] / self.data['Close'].shift(5) - 1
        
        # Trend strength indicators
        self.data['Trend_Strength'] = abs(self.data['SMA_20'] - self.data['SMA_50']) / self.data['SMA_50']
        self.data['Price_vs_SMA20'] = (self.data['Close'] - self.data['SMA_20']) / self.data['SMA_20']
        self.data['Price_vs_SMA50'] = (self.data['Close'] - self.data['SMA_50']) / self.data['SMA_50']
        
        print("Advanced technical indicators calculated successfully!")
    
    def create_features(self):
        """
        Create comprehensive feature set for ML model
        """
        features = []
        
        # Price-based features
        price_features = ['Returns', 'Log_Returns', 'Price_Range_Ratio', 'Price_vs_SMA20', 'Price_vs_SMA50']
        features.extend(price_features)
        
        # Moving average features
        ma_features = [col for col in self.data.columns if 'SMA_' in col or 'EMA_' in col]
        features.extend(ma_features)
        
        # MACD features
        macd_features = ['MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Zero_Cross', 'MACD_Signal_Cross']
        features.extend(macd_features)
        
        # RSI features
        rsi_features = [col for col in self.data.columns if 'RSI_' in col]
        features.extend(rsi_features)
        
        # Bollinger Bands features
        bb_features = [col for col in self.data.columns if 'BB_' in col]
        features.extend(bb_features)
        
        # Volume features
        volume_features = ['Volume_Ratio', 'OBV', 'ADI']
        features.extend(volume_features)
        
        # Momentum features
        momentum_features = ['Williams_R', 'ROC', 'Stoch_K', 'Stoch_D', 'Stoch_Overbought', 'Stoch_Oversold']
        features.extend(momentum_features)
        
        # Volatility features
        volatility_features = ['ATR', 'ATR_Ratio'] + [col for col in self.data.columns if 'Volatility_' in col]
        features.extend(volatility_features)
        
        # Trend features
        trend_features = ['ADX', 'DI_Plus', 'DI_Minus', 'Trend_Strength']
        features.extend(trend_features)
        
        return features
    
    def create_target_variable(self, holding_period=10, profit_threshold=0.03):
        """
        Create target variable for ML model (1 if profitable within holding period)
        """
        future_returns = self.data['Close'].shift(-holding_period) / self.data['Close'] - 1
        self.data['Target'] = np.where(future_returns > profit_threshold, 1, 0)
        return 'Target'
    
    def prepare_ml_data(self, holding_period=10, profit_threshold=0.03):
        """
        Prepare data for machine learning model
        """
        features = self.create_features()
        target = self.create_target_variable(holding_period, profit_threshold)
        
        # Remove rows with NaN values
        ml_data = self.data[features + [target]].dropna()
        
        X = ml_data[features]
        y = ml_data[target]
        
        return X, y
    
    def train_ml_model(self, holding_period=10, profit_threshold=0.03):
        """
        Train machine learning model for prediction
        """
        X, y = self.prepare_ml_data(holding_period, profit_threshold)
        
        if len(X) < 100:
            print("Insufficient data for ML model training")
            return False
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Store model information
        self.model_info = {
            'train_score': train_score,
            'test_score': test_score,
            'features_count': len(X.columns),
            'data_points': len(X),
            'last_trained': datetime.now().isoformat(),
            'holding_period': holding_period,
            'profit_threshold': profit_threshold
        }
        
        print(f"ML Model Training Results:")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        
        return test_score > 0.6  # Return True if model is reasonably accurate
    
    def calculate_technical_score(self, index):
        """
        Calculate technical analysis score (0-1)
        """
        if index < 50:
            return 0.5
        
        current = self.data.iloc[index]
        
        score = 0.5  # Base score
        
        # RSI analysis
        if current['RSI_14'] < 30:
            score += 0.1  # Oversold
        elif current['RSI_14'] > 70:
            score -= 0.1  # Overbought
        elif 40 <= current['RSI_14'] <= 60:
            score += 0.05  # Neutral zone
        
        # MACD analysis
        if current['MACD'] > current['MACD_Signal']:
            score += 0.1
        else:
            score -= 0.1
        
        # Moving average analysis
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            score += 0.1
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            score -= 0.1
        
        # Bollinger Bands analysis
        if current['BB_Position_20'] < 0.2:
            score += 0.1  # Near lower band
        elif current['BB_Position_20'] > 0.8:
            score -= 0.1  # Near upper band
        
        # Volume analysis
        if current['Volume_Ratio'] > 1.5:
            score += 0.05
        elif current['Volume_Ratio'] < 0.5:
            score -= 0.05
        
        # Stochastic analysis
        if current['Stoch_Oversold'] == 1:
            score += 0.05
        elif current['Stoch_Overbought'] == 1:
            score -= 0.05
        
        return max(0, min(1, score))
    
    def backtest_strategy(self, holding_period=10, profit_threshold=0.03, stop_loss=0.02):
        """
        Comprehensive backtesting of the strategy
        """
        if self.data is None:
            print("No data available for backtesting")
            return None
        
        # Calculate all indicators
        self.calculate_advanced_indicators()
        
        # Train ML model
        ml_success = self.train_ml_model(holding_period, profit_threshold)
        
        # Initialize backtest results
        trades = []
        current_position = None
        entry_price = 0
        entry_date = None
        
        # Get features for prediction
        features = self.create_features()
        X = self.data[features].dropna()
        
        if len(X) < 50:
            print("Insufficient data for backtesting")
            return None
        
        # Start from day 50 to allow for indicator calculation
        for i in range(50, len(self.data) - holding_period):
            current_date = self.data.index[i]
            current_price = self.data.iloc[i]['Close']
            
            # Get current features
            current_features = X.iloc[i-50:i].iloc[-1:].values
            
            if len(current_features) == 0:
                continue
                
            # Scale features
            current_features_scaled = self.scaler.transform(current_features)
            
            # Get ML prediction
            ml_prediction = self.model.predict_proba(current_features_scaled)[0][1] if self.model else 0.5
            
            # Calculate technical score
            tech_score = self.calculate_technical_score(i)
            
            # Combined score (70% ML, 30% Technical)
            combined_score = 0.7 * ml_prediction + 0.3 * tech_score
            
            # Entry conditions
            if current_position is None and combined_score > 0.7:
                current_position = 'LONG'
                entry_price = current_price
                entry_date = current_date
                
            # Exit conditions
            elif current_position == 'LONG':
                exit_price = current_price
                exit_date = current_date
                returns = (exit_price - entry_price) / entry_price
                
                # Check if we should exit
                should_exit = False
                exit_reason = ""
                
                # Profit target hit
                if returns >= profit_threshold:
                    should_exit = True
                    exit_reason = "Profit Target"
                
                # Stop loss hit
                elif returns <= -stop_loss:
                    should_exit = True
                    exit_reason = "Stop Loss"
                
                # Technical indicators turn bearish
                elif combined_score < 0.3:
                    should_exit = True
                    exit_reason = "Technical Exit"
                
                # Holding period exceeded
                elif (current_date - entry_date).days >= holding_period:
                    should_exit = True
                    exit_reason = "Time Exit"
                
                if should_exit:
                    trades.append({
                        'Entry_Date': entry_date,
                        'Exit_Date': exit_date,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Returns': returns,
                        'Exit_Reason': exit_reason,
                        'Success': returns > 0
                    })
                    
                    current_position = None
                    entry_price = 0
                    entry_date = None
        
        # Calculate performance metrics
        if trades:
            total_trades = len(trades)
            successful_trades = sum(1 for trade in trades if trade['Success'])
            success_rate = successful_trades / total_trades
            avg_return = np.mean([trade['Returns'] for trade in trades])
            total_return = np.prod([1 + trade['Returns'] for trade in trades]) - 1
            
            self.backtest_results = {
                'Total_Trades': total_trades,
                'Successful_Trades': successful_trades,
                'Success_Rate': success_rate,
                'Average_Return': avg_return,
                'Total_Return': total_return,
                'Trades': trades
            }
            
            print(f"\n{'='*60}")
            print(f"BACKTEST RESULTS FOR {self.symbol}")
            print(f"{'='*60}")
            print(f"Total Trades: {total_trades}")
            print(f"Successful Trades: {successful_trades}")
            print(f"Success Rate: {success_rate:.2%}")
            print(f"Average Return per Trade: {avg_return:.2%}")
            print(f"Total Return: {total_return:.2%}")
            print(f"ML Model Used: {ml_success}")
            
            return self.backtest_results
        
        return None
    
    def estimate_highest_price_10_days(self, symbol, current_price):
        """
        Estimate the highest price in the next 10 days using historical volatility and momentum
        """
        if self.data is None or len(self.data) < 50:
            return current_price * 1.03  # Default 3% estimate
        
        # Calculate historical volatility
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['Close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods
        recent_highs = []
        for i in range(max(0, len(self.data) - 20), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_high = self.data['High'].iloc[i:i+10].max()
                period_start = self.data['Close'].iloc[i]
                recent_highs.append(period_high / period_start - 1)
        
        avg_high_return = np.mean(recent_highs) if recent_highs else 0.05
        
        # Calculate technical indicators for price direction
        current = self.data.iloc[-1]
        
        # RSI-based adjustment
        rsi_adjustment = 1.0
        if 'RSI_14' in current:
            if current['RSI_14'] < 30:  # Oversold
                rsi_adjustment = 1.1  # Higher potential
            elif current['RSI_14'] > 70:  # Overbought
                rsi_adjustment = 0.9  # Lower potential
        
        # MACD-based adjustment
        macd_adjustment = 1.0
        if 'MACD' in current and 'MACD_Signal' in current:
            if current['MACD'] > current['MACD_Signal']:
                macd_adjustment = 1.05  # Bullish
            else:
                macd_adjustment = 0.95  # Bearish
        
        # Volume-based adjustment
        volume_adjustment = 1.0
        if 'Volume_Ratio' in current:
            if current['Volume_Ratio'] > 1.5:
                volume_adjustment = 1.05  # High volume supports price
            elif current['Volume_Ratio'] < 0.5:
                volume_adjustment = 0.95  # Low volume
        
        # Calculate estimated highest price
        base_estimate = current_price * (1 + avg_high_return)
        
        # Apply adjustments
        adjusted_estimate = base_estimate * rsi_adjustment * macd_adjustment * volume_adjustment
        
        # Ensure reasonable bounds (between 1% and 15% increase)
        min_increase = current_price * 1.01
        max_increase = current_price * 1.15
        
        estimated_high = max(min_increase, min(adjusted_estimate, max_increase))
        
        return estimated_high
    
    def estimate_lowest_price_10_days(self, symbol, current_price):
        """
        Estimate the lowest price in the next 10 days using historical volatility and momentum
        """
        if self.data is None or len(self.data) < 50:
            return current_price * 0.97  # Default 3% decline estimate
        
        # Calculate historical volatility
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['Close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods
        recent_lows = []
        for i in range(max(0, len(self.data) - 20), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_low = self.data['Low'].iloc[i:i+10].min()
                period_start = self.data['Close'].iloc[i]
                recent_lows.append(period_low / period_start - 1)
        
        avg_low_return = np.mean(recent_lows) if recent_lows else -0.03
        
        # Calculate technical indicators for price direction
        current = self.data.iloc[-1]
        
        # RSI-based adjustment
        rsi_adjustment = 1.0
        if 'RSI_14' in current:
            if current['RSI_14'] > 70:  # Overbought
                rsi_adjustment = 0.95  # Higher potential for decline
            elif current['RSI_14'] < 30:  # Oversold
                rsi_adjustment = 1.05  # Lower potential for decline
        
        # MACD-based adjustment
        macd_adjustment = 1.0
        if 'MACD' in current and 'MACD_Signal' in current:
            if current['MACD'] < current['MACD_Signal']:
                macd_adjustment = 0.95  # Bearish
            else:
                macd_adjustment = 1.05  # Bullish
        
        # Volume-based adjustment
        volume_adjustment = 1.0
        if 'Volume_Ratio' in current:
            if current['Volume_Ratio'] > 1.5:
                volume_adjustment = 0.95  # High volume often precedes decline
            elif current['Volume_Ratio'] < 0.5:
                volume_adjustment = 1.02  # Low volume may indicate stability
        
        # Calculate estimated lowest price
        base_estimate = current_price * (1 + avg_low_return)
        
        # Apply adjustments
        adjusted_estimate = base_estimate * rsi_adjustment * macd_adjustment * volume_adjustment
        
        # Ensure reasonable bounds (between -8% and +2%)
        min_decline = current_price * 0.92
        max_decline = current_price * 1.02
        
        estimated_low = max(min_decline, min(adjusted_estimate, max_decline))
        
        return estimated_low
    
    def analyze_stock_with_backtest(self, symbol, holding_period=10, profit_threshold=0.03):
        """
        Complete analysis with backtesting and model persistence
        """
        print(f"\n{'='*60}")
        print(f"ENHANCED ANALYSIS WITH BACKTESTING: {symbol}")
        print(f"{'='*60}")
        
        # Try to load existing model first
        model_loaded = self.load_model(symbol)
        
        if not model_loaded:
            # Download data and train new model
            if not self.download_data(symbol):
                return None
            
            # Run backtest
            results = self.backtest_strategy(holding_period, profit_threshold)
            
            # Save the model if backtest was successful
            if results and results['Success_Rate'] >= 0.4:
                self.save_model(symbol)
        else:
            # Model loaded successfully, just download latest data for analysis
            if not self.download_data(symbol):
                return None
            
            # Use existing backtest results if available
            results = self.backtest_results
            if results is None:
                results = self.backtest_strategy(holding_period, profit_threshold)
        
        # Calculate estimated highest and lowest prices
        current_price = self.data.iloc[-1]['Close']
        estimated_high = self.estimate_highest_price_10_days(symbol, current_price)
        estimated_low = self.estimate_lowest_price_10_days(symbol, current_price)
        potential_gain = (estimated_high - current_price) / current_price
        potential_loss = (estimated_low - current_price) / current_price
        
        if results and results['Success_Rate'] >= 0.8:
            print(f"✅ EXCELLENT: Success rate {results['Success_Rate']:.2%} >= 80%")
            recommendation = "STRONG BUY"
        elif results and results['Success_Rate'] >= 0.6:
            print(f"✅ GOOD: Success rate {results['Success_Rate']:.2%} >= 60%")
            recommendation = "BUY"
        elif results and results['Success_Rate'] >= 0.4:
            print(f"⚠️  MODERATE: Success rate {results['Success_Rate']:.2%} >= 40%")
            recommendation = "HOLD"
        else:
            print(f"❌ POOR: Success rate {results['Success_Rate']:.2%} < 40%")
            recommendation = "AVOID"
        
        return {
            'symbol': symbol,
            'backtest_results': results,
            'recommendation': recommendation,
            'current_price': current_price,
            'estimated_high_10d': estimated_high,
            'estimated_low_10d': estimated_low,
            'potential_gain_10d': potential_gain,
            'potential_loss_10d': potential_loss,
            'model_loaded': model_loaded
        }
    
    def compare_stocks_enhanced(self, symbols, holding_period=10, profit_threshold=0.03):
        """
        Compare multiple stocks with enhanced backtesting
        """
        print(f"\n{'='*70}")
        print("ENHANCED STOCK COMPARISON WITH BACKTESTING")
        print(f"Target: {profit_threshold:.1%} profit in {holding_period} days")
        print(f"{'='*70}")
        
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_stock_with_backtest(symbol, holding_period, profit_threshold)
                if result and result['backtest_results']:
                    results.append(result)
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        if not results:
            print("No valid results generated")
            return None
        
        # Sort by success rate
        results.sort(key=lambda x: x['backtest_results']['Success_Rate'], reverse=True)
        
        # Display results
        print(f"\n{'='*70}")
        print("ENHANCED RECOMMENDATION SUMMARY")
        print(f"{'='*70}")
        
        for i, result in enumerate(results, 1):
            bt = result['backtest_results']
            print(f"\n{i}. {result['symbol']}")
            print(f"   Current Price: ${result['current_price']:.2f}")
            print(f"   Estimated High (10d): ${result['estimated_high_10d']:.2f}")
            print(f"   Estimated Low (10d): ${result['estimated_low_10d']:.2f}")
            print(f"   Potential Gain: {result['potential_gain_10d']:.2%}")
            print(f"   Potential Loss: {result['potential_loss_10d']:.2%}")
            print(f"   Success Rate: {bt['Success_Rate']:.2%}")
            print(f"   Total Trades: {bt['Total_Trades']}")
            print(f"   Avg Return: {bt['Average_Return']:.2%}")
            print(f"   Total Return: {bt['Total_Return']:.2%}")
            print(f"   Recommendation: {result['recommendation']}")
        
        # Best recommendation
        best = results[0]
        bt = best['backtest_results']
        
        print(f"\n{'='*70}")
        print(f"🎯 BEST RECOMMENDATION: {best['symbol']}")
        print(f"{'='*70}")
        print(f"Current Price: ${best['current_price']:.2f}")
        print(f"Estimated High (10d): ${best['estimated_high_10d']:.2f}")
        print(f"Estimated Low (10d): ${best['estimated_low_10d']:.2f}")
        print(f"Potential Gain: {best['potential_gain_10d']:.2%}")
        print(f"Potential Loss: {best['potential_loss_10d']:.2%}")
        print(f"Success Rate: {bt['Success_Rate']:.2%}")
        print(f"Total Trades: {bt['Total_Trades']}")
        print(f"Average Return: {bt['Average_Return']:.2%}")
        print(f"Total Return: {bt['Total_Return']:.2%}")
        print(f"Recommendation: {best['recommendation']}")
        
        if bt['Success_Rate'] >= 0.8:
            print(f"✅ EXCELLENT CHOICE - Success rate over 80%!")
        elif bt['Success_Rate'] >= 0.6:
            print(f"✅ GOOD CHOICE - Success rate over 60%")
        else:
            print(f"⚠️  CAUTION - Success rate below 60%")
        
        return best
    
    def estimate_sell_point(self, symbol, buy_price, buy_date=None, holding_period=10):
        """
        Estimate optimal sell point and price using trained model
        """
        print(f"\n{'='*60}")
        print(f"SELL POINT ESTIMATION: {symbol}")
        print(f"{'='*60}")
        print(f"Buy Price: ${buy_price:.2f}")
        if buy_date:
            print(f"Buy Date: {buy_date}")
        print(f"Target Holding Period: {holding_period} days")
        
        # Download latest data
        if not self.download_data(symbol):
            return None
        
        # Calculate indicators
        self.calculate_advanced_indicators()
        
        # Train ML model for sell prediction
        ml_success = self.train_ml_model(holding_period=holding_period, profit_threshold=0.02)
        
        if not ml_success:
            print("ML model training failed. Using technical analysis only.")
        
        # Get current market position
        current_price = self.data.iloc[-1]['Close']
        current_return = (current_price - buy_price) / buy_price
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Current Return: {current_return:.2%}")
        
        # Analyze sell signals
        sell_analysis = self.analyze_sell_signals(buy_price, holding_period)
        
        # Generate sell recommendation
        recommendation = self.generate_sell_recommendation(sell_analysis, current_return, buy_price)
        
        return {
            'symbol': symbol,
            'buy_price': buy_price,
            'current_price': current_price,
            'current_return': current_return,
            'sell_analysis': sell_analysis,
            'recommendation': recommendation,
            'ml_model_used': ml_success
        }
    
    def analyze_sell_signals(self, buy_price, holding_period=10):
        """
        Analyze various sell signals using technical indicators and ML
        """
        if self.data is None or len(self.data) < 50:
            return None
        
        current = self.data.iloc[-1]
        analysis = {
            'technical_score': 0,
            'ml_score': 0,
            'sell_signals': [],
            'hold_signals': [],
            'risk_factors': [],
            'profit_potential': 0,
            'stop_loss_triggered': False
        }
        
        # Calculate technical sell score
        tech_score = self.calculate_sell_technical_score()
        analysis['technical_score'] = tech_score
        
        # Get ML prediction if model is available
        if self.model:
            features = self.create_features()
            X = self.data[features].dropna()
            
            if len(X) > 0:
                current_features = X.iloc[-1:].values
                current_features_scaled = self.scaler.transform(current_features)
                ml_prediction = self.model.predict_proba(current_features_scaled)[0][1]
                analysis['ml_score'] = ml_prediction
                
                if ml_prediction < 0.3:
                    analysis['sell_signals'].append("ML model predicts low profit probability")
                elif ml_prediction > 0.7:
                    analysis['hold_signals'].append("ML model predicts high profit probability")
        
        # Current return analysis
        current_return = (current['Close'] - buy_price) / buy_price
        
        # Profit target analysis
        if current_return >= 0.03:  # 3% profit target
            analysis['sell_signals'].append("Profit target reached (3%+)")
            analysis['profit_potential'] = current_return
        
        # Stop loss analysis
        if current_return <= -0.02:  # 2% stop loss
            analysis['sell_signals'].append("Stop loss triggered (-2%+)")
            analysis['stop_loss_triggered'] = True
            analysis['risk_factors'].append("Significant loss position")
        
        # Technical indicator analysis
        if current['RSI_14'] > 70:
            analysis['sell_signals'].append("RSI overbought (>70)")
        elif current['RSI_14'] < 30:
            analysis['hold_signals'].append("RSI oversold (<30)")
        
        if current['MACD'] < current['MACD_Signal']:
            analysis['sell_signals'].append("MACD bearish crossover")
        else:
            analysis['hold_signals'].append("MACD bullish")
        
        if current['Close'] < current['SMA_20']:
            analysis['sell_signals'].append("Price below 20-day SMA")
        else:
            analysis['hold_signals'].append("Price above 20-day SMA")
        
        # Volume analysis
        if current['Volume_Ratio'] > 2.0:
            analysis['sell_signals'].append("High volume - potential distribution")
        elif current['Volume_Ratio'] < 0.5:
            analysis['hold_signals'].append("Low volume - accumulation possible")
        
        # Bollinger Bands analysis
        if current['BB_Position_20'] > 0.8:
            analysis['sell_signals'].append("Price near upper Bollinger Band")
        elif current['BB_Position_20'] < 0.2:
            analysis['hold_signals'].append("Price near lower Bollinger Band")
        
        # Momentum analysis
        if current['Price_Momentum_5'] < -0.03:
            analysis['sell_signals'].append("Negative 5-day momentum")
        elif current['Price_Momentum_5'] > 0.03:
            analysis['hold_signals'].append("Positive 5-day momentum")
        
        return analysis
    
    def calculate_sell_technical_score(self):
        """
        Calculate technical score for sell decision (0-100)
        """
        if self.data is None or len(self.data) < 50:
            return 50
        
        current = self.data.iloc[-1]
        score = 50  # Neutral base score
        
        # RSI analysis
        if current['RSI_14'] > 70:
            score += 20  # Strong sell signal
        elif current['RSI_14'] > 60:
            score += 10  # Moderate sell signal
        elif current['RSI_14'] < 30:
            score -= 20  # Strong buy signal (hold)
        elif current['RSI_14'] < 40:
            score -= 10  # Moderate buy signal (hold)
        
        # MACD analysis
        if current['MACD'] < current['MACD_Signal']:
            score += 15  # Bearish MACD
        else:
            score -= 15  # Bullish MACD
        
        # Moving average analysis
        if current['Close'] < current['SMA_20']:
            score += 10  # Below short-term MA
        else:
            score -= 10  # Above short-term MA
        
        if current['Close'] < current['SMA_50']:
            score += 15  # Below long-term MA
        else:
            score -= 15  # Above long-term MA
        
        # Volume analysis
        if current['Volume_Ratio'] > 1.5:
            score += 5  # High volume
        elif current['Volume_Ratio'] < 0.5:
            score -= 5  # Low volume
        
        # Bollinger Bands analysis
        if current['BB_Position_20'] > 0.8:
            score += 15  # Near upper band
        elif current['BB_Position_20'] < 0.2:
            score -= 15  # Near lower band
        
        # Momentum analysis
        if current['Price_Momentum_5'] < -0.02:
            score += 10  # Negative momentum
        elif current['Price_Momentum_5'] > 0.02:
            score -= 10  # Positive momentum
        
        return max(0, min(100, score))
    
    def generate_sell_recommendation(self, sell_analysis, current_return, buy_price):
        """
        Generate comprehensive sell recommendation
        """
        if not sell_analysis:
            return None
        
        # Calculate combined score
        tech_score = sell_analysis['technical_score']
        ml_score = sell_analysis['ml_score']
        
        # Weighted score (60% technical, 40% ML if available)
        if ml_score > 0:
            combined_score = 0.6 * tech_score + 0.4 * (1 - ml_score) * 100
        else:
            combined_score = tech_score
        
        # Determine sell action
        if combined_score >= 75:
            action = "SELL NOW"
            urgency = "HIGH"
            reasoning = "Strong technical and ML signals indicate selling"
        elif combined_score >= 60:
            action = "SELL SOON"
            urgency = "MEDIUM"
            reasoning = "Moderate sell signals detected"
        elif combined_score >= 40:
            action = "HOLD"
            urgency = "LOW"
            reasoning = "Mixed signals - consider holding"
        else:
            action = "HOLD/ADD"
            urgency = "LOW"
            reasoning = "Strong buy signals - consider adding to position"
        
        # Calculate target sell price
        if current_return >= 0.03:
            # Already at profit target, suggest selling at current price
            target_price = buy_price * 1.03  # 3% profit
        elif current_return >= 0.01:
            # Small profit, suggest waiting for 3% target
            target_price = buy_price * 1.03
        elif current_return <= -0.02:
            # At stop loss, suggest immediate selling
            target_price = current_return * buy_price
        else:
            # Calculate based on technical analysis
            if combined_score >= 60:
                target_price = buy_price * (1 + max(current_return, 0.01))
            else:
                target_price = buy_price * 1.03
        
        # Risk assessment
        risk_level = "LOW"
        if sell_analysis['stop_loss_triggered']:
            risk_level = "HIGH"
        elif len(sell_analysis['risk_factors']) > 2:
            risk_level = "MEDIUM"
        
        return {
            'action': action,
            'urgency': urgency,
            'reasoning': reasoning,
            'combined_score': combined_score,
            'technical_score': tech_score,
            'ml_score': ml_score,
            'target_price': target_price,
            'risk_level': risk_level,
            'sell_signals': sell_analysis['sell_signals'],
            'hold_signals': sell_analysis['hold_signals'],
            'risk_factors': sell_analysis['risk_factors'],
            'profit_potential': sell_analysis['profit_potential']
        } 