import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from datetime import datetime

class DynamicModelAnalyzer:
    def __init__(self):
        self.data = None
        self.symbol = None
        self.model = None
        self.scaler = StandardScaler()
        self.model_version = 0
        self.last_update_date = None
        self.initial_training_complete = False
        self.model_performance = []
        
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
        
    def initialize_model(self, symbol, period="2y", market='A'):
        """
        Initialize dynamic model with 2-year data
        """
        print(f"\n{'='*50}")
        print(f"DYNAMIC MODEL INITIALIZATION: {symbol}")
        print(f"{'='*50}")
        
        self.symbol = symbol.upper()
        
        # Download initial data
        if not self.download_data(symbol, period, market):
            return False
        
        # Calculate indicators
        self.calculate_indicators()
        
        # Train initial model
        success = self.train_initial_model()
        
        if success:
            self.initial_training_complete = True
            self.last_update_date = self.data.index[-1].date()
            print(f"✅ Initial model trained successfully!")
            print(f"Model Version: {self.model_version}")
            print(f"Data Points: {len(self.data)}")
        
        return success
    
    def download_data(self, symbol, period="2y", market='A'):
        """
        Download stock data
        """
        try:
            # Format symbol for Chinese stocks
            if market == 'A' or market == 'H':
                formatted_symbol = self.get_chinese_stock_symbol(symbol, market)
                print(f"Downloading data for {symbol} ({market}-shares) as {formatted_symbol}")
            else:
                formatted_symbol = symbol
                print(f"Downloading data for {symbol}")
            
            ticker = yf.Ticker(formatted_symbol)
            self.data = ticker.history(period=period)
            
            if self.data.empty:
                print(f"No data found for {formatted_symbol}")
                return False
            
            print(f"Downloaded {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return False
    
    def calculate_indicators(self):
        """
        Calculate technical indicators
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
        
        # Momentum
        self.data['Price_Momentum_5'] = self.data['Close'] / self.data['Close'].shift(5) - 1
        self.data['Price_Momentum_10'] = self.data['Close'] / self.data['Close'].shift(10) - 1
        
        # Volatility
        self.data['Volatility_20'] = self.data['Returns'].rolling(window=20).std()
        
        print("Technical indicators calculated successfully!")
    
    def create_features(self):
        """
        Create feature set for ML model
        """
        features = ['Returns', 'Price_Momentum_5', 'Price_Momentum_10', 
                   'SMA_20', 'SMA_50', 'Volume_Ratio', 'Volatility_20']
        return features
    
    def create_target_variable(self, holding_period=10, profit_threshold=0.03):
        """
        Create target variable for ML model
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
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Store performance
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
            
            return test_score > 0.55  # Lower threshold for Chinese stocks
            
        except Exception as e:
            print(f"Error in initial model training: {str(e)}")
            return False
    
    def update_model(self, new_data_period="3m"):
        """
        Update the model with new data
        """
        if not self.initial_training_complete:
            print("❌ Initial model not trained. Run initialize_model first.")
            return False
        
        print(f"\n{'='*50}")
        print(f"MODEL UPDATE: {self.symbol}")
        print(f"{'='*50}")
        
        # Download new data
        new_data = self.download_new_data(new_data_period)
        if new_data is None or len(new_data) < 20:
            print("❌ Insufficient new data for model update")
            return False
        
        # Combine with existing data
        self.data = pd.concat([self.data, new_data]).drop_duplicates()
        
        # Recalculate indicators
        self.calculate_indicators()
        
        # Update model
        success = self.update_model_weights()
        
        if success:
            self.model_version += 1
            self.last_update_date = self.data.index[-1].date()
            print(f"✅ Model updated successfully!")
            print(f"New Model Version: {self.model_version}")
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
                new_data = ticker.history(start=self.last_update_date, end=None)
            else:
                new_data = ticker.history(period=period)
            
            if new_data.empty:
                return None
            
            # Remove overlap with existing data
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
            
            if len(X) < 150:
                print("Insufficient data for model update")
                return False
            
            # Split data (85% train, 15% test for updates)
            split_idx = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Retrain model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate updated model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Store performance
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
            
            # Check improvement
            if len(self.model_performance) > 1:
                prev_score = self.model_performance[-2]['test_score']
                improvement = test_score - prev_score
                print(f"Model Improvement: {improvement:+.3f}")
                
                if improvement > 0:
                    print("✅ Model accuracy improved!")
                else:
                    print("➡️  Model accuracy stable")
            
            return test_score > 0.6
            
        except Exception as e:
            print(f"Error in model update: {str(e)}")
            return False
    
    def estimate_highest_price_10_days(self, symbol, current_price):
        """
        Estimate the highest price in the next 10 days using dynamic model insights
        """
        if self.data is None or len(self.data) < 20:
            return current_price * 1.03  # Default 3% estimate
        
        # Calculate historical volatility
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate recent momentum
        recent_momentum = self.data['Close'].iloc[-5:].pct_change().mean()
        
        # Calculate price range in recent periods
        recent_highs = []
        for i in range(max(0, len(self.data) - 15), len(self.data) - 1):
            if i + 10 < len(self.data):
                period_high = self.data['High'].iloc[i:i+10].max()
                period_start = self.data['Close'].iloc[i]
                recent_highs.append(period_high / period_start - 1)
        
        avg_high_return = np.mean(recent_highs) if recent_highs else 0.05
        
        # Calculate technical indicators for price direction
        current = self.data.iloc[-1]
        
        # Momentum-based adjustment
        momentum_adjustment = 1.0
        if current['Price_Momentum_5'] > 0.05:
            momentum_adjustment = 1.1  # Strong upward momentum
        elif current['Price_Momentum_5'] > 0.02:
            momentum_adjustment = 1.05  # Moderate upward momentum
        elif current['Price_Momentum_5'] < -0.05:
            momentum_adjustment = 0.9  # Strong downward momentum
        elif current['Price_Momentum_5'] < -0.02:
            momentum_adjustment = 0.95  # Moderate downward momentum
        
        # Moving average-based adjustment
        ma_adjustment = 1.0
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            ma_adjustment = 1.05  # Strong uptrend
        elif current['Close'] > current['SMA_20']:
            ma_adjustment = 1.02  # Moderate uptrend
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            ma_adjustment = 0.95  # Strong downtrend
        elif current['Close'] < current['SMA_20']:
            ma_adjustment = 0.98  # Moderate downtrend
        
        # Volume-based adjustment
        volume_adjustment = 1.0
        if current['Volume_Ratio'] > 2.0:
            volume_adjustment = 1.08  # Very high volume
        elif current['Volume_Ratio'] > 1.5:
            volume_adjustment = 1.05  # High volume
        elif current['Volume_Ratio'] < 0.5:
            volume_adjustment = 0.95  # Low volume
        
        # Volatility-based adjustment
        volatility_adjustment = 1.0
        avg_volatility = self.data['Volatility_20'].mean()
        if current['Volatility_20'] > avg_volatility * 1.5:
            volatility_adjustment = 1.05  # High volatility = higher potential
        elif current['Volatility_20'] < avg_volatility * 0.5:
            volatility_adjustment = 0.98  # Low volatility = lower potential
        
        # Calculate estimated highest price
        base_estimate = current_price * (1 + avg_high_return)
        
        # Apply adjustments
        adjusted_estimate = base_estimate * momentum_adjustment * ma_adjustment * volume_adjustment * volatility_adjustment
        
        # Ensure reasonable bounds (between 1% and 15% increase)
        min_increase = current_price * 1.01
        max_increase = current_price * 1.15
        
        estimated_high = max(min_increase, min(adjusted_estimate, max_increase))
        
        return estimated_high
    
    def analyze_stock(self, symbol, buy_price=None, market='A'):
        """
        Analyze stock using dynamic model
        """
        print(f"\n{'='*50}")
        print(f"DYNAMIC MODEL ANALYSIS: {symbol}")
        print(f"{'='*50}")
        
        # Initialize model if needed
        if not self.initial_training_complete or self.symbol != symbol.upper():
            print("Initializing dynamic model...")
            if not self.initialize_model(symbol, market=market):
                return None
        
        # Update model if needed (more than 30 days since last update)
        if self.last_update_date:
            days_since_update = (pd.Timestamp.now().date() - self.last_update_date).days
            if days_since_update > 30:
                print(f"Model last updated {days_since_update} days ago. Updating...")
                self.update_model()
        
        # Get current prediction
        if self.model is None:
            print("❌ Model not available")
            return None
        
        # Download latest data
        if not self.download_data(symbol, period="1y", market=market):
            return None
        
        self.calculate_indicators()
        
        # Get current features
        features = self.create_features()
        X = self.data[features].dropna()
        
        if len(X) == 0:
            print("❌ No features available for prediction")
            return None
        
        current_features = X.iloc[-1:].values
        prediction, probability = self.predict(current_features)
        
        if prediction is None:
            print(f"❌ Prediction failed")
            return None
        
        # Calculate technical score
        tech_score = self.calculate_technical_score()
        
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
        
        # Calculate estimated highest price
        current_price = self.data.iloc[-1]['Close']
        estimated_high = self.estimate_highest_price_10_days(symbol, current_price)
        potential_gain = (estimated_high - current_price) / current_price
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'probability': probability,
            'technical_score': tech_score,
            'combined_score': combined_score,
            'recommendation': recommendation,
            'confidence': confidence,
            'current_price': current_price,
            'estimated_high_10d': estimated_high,
            'potential_gain_10d': potential_gain,
            'model_version': self.model_version,
            'data_points': len(self.data)
        }
    
    def predict(self, features):
        """
        Make prediction using the dynamic model
        """
        if self.model is None:
            return None, None
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            return prediction, probability
            
        except Exception as e:
            return None, None
    
    def calculate_technical_score(self):
        """
        Calculate technical analysis score (0-1)
        """
        if self.data is None or len(self.data) < 50:
            return 0.5
        
        current = self.data.iloc[-1]
        score = 0.5  # Base score
        
        # Momentum analysis
        if current['Price_Momentum_5'] > 0.05:
            score += 0.1
        elif current['Price_Momentum_5'] < -0.05:
            score -= 0.1
        
        # Moving average analysis
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            score += 0.1
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            score -= 0.1
        
        # Volume analysis
        if current['Volume_Ratio'] > 1.5:
            score += 0.05
        elif current['Volume_Ratio'] < 0.5:
            score -= 0.05
        
        return max(0, min(1, score))
    
    def list_saved_models(self):
        """
        List all saved models
        """
        saved_models = []
        
        # Check Chinese models
        chinese_dir = "chinese_models"
        if os.path.exists(chinese_dir):
            for file in os.listdir(chinese_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('_model.pkl', '')
                    saved_models.append(f"{model_name} (Chinese)")
        
        # Check US models
        us_dir = "us_models"
        if os.path.exists(us_dir):
            for file in os.listdir(us_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('_model.pkl', '')
                    saved_models.append(f"{model_name} (US)")
        
        return saved_models
    
    def get_model_status(self, model_name=None):
        """
        Get current model status or status of a specific model
        """
        if model_name:
            # Get status of a specific saved model
            try:
                # Try Chinese model first
                chinese_file = f"chinese_models/{model_name}_model.pkl"
                if os.path.exists(chinese_file):
                    with open(chinese_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    last_trained = model_data['last_trained']
                    days_old = (datetime.now() - last_trained).days
                    
                    return {
                        'status': 'ACTIVE' if days_old <= 30 else 'EXPIRED',
                        'symbol': model_data['symbol'],
                        'market': model_data.get('market', 'A'),
                        'model_version': 0,
                        'last_update': last_trained.date(),
                        'data_points': model_data['data_points'],
                        'current_accuracy': model_data.get('model_info', {}).get('test_score', 'N/A'),
                        'model_file': chinese_file,
                        'days_old': days_old
                    }
                
                # Try US model
                us_file = f"us_models/{model_name}_model.pkl"
                if os.path.exists(us_file):
                    with open(us_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    last_trained = model_data['last_trained']
                    days_old = (datetime.now() - last_trained).days
                    
                    return {
                        'status': 'ACTIVE' if days_old <= 30 else 'EXPIRED',
                        'symbol': model_data['symbol'],
                        'market': 'US',
                        'model_version': 0,
                        'last_update': last_trained.date(),
                        'data_points': model_data['data_points'],
                        'current_accuracy': model_data.get('model_info', {}).get('test_score', 'N/A'),
                        'model_file': us_file,
                        'days_old': days_old
                    }
                
                return None
                
            except Exception as e:
                return None
        else:
            # Get status of current dynamic model
            if not self.initial_training_complete:
                return {
                    'status': 'NOT_INITIALIZED',
                    'message': 'Model not initialized'
                }
            
            status = {
                'status': 'ACTIVE',
                'symbol': self.symbol,
                'model_version': self.model_version,
                'last_update': self.last_update_date,
                'data_points': len(self.data) if self.data is not None else 0
            }
            
            if self.model_performance:
                latest = self.model_performance[-1]
                status.update({
                    'current_accuracy': latest['test_score'],
                    'training_accuracy': latest['train_score']
                })
            
            return status
    
    def delete_model(self, model_name):
        """
        Delete a specific model
        """
        try:
            # Try Chinese model first
            chinese_file = f"chinese_models/{model_name}_model.pkl"
            if os.path.exists(chinese_file):
                os.remove(chinese_file)
                return True
            
            # Try US model
            us_file = f"us_models/{model_name}_model.pkl"
            if os.path.exists(us_file):
                os.remove(us_file)
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def get_performance_summary(self):
        """
        Get model performance summary
        """
        if not self.model_performance:
            return "No performance data available"
        
        summary = f"\n{'='*50}\n"
        summary += f"MODEL PERFORMANCE: {self.symbol}\n"
        summary += f"{'='*50}\n"
        
        for perf in self.model_performance:
            summary += f"Version {perf['version']}: {perf['date']}\n"
            summary += f"  Test Accuracy: {perf['test_score']:.3f}\n"
            summary += f"  Data Points: {perf['data_points']}\n"
            
            if perf['version'] > 0:
                prev_perf = self.model_performance[perf['version'] - 1]
                improvement = perf['test_score'] - prev_perf['test_score']
                summary += f"  Improvement: {improvement:+.3f}\n"
            summary += "\n"
        
        return summary 