#!/usr/bin/env python3
"""
ML-Based Scoring Model for Stock Recommendations
Uses ML to calculate final recommendation score from technical and ML scores
Trained on multiple stocks for better cross-stock insights
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MLScoringModel:
    def __init__(self, model_path='ml_scoring_model.pkl'):
        """
        Initialize ML-based scoring model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.backtest_results = []
        self.stock_insights = {}
        
    def create_training_data_from_multiple_stocks(self, stock_data_dict, technical_scores_dict, ml_scores_dict, actual_returns_dict):
        """
        Create training data from multiple stocks for better cross-stock insights
        stock_data_dict: Dict of {symbol: DataFrame} with stock price data
        technical_scores_dict: Dict of {symbol: list} of technical scores for each day
        ml_scores_dict: Dict of {symbol: list} of ML scores for each day  
        actual_returns_dict: Dict of {symbol: list} of actual returns (target variable)
        """
        training_data = []
        
        for symbol in stock_data_dict.keys():
            stock_data = stock_data_dict[symbol]
            technical_scores = technical_scores_dict.get(symbol, [])
            ml_scores = ml_scores_dict.get(symbol, [])
            actual_returns = actual_returns_dict.get(symbol, [])
            
            if len(technical_scores) != len(ml_scores) or len(technical_scores) != len(actual_returns):
                continue
                
            for i in range(len(technical_scores)):
                if i < 20:  # Skip first 20 days to allow for indicator calculation
                    continue
                    
                # Features: technical score, ml score, market conditions, and stock-specific features
                features = {
                    'technical_score': technical_scores[i],
                    'ml_score': ml_scores[i],
                    'price_momentum': stock_data.iloc[i]['Price_Momentum_5'] if 'Price_Momentum_5' in stock_data.columns else 0,
                    'volume_ratio': stock_data.iloc[i]['Volume_Ratio'] if 'Volume_Ratio' in stock_data.columns else 1.0,
                    'rsi': stock_data.iloc[i]['RSI'] if 'RSI' in stock_data.columns else 50,
                    'volatility': stock_data.iloc[i]['Volatility_20'] if 'Volatility_20' in stock_data.columns else 0.02,
                    'macd_signal': stock_data.iloc[i]['MACD_Signal'] if 'MACD_Signal' in stock_data.columns else 0,
                    'bollinger_position': stock_data.iloc[i]['Bollinger_Position'] if 'Bollinger_Position' in stock_data.columns else 0.5,
                    'stock_price_level': stock_data.iloc[i]['close'] if 'close' in stock_data.columns else 100,
                    'stock_volume_level': stock_data.iloc[i]['volume'] if 'volume' in stock_data.columns else 1000000,
                    'symbol_hash': hash(symbol) % 1000  # Encode stock identity
                }
                
                # Target: actual return (normalized to 0-100 scale)
                target_return = actual_returns[i]
                target_score = self._return_to_score(target_return)
                
                training_data.append({
                    'symbol': symbol,
                    'features': features,
                    'target': target_score,
                    'actual_return': target_return,
                    'date_index': i
                })
        
        return training_data
    
    def _return_to_score(self, return_value):
        """
        Convert return to 0-100 score
        Positive returns get higher scores, negative returns get lower scores
        """
        # Sigmoid-like transformation
        if return_value >= 0.05:  # 5%+ gain
            return 90 + (return_value - 0.05) * 200  # 90-100 for 5%+ gains
        elif return_value >= 0.02:  # 2-5% gain
            return 70 + (return_value - 0.02) * 667  # 70-90 for 2-5% gains
        elif return_value >= 0:  # 0-2% gain
            return 50 + return_value * 1000  # 50-70 for 0-2% gains
        elif return_value >= -0.02:  # 0 to -2% loss
            return 30 + (return_value + 0.02) * 1000  # 30-50 for 0 to -2% losses
        elif return_value >= -0.05:  # -2 to -5% loss
            return 10 + (return_value + 0.05) * 667  # 10-30 for -2 to -5% losses
        else:  # -5%+ loss
            return max(0, 10 + (return_value + 0.05) * 200)  # 0-10 for -5%+ losses
    
    def train_model_on_multiple_stocks(self, training_data):
        """
        Train the ML scoring model on data from multiple stocks
        """
        if not training_data:
            print("❌ No training data provided")
            return False
        
        print(f"🤖 Training ML scoring model with {len(training_data)} samples from multiple stocks...")
        
        # Prepare features and targets
        X = []
        y = []
        symbols = []
        
        for sample in training_data:
            features = sample['features']
            X.append([
                features['technical_score'],
                features['ml_score'],
                features['price_momentum'],
                features['volume_ratio'],
                features['rsi'],
                features['volatility'],
                features['macd_signal'],
                features['bollinger_position'],
                features['stock_price_level'],
                features['stock_volume_level'],
                features['symbol_hash']
            ])
            y.append(sample['target'])
            symbols.append(sample['symbol'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data (stratified by symbol to ensure representation)
        unique_symbols = list(set(symbols))
        train_symbols, test_symbols = train_test_split(unique_symbols, test_size=0.2, random_state=42)
        
        train_indices = [i for i, s in enumerate(symbols) if s in train_symbols]
        test_indices = [i for i, s in enumerate(symbols) if s in test_symbols]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create ensemble model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully on {len(unique_symbols)} stocks!")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   📊 Test samples: {len(X_test)}")
        print(f"   📊 R² Score: {r2:.3f}")
        print(f"   📊 Mean Squared Error: {mse:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"   📊 Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Analyze performance by stock
        self._analyze_stock_performance(training_data, y_pred, test_indices)
        
        self.is_trained = True
        self.training_data = training_data
        
        return True
    
    def _analyze_stock_performance(self, training_data, predictions, test_indices):
        """Analyze model performance for each stock"""
        print(f"\n📊 STOCK-SPECIFIC PERFORMANCE ANALYSIS:")
        print("=" * 60)
        
        stock_performance = {}
        
        for i, idx in enumerate(test_indices):
            sample = training_data[idx]
            symbol = sample['symbol']
            actual_score = sample['target']
            predicted_score = predictions[i]
            
            if symbol not in stock_performance:
                stock_performance[symbol] = {'actual': [], 'predicted': [], 'errors': []}
            
            stock_performance[symbol]['actual'].append(actual_score)
            stock_performance[symbol]['predicted'].append(predicted_score)
            stock_performance[symbol]['errors'].append(abs(actual_score - predicted_score))
        
        # Calculate metrics for each stock
        for symbol, data in stock_performance.items():
            if len(data['actual']) >= 3:  # Need minimum samples
                avg_error = np.mean(data['errors'])
                correlation = np.corrcoef(data['actual'], data['predicted'])[0, 1]
                
                print(f"   📈 {symbol}:")
                print(f"      Samples: {len(data['actual'])}")
                print(f"      Avg Error: {avg_error:.1f}")
                print(f"      Correlation: {correlation:.3f}")
                
                # Store insights
                self.stock_insights[symbol] = {
                    'avg_error': avg_error,
                    'correlation': correlation,
                    'sample_count': len(data['actual'])
                }
    
    def predict_score(self, technical_score, ml_score, market_features=None, symbol=None):
        """
        Predict final recommendation score using trained model
        """
        if not self.is_trained or self.model is None:
            print("⚠️  Model not trained, using fallback calculation")
            return self._fallback_score(technical_score, ml_score)
        
        # Prepare features
        if market_features is None:
            market_features = {
                'price_momentum': 0,
                'volume_ratio': 1.0,
                'rsi': 50,
                'volatility': 0.02,
                'macd_signal': 0,
                'bollinger_position': 0.5,
                'stock_price_level': 100,
                'stock_volume_level': 1000000
            }
        
        # Add symbol hash if provided
        symbol_hash = hash(symbol) % 1000 if symbol else 0
        
        features = [
            technical_score,
            ml_score,
            market_features['price_momentum'],
            market_features['volume_ratio'],
            market_features['rsi'],
            market_features['volatility'],
            market_features['macd_signal'],
            market_features['bollinger_position'],
            market_features['stock_price_level'],
            market_features['stock_volume_level'],
            symbol_hash
        ]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict score
        predicted_score = self.model.predict(features_scaled)[0]
        
        # Ensure score is within 0-100 range
        predicted_score = max(0, min(100, predicted_score))
        
        return int(predicted_score)
    
    def _fallback_score(self, technical_score, ml_score):
        """
        Fallback calculation when ML model is not available
        """
        # Dynamic weighting based on score confidence
        if technical_score > 80 and ml_score > 80:
            # Both scores are high - give more weight to technical
            return int(technical_score * 0.8 + ml_score * 0.2)
        elif technical_score < 30 and ml_score < 30:
            # Both scores are low - give more weight to ML
            return int(technical_score * 0.3 + ml_score * 0.7)
        else:
            # Mixed signals - balanced approach
            return int(technical_score * 0.6 + ml_score * 0.4)
    
    def backtest_model_on_top_stocks(self, stock_symbols, analyzer, period_days=60):
        """
        Backtest the ML scoring model on top stocks together
        """
        print(f"🔬 Backtesting ML scoring model on {len(stock_symbols)} top stocks...")
        
        # Collect data from all stocks
        stock_data_dict = {}
        technical_scores_dict = {}
        ml_scores_dict = {}
        actual_returns_dict = {}
        
        for symbol in stock_symbols:
            try:
                print(f"   📊 Collecting data for {symbol}...")
                
                # Download data
                success, stock_name = analyzer.download_chinese_stock_data(symbol, 'A')
                if not success:
                    continue
                
                # Calculate indicators
                analyzer.calculate_chinese_indicators()
                
                # Get data for backtesting
                data = analyzer.data.copy()
                if len(data) < period_days + 20:
                    continue
                
                # Prepare for backtesting
                technical_scores = []
                ml_scores = []
                actual_returns = []
                
                # Calculate scores for each day
                for i in range(20, len(data) - 10):  # Skip first 20 days, leave 10 days for return calculation
                    # Set current data point
                    analyzer.data = data.iloc[:i+1]
                    
                    # Calculate technical score
                    tech_score = analyzer.calculate_chinese_technical_score()
                    technical_scores.append(tech_score)
                    
                    # Get ML score
                    try:
                        # Train model for this stock if needed
                        if analyzer.model is None:
                            analyzer.train_ml_model(holding_period=10, profit_threshold=0.03)
                        
                        ml_pred, ml_prob = analyzer.get_ml_prediction()
                        ml_score = int(ml_prob * 100) if ml_prob is not None else 50
                    except:
                        ml_score = 50
                    
                    ml_scores.append(ml_score)
                    
                    # Calculate actual return (10-day forward return)
                    current_price = data.iloc[i]['close']
                    future_price = data.iloc[i+10]['close']
                    actual_return = (future_price - current_price) / current_price
                    actual_returns.append(actual_return)
                
                # Store data
                stock_data_dict[symbol] = data.iloc[20:len(data)-10]
                technical_scores_dict[symbol] = technical_scores
                ml_scores_dict[symbol] = ml_scores
                actual_returns_dict[symbol] = actual_returns
                
                print(f"      ✅ {symbol}: {len(technical_scores)} samples collected")
                
            except Exception as e:
                print(f"      ❌ {symbol}: {str(e)}")
                continue
        
        # Create training data from all stocks
        training_data = self.create_training_data_from_multiple_stocks(
            stock_data_dict, technical_scores_dict, ml_scores_dict, actual_returns_dict
        )
        
        if len(training_data) > 50:  # Need minimum data
            # Train model on all stocks together
            model_trained = self.train_model_on_multiple_stocks(training_data)
            
            if model_trained:
                # Test predictions on each stock
                self._test_predictions_by_stock(training_data)
                
                # Save model
                self.save_model()
        
        return training_data
    
    def _test_predictions_by_stock(self, training_data):
        """Test predictions for each stock separately"""
        print(f"\n📊 PREDICTION TESTING BY STOCK:")
        print("=" * 60)
        
        # Group by stock
        stock_groups = {}
        for sample in training_data:
            symbol = sample['symbol']
            if symbol not in stock_groups:
                stock_groups[symbol] = []
            stock_groups[symbol].append(sample)
        
        # Test each stock
        for symbol, samples in stock_groups.items():
            if len(samples) < 5:  # Need minimum samples
                continue
                
            test_results = []
            for sample in samples[-5:]:  # Test on last 5 samples
                predicted_score = self.predict_score(
                    sample['features']['technical_score'],
                    sample['features']['ml_score'],
                    sample['features'],
                    symbol
                )
                
                test_results.append({
                    'predicted_score': predicted_score,
                    'actual_score': sample['target'],
                    'actual_return': sample['actual_return'],
                    'technical_score': sample['features']['technical_score'],
                    'ml_score': sample['features']['ml_score']
                })
            
            # Calculate metrics
            avg_prediction_error = np.mean([abs(r['predicted_score'] - r['actual_score']) for r in test_results])
            correlation = np.corrcoef([r['predicted_score'] for r in test_results], 
                                    [r['actual_score'] for r in test_results])[0, 1]
            
            print(f"   📈 {symbol}:")
            print(f"      Test samples: {len(test_results)}")
            print(f"      Avg error: {avg_prediction_error:.1f}")
            print(f"      Correlation: {correlation:.3f}")
    
    def save_model(self):
        """Save the trained model"""
        if self.is_trained and self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_data_size': len(self.training_data),
                'stock_insights': self.stock_insights,
                'trained_date': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            print(f"✅ Model saved to {self.model_path}")
            return True
        return False
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.training_data = model_data.get('training_data_size', 0)
                self.stock_insights = model_data.get('stock_insights', {})
                self.is_trained = True
                print(f"✅ Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
        return False
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            feature_names = [
                'Technical Score', 'ML Score', 'Price Momentum', 
                'Volume Ratio', 'RSI', 'Volatility', 'MACD Signal', 'Bollinger Position',
                'Stock Price Level', 'Stock Volume Level', 'Symbol Hash'
            ]
            importance = self.model.feature_importances_
            
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 FEATURE IMPORTANCE:")
            for feature, imp in feature_importance:
                print(f"   {feature}: {imp:.3f}")
            
            return feature_importance
        return None
    
    def get_stock_insights(self):
        """Get insights about model performance for each stock"""
        if self.stock_insights:
            print(f"\n📊 STOCK-SPECIFIC INSIGHTS:")
            print("=" * 50)
            
            # Sort by correlation
            sorted_insights = sorted(self.stock_insights.items(), 
                                   key=lambda x: x[1]['correlation'], reverse=True)
            
            for symbol, insights in sorted_insights:
                print(f"   📈 {symbol}:")
                print(f"      Correlation: {insights['correlation']:.3f}")
                print(f"      Avg Error: {insights['avg_error']:.1f}")
                print(f"      Samples: {insights['sample_count']}")
            
            return self.stock_insights
        return None
