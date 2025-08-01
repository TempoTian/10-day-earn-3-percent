#!/usr/bin/env python3
"""
Improved ML Scoring System with Better Reliability
Enhanced features and validation for recommendation system integration
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedMLScoringModel:
    def __init__(self, model_path='improved_ml_scoring_model.pkl'):
        """
        Initialize improved ML-based scoring model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.is_trained = False
        self.training_data = []
        self.backtest_results = []
        self.reliability_score = 0.0
        
    def create_enhanced_training_data(self, stock_data_dict, technical_scores_dict, ml_scores_dict, actual_returns_dict):
        """
        Create enhanced training data with better features
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
                if i < 30:  # Skip first 30 days for better indicator stability
                    continue
                    
                # Enhanced features with better market context
                features = {
                    'technical_score': technical_scores[i],
                    'ml_score': ml_scores[i],
                    
                    # Price momentum features
                    'price_momentum_5': stock_data.iloc[i].get('Price_Momentum_5', 0),
                    'price_momentum_10': stock_data.iloc[i].get('Price_Momentum_10', 0),
                    'price_momentum_20': stock_data.iloc[i].get('Price_Momentum_20', 0),
                    
                    # Volume features
                    'volume_ratio': stock_data.iloc[i].get('Volume_Ratio', 1.0),
                    'volume_ma_20': stock_data.iloc[i].get('Volume_MA_20', 1000000),
                    
                    # Technical indicators
                    'rsi': stock_data.iloc[i].get('RSI', 50),
                    'macd': stock_data.iloc[i].get('MACD', 0),
                    'macd_signal': stock_data.iloc[i].get('MACD_Signal', 0),
                    'macd_histogram': stock_data.iloc[i].get('MACD_Histogram', 0),
                    'bollinger_position': stock_data.iloc[i].get('BB_Position', 0.5),
                    
                    # Volatility features
                    'volatility_10': stock_data.iloc[i].get('Volatility_10', 0.02),
                    'volatility_20': stock_data.iloc[i].get('Volatility_20', 0.02),
                    'volatility_50': stock_data.iloc[i].get('Volatility_50', 0.02),
                    
                    # Moving averages
                    'sma_20': stock_data.iloc[i].get('SMA_20', 0),
                    'sma_50': stock_data.iloc[i].get('SMA_50', 0),
                    'ema_12': stock_data.iloc[i].get('EMA_12', 0),
                    'ema_26': stock_data.iloc[i].get('EMA_26', 0),
                    
                    # Support/Resistance
                    'support_20': stock_data.iloc[i].get('Support_20', 0),
                    'resistance_20': stock_data.iloc[i].get('Resistance_20', 0),
                    
                    # Market regime features
                    'price_vs_sma20': (stock_data.iloc[i]['close'] - stock_data.iloc[i].get('SMA_20', stock_data.iloc[i]['close'])) / stock_data.iloc[i].get('SMA_20', stock_data.iloc[i]['close']),
                    'price_vs_sma50': (stock_data.iloc[i]['close'] - stock_data.iloc[i].get('SMA_50', stock_data.iloc[i]['close'])) / stock_data.iloc[i].get('SMA_50', stock_data.iloc[i]['close']),
                    
                    # Stock-specific features
                    'stock_price_level': stock_data.iloc[i]['close'],
                    'stock_volume_level': stock_data.iloc[i]['volume'],
                    'symbol_hash': hash(symbol) % 1000
                }
                
                # Target: actual return (normalized to 0-100 scale)
                target_return = actual_returns[i]
                target_score = self._enhanced_return_to_score(target_return)
                
                training_data.append({
                    'symbol': symbol,
                    'features': features,
                    'target': target_score,
                    'actual_return': target_return,
                    'date_index': i
                })
        
        return training_data
    
    def _enhanced_return_to_score(self, return_value):
        """
        Enhanced return to score conversion with better scaling
        """
        # More conservative transformation with proper bounds
        if return_value >= 0.08:  # 8%+ gain
            return min(100, 90 + (return_value - 0.08) * 125)  # 90-100 for 8%+ gains
        elif return_value >= 0.05:  # 5-8% gain
            return 80 + (return_value - 0.05) * 333  # 80-90 for 5-8% gains
        elif return_value >= 0.03:  # 3-5% gain
            return 70 + (return_value - 0.03) * 500  # 70-80 for 3-5% gains
        elif return_value >= 0.01:  # 1-3% gain
            return 55 + (return_value - 0.01) * 750  # 55-70 for 1-3% gains
        elif return_value >= 0:  # 0-1% gain
            return 45 + return_value * 1000  # 45-55 for 0-1% gains
        elif return_value >= -0.01:  # 0 to -1% loss
            return 35 + (return_value + 0.01) * 1000  # 35-45 for 0 to -1% losses
        elif return_value >= -0.03:  # -1 to -3% loss
            return 20 + (return_value + 0.03) * 750  # 20-35 for -1 to -3% losses
        elif return_value >= -0.05:  # -3 to -5% loss
            return 5 + (return_value + 0.05) * 750  # 5-20 for -3 to -5% losses
        else:  # -5%+ loss
            return max(0, 5 + (return_value + 0.05) * 100)  # 0-5 for -5%+ losses
    
    def train_improved_model(self, training_data):
        """
        Train improved ML scoring model with better validation
        """
        if not training_data:
            print("❌ No training data provided")
            return False
        
        print(f"🤖 Training improved ML scoring model with {len(training_data)} samples...")
        
        # Prepare features and targets
        X = []
        y = []
        
        feature_names = [
            'technical_score', 'ml_score', 'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'volume_ratio', 'volume_ma_20', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_position', 'volatility_10', 'volatility_20', 'volatility_50',
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'support_20', 'resistance_20',
            'price_vs_sma20', 'price_vs_sma50', 'stock_price_level', 'stock_volume_level', 'symbol_hash'
        ]
        
        for sample in training_data:
            features = sample['features']
            X.append([features.get(name, 0) for name in feature_names])
            y.append(sample['target'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Time series split for better validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Use Gradient Boosting for better performance
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Enhanced validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='r2')
        y_pred = self.model.predict(X_scaled)
        
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"✅ Improved model trained successfully!")
        print(f"   📊 R² Score: {r2:.3f}")
        print(f"   📊 Mean Squared Error: {mse:.3f}")
        print(f"   📊 Mean Absolute Error: {mae:.3f}")
        print(f"   📊 Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Calculate reliability score
        self.reliability_score = max(0, min(1, (r2 + 1) / 2))  # Convert to 0-1 scale
        
        if self.reliability_score > 0.6:
            print(f"   ✅ Model reliability: {self.reliability_score:.1%} (GOOD)")
        elif self.reliability_score > 0.4:
            print(f"   ⚠️  Model reliability: {self.reliability_score:.1%} (MODERATE)")
        else:
            print(f"   ❌ Model reliability: {self.reliability_score:.1%} (POOR)")
        
        self.is_trained = True
        self.training_data = training_data
        
        return True
    
    def predict_improved_score(self, technical_score, ml_score, market_features=None, symbol=None):
        """
        Predict final recommendation score using improved model
        """
        if not self.is_trained or self.model is None:
            print("⚠️  Model not trained, using fallback calculation")
            return self._enhanced_fallback_score(technical_score, ml_score)
        
        # Prepare features
        if market_features is None:
            market_features = {
                'price_momentum_5': 0, 'price_momentum_10': 0, 'price_momentum_20': 0,
                'volume_ratio': 1.0, 'volume_ma_20': 1000000,
                'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
                'bollinger_position': 0.5, 'volatility_10': 0.02, 'volatility_20': 0.02, 'volatility_50': 0.02,
                'sma_20': 100, 'sma_50': 100, 'ema_12': 100, 'ema_26': 100,
                'support_20': 90, 'resistance_20': 110,
                'price_vs_sma20': 0, 'price_vs_sma50': 0,
                'stock_price_level': 100, 'stock_volume_level': 1000000,
                'symbol_hash': hash(symbol) % 1000 if symbol else 0
            }
        
        feature_names = [
            'technical_score', 'ml_score', 'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'volume_ratio', 'volume_ma_20', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_position', 'volatility_10', 'volatility_20', 'volatility_50',
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'support_20', 'resistance_20',
            'price_vs_sma20', 'price_vs_sma50', 'stock_price_level', 'stock_volume_level', 'symbol_hash'
        ]
        
        features = [technical_score, ml_score] + [market_features.get(name, 0) for name in feature_names[2:]]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict score
        predicted_score = self.model.predict(features_scaled)[0]
        
        # Conservative bounds checking with fallback to technical score if prediction is unrealistic
        if predicted_score < 0 or predicted_score > 100:
            print(f"⚠️  ML prediction out of bounds ({predicted_score:.1f}), using conservative fallback")
            # Use a conservative blend of technical and ML scores
            conservative_score = technical_score * 0.7 + ml_score * 0.3
            return max(0, min(100, int(conservative_score)))
        
        # Additional sanity check: if prediction is too far from technical score, use conservative approach
        score_diff = abs(predicted_score - technical_score)
        if score_diff > 30:  # If ML prediction differs by more than 30 points from technical
            print(f"⚠️  Large score difference detected (ML: {predicted_score:.1f}, Tech: {technical_score}), using conservative blend")
            # Use weighted average with more weight on technical score
            conservative_score = technical_score * 0.8 + predicted_score * 0.2
            return max(0, min(100, int(conservative_score)))
        
        # Ensure score is within 0-100 range and return as integer
        final_score = max(0, min(100, predicted_score))
        return int(final_score)
    
    def _enhanced_fallback_score(self, technical_score, ml_score):
        """
        Enhanced fallback calculation when ML model is not available
        """
        # More sophisticated fallback logic
        if technical_score > 80 and ml_score > 80:
            # Both scores are high - give more weight to technical
            return int(technical_score * 0.75 + ml_score * 0.25)
        elif technical_score < 30 and ml_score < 30:
            # Both scores are low - give more weight to ML
            return int(technical_score * 0.25 + ml_score * 0.75)
        elif abs(technical_score - ml_score) > 30:
            # Large discrepancy - use the higher score with some weight from the other
            if technical_score > ml_score:
                return int(technical_score * 0.8 + ml_score * 0.2)
            else:
                return int(technical_score * 0.2 + ml_score * 0.8)
        else:
            # Balanced approach
            return int(technical_score * 0.6 + ml_score * 0.4)
    
    def get_reliability_score(self):
        """Get model reliability score"""
        return self.reliability_score
    
    def is_reliable(self, threshold=0.5):
        """Check if model is reliable enough for production use"""
        return self.reliability_score >= threshold
    
    def save_model(self):
        """Save the trained model"""
        if self.is_trained and self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_data_size': len(self.training_data),
                'reliability_score': self.reliability_score,
                'trained_date': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            print(f"✅ Improved model saved to {self.model_path}")
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
                self.reliability_score = model_data.get('reliability_score', 0.0)
                self.is_trained = True
                print(f"✅ Improved model loaded from {self.model_path}")
                print(f"   📊 Reliability score: {self.reliability_score:.1%}")
                return True
            except Exception as e:
                print(f"❌ Error loading improved model: {str(e)}")
        return False

def test_improved_ml_scoring():
    """Test the improved ML scoring system"""
    print("🚀 Testing Improved ML Scoring System")
    print("=" * 60)
    
    # Import test functions
    from test_ml_scoring_system import load_cached_stocks, load_stock_data_from_cache, select_top_20_stocks
    from chinese_stock_analyzer import ChineseStockAnalyzer
    
    # Initialize components
    analyzer = ChineseStockAnalyzer(data_source='akshare')
    improved_model = ImprovedMLScoringModel()
    
    # Load cached stocks
    cached_stocks = load_cached_stocks()
    if not cached_stocks:
        print("❌ No cached stocks available")
        return
    
    # Select top 20 stocks
    top_20_stocks = select_top_20_stocks(cached_stocks, analyzer)
    if not top_20_stocks:
        print("❌ Failed to select top 20 stocks")
        return
    
    # Train improved model
    print(f"\nTraining improved ML scoring model...")
    
    # Collect data from all stocks
    stock_data_dict = {}
    technical_scores_dict = {}
    ml_scores_dict = {}
    actual_returns_dict = {}
    
    for i, stock_info in enumerate(top_20_stocks, 1):
        symbol = stock_info['symbol']
        data_file = stock_info['data_file']
        
        print(f"   Processing {symbol} ({i}/20)...")
        
        # Load data
        data = load_stock_data_from_cache(data_file)
        if data is None or len(data) < 60:
            continue
        
        try:
            # Set data in analyzer and calculate indicators
            analyzer.data = data
            indicators_calculated = analyzer.calculate_chinese_indicators()
            
            if not indicators_calculated:
                continue
            
            # Work with a copy of the data
            data_with_indicators = analyzer.data.copy()
            
            # Prepare for backtesting
            technical_scores = []
            ml_scores = []
            actual_returns = []
            
            # Calculate scores for each day
            for j in range(30, len(data_with_indicators) - 10):
                # Create a temporary analyzer
                temp_analyzer = ChineseStockAnalyzer(data_source='akshare')
                temp_analyzer.data = data_with_indicators.iloc[:j+1]
                
                # Calculate technical score
                tech_score = temp_analyzer.calculate_chinese_technical_score()
                technical_scores.append(tech_score)
                
                # Get ML score
                try:
                    current = data_with_indicators.iloc[j]
                    rsi = current.get('RSI', 50)
                    momentum = current.get('Price_Momentum_5', 0)
                    volume = current.get('Volume_Ratio', 1.0)
                    
                    ml_score = 50
                    if rsi > 70:
                        ml_score += 20
                    elif rsi < 30:
                        ml_score -= 20
                    
                    if momentum > 0.02:
                        ml_score += 15
                    elif momentum < -0.02:
                        ml_score -= 15
                    
                    if volume > 1.5:
                        ml_score += 10
                    elif volume < 0.5:
                        ml_score -= 10
                    
                    ml_score = max(0, min(100, ml_score))
                    
                except:
                    ml_score = 50
                
                ml_scores.append(ml_score)
                
                # Calculate actual return
                current_price = data_with_indicators.iloc[j]['close']
                future_price = data_with_indicators.iloc[j+10]['close']
                actual_return = (future_price - current_price) / current_price
                actual_returns.append(actual_return)
            
            # Store data
            stock_data_dict[symbol] = data_with_indicators.iloc[30:len(data_with_indicators)-10]
            technical_scores_dict[symbol] = technical_scores
            ml_scores_dict[symbol] = ml_scores
            actual_returns_dict[symbol] = actual_returns
            
            print(f"      ✅ Successfully processed {symbol} with {len(technical_scores)} samples")
            
        except Exception as e:
            print(f"   Error processing {symbol}: {str(e)}")
            continue
    
    print(f"   Processed {len(stock_data_dict)} stocks")
    
    # Create training data and train model
    if len(stock_data_dict) > 0:
        training_data = improved_model.create_enhanced_training_data(
            stock_data_dict, technical_scores_dict, ml_scores_dict, actual_returns_dict
        )
        
        print(f"   Total training samples: {len(training_data)}")
        
        if len(training_data) > 100:
            model_trained = improved_model.train_improved_model(training_data)
            
            if model_trained:
                # Save the model
                improved_model.save_model()
                
                # Test reliability
                if improved_model.is_reliable(threshold=0.5):
                    print(f"\n✅ Model is reliable enough for production use!")
                    print(f"💡 Ready to integrate into recommendation system")
                    return improved_model
                else:
                    print(f"\n⚠️  Model reliability below threshold")
                    print(f"💡 Consider improving features or using fallback")
                    return improved_model
            else:
                print("❌ Failed to train improved model")
                return None
        else:
            print(f"❌ Insufficient training data ({len(training_data)} samples)")
            return None
    else:
        print("❌ No valid stock data found")
        return None

if __name__ == "__main__":
    improved_model = test_improved_ml_scoring()
    
    if improved_model and improved_model.is_reliable():
        print(f"\n🎉 Improved ML scoring system is ready for integration!")
    else:
        print(f"\n❌ Improved ML scoring system needs further refinement") 