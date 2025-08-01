#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced ML Scoring Model for Chinese Stock Recommendations
Improved version with better regularization and validation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedMLScoringModel:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_names = None
        self.is_trained = False
        self.model_info = {}
        self.model_file = "improved_ml_scoring_model.pkl"
        
    def create_enhanced_training_data(self, data_dict, technical_scores_dict=None, ml_scores_dict=None, actual_returns_dict=None):
        """
        Create enhanced training data from multiple stocks
        """
        return self.create_training_data_from_multiple_stocks(data_dict)
    
    def create_enhanced_features(self, data_dict):
        """
        Create enhanced features with advanced engineering for better accuracy
        """
        all_features = []
        
        for symbol, data in data_dict.items():
            if data is None or len(data) < 50:
                continue
                
            # Advanced price features
            data['Price_Change_1d'] = data['close'].pct_change(1)
            data['Price_Change_3d'] = data['close'].pct_change(3)
            data['Price_Change_5d'] = data['close'].pct_change(5)
            data['Price_Change_10d'] = data['close'].pct_change(10)
            data['Price_Change_20d'] = data['close'].pct_change(20)
            
            # Advanced volatility features with different windows
            data['Volatility_5d'] = data['Price_Change_1d'].rolling(5).std()
            data['Volatility_10d'] = data['Price_Change_1d'].rolling(10).std()
            data['Volatility_20d'] = data['Price_Change_1d'].rolling(20).std()
            data['Volatility_30d'] = data['Price_Change_1d'].rolling(30).std()
            
            # Volatility ratio features
            data['Volatility_Ratio_5_20'] = data['Volatility_5d'] / data['Volatility_20d']
            data['Volatility_Ratio_10_30'] = data['Volatility_10d'] / data['Volatility_30d']
            
            # Advanced volume features
            data['Volume_MA_5'] = data['volume'].rolling(5).mean()
            data['Volume_MA_10'] = data['volume'].rolling(10).mean()
            data['Volume_MA_20'] = data['volume'].rolling(20).mean()
            data['Volume_Ratio_5'] = data['volume'] / data['Volume_MA_5']
            data['Volume_Ratio_10'] = data['volume'] / data['Volume_MA_10']
            data['Volume_Ratio_20'] = data['volume'] / data['Volume_MA_20']
            
            # Volume trend features
            data['Volume_Trend_5'] = data['Volume_MA_5'] / data['Volume_MA_20']
            data['Volume_Trend_10'] = data['Volume_MA_10'] / data['Volume_MA_20']
            
            # Advanced moving average features
            data['SMA_5'] = data['close'].rolling(5).mean()
            data['SMA_10'] = data['close'].rolling(10).mean()
            data['SMA_20'] = data['close'].rolling(20).mean()
            data['SMA_50'] = data['close'].rolling(50).mean()
            data['EMA_12'] = data['close'].ewm(span=12).mean()
            data['EMA_26'] = data['close'].ewm(span=26).mean()
            
            # Advanced price position features
            data['Price_vs_SMA5'] = (data['close'] - data['SMA_5']) / data['SMA_5']
            data['Price_vs_SMA10'] = (data['close'] - data['SMA_10']) / data['SMA_10']
            data['Price_vs_SMA20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
            data['Price_vs_SMA50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
            data['Price_vs_EMA12'] = (data['close'] - data['EMA_12']) / data['EMA_12']
            data['Price_vs_EMA26'] = (data['close'] - data['EMA_26']) / data['EMA_26']
            
            # Moving average crossovers
            data['SMA_5_vs_20'] = (data['SMA_5'] - data['SMA_20']) / data['SMA_20']
            data['SMA_10_vs_50'] = (data['SMA_10'] - data['SMA_50']) / data['SMA_50']
            data['EMA_12_vs_26'] = (data['EMA_12'] - data['EMA_26']) / data['EMA_26']
            
            # Advanced momentum features
            data['Momentum_3d'] = data['close'] / data['close'].shift(3) - 1
            data['Momentum_5d'] = data['close'] / data['close'].shift(5) - 1
            data['Momentum_10d'] = data['close'] / data['close'].shift(10) - 1
            data['Momentum_20d'] = data['close'] / data['close'].shift(20) - 1
            
            # Momentum acceleration
            data['Momentum_Accel_5_10'] = data['Momentum_5d'] - data['Momentum_10d']
            data['Momentum_Accel_10_20'] = data['Momentum_10d'] - data['Momentum_20d']
            
            # RSI features (if available)
            if 'RSI' in data.columns:
                data['RSI_MA_5'] = data['RSI'].rolling(5).mean()
                data['RSI_MA_10'] = data['RSI'].rolling(10).mean()
                data['RSI_vs_MA5'] = data['RSI'] - data['RSI_MA_5']
                data['RSI_vs_MA10'] = data['RSI'] - data['RSI_MA_10']
                data['RSI_Trend'] = data['RSI_MA_5'] - data['RSI_MA_10']
                
                # RSI divergence features
                data['RSI_Price_Divergence'] = data['Price_Change_5d'] - data['RSI_vs_MA5']
            
            # MACD features (if available)
            if 'MACD' in data.columns:
                data['MACD_MA_5'] = data['MACD'].rolling(5).mean()
                data['MACD_MA_10'] = data['MACD'].rolling(10).mean()
                data['MACD_vs_MA5'] = data['MACD'] - data['MACD_MA_5']
                data['MACD_vs_MA10'] = data['MACD'] - data['MACD_MA_10']
                data['MACD_Trend'] = data['MACD_MA_5'] - data['MACD_MA_10']
                
                # MACD histogram features
                if 'MACD_Histogram' in data.columns:
                    data['MACD_Hist_MA_5'] = data['MACD_Histogram'].rolling(5).mean()
                    data['MACD_Hist_Trend'] = data['MACD_Histogram'] - data['MACD_Hist_MA_5']
            
            # Bollinger Bands features (if available)
            if 'BB_Position' in data.columns:
                data['BB_Position_MA_5'] = data['BB_Position'].rolling(5).mean()
                data['BB_Position_MA_10'] = data['BB_Position'].rolling(10).mean()
                data['BB_Trend'] = data['BB_Position_MA_5'] - data['BB_Position_MA_10']
                
                # Bollinger Band width and squeeze
                if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['SMA_20']
                    data['BB_Width_MA_10'] = data['BB_Width'].rolling(10).mean()
                    data['BB_Squeeze'] = data['BB_Width'] / data['BB_Width_MA_10']
            
            # Advanced market regime features
            data['Bull_Market'] = ((data['close'] > data['SMA_20']) & 
                                  (data['SMA_20'] > data['SMA_50']) &
                                  (data['EMA_12'] > data['EMA_26'])).astype(int)
            data['Bear_Market'] = ((data['close'] < data['SMA_20']) & 
                                  (data['SMA_20'] < data['SMA_50']) &
                                  (data['EMA_12'] < data['EMA_26'])).astype(int)
            data['Sideways_Market'] = ((data['close'] > data['SMA_20']) & 
                                      (data['SMA_20'] < data['SMA_50'])).astype(int)
            
            # Trend strength features
            data['Trend_Strength_5'] = abs(data['SMA_5'] - data['SMA_20']) / data['SMA_20']
            data['Trend_Strength_10'] = abs(data['SMA_10'] - data['SMA_50']) / data['SMA_50']
            data['Trend_Strength_20'] = abs(data['EMA_12'] - data['EMA_26']) / data['EMA_26']
            
            # Advanced support/resistance features
            data['Support_20'] = data['low'].rolling(20).min()
            data['Resistance_20'] = data['high'].rolling(20).max()
            data['Support_50'] = data['low'].rolling(50).min()
            data['Resistance_50'] = data['high'].rolling(50).max()
            
            data['Price_vs_Support_20'] = (data['close'] - data['Support_20']) / data['Support_20']
            data['Price_vs_Resistance_20'] = (data['close'] - data['Resistance_20']) / data['Resistance_20']
            data['Price_vs_Support_50'] = (data['close'] - data['Support_50']) / data['Support_50']
            data['Price_vs_Resistance_50'] = (data['close'] - data['Resistance_50']) / data['Resistance_50']
            
            # Support/Resistance strength
            data['Support_Strength'] = (data['close'] - data['Support_20']) / (data['Resistance_20'] - data['Support_20'])
            
            # Advanced price level features (normalized)
            data['Price_Level_20'] = (data['close'] - data['close'].rolling(252).min()) / \
                                    (data['close'].rolling(252).max() - data['close'].rolling(252).min())
            data['Price_Level_50'] = (data['close'] - data['close'].rolling(126).min()) / \
                                    (data['close'].rolling(126).max() - data['close'].rolling(126).min())
            
            # Volume level features (normalized)
            data['Volume_Level_20'] = (data['volume'] - data['volume'].rolling(252).min()) / \
                                     (data['volume'].rolling(252).max() - data['volume'].rolling(252).min())
            
            # Advanced time-based features
            data['Day_of_Week'] = pd.to_datetime(data.index).dayofweek
            data['Month'] = pd.to_datetime(data.index).month
            data['Quarter'] = pd.to_datetime(data.index).quarter
            
            # Cyclical encoding for time features
            data['Day_of_Week_Sin'] = np.sin(2 * np.pi * data['Day_of_Week'] / 7)
            data['Day_of_Week_Cos'] = np.cos(2 * np.pi * data['Day_of_Week'] / 7)
            data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
            data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
            
            # Symbol hash for categorical encoding
            data['Symbol_Hash'] = hash(symbol) % 1000 / 1000.0
            
            all_features.append(data)
        
        return all_features
    
    def create_training_data_from_multiple_stocks(self, data_dict):
        """
        Create training data from multiple stocks with enhanced features
        """
        enhanced_data = self.create_enhanced_features(data_dict)
        
        X_list = []
        y_list = []
        
        for data in enhanced_data:
            if data is None or len(data) < 50:
                continue
            
            # Define feature columns (enhanced list with more sophisticated features)
            feature_columns = [
                'Price_Change_1d', 'Price_Change_3d', 'Price_Change_5d', 'Price_Change_10d', 'Price_Change_20d',
                'Volatility_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_30d',
                'Volatility_Ratio_5_20', 'Volatility_Ratio_10_30',
                'Volume_Ratio_5', 'Volume_Ratio_10', 'Volume_Ratio_20',
                'Volume_Trend_5', 'Volume_Trend_10',
                'Price_vs_SMA5', 'Price_vs_SMA10', 'Price_vs_SMA20', 'Price_vs_SMA50',
                'Price_vs_EMA12', 'Price_vs_EMA26',
                'SMA_5_vs_20', 'SMA_10_vs_50', 'EMA_12_vs_26',
                'Momentum_3d', 'Momentum_5d', 'Momentum_10d', 'Momentum_20d',
                'Momentum_Accel_5_10', 'Momentum_Accel_10_20',
                'Bull_Market', 'Bear_Market', 'Sideways_Market',
                'Trend_Strength_5', 'Trend_Strength_10', 'Trend_Strength_20',
                'Price_vs_Support_20', 'Price_vs_Resistance_20',
                'Price_vs_Support_50', 'Price_vs_Resistance_50',
                'Support_Strength',
                'Price_Level_20', 'Price_Level_50', 'Volume_Level_20',
                'Day_of_Week_Sin', 'Day_of_Week_Cos', 'Month_Sin', 'Month_Cos',
                'Symbol_Hash'
            ]
            
            # Add RSI features if available
            if 'RSI' in data.columns:
                feature_columns.extend(['RSI', 'RSI_MA_5', 'RSI_MA_10', 'RSI_vs_MA5', 'RSI_vs_MA10', 'RSI_Trend', 'RSI_Price_Divergence'])
            
            # Add MACD features if available
            if 'MACD' in data.columns:
                feature_columns.extend(['MACD', 'MACD_MA_5', 'MACD_MA_10', 'MACD_vs_MA5', 'MACD_vs_MA10', 'MACD_Trend'])
                if 'MACD_Histogram' in data.columns:
                    feature_columns.extend(['MACD_Histogram', 'MACD_Hist_MA_5', 'MACD_Hist_Trend'])
            
            # Add Bollinger Bands features if available
            if 'BB_Position' in data.columns:
                feature_columns.extend(['BB_Position', 'BB_Position_MA_5', 'BB_Position_MA_10', 'BB_Trend'])
                if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                    feature_columns.extend(['BB_Width', 'BB_Width_MA_10', 'BB_Squeeze'])
            
            # Check which features are available
            available_features = [col for col in feature_columns if col in data.columns]
            
            if len(available_features) < 10:  # Need minimum features
                continue
            
            # Create multiple target variables for ensemble learning
            data['Future_Return_5d'] = data['close'].shift(-5) / data['close'] - 1
            data['Future_Return_10d'] = data['close'].shift(-10) / data['close'] - 1
            data['Future_Return_15d'] = data['close'].shift(-15) / data['close'] - 1
            
            # Create volatility-adjusted returns
            data['Volatility_Adjusted_Return_10d'] = data['Future_Return_10d'] / (data['Volatility_20d'] + 0.001)
            
            # Create risk-adjusted returns (Sharpe-like)
            data['Risk_Adjusted_Return_10d'] = data['Future_Return_10d'] / (data['Volatility_20d'] * np.sqrt(10) + 0.001)
            
            # Remove rows with NaN values
            target_columns = ['Future_Return_5d', 'Future_Return_10d', 'Future_Return_15d', 
                            'Volatility_Adjusted_Return_10d', 'Risk_Adjusted_Return_10d']
            clean_data = data[available_features + target_columns].dropna()
            
            if len(clean_data) < 20:  # Need minimum data points
                continue
            
            X = clean_data[available_features]
            
            # Use the most predictive target variable
            y = clean_data['Future_Return_10d']  # Primary target
            
            # Convert target to score (0-100) - more sophisticated approach
            y_score = self._enhanced_return_to_score(y)
            
            # Filter out extreme outliers - more sophisticated filtering
            outlier_mask = (y_score >= -60) & (y_score <= 60) & (abs(y) <= 0.15)  # 15% max return
            y_score_clean = y_score[outlier_mask]
            X_clean = X.iloc[outlier_mask.values]  # Convert to numpy array for indexing
            
            if len(y_score_clean) < 15:  # Need minimum clean data points
                continue
            
            X_list.append(X_clean)
            y_list.append(pd.Series(y_score_clean, index=X_clean.index))
        
        if not X_list:
            return None, None
        
        # Combine all data
        X_combined = pd.concat(X_list, ignore_index=True)
        y_combined = pd.concat(y_list, ignore_index=True)
        
        # Store feature names
        self.feature_names = X_combined.columns.tolist()
        
        return X_combined, y_combined
    
    def _enhanced_return_to_score(self, returns):
        """
        Enhanced conversion from returns to scores with more conservative scaling
        """
        # More conservative and realistic scoring
        scores = np.zeros_like(returns)
        
        # Positive returns (gains) - more conservative
        positive_mask = returns > 0
        scores[positive_mask] = np.where(
            returns[positive_mask] <= 0.01,  # 0-1%
            returns[positive_mask] * 2000,   # 0-20 points
            np.where(
                returns[positive_mask] <= 0.03,  # 1-3%
                20 + (returns[positive_mask] - 0.01) * 1000,  # 20-40 points
                np.where(
                    returns[positive_mask] <= 0.05,  # 3-5%
                    40 + (returns[positive_mask] - 0.03) * 800,  # 40-56 points
                    np.where(
                        returns[positive_mask] <= 0.08,  # 5-8%
                        56 + (returns[positive_mask] - 0.05) * 400,  # 56-68 points
                        np.where(
                            returns[positive_mask] <= 0.12,  # 8-12%
                            68 + (returns[positive_mask] - 0.08) * 200,  # 68-76 points
                            76  # 12%+
                        )
                    )
                )
            )
        )
        
        # Negative returns (losses) - more conservative
        negative_mask = returns < 0
        scores[negative_mask] = np.where(
            returns[negative_mask] >= -0.01,  # 0 to -1%
            returns[negative_mask] * 2000,    # 0 to -20 points
            np.where(
                returns[negative_mask] >= -0.03,  # -1 to -3%
                -20 + (returns[negative_mask] + 0.01) * 1000,  # -20 to -40 points
                np.where(
                    returns[negative_mask] >= -0.05,  # -3 to -5%
                    -40 + (returns[negative_mask] + 0.03) * 800,  # -40 to -56 points
                    np.where(
                        returns[negative_mask] >= -0.08,  # -5 to -8%
                        -56 + (returns[negative_mask] + 0.05) * 400,  # -56 to -68 points
                        np.where(
                            returns[negative_mask] >= -0.12,  # -8 to -12%
                            -68 + (returns[negative_mask] + 0.08) * 200,  # -68 to -76 points
                            -76  # -12% and below
                        )
                    )
                )
            )
        )
        
        # Ensure scores are within bounds
        scores = np.clip(scores, -100, 100)
        
        return scores
    
    def train_improved_model(self, X, y):
        """
        Train an improved model with advanced ensemble methods and better validation
        """
        if len(X) < 100:
            print("❌ Insufficient data for training")
            return False
        
        print(f"🤖 Training enhanced ML scoring model with {len(X)} samples...")
        
        # Advanced feature selection with multiple methods
        from sklearn.feature_selection import SelectFromModel, RFE
        from sklearn.ensemble import ExtraTreesRegressor
        
        # Use multiple feature selection methods
        selector1 = SelectKBest(score_func=mutual_info_regression, k=min(25, len(X.columns)))
        selector2 = SelectFromModel(ExtraTreesRegressor(n_estimators=50, random_state=42), max_features=min(20, len(X.columns)))
        
        # Create advanced models with better hyperparameters
        models = {
            'ridge': Ridge(alpha=5.0, random_state=42),  # Balanced regularization
            'lasso': Lasso(alpha=0.05, random_state=42),  # Balanced regularization
            'elastic': Ridge(alpha=2.0, random_state=42),  # Elastic net equivalent
            'rf': RandomForestRegressor(
                n_estimators=100,      # Increased for better ensemble
                max_depth=8,           # Balanced depth
                min_samples_split=25,  # Balanced split
                min_samples_leaf=15,   # Balanced leaf
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,        # Out-of-bag scoring
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,      # Increased for better ensemble
                learning_rate=0.05,    # Balanced learning rate
                max_depth=6,           # Balanced depth
                min_samples_split=30,  # Balanced split
                min_samples_leaf=20,   # Balanced leaf
                subsample=0.8,         # Balanced subsample
                random_state=42
            ),
            'xgb': None  # Will be added if available
        }
        
        # Try to add XGBoost if available
        try:
            import xgboost as xgb
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            print("   ⚠️  XGBoost not available, using other models")
        
        # Advanced cross-validation with multiple strategies
        from sklearn.model_selection import TimeSeriesSplit, KFold
        
        # Time series CV for temporal data
        tscv = TimeSeriesSplit(n_splits=5)
        # K-fold CV for robustness
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        best_cv_method = None
        
        for name, model in models.items():
            if model is None:
                continue
                
            # Test both CV methods
            for cv_method, cv_splitter in [('TimeSeries', tscv), ('KFold', kfold)]:
                # Create pipeline with feature selection
                if name in ['ridge', 'lasso', 'elastic']:
                    # Linear models with KBest selection
                    pipeline = Pipeline([
                        ('scaler', self.scaler),
                        ('feature_selector', selector1),
                        ('regressor', model)
                    ])
                else:
                    # Tree-based models with model-based selection
                    pipeline = Pipeline([
                        ('scaler', self.scaler),
                        ('feature_selector', selector2),
                        ('regressor', model)
                    ])
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='r2')
                mean_cv_score = cv_scores.mean()
                
                print(f"   📊 {name.upper()} ({cv_method}): CV R² = {mean_cv_score:.3f} (+/- {cv_scores.std()*2:.3f})")
                
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_model = pipeline
                    best_model_name = name
                    best_cv_method = cv_method
        
        # Train the best model on full dataset
        best_model.fit(X, y)
        
        # Evaluate on training data
        y_pred = best_model.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_mse = mean_squared_error(y, y_pred)
        train_mae = mean_absolute_error(y, y_pred)
        
        # Advanced validation metrics
        from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score
        
        train_mape = mean_absolute_percentage_error(y, y_pred)
        train_explained_var = explained_variance_score(y, y_pred)
        
        # Cross-validation on full dataset with best method
        if best_cv_method == 'TimeSeries':
            cv_splitter = tscv
        else:
            cv_splitter = kfold
            
        cv_scores = cross_val_score(best_model, X, y, cv=cv_splitter, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store model and info
        self.model = best_model
        self.is_trained = True
        
        self.model_info = {
            'model_type': f'Advanced {best_model_name.upper()} ({best_cv_method})',
            'train_r2': train_r2,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_mape': train_mape,
            'train_explained_var': train_explained_var,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat()
        }
        
        # Calculate enhanced reliability score
        reliability_score = self._calculate_enhanced_reliability_score(cv_mean, cv_std, train_r2, train_explained_var)
        
        print(f"✅ Advanced model trained successfully!")
        print(f"   📊 Model: {best_model_name.upper()} ({best_cv_method})")
        print(f"   📊 Train R²: {train_r2:.3f}")
        print(f"   📊 CV R²: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
        print(f"   📊 MSE: {train_mse:.3f}")
        print(f"   📊 MAE: {train_mae:.3f}")
        print(f"   📊 MAPE: {train_mape:.3f}")
        print(f"   📊 Explained Variance: {train_explained_var:.3f}")
        print(f"   📊 Features: {len(self.feature_names)}")
        print(f"   ✅ Reliability: {reliability_score:.1f}% ({self._get_reliability_level(reliability_score)})")
        
        return True
    
    def _calculate_enhanced_reliability_score(self, cv_mean, cv_std, train_r2, explained_var):
        """
        Calculate enhanced reliability score with multiple metrics for better accuracy assessment
        """
        # Base score from CV R² - more sophisticated
        if cv_mean > 0:
            base_score = min(85, cv_mean * 120)  # Allow higher scores for good models
        else:
            base_score = 0
        
        # Penalty for high variance - adaptive
        variance_penalty = min(20, cv_std * 60)  # Increased penalty for high variance
        
        # Penalty for overfitting - more sophisticated
        overfitting_penalty = min(25, max(0, (train_r2 - cv_mean) * 60))
        
        # Bonus for consistency and explained variance
        consistency_bonus = 8 if cv_std < 0.08 else 4 if cv_std < 0.15 else 0
        explained_var_bonus = min(10, explained_var * 20) if explained_var > 0 else 0
        
        # Penalty for poor explained variance
        explained_var_penalty = max(0, (0.5 - explained_var) * 20) if explained_var < 0.5 else 0
        
        # Final reliability score
        reliability = base_score - variance_penalty - overfitting_penalty + consistency_bonus + explained_var_bonus - explained_var_penalty
        
        return max(0, min(100, reliability))
    
    def _calculate_reliability_score(self, cv_mean, cv_std, train_r2):
        """
        Calculate reliability score based on multiple metrics with more realistic assessment
        """
        # Base score from CV R² - more conservative
        if cv_mean > 0:
            base_score = min(80, cv_mean * 100)  # Cap at 80% for realistic expectations
        else:
            base_score = 0
        
        # Penalty for high variance - reduced penalty
        variance_penalty = min(15, cv_std * 50)  # Reduced from 20
        
        # Penalty for overfitting - more lenient
        overfitting_penalty = min(20, max(0, (train_r2 - cv_mean) * 50))  # Reduced from 30
        
        # Bonus for consistency
        consistency_bonus = 5 if cv_std < 0.1 else 0
        
        # Final reliability score
        reliability = base_score - variance_penalty - overfitting_penalty + consistency_bonus
        
        return max(0, min(100, reliability))
    
    def _enhanced_fallback_score(self, technical_score, ml_score):
        """
        Enhanced fallback scoring when ML model is not available
        """
        # Conservative weighting with technical score having more weight
        technical_weight = 0.7
        ml_weight = 0.3
        
        # Ensure scores are within bounds
        technical_score = max(0, min(100, technical_score))
        ml_score = max(0, min(100, ml_score))
        
        # Calculate weighted score
        fallback_score = (technical_score * technical_weight) + (ml_score * ml_weight)
        
        return int(round(fallback_score))
    
    def _get_reliability_level(self, score):
        """Get reliability level description"""
        if score >= 80:
            return "EXCELLENT"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    
    def predict_improved_score(self, technical_score, ml_score, market_features):
        """
        Predict improved final score with ensemble methods and advanced feature engineering
        """
        if not self.is_trained or self.model is None:
            # Enhanced fallback with market context
            return self._enhanced_fallback_score_with_context(technical_score, ml_score, market_features)
        
        try:
            # Enhanced feature engineering for prediction
            features = self._create_enhanced_prediction_features(technical_score, ml_score, market_features)
            
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0.0)  # Default value
            
            # Make prediction
            predicted_score = self.model.predict([feature_vector])[0]
            
            # Enhanced bounds checking with market context
            if predicted_score < 0 or predicted_score > 100:
                # Use context-aware fallback
                return self._enhanced_fallback_score_with_context(technical_score, ml_score, market_features)
            
            # Advanced reasonableness check
            score_diff = abs(predicted_score - technical_score)
            max_allowed_diff = 35  # Increased tolerance
            
            # Adjust tolerance based on market conditions
            if market_features.get('volatility_20d', 0.02) > 0.03:
                max_allowed_diff = 40  # Higher tolerance in volatile markets
            
            if score_diff > max_allowed_diff:
                # Use weighted blend based on confidence
                confidence = self._calculate_prediction_confidence(features)
                blend_weight = min(0.3, confidence * 0.5)
                return int(technical_score * (1 - blend_weight) + predicted_score * blend_weight)
            
            return int(predicted_score)
            
        except Exception as e:
            print(f"⚠️  Prediction error: {str(e)}")
            # Enhanced fallback
            return self._enhanced_fallback_score_with_context(technical_score, ml_score, market_features)
    
    def _create_enhanced_prediction_features(self, technical_score, ml_score, market_features):
        """
        Create enhanced features for prediction with better engineering
        """
        features = {
            'technical_score': technical_score,
            'ml_score': ml_score,
            'stock_price_level': market_features.get('stock_price_level', 0.5),
            'stock_volume_level': market_features.get('stock_volume_level', 0.5),
            'symbol_hash': market_features.get('symbol_hash', 0.5)
        }
        
        # Add enhanced market features
        enhanced_features = [
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'volume_ratio', 'volume_ma_20', 'rsi', 'macd', 'macd_signal',
            'bollinger_position', 'volatility_10', 'volatility_20',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'support_20', 'resistance_20', 'price_vs_sma20', 'price_vs_sma50'
        ]
        
        for feature_name in enhanced_features:
            if feature_name in market_features:
                features[feature_name] = market_features[feature_name]
        
        # Add derived features
        if 'price_momentum_5' in features and 'price_momentum_10' in features:
            features['momentum_acceleration'] = features['price_momentum_5'] - features['price_momentum_10']
        
        if 'rsi' in features:
            features['rsi_extreme'] = 1 if features['rsi'] > 70 or features['rsi'] < 30 else 0
        
        if 'volume_ratio' in features:
            features['volume_spike'] = 1 if features['volume_ratio'] > 2.0 else 0
        
        # Add interaction features
        if 'technical_score' in features and 'ml_score' in features:
            features['score_interaction'] = features['technical_score'] * features['ml_score'] / 100
        
        return features
    
    def _enhanced_fallback_score_with_context(self, technical_score, ml_score, market_features):
        """
        Enhanced fallback scoring with market context
        """
        # Base weights
        technical_weight = 0.6
        ml_weight = 0.4
        
        # Adjust weights based on market conditions
        volatility = market_features.get('volatility_20d', 0.02)
        if volatility > 0.04:  # High volatility
            technical_weight = 0.7  # Trust technical more in volatile markets
            ml_weight = 0.3
        elif volatility < 0.015:  # Low volatility
            technical_weight = 0.5  # Trust ML more in stable markets
            ml_weight = 0.5
        
        # Market regime adjustment
        if market_features.get('bull_market', 0) == 1:
            ml_weight += 0.1  # Trust ML more in bull markets
            technical_weight -= 0.1
        elif market_features.get('bear_market', 0) == 1:
            technical_weight += 0.1  # Trust technical more in bear markets
            ml_weight -= 0.1
        
        # Ensure scores are within bounds
        technical_score = max(0, min(100, technical_score))
        ml_score = max(0, min(100, ml_score))
        
        # Calculate weighted score
        fallback_score = (technical_score * technical_weight) + (ml_score * ml_weight)
        
        return int(round(fallback_score))
    
    def _calculate_prediction_confidence(self, features):
        """
        Calculate confidence in prediction based on feature quality
        """
        confidence = 0.5  # Base confidence
        
        # Higher confidence for extreme technical scores
        tech_score = features.get('technical_score', 50)
        if tech_score > 80 or tech_score < 20:
            confidence += 0.2
        
        # Higher confidence for extreme ML scores
        ml_score = features.get('ml_score', 50)
        if ml_score > 80 or ml_score < 20:
            confidence += 0.2
        
        # Higher confidence for clear market signals
        if features.get('rsi_extreme', 0) == 1:
            confidence += 0.1
        
        if features.get('volume_spike', 0) == 1:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def save_model(self):
        """Save the enhanced model"""
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names,
                'model_info': self.model_info,
                'is_trained': self.is_trained
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Enhanced model saved to {self.model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """Load the enhanced model"""
        try:
            if not os.path.exists(self.model_file):
                return False
            
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_names = model_data['feature_names']
            self.model_info = model_data['model_info']
            self.is_trained = model_data['is_trained']
            
            reliability_score = self._calculate_reliability_score(
                self.model_info['cv_r2_mean'],
                self.model_info['cv_r2_std'],
                self.model_info['train_r2']
            )
            
            print(f"✅ Enhanced model loaded from {self.model_file}")
            print(f"   📊 Reliability score: {reliability_score:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def is_reliable(self):
        """Check if the model is reliable"""
        if not self.is_trained:
            return False
        
        reliability_score = self._calculate_reliability_score(
            self.model_info['cv_r2_mean'],
            self.model_info['cv_r2_std'],
            self.model_info['train_r2']
        )
        
        return reliability_score >= 50  # Minimum threshold
    
    def get_reliability_score(self):
        """Get the current reliability score"""
        if not self.is_trained:
            return 0.0
        
        return self._calculate_reliability_score(
            self.model_info['cv_r2_mean'],
            self.model_info['cv_r2_std'],
            self.model_info['train_r2']
        ) 