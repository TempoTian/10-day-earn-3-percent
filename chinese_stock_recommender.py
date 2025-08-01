#!/usr/bin/env python3
"""
Chinese Stock Recommendation System
Analyzes A500 Chinese stocks using multiple strategies and ML predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import time
from datetime import datetime, timedelta
from chinese_stock_cache import ChineseStockCache
from chinese_stock_analyzer import ChineseStockAnalyzer
from chinese_stock_downloader import ChineseStockDownloader
from improved_ml_scoring import ImprovedMLScoringModel

warnings.filterwarnings('ignore')

class ChineseStockRecommender:
    def __init__(self, data_source='yfinance'):
        """
        Initialize Chinese stock recommender
        """
        self.data_source = data_source
        self.analyzer = ChineseStockAnalyzer(data_source)
        self.downloader = ChineseStockDownloader(data_source)
        self.cache = ChineseStockCache()
        
        # Initialize improved ML scoring model
        self.ml_scoring_model = ImprovedMLScoringModel()
        self.ml_scoring_model.load_model()  # Try to load existing model
        
        # A500 typical Chinese stock symbols (major stocks) - extracted from stock_info.txt
        self.a500_symbols = [
            # Shenzhen A-shares (000xxx, 002xxx, 300xxx)
            '000001', '000002', '000063', '000100', '000157', '000166', '000301', '000333', '000338',
            '000408', '000425', '000538', '000568', '000596', '000617', '000625', '000630', '000651',
            '000661', '000708', '000725', '000768', '000776', '000792', '000800', '000807', '000858',
            '000876', '000895', '000938', '000963', '000975', '000977', '000983', '000999',
            
            # Shenzhen A-shares (001xxx, 002xxx)
            '001289', '001391', '001965', '001979',
            '002001', '002027', '002028', '002049', '002074', '002129', '002142', '002179', '002180',
            '002230', '002236', '002241', '002252', '002304', '002311', '002352', '002371', '002415',
            '002422', '002459', '002460', '002463', '002466', '002475', '002594', '002600', '002601',
            '002648', '002709', '002714', '002736', '002916', '002920', '002938',
            
            # Shenzhen A-shares (003xxx)
            '003816',
            
            # Shenzhen A-shares (300xxx)
            '300014', '300015', '300033', '300059', '300122', '300124', '300274', '300308', '300316',
            '300347', '300394', '300413', '300418', '300442', '300498', '300502', '300628', '300661',
            '300750', '300759', '300760', '300782', '300832', '300896', '300979', '300999',
            
            # Shenzhen A-shares (301xxx, 302xxx)
            '301236', '302132',
            
            # Shanghai A-shares (600xxx)
            '600009', '600010', '600011', '600015', '600016', '600018', '600019', '600023', '600025',
            '600026', '600027', '600028', '600029', '600030', '600031', '600036', '600039', '600048',
            '600050', '600061', '600066', '600089', '600111', '600115', '600150', '600160', '600161',
            '600176', '600183', '600188', '600219', '600233', '600276', '600332', '600346', '600362',
            '600372', '600377', '600406', '600415', '600426', '600436', '600438', '600460', '600482',
            '600489', '600515', '600519', '600547', '600570', '600584', '600585', '600588', '600600',
            '600660', '600674', '600690', '600760', '600803', '600809', '600845', '600875', '600886',
            '600887', '600893', '600905', '600918', '600919', '600926', '600938', '600941', '600958',
            '600989', '600999',
            
            # Shanghai A-shares (601xxx)
            '601006', '601009', '601012', '601021', '601058', '601059', '601066', '601077', '601088',
            '601100', '601111', '601117', '601127', '601136', '601138', '601166', '601169', '601186',
            '601211', '601225', '601229', '601236', '601238', '601288', '601318', '601319', '601328',
            '601336', '601360', '601377', '601390', '601398', '601600', '601601', '601607', '601618',
            '601628', '601633', '601658', '601668', '601669', '601688', '601689', '601698', '601699',
            '601728', '601766', '601788', '601799', '601800', '601816', '601818', '601825', '601838',
            '601857', '601868', '601872', '601877', '601878', '601881', '601888', '601898', '601899',
            '601901', '601916', '601919', '601939', '601985', '601988', '601989', '601995', '601998',
            
            # Shanghai A-shares (603xxx)
            '603019', '603195', '603259', '603260', '603288', '603296', '603369', '603392', '603501',
            '603799', '603806', '603833', '603986', '603993',
            
            # Shanghai A-shares (605xxx)
            '605117', '605499',
            
            # Shanghai A-shares (688xxx) - STAR Market
            '688008', '688009', '688012', '688036', '688047', '688082', '688111', '688126', '688169',
            '688187', '688223', '688271', '688303', '688396', '688472', '688506', '688599', '688981'
        ]
        try:
            import akshare as ak
           # bse_stock_list_df = ak.stock_info_bj_name_code()
            #self.a500_symbols = bse_stock_list_df["证券代码"]
        except Exception as e:
            print(f"Failed to download stock list: {e}")


        # Remove duplicates while preserving order
        self.a500_symbols = list(dict.fromkeys(self.a500_symbols))
        
        # Strategy definitions
        self.strategies = {
            '1': {'name': 'All strategies', 'description': 'Combined analysis using all strategies'},
            '2': {'name': '强中选强 (Strong Among Strong)', 'description': 'Focus on stocks with strong momentum and volume'},
            '3': {'name': '中位破局 (Mid-range Breakout)', 'description': 'Stocks breaking out from mid-range positions'},
            '4': {'name': '低位反弹 (Low Position Rebound)', 'description': 'Stocks rebounding from low positions'},
            '5': {'name': '技术突破 (Technical Breakout)', 'description': 'Stocks with technical breakout patterns'},
            '6': {'name': '价值回归 (Value Reversion)', 'description': 'Undervalued stocks with potential reversion'},
            '7': {'name': '成长加速 (Growth Acceleration)', 'description': 'High-growth stocks with accelerating momentum'}
        }
    
    def get_chinese_stock_symbol(self, symbol, market='A'):
        """Convert Chinese stock symbol to yfinance format"""
        symbol = symbol.strip().upper()
        
        if market.upper() == 'A':
            # A-shares
            if symbol.startswith('000') or symbol.startswith('002') or symbol.startswith('300'):
                return f"{symbol}.SZ"  # Shenzhen
            elif symbol.startswith('600') or symbol.startswith('900'):
                return f"{symbol}.SS"  # Shanghai
            else:
                return f"{symbol}.SS"  # Default to Shanghai
        elif market.upper() == 'H':
            # H-shares
            return f"{symbol}.HK"
        else:
            return f"{symbol}.SS"
    
    def download_stock_data(self, symbol, market='A', period="1mo"):
        """Download stock data with cache support"""
     
        # Check if download failed recently
        if self.cache.is_failed_download(symbol, market):
            print(f"⏭️  Skipping {symbol} ({market}-shares) - recent download failed")
            return None
        
        # Check cache first
        cached_data = self.cache.get_cached_stock_data(symbol, market, period)
        if cached_data is not None:
            return cached_data
        
        # Add delay to avoid server resistance
        time.sleep(0.3)
        
        try:
            # Use the new downloader
            data = self.downloader.download_stock_data(symbol, market, period)
            
            if data is not None:
                # Cache the data
                self.cache.cache_stock_data(symbol, market, period, data)
                return data
            else:
                self.cache.mark_download_failed(symbol, market, "No data available")
                return None
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {str(e)}")
            self.cache.mark_download_failed(symbol, market, str(e))
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for analysis"""
        if data is None or data.empty:
            return None
        
        # Basic indicators
        data = data.copy()
        data['Returns'] = data['close'].pct_change()
        data['Volume_MA_20'] = data['volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['volume'] / data['Volume_MA_20']
        
        # Moving averages
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['EMA_12'] = data['close'].ewm(span=12).mean()
        data['EMA_26'] = data['close'].ewm(span=26).mean()
        
        # Momentum indicators
        data['RSI'] = self.calculate_rsi(data['close'])
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Price momentum
        data['Price_Momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['Price_Momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['Price_Momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        # Volatility
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # Bollinger Bands
        data['BB_Upper'] = data['SMA_20'] + (data['close'].rolling(window=20).std() * 2)
        data['BB_Lower'] = data['SMA_20'] - (data['close'].rolling(window=20).std() * 2)
        data['BB_Position'] = (data['close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Support and resistance
        data['Support_20'] = data['low'].rolling(window=20).min()
        data['Resistance_20'] = data['high'].rolling(window=20).max()
        
        return data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def apply_strategy_filter(self, data, strategy_type):
        """Apply strategy-specific filters"""
        if data is None or data.empty:
            return 0, "No data available"
        
        current_price = data['close'].iloc[-1]
        score = 0
        reasons = []
        
        if strategy_type == '1':  # All strategies
            score, reasons = self.apply_all_strategies(data)
        elif strategy_type == '2':  # 强中选强 (Strong Among Strong)
            score, reasons = self.apply_strong_among_strong(data)
        elif strategy_type == '3':  # 中位破局 (Mid-range Breakout)
            score, reasons = self.apply_mid_range_breakout(data)
        elif strategy_type == '4':  # 低位反弹 (Low Position Rebound)
            score, reasons = self.apply_low_position_rebound(data)
        elif strategy_type == '5':  # 技术突破 (Technical Breakout)
            score, reasons = self.apply_technical_breakout(data)
        elif strategy_type == '6':  # 价值回归 (Value Reversion)
            score, reasons = self.apply_value_reversion(data)
        elif strategy_type == '7':  # 成长加速 (Growth Acceleration)
            score, reasons = self.apply_growth_acceleration(data)
        
        return score, reasons
    
    def apply_all_strategies(self, data):
        """Apply all strategies and combine scores"""
        scores = []
        reasons = []
        
        for strategy in ['2', '3', '4', '5', '6', '7']:
            score, reason = self.apply_strategy_filter(data, strategy)
            scores.append(score)
            reasons.extend(reason)
        
        # Average score with bonus for consistency
        avg_score = np.mean(scores)
        consistency_bonus = np.std(scores) < 20  # Bonus if scores are consistent
        
        final_score = avg_score + (10 if consistency_bonus else 0)
        return round(min(final_score, 100), 2), reasons
    
    def apply_strong_among_strong(self, data):
        """强中选强 (Strong Among Strong) strategy"""
        score = 0
        reasons = []
        
        # Strong momentum
        if data['Price_Momentum_5'].iloc[-1] > 0.05:
            score += 20
            reasons.append("Strong 5-day momentum")
        
        if data['Price_Momentum_10'].iloc[-1] > 0.08:
            score += 15
            reasons.append("Strong 10-day momentum")
        
        # High volume
        if data['Volume_Ratio'].iloc[-1] > 1.5:
            score += 15
            reasons.append("High volume ratio")
        
        # Price above moving averages
        if data['close'].iloc[-1] > data['SMA_20'].iloc[-1]:
            score += 10
            reasons.append("Price above 20-day MA")
        
        if data['close'].iloc[-1] > data['SMA_50'].iloc[-1]:
            score += 10
            reasons.append("Price above 50-day MA")
        
        # Strong RSI
        if 50 < data['RSI'].iloc[-1] < 80:
            score += 15
            reasons.append("Strong RSI")
        
        # MACD positive
        if data['MACD'].iloc[-1] > 0:
            score += 10
            reasons.append("Positive MACD")
        
        # Low volatility
        if data['Volatility_20'].iloc[-1] < 0.03:
            score += 5
            reasons.append("Low volatility")
        
        return round(score, 2), reasons
    
    def apply_mid_range_breakout(self, data):
        """中位破局 (Mid-range Breakout) strategy"""
        score = 0
        reasons = []
        
        current_price = data['close'].iloc[-1]
        
        # Price in mid-range
        bb_position = data['BB_Position'].iloc[-1]
        if 0.3 < bb_position < 0.7:
            score += 15
            reasons.append("Price in mid-range")
        
        # Breaking out
        if data['close'].iloc[-1] > data['SMA_20'].iloc[-1] and data['close'].iloc[-2] <= data['SMA_20'].iloc[-2]:
            score += 20
            reasons.append("Breaking above 20-day MA")
        
        # Volume confirmation
        if data['Volume_Ratio'].iloc[-1] > 1.2:
            score += 15
            reasons.append("Volume confirmation")
        
        # RSI not overbought
        if 40 < data['RSI'].iloc[-1] < 70:
            score += 15
            reasons.append("RSI not overbought")
        
        # Positive momentum
        if data['Price_Momentum_5'].iloc[-1] > 0.02:
            score += 15
            reasons.append("Positive momentum")
        
        # MACD turning positive
        if data['MACD'].iloc[-1] > data['MACD'].iloc[-2]:
            score += 10
            reasons.append("MACD improving")
        
        # Price near resistance
        resistance = data['Resistance_20'].iloc[-1]
        if 0.95 < current_price / resistance < 1.05:
            score += 10
            reasons.append("Near resistance level")
        
        return round(score, 2), reasons
    
    def apply_low_position_rebound(self, data):
        """低位反弹 (Low Position Rebound) strategy"""
        score = 0
        reasons = []
        
        current_price = data['close'].iloc[-1]
        
        # Price near support
        support = data['Support_20'].iloc[-1]
        if 0.95 < current_price / support < 1.05:
            score += 20
            reasons.append("Price near support")
        
        # RSI oversold
        if data['RSI'].iloc[-1] < 30:
            score += 20
            reasons.append("RSI oversold")
        
        # Volume spike
        if data['Volume_Ratio'].iloc[-1] > 1.5:
            score += 15
            reasons.append("Volume spike")
        
        # Price below moving averages but showing momentum
        if data['close'].iloc[-1] < data['SMA_20'].iloc[-1] and data['Price_Momentum_5'].iloc[-1] > 0:
            score += 15
            reasons.append("Below MA but showing momentum")
        
        # MACD turning positive
        if data['MACD'].iloc[-1] > data['MACD'].iloc[-2]:
            score += 10
            reasons.append("MACD turning positive")
        
        # Low volatility
        if data['Volatility_20'].iloc[-1] < 0.03:
            score += 10
            reasons.append("Low volatility")
        
        # Price near Bollinger lower band
        if data['BB_Position'].iloc[-1] < 0.2:
            score += 10
            reasons.append("Near Bollinger lower band")
        
        return round(score, 2), reasons
    
    def apply_technical_breakout(self, data):
        """技术突破 (Technical Breakout) strategy"""
        score = 0
        reasons = []
        
        current_price = data['close'].iloc[-1]
        
        # Breaking above resistance
        resistance = data['Resistance_20'].iloc[-1]
        if current_price > resistance * 1.02:
            score += 25
            reasons.append("Breaking above resistance")
        
        # High volume breakout
        if data['Volume_Ratio'].iloc[-1] > 2.0:
            score += 20
            reasons.append("High volume breakout")
        
        # Strong momentum
        if data['Price_Momentum_5'].iloc[-1] > 0.05:
            score += 15
            reasons.append("Strong momentum")
        
        # Price above all moving averages
        if (current_price > data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]):
            score += 15
            reasons.append("Above all moving averages")
        
        # RSI strong but not overbought
        if 60 < data['RSI'].iloc[-1] < 80:
            score += 15
            reasons.append("Strong RSI")
        
        # MACD positive and rising
        if data['MACD'].iloc[-1] > 0 and data['MACD'].iloc[-1] > data['MACD'].iloc[-2]:
            score += 10
            reasons.append("Positive and rising MACD")
        
        return round(score, 2), reasons
    
    def apply_value_reversion(self, data):
        """价值回归 (Value Reversion) strategy"""
        score = 0
        reasons = []
        
        current_price = data['close'].iloc[-1]
        
        # Price below moving averages (potential value)
        if current_price < data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1]:
            score += 20
            reasons.append("Price below moving averages")
        
        # RSI oversold
        if data['RSI'].iloc[-1] < 30:
            score += 20
            reasons.append("RSI oversold")
        
        # Low volatility (stable)
        if data['Volatility_20'].iloc[-1] < 0.025:
            score += 15
            reasons.append("Low volatility")
        
        # Price near support
        support = data['Support_20'].iloc[-1]
        if 0.95 < current_price / support < 1.05:
            score += 15
            reasons.append("Price near support")
        
        # Volume increasing
        if data['Volume_Ratio'].iloc[-1] > 1.2:
            score += 10
            reasons.append("Volume increasing")
        
        # MACD showing reversal
        if data['MACD'].iloc[-1] > data['MACD'].iloc[-2]:
            score += 10
            reasons.append("MACD showing reversal")
        
        # Price near Bollinger lower band
        if data['BB_Position'].iloc[-1] < 0.2:
            score += 10
            reasons.append("Near Bollinger lower band")
        
        return round(score, 2), reasons
    
    def apply_growth_acceleration(self, data):
        """成长加速 (Growth Acceleration) strategy"""
        score = 0
        reasons = []
        
        # Accelerating momentum
        momentum_5 = data['Price_Momentum_5'].iloc[-1]
        momentum_10 = data['Price_Momentum_10'].iloc[-1]
        
        if momentum_5 > momentum_10 and momentum_5 > 0.03:
            score += 25
            reasons.append("Accelerating momentum")
        
        # High volume growth
        if data['Volume_Ratio'].iloc[-1] > 1.8:
            score += 20
            reasons.append("High volume growth")
        
        # Price above moving averages with gap
        current_price = data['close'].iloc[-1]
        if current_price > data['SMA_20'].iloc[-1] * 1.05:
            score += 15
            reasons.append("Price well above 20-day MA")
        
        # Strong RSI
        if 60 < data['RSI'].iloc[-1] < 85:
            score += 15
            reasons.append("Strong RSI")
        
        # MACD positive and accelerating
        if data['MACD'].iloc[-1] > 0 and data['MACD'].iloc[-1] > data['MACD'].iloc[-2] * 1.1:
            score += 15
            reasons.append("MACD accelerating")
        
        # Low volatility (stable growth)
        if data['Volatility_20'].iloc[-1] < 0.03:
            score += 10
            reasons.append("Low volatility growth")
        
        return round(score, 2), reasons
    
    def analyze_stock_with_ml(self, symbol, market='A'):
        """Analyze stock using ML model"""
        try:
            # Use the full analysis from ChineseStockAnalyzer
            result = self.analyzer.analyze_chinese_stock(symbol, market)
            
            if result:
                # Get ML prediction
                ml_prediction = result.get('ml_prediction')
                ml_probability = result.get('ml_probability')
                
                if ml_probability is not None:
                    prediction = 'BUY' if ml_prediction == 1 else 'HOLD'
                    return {
                        'stock_name': result.get('stock_name'),
                        'prediction': prediction,
                        'probability': ml_probability,
                        'estimated_high_10d': result.get('estimated_high_10d'),
                        'estimated_low_10d': result.get('estimated_low_10d'),
                        'potential_gain_10d': result.get('potential_gain_10d'),
                        'potential_loss_10d': result.get('potential_loss_10d'),
                        'high_confidence': result.get('high_confidence'),
                        'low_confidence': result.get('low_confidence')
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def get_top_stocks_by_strategy(self, strategy_type, top_n=20):
        """Get top stocks for a specific strategy"""
        print(f"\n🔍 Analyzing A500 stocks using strategy: {self.strategies[strategy_type]['name']}")
        print(f"📝 {self.strategies[strategy_type]['description']}")
        
        # Check cache first
        cached_results = self.cache.get_cached_recommendation(strategy_type)
        if cached_results is not None:
            print(f"✅ Using cached recommendations for strategy {strategy_type}")
            return cached_results[:top_n]
        
        stock_scores = []
        filtered_out = 0
        total_stocks = len(self.a500_symbols)
        
        for i, symbol in enumerate(self.a500_symbols, 1):
            print(f"📊 Analyzing {symbol} ({i}/{total_stocks})", end='\r')
            
            # Add delay every 10 stocks to avoid server resistance
            if i % 10 == 0:
                time.sleep(1.0)
            
            # Download 1-month data for initial screening
            data = self.download_stock_data(symbol, 'A', "1mo")
            if data is None or len(data) < 20:
                filtered_out += 1
                continue
            
            # Calculate indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            if data_with_indicators is None:
                filtered_out += 1
                continue
            
            # Apply strategy filter
            score, reasons = self.apply_strategy_filter(data_with_indicators, strategy_type)
            
            if score > 30:  # Only consider stocks with decent scores
                stock_scores.append({
                    'symbol': symbol,
                    'market': 'A',
                    'score': score,
                    'reasons': reasons,
                    'current_price': data['close'].iloc[-1],
                    'volume_ratio': data_with_indicators['Volume_Ratio'].iloc[-1],
                    'momentum_5d': data_with_indicators['Price_Momentum_5'].iloc[-1],
                    'rsi': data_with_indicators['RSI'].iloc[-1]
                })
            else:
                filtered_out += 1
        
        print(f"\n✅ Analyzed {total_stocks} stocks:")
        print(f"   📊 Found {len(stock_scores)} candidates (score > 30)")
        print(f"   ❌ Filtered out {filtered_out} stocks (low score or no data)")
        print(f"   📈 Success rate: {len(stock_scores)/total_stocks*100:.1f}%")
        
        # Sort by score and get top N
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        top_stocks = stock_scores[:top_n]
        
        # Cache results
        self.cache.cache_recommendation(strategy_type, stock_scores)
        
        return top_stocks
    
    def get_final_recommendations(self, top_stocks, strategy_name):
        """
        Get final recommendations with ML integration (using already retrained model)
        """
 
        # First, generate pre-output for all stocks with combined technical + ML scores
        print(f"📊 STEP 1: PRELIMINARY ANALYSIS FOR ALL TOP STOCKS")
        pre_output_results = self._generate_pre_output_for_all_stocks(top_stocks)
        
        # Store intermediate results to avoid re-running
        self._intermediate_results = pre_output_results
          # Display pre-output results if available
        if hasattr(self, '_intermediate_results') and self._intermediate_results:
            self.display_pre_output_results(self._intermediate_results)
        
        print(f"✅ Preliminary analysis completed for {len(pre_output_results)} stocks")
        print(f"📊 Average preliminary score: {np.mean([r['pre_final_score'] for r in pre_output_results]):.1f}/100")
        
         # Retrain ML model with the selected top stocks
        model_retrained = self.retrain_ml_model_with_top_stocks(top_stocks)
        
        if model_retrained:
            print(f"🤖 Using retrained ML model for recommendations")
            self._model_retrained = True
        else:
            print(f"🤖 Using existing ML model for recommendations")
            self._model_retrained = False
        # Now generate final output using the retrained model
        print(f"\n🎯 STEP 2: OPTIMIZED RECOMMENDATIONS WITH RETRAINED MODEL")
        final_recommendations = self._generate_final_output_from_pre_results(pre_output_results, strategy_name)
        
        return final_recommendations
    
    def _generate_pre_output_for_all_stocks(self, top_stocks):
        """
        Generate pre-output for all top stocks using combined technical + ML scores
        """
        pre_output_results = []
        
        for i, stock in enumerate(top_stocks, 1):
            symbol = stock['symbol']
            print(f"   📊 Pre-analyzing {symbol} ({i}/{len(top_stocks)})...", end='\r')
            
            try:
                # Check cache first for 2-year data
                cached_data = self.cache.get_cached_stock_data(symbol, 'A', "2y")
                if cached_data is not None and len(cached_data) >= 100:
                    data = cached_data
                else:
                    # Download 2-year data for analysis
                    data = self.downloader.download_stock_data(symbol, 'A', period="2y")
                    if data is None or len(data) < 100:
                        continue
                    
                    # Cache the data
                    self.cache.cache_stock_data(symbol, 'A', "2y", data)
                
                # Analyze with ML (this will use existing model or fallback)
                analysis_result = self.analyzer.analyze_chinese_stock(symbol, 'A')
                
                if analysis_result:
                    # Extract scores and data
                    technical_score = analysis_result.get('technical_score', 0)
                    ml_score = analysis_result.get('ml_score', 0)
                    ml_probability = analysis_result.get('ml_probability', 0.5)
                    stock_name = analysis_result.get('stock_name', symbol)
                    current_price = analysis_result.get('current_price', 0)
                    estimated_high_10d = analysis_result.get('estimated_high_10d', 0)
                    estimated_low_10d = analysis_result.get('estimated_low_10d', 0)
                    potential_gain_10d = analysis_result.get('potential_gain_10d', 0)
                    potential_loss_10d = analysis_result.get('potential_loss_10d', 0)
                    high_confidence = analysis_result.get('high_confidence', 0)
                    low_confidence = analysis_result.get('low_confidence', 0)
                    
                    # Calculate pre-final score using combined technical + ML
                    pre_final_score = self._calculate_pre_final_score(technical_score, ml_score, ml_probability)
                    
                    # Store pre-output result
                    pre_output_results.append({
                        'symbol': symbol,
                        'stock_name': stock_name,
                        'technical_score': technical_score,
                        'ml_score': ml_score,
                        'ml_probability': ml_probability,
                        'pre_final_score': pre_final_score,
                        'current_price': current_price,
                        'estimated_high_10d': estimated_high_10d,
                        'estimated_low_10d': estimated_low_10d,
                        'potential_gain_10d': potential_gain_10d,
                        'potential_loss_10d': potential_loss_10d,
                        'high_confidence': high_confidence,
                        'low_confidence': low_confidence,
                        'data': data  # Store data for later use
                    })
                    
                    print(f"      ✅ {stock_name} - Pre-score: {pre_final_score:.1f}/100")
                
                # Add delay to avoid server resistance
                time.sleep(0.3)
                
            except Exception as e:
                print(f"      ❌ Error pre-analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by pre-final score
        pre_output_results.sort(key=lambda x: x['pre_final_score'], reverse=True)
        
        return pre_output_results
    
    def _calculate_pre_final_score(self, technical_score, ml_score, ml_probability):
        """
        Calculate pre-final score using combined technical + ML approach
        """
        # Convert ML probability to score (0-100)
        ml_prob_score = ml_probability * 100
        
        # Enhanced weighting for pre-output
        # Technical score gets more weight initially, ML gets weight based on confidence
        ml_confidence_weight = min(ml_probability * 2, 0.4)  # Max 40% weight for ML
        technical_weight = 1.0 - ml_confidence_weight
        
        pre_final_score = (technical_score * technical_weight) + (ml_prob_score * ml_confidence_weight)
        
        return round(pre_final_score, 1)
    
    def _generate_final_output_from_pre_results(self, pre_output_results, strategy_name):
        """
        Generate optimized recommendations using retrained model from preliminary results
        """
        final_recommendations = []
        
        for i, pre_result in enumerate(pre_output_results, 1):
            symbol = pre_result['symbol']
            print(f"   🎯 Optimizing {symbol} ({i}/{len(pre_output_results)})...", end='\r')
            
            try:
                # Get market features for ML scoring
                data = pre_result['data']
                current = data.iloc[-1]
                current_price = pre_result['current_price']
                
                market_features = {
                    'price_momentum_5': current.get('Price_Momentum_5', 0),
                    'price_momentum_10': current.get('Price_Momentum_10', 0),
                    'price_momentum_20': current.get('Price_Momentum_20', 0),
                    'volume_ratio': current.get('Volume_Ratio', 1.0),
                    'volume_ma_20': current.get('Volume_MA_20', 1000000),
                    'rsi': current.get('RSI', 50),
                    'macd': current.get('MACD', 0),
                    'macd_signal': current.get('MACD_Signal', 0),
                    'macd_histogram': current.get('MACD_Histogram', 0),
                    'bollinger_position': current.get('BB_Position', 0.5),
                    'volatility_10': current.get('Volatility_10', 0.02),
                    'volatility_20': current.get('Volatility_20', 0.02),
                    'volatility_50': current.get('Volatility_50', 0.02),
                    'sma_20': current.get('SMA_20', current_price),
                    'sma_50': current.get('SMA_50', current_price),
                    'ema_12': current.get('EMA_12', current_price),
                    'ema_26': current.get('EMA_26', current_price),
                    'support_20': current.get('Support_20', current_price * 0.9),
                    'resistance_20': current.get('Resistance_20', current_price * 1.1),
                    'price_vs_sma20': (current_price - current.get('SMA_20', current_price)) / current.get('SMA_20', current_price),
                    'price_vs_sma50': (current_price - current.get('SMA_50', current_price)) / current.get('SMA_50', current_price),
                    'stock_price_level': current_price,
                    'stock_volume_level': current.get('volume', 1000000),
                    'symbol_hash': hash(symbol) % 1000
                }
                
                # Use improved ML scoring model for final score
                if self.ml_scoring_model.is_trained and self.ml_scoring_model.is_reliable():
                    final_score = self.ml_scoring_model.predict_improved_score(
                        pre_result['technical_score'], pre_result['ml_score'], market_features, symbol
                    )
                    # Check if model was recently retrained (within this session)
                    if hasattr(self, '_model_retrained') and self._model_retrained:
                        scoring_method = "Retrained ML Model"
                    else:
                        scoring_method = "ML Model"
                else:
                    # Fallback to enhanced weighting
                    final_score = self.ml_scoring_model._enhanced_fallback_score(
                        pre_result['technical_score'], pre_result['ml_score']
                    )
                    scoring_method = "Enhanced Weighting"
                
                # Determine recommendation
                if final_score >= 80:
                    recommendation = "STRONG BUY"
                    emoji = "🚀"
                elif final_score >= 65:
                    recommendation = "BUY"
                    emoji = "📈"
                elif final_score >= 50:
                    recommendation = "HOLD"
                    emoji = "⏸️"
                elif final_score >= 35:
                    recommendation = "SELL"
                    emoji = "📉"
                else:
                    recommendation = "STRONG SELL"
                    emoji = "💥"
                
                final_recommendations.append({
                    'symbol': symbol,
                    'stock_name': pre_result['stock_name'],
                    'strategy': strategy_name,
                    'technical_score': pre_result['technical_score'],
                    'ml_score': pre_result['ml_score'],
                    'ml_probability': pre_result['ml_probability'],
                    'final_score': final_score,
                    'scoring_method': scoring_method,
                    'recommendation': recommendation,
                    'emoji': emoji,
                    'current_price': pre_result['current_price'],
                    'estimated_high_10d': pre_result['estimated_high_10d'],
                    'estimated_low_10d': pre_result['estimated_low_10d'],
                    'potential_gain_10d': pre_result['potential_gain_10d'],
                    'potential_loss_10d': pre_result['potential_loss_10d'],
                    'high_confidence': pre_result['high_confidence'],
                    'low_confidence': pre_result['low_confidence'],
                    'pre_final_score': pre_result['pre_final_score']  # Include for comparison
                })
                
                print(f"      ✅ {pre_result['stock_name']} - {recommendation} {emoji} (Score: {final_score}, Pre: {pre_result['pre_final_score']:.1f})")
                
            except Exception as e:
                print(f"      ❌ Error final-analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by final score and return top 5
        final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        top_5_recommendations = final_recommendations[:5]
        
        return top_5_recommendations
    
    def display_recommendations(self, recommendations):
        """
        Display recommendations with improved ML scoring information and save to file
        """
        if not recommendations:
            print("❌ No recommendations to display")
            return
        
        # Get strategy name from first recommendation
        strategy_name = recommendations[0].get('strategy', 'Unknown Strategy')
        
        print(f"\n{'='*100}")
        print(f"🎯 FINAL RECOMMENDATIONS")
        print(f"{'='*100}")
        
        # Check if ML model is being used
        ml_model_used = any(rec.get('scoring_method') == 'ML Model' for rec in recommendations)
        
        if ml_model_used:
            print(f"🤖 ML Scoring Model: ✅ ACTIVE")
        else:
            print(f"🤖 ML Scoring Model: ⚠️  Enhanced Weighting (Fallback)")
        
        print(f"📊 Total Recommendations: {len(recommendations)}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
        
        # Display each recommendation
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['emoji']} {rec['stock_name']} ({rec['symbol']})")
            print(f"   Strategy: {rec['strategy']}")
            print(f"   Recommendation: {rec['recommendation']} {rec['emoji']}")
            print(f"   Current Price: ¥{rec['current_price']:.2f}")
            
            # Scoring information
            print(f"   📊 Scoring:")
            print(f"      Technical Score: {rec['technical_score']:.1f}/100")
            print(f"      ML Score: {rec['ml_score']:.1f}/100")
            print(f"      ML Probability: {rec['ml_probability']:.1%}")
            print(f"      Final Score: {rec['final_score']}/100 ({rec['scoring_method']})")
            
            # Show pre-output comparison if available
            if 'pre_final_score' in rec:
                score_change = rec['final_score'] - rec['pre_final_score']
                change_emoji = "📈" if score_change > 0 else "📉" if score_change < 0 else "➡️"
                print(f"      Pre-Output Score: {rec['pre_final_score']:.1f}/100 (Change: {change_emoji} {score_change:+.1f})")
            
            # Price estimates and confidence
            if rec['estimated_high_10d'] > 0 and rec['estimated_low_10d'] > 0:
                print(f"   📈 10-Day Price Estimates:")
                print(f"      High: ¥{rec['estimated_high_10d']:.2f} (Confidence: {rec['high_confidence']:.1f}%)")
                print(f"      Low: ¥{rec['estimated_low_10d']:.2f} (Confidence: {rec['low_confidence']:.1f}%)")
                
                if rec['potential_gain_10d'] > 0:
                    print(f"      Potential Gain: +{rec['potential_gain_10d']:.1%}")
                if rec['potential_loss_10d'] > 0:
                    print(f"      Potential Loss: -{rec['potential_loss_10d']:.1%}")
            
            print(f"   {'─'*80}")
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Average Technical Score: {np.mean([r['technical_score'] for r in recommendations]):.1f}")
        print(f"   Average ML Score: {np.mean([r['ml_score'] for r in recommendations]):.1f}")
        print(f"   Average Final Score: {np.mean([r['final_score'] for r in recommendations]):.1f}")
        print(f"   Average ML Probability: {np.mean([r['ml_probability'] for r in recommendations]):.1%}")
        
        # Show pre-output comparison if available
        if 'pre_final_score' in recommendations[0]:
            avg_pre_score = np.mean([r['pre_final_score'] for r in recommendations])
            avg_final_score = np.mean([r['final_score'] for r in recommendations])
            avg_change = avg_final_score - avg_pre_score
            print(f"   Average Pre-Output Score: {avg_pre_score:.1f}")
            print(f"   Average Score Change: {avg_change:+.1f}")
        
        # Count recommendations by type
        strong_buy = len([r for r in recommendations if r['recommendation'] == 'STRONG BUY'])
        buy = len([r for r in recommendations if r['recommendation'] == 'BUY'])
        hold = len([r for r in recommendations if r['recommendation'] == 'HOLD'])
        sell = len([r for r in recommendations if r['recommendation'] == 'SELL'])
        strong_sell = len([r for r in recommendations if r['recommendation'] == 'STRONG SELL'])
        
        print(f"\n🎯 RECOMMENDATION BREAKDOWN:")
        if strong_buy > 0:
            print(f"   🚀 Strong Buy: {strong_buy}")
        if buy > 0:
            print(f"   📈 Buy: {buy}")
        if hold > 0:
            print(f"   ⏸️  Hold: {hold}")
        if sell > 0:
            print(f"   📉 Sell: {sell}")
        if strong_sell > 0:
            print(f"   💥 Strong Sell: {strong_sell}")
        
        # ML model status
        if ml_model_used:
            print(f"\n🤖 ML MODEL STATUS: ✅ Active and Reliable")
            print(f"   The final scores are calculated using the trained ML model")
            print(f"   which considers technical indicators, ML predictions, and market features.")
        else:
            print(f"\n🤖 ML MODEL STATUS: ⚠️  Using Enhanced Weighting")
            print(f"   The final scores use an enhanced weighting algorithm")
            print(f"   as the ML model is not available or not reliable enough.")
        
        print(f"\n{'='*100}")
        print(f"💡 DISCLAIMER: These recommendations are for educational purposes only.")
        print(f"   Always conduct your own research before making investment decisions.")
        print(f"{'='*100}")
        
        # Save recommendations to file
        self.save_recommendations_to_file(recommendations, strategy_name)

    def save_recommendations_to_file(self, recommendations, strategy_name):
        """
        Save recommendations to a single file with timestamp
        """
        if not recommendations:
            return False
        
        try:
            # Use a single file for all recommendations
            filename = "chinese_stock_recommendations.txt"
            
            with open(filename, 'a', encoding='utf-8') as f:
                # Add separator and timestamp
                f.write(f"\n{'='*80}\n")
                f.write(f"🎯 CHINESE STOCK RECOMMENDATIONS\n")
                f.write(f"Strategy: {strategy_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Recommendations: {len(recommendations)}\n")
                f.write("=" * 80 + "\n\n")
                
                # Check ML model status
                ml_model_used = any(rec.get('scoring_method') == 'ML Model' for rec in recommendations)
                if ml_model_used:
                    f.write("🤖 ML Scoring Model: ✅ ACTIVE\n")
                else:
                    f.write("🤖 ML Scoring Model: ⚠️  Enhanced Weighting (Fallback)\n")
                f.write("\n")
                
                # Write each recommendation
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec['emoji']} {rec['stock_name']} ({rec['symbol']})\n")
                    f.write(f"   Strategy: {rec['strategy']}\n")
                    f.write(f"   Recommendation: {rec['recommendation']} {rec['emoji']}\n")
                    f.write(f"   Current Price: ¥{rec['current_price']:.2f}\n")
                    
                    # Scoring information
                    f.write(f"   📊 Scoring:\n")
                    f.write(f"      Technical Score: {rec['technical_score']:.1f}/100\n")
                    f.write(f"      ML Score: {rec['ml_score']:.1f}/100\n")
                    f.write(f"      ML Probability: {rec['ml_probability']:.1%}\n")
                    f.write(f"      Final Score: {rec['final_score']}/100 ({rec['scoring_method']})\n")
                    
                    # Show pre-output comparison if available
                    if 'pre_final_score' in rec:
                        score_change = rec['final_score'] - rec['pre_final_score']
                        change_emoji = "📈" if score_change > 0 else "📉" if score_change < 0 else "➡️"
                        f.write(f"      Pre-Output Score: {rec['pre_final_score']:.1f}/100 (Change: {change_emoji} {score_change:+.1f})\n")
                    
                    # Price estimates and confidence
                    if rec['estimated_high_10d'] > 0 and rec['estimated_low_10d'] > 0:
                        f.write(f"   📈 10-Day Price Estimates:\n")
                        f.write(f"      High: ¥{rec['estimated_high_10d']:.2f} (Confidence: {rec['high_confidence']:.1f}%)\n")
                        f.write(f"      Low: ¥{rec['estimated_low_10d']:.2f} (Confidence: {rec['low_confidence']:.1f}%)\n")
                        
                        if rec['potential_gain_10d'] > 0:
                            f.write(f"      Potential Gain: +{rec['potential_gain_10d']:.1%}\n")
                        if rec['potential_loss_10d'] > 0:
                            f.write(f"      Potential Loss: -{rec['potential_loss_10d']:.1%}\n")
                    
                    f.write(f"   {'─' * 80}\n\n")
                
                # Summary statistics
                f.write(f"📊 SUMMARY STATISTICS:\n")
                f.write(f"   Average Technical Score: {np.mean([r['technical_score'] for r in recommendations]):.1f}\n")
                f.write(f"   Average ML Score: {np.mean([r['ml_score'] for r in recommendations]):.1f}\n")
                f.write(f"   Average Final Score: {np.mean([r['final_score'] for r in recommendations]):.1f}\n")
                f.write(f"   Average ML Probability: {np.mean([r['ml_probability'] for r in recommendations]):.1%}\n")
                
                # Show pre-output comparison if available
                if 'pre_final_score' in recommendations[0]:
                    avg_pre_score = np.mean([r['pre_final_score'] for r in recommendations])
                    avg_final_score = np.mean([r['final_score'] for r in recommendations])
                    avg_change = avg_final_score - avg_pre_score
                    f.write(f"   Average Pre-Output Score: {avg_pre_score:.1f}\n")
                    f.write(f"   Average Score Change: {avg_change:+.1f}\n")
                
                # Count recommendations by type
                strong_buy = len([r for r in recommendations if r['recommendation'] == 'STRONG BUY'])
                buy = len([r for r in recommendations if r['recommendation'] == 'BUY'])
                hold = len([r for r in recommendations if r['recommendation'] == 'HOLD'])
                sell = len([r for r in recommendations if r['recommendation'] == 'SELL'])
                strong_sell = len([r for r in recommendations if r['recommendation'] == 'STRONG SELL'])
                
                f.write(f"\n🎯 RECOMMENDATION BREAKDOWN:\n")
                if strong_buy > 0:
                    f.write(f"   🚀 Strong Buy: {strong_buy}\n")
                if buy > 0:
                    f.write(f"   📈 Buy: {buy}\n")
                if hold > 0:
                    f.write(f"   ⏸️  Hold: {hold}\n")
                if sell > 0:
                    f.write(f"   📉 Sell: {sell}\n")
                if strong_sell > 0:
                    f.write(f"   💥 Strong Sell: {strong_sell}\n")
                
                # ML model status
                if ml_model_used:
                    f.write(f"\n🤖 ML MODEL STATUS: ✅ Active and Reliable\n")
                    f.write(f"   The final scores are calculated using the trained ML model\n")
                    f.write(f"   which considers technical indicators, ML predictions, and market features.\n")
                else:
                    f.write(f"\n🤖 ML MODEL STATUS: ⚠️  Using Enhanced Weighting\n")
                    f.write(f"   The final scores use an enhanced weighting algorithm\n")
                    f.write(f"   as the ML model is not available or not reliable enough.\n")
                
                f.write(f"\n{'=' * 80}\n")
                f.write(f"💡 DISCLAIMER: These recommendations are for educational purposes only.\n")
                f.write(f"   Always conduct your own research before making investment decisions.\n")
                f.write(f"{'=' * 80}\n")
            
            print(f"💾 Recommendations saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving recommendations to file: {str(e)}")
            return False
    
    def display_pre_output_results(self, pre_output_results):
        """
        Display preliminary analysis results for all analyzed stocks
        """
        if not pre_output_results:
            print("❌ No preliminary results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 PRELIMINARY ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"📊 Total Stocks Analyzed: {len(pre_output_results)}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Display top 10 preliminary results
        for i, result in enumerate(pre_output_results[:10], 1):
            print(f"\n{i}. {result['stock_name']} ({result['symbol']})")
            print(f"   Preliminary Score: {result['pre_final_score']:.1f}/100")
            print(f"   Technical Score: {result['technical_score']:.1f}/100")
            print(f"   ML Score: {result['ml_score']:.1f}/100")
            print(f"   ML Probability: {result['ml_probability']:.1%}")
            print(f"   Current Price: ¥{result['current_price']:.2f}")
            
            if result['estimated_high_10d'] > 0 and result['estimated_low_10d'] > 0:
                print(f"   📈 10-Day Estimates: ¥{result['estimated_low_10d']:.2f} - ¥{result['estimated_high_10d']:.2f}")
            
            print(f"   {'─'*60}")
        
        # Summary statistics
        print(f"\n📊 PRELIMINARY SUMMARY:")
        print(f"   Average Preliminary Score: {np.mean([r['pre_final_score'] for r in pre_output_results]):.1f}")
        print(f"   Average Technical Score: {np.mean([r['technical_score'] for r in pre_output_results]):.1f}")
        print(f"   Average ML Score: {np.mean([r['ml_score'] for r in pre_output_results]):.1f}")
        print(f"   Average ML Probability: {np.mean([r['ml_probability'] for r in pre_output_results]):.1%}")
        
        # Score distribution
        excellent = len([r for r in pre_output_results if r['pre_final_score'] >= 80])
        good = len([r for r in pre_output_results if 65 <= r['pre_final_score'] < 80])
        moderate = len([r for r in pre_output_results if 50 <= r['pre_final_score'] < 65])
        poor = len([r for r in pre_output_results if r['pre_final_score'] < 50])
        
        print(f"\n📈 SCORE DISTRIBUTION:")
        print(f"   🚀 Excellent (80+): {excellent}")
        print(f"   📈 Good (65-79): {good}")
        print(f"   ⏸️  Moderate (50-64): {moderate}")
        print(f"   📉 Poor (<50): {poor}")
        
        print(f"\n{'='*80}")
        print(f"💡 These are preliminary results before ML model optimization.")
        print(f"   Final optimized recommendations will be available after model retraining.")
        print(f"{'='*80}")

        # Save preliminary results to file
        self.save_preliminary_results_to_file(pre_output_results)
    
    def save_preliminary_results_to_file(self, pre_output_results):
        """
        Save preliminary analysis results to a single file with timestamp
        """
        if not pre_output_results:
            return False
        
        try:
            # Use a single file for all preliminary results
            filename = "chinese_stock_preliminary_results.txt"
            
            with open(filename, 'a', encoding='utf-8') as f:
                # Add separator and timestamp
                f.write(f"\n{'='*80}\n")
                f.write(f"📊 PRELIMINARY ANALYSIS RESULTS\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Stocks Analyzed: {len(pre_output_results)}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write top 10 preliminary results
                for i, result in enumerate(pre_output_results[:10], 1):
                    f.write(f"{i}. {result['stock_name']} ({result['symbol']})\n")
                    f.write(f"   Preliminary Score: {result['pre_final_score']:.1f}/100\n")
                    f.write(f"   Technical Score: {result['technical_score']:.1f}/100\n")
                    f.write(f"   ML Score: {result['ml_score']:.1f}/100\n")
                    f.write(f"   ML Probability: {result['ml_probability']:.1%}\n")
                    f.write(f"   Current Price: ¥{result['current_price']:.2f}\n")
                    
                    if result['estimated_high_10d'] > 0 and result['estimated_low_10d'] > 0:
                        f.write(f"   📈 10-Day Estimates: ¥{result['estimated_low_10d']:.2f} - ¥{result['estimated_high_10d']:.2f}\n")
                    
                    f.write(f"   {'─' * 60}\n\n")
                
                # Summary statistics
                f.write(f"📊 PRELIMINARY SUMMARY:\n")
                f.write(f"   Average Preliminary Score: {np.mean([r['pre_final_score'] for r in pre_output_results]):.1f}\n")
                f.write(f"   Average Technical Score: {np.mean([r['technical_score'] for r in pre_output_results]):.1f}\n")
                f.write(f"   Average ML Score: {np.mean([r['ml_score'] for r in pre_output_results]):.1f}\n")
                f.write(f"   Average ML Probability: {np.mean([r['ml_probability'] for r in pre_output_results]):.1%}\n")
                
                # Score distribution
                excellent = len([r for r in pre_output_results if r['pre_final_score'] >= 80])
                good = len([r for r in pre_output_results if 65 <= r['pre_final_score'] < 80])
                moderate = len([r for r in pre_output_results if 50 <= r['pre_final_score'] < 65])
                poor = len([r for r in pre_output_results if r['pre_final_score'] < 50])
                
                f.write(f"\n📈 SCORE DISTRIBUTION:\n")
                f.write(f"   🚀 Excellent (80+): {excellent}\n")
                f.write(f"   📈 Good (65-79): {good}\n")
                f.write(f"   ⏸️  Moderate (50-64): {moderate}\n")
                f.write(f"   📉 Poor (<50): {poor}\n")
                
                f.write(f"\n{'=' * 80}\n")
                f.write(f"💡 These are preliminary results before ML model optimization.\n")
                f.write(f"   Final optimized recommendations will be available after model retraining.\n")
                f.write(f"{'=' * 80}\n")
            
            print(f"💾 Preliminary results saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving preliminary results to file: {str(e)}")
            return False
    
    def recommend(self, strategy_type, top_n=5):
        """
        Main recommendation method with improved ML scoring and model retraining
        """
        print(f"\n🎯 Chinese Stock Recommendation System")
        print(f"Strategy: {self.strategies[strategy_type]['name']}")
        print(f"Data Source: {self.data_source}")
        print("=" * 60)
        
        # Check ML model status
        if self.ml_scoring_model.is_trained and self.ml_scoring_model.is_reliable():
            print(f"🤖 ML Scoring Model: ✅ Active (Reliability: {self.ml_scoring_model.get_reliability_score():.1%})")
        else:
            print(f"🤖 ML Scoring Model: ⚠️  Using Enhanced Weighting (Fallback)")
        
        # Get top stocks from initial screening
        top_stocks = self.get_top_stocks_by_strategy(strategy_type, top_n=20)
        
        if not top_stocks:
            print("❌ No suitable stocks found for this strategy")
            return []
        
        print(f"\n📊 Initial screening completed: {len(top_stocks)} stocks selected")
        
        # Get final recommendations with ML integration (no retraining here)
        strategy_name = self.strategies[strategy_type]['name']
        final_recommendations = self.get_final_recommendations(top_stocks, strategy_name)
        
        if not final_recommendations:
            print("❌ No final recommendations generated")
            return []
        
        
        # Display recommendations
        self.display_recommendations(final_recommendations)
        
        return final_recommendations

    def retrain_ml_model_with_top_stocks(self, top_stocks):
        """
        Retrain ML model with data from selected top stocks
        """
        if not top_stocks:
            print("❌ No top stocks provided for retraining")
            return False
        
        print(f"\n🤖 RETRAINING ML MODEL WITH TOP {len(top_stocks)} STOCKS")
        print(f"📊 Collecting training data...")
        
        # Collect 2-year data for selected stocks
        training_data = []
        collected_stocks = 0
        
        for i, stock_info in enumerate(top_stocks, 1):
            symbol = stock_info['symbol']
            market = stock_info.get('market', 'A')
            
            print(f"   📈 Processing {symbol} ({i}/{len(top_stocks)})...", end='\r')
            
            try:
                # Check cache first for 2-year data
                cached_data = self.cache.get_cached_stock_data(symbol, market, "2y")
                
                if cached_data is not None and len(cached_data) > 400:  # At least 2 years
                    print(f"      ✅ Using cached 2-year data ({len(cached_data)} days)")
                    data = cached_data
                else:
                    print(f"      📥 Downloading 2-year data...")
                    data = self.downloader.download_stock_data(symbol, market, "2y")
                    
                    if data is not None and len(data) > 400:
                        # Cache the downloaded data
                        self.cache.cache_stock_data(symbol, market, "2y", data)
                        print(f"      ✅ Downloaded and cached ({len(data)} days)")
                    else:
                        print(f"      ⚠️  Insufficient data for {symbol}, skipping")
                        continue
                
                # Calculate indicators
                data_with_indicators = self.calculate_technical_indicators(data.copy())
                
                if data_with_indicators is None:
                    print(f"      ⚠️  Failed to calculate indicators for {symbol}, skipping")
                    continue
                
                # Create training data for this stock
                stock_training_data = self.create_training_data_from_stock(
                    data_with_indicators, symbol, market
                )
                
                if stock_training_data:
                    training_data.extend(stock_training_data)
                    collected_stocks += 1
                    print(f"      ✅ Added {len(stock_training_data)} training samples")
                else:
                    print(f"      ⚠️  No training data generated for {symbol}")
                
            except Exception as e:
                print(f"      ❌ Error processing {symbol}: {str(e)}")
                continue
        
        print(f"\n📊 Data collection completed!")
        print(f"   ✅ Successfully processed {collected_stocks}/{len(top_stocks)} stocks")
        print(f"   📈 Total training samples: {len(training_data)}")
        
        if len(training_data) < 100:
            print(f"❌ Insufficient training data ({len(training_data)} samples)")
            return False
        
        # Train the improved ML model
        print(f"\n🤖 Training improved ML scoring model...")
        success = self.ml_scoring_model.train_improved_model(training_data)
        
        if success:
            print(f"✅ ML model retraining completed successfully!")
            print(f"📊 Model trained on {len(training_data)} samples from {collected_stocks} stocks")
            return True
        else:
            print(f"❌ ML model retraining failed")
            return False

    def create_training_data_from_stock(self, data_with_indicators, symbol, market):
        """
        Create training data from a single stock's historical data
        """
        if data_with_indicators is None or len(data_with_indicators) < 50:
            return None
        
        training_samples = []
        
        try:
            # Calculate scores for each day (skip first 30, leave 10 for return calculation)
            for j in range(30, len(data_with_indicators) - 10):
                current = data_with_indicators.iloc[j]
                
                # Calculate technical score for this point
                tech_score = self.calculate_technical_score_for_point(data_with_indicators, j)
                
                # Calculate ML-like score based on indicators
                ml_score = self.calculate_ml_like_score_for_point(current)
                
                # Calculate actual return (10-day forward return)
                current_price = data_with_indicators.iloc[j]['close']
                future_price = data_with_indicators.iloc[j+10]['close']
                actual_return = (future_price - current_price) / current_price
                
                # Convert return to target score
                target_score = self._enhanced_return_to_score(actual_return)
                
                # Create market features
                market_features = {
                    'stock_price_level': self._normalize_price_level(current_price),
                    'stock_volume_level': self._normalize_volume_level(current.get('Volume_Ratio', 1.0)),
                    'volatility': current.get('Volatility_20', 0.02),
                    'momentum': current.get('Price_Momentum_5', 0),
                    'macd': current.get('MACD', 0),
                    'rsi': current.get('RSI', 50),
                    'bollinger_position': current.get('BB_Position', 0.5),
                    'symbol_hash': hash(symbol) % 1000  # Simple hash for symbol
                }
                
                # Create training sample
                sample = {
                    'technical_score': tech_score,
                    'ml_score': ml_score,
                    'target_score': target_score,
                    'actual_return': actual_return,
                    **market_features
                }
                
                training_samples.append(sample)
            
            return training_samples
            
        except Exception as e:
            print(f"      ❌ Error creating training data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_score_for_point(self, data, index):
        """
        Calculate technical score for a specific point in time
        """
        try:
            current = data.iloc[index]
            
            score = 50  # Base score
            
            # Momentum scoring
            momentum_5 = current.get('Price_Momentum_5', 0)
            if momentum_5 > 0.05:
                score += 15
            elif momentum_5 > 0.02:
                score += 10
            elif momentum_5 < -0.05:
                score -= 15
            
            # Volume scoring
            volume_ratio = current.get('Volume_Ratio', 1.0)
            if volume_ratio > 1.5:
                score += 10
            elif volume_ratio < 0.5:
                score -= 5
            
            # Moving average scoring
            close = current['close']
            sma_20 = current.get('SMA_20', close)
            sma_50 = current.get('SMA_50', close)
            
            if close > sma_20 > sma_50:
                score += 15
            elif close < sma_20 < sma_50:
                score -= 15
            
            # RSI scoring
            rsi = current.get('RSI', 50)
            if rsi > 70:
                score -= 10
            elif rsi < 30:
                score += 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50  # Default score on error
    
    def calculate_ml_like_score_for_point(self, current):
        """
        Calculate ML-like score based on technical indicators
        """
        try:
            score = 50  # Base score
            
            # RSI-based adjustment
            rsi = current.get('RSI', 50)
            if rsi > 70:
                score -= 20  # Overbought
            elif rsi > 60:
                score -= 10
            elif rsi < 30:
                score += 20  # Oversold
            elif rsi < 40:
                score += 10
            
            # Momentum-based adjustment
            momentum = current.get('Price_Momentum_5', 0)
            if momentum > 0.05:
                score += 15
            elif momentum > 0.02:
                score += 10
            elif momentum < -0.05:
                score -= 15
            elif momentum < -0.02:
                score -= 10
            
            # Volume-based adjustment
            volume_ratio = current.get('Volume_Ratio', 1.0)
            if volume_ratio > 1.5:
                score += 10
            elif volume_ratio < 0.5:
                score -= 5
            
            # MACD-based adjustment
            macd = current.get('MACD', 0)
            if macd > 0:
                score += 5
            else:
                score -= 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50  # Default score on error
    
    def _enhanced_return_to_score(self, return_value):
        """
        Convert return value to target score (0-100)
        """
        if return_value >= 0.08:  # 8%+ gain
            return 90 + min(10, (return_value - 0.08) * 100)  # 90-100 for 8%+ gains
        elif return_value >= 0.05:  # 5-8% gain
            return 80 + (return_value - 0.05) * 333  # 80-90 for 5-8% gains
        elif return_value >= 0.03:  # 3-5% gain
            return 70 + (return_value - 0.03) * 500  # 70-80 for 3-5% gains
        elif return_value >= 0.01:  # 1-3% gain
            return 60 + (return_value - 0.01) * 500  # 60-70 for 1-3% gains
        elif return_value >= -0.01:  # -1% to 1%
            return 50 + return_value * 500  # 45-55 for -1% to 1%
        elif return_value >= -0.03:  # -3% to -1%
            return 40 + (return_value + 0.03) * 500  # 40-45 for -3% to -1%
        elif return_value >= -0.05:  # -5% to -3%
            return 20 + (return_value + 0.05) * 1000  # 20-40 for -5% to -3%
        else:  # -5% and below
            return max(0, 5 + (return_value + 0.05) * 100)  # 0-5 for -5%+ losses
    
    def _normalize_price_level(self, price):
        """
        Normalize price level to 0-1 range
        """
        # Simple normalization based on typical Chinese stock price ranges
        if price <= 10:
            return price / 10
        elif price <= 50:
            return 1.0 + (price - 10) / 40
        elif price <= 100:
            return 2.0 + (price - 50) / 50
        else:
            return 3.0 + min(2.0, (price - 100) / 100)
    
    def _normalize_volume_level(self, volume_ratio):
        """
        Normalize volume ratio to 0-1 range
        """
        return min(1.0, max(0.0, volume_ratio / 3.0))
