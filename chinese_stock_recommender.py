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

warnings.filterwarnings('ignore')

class ChineseStockRecommender:
    def __init__(self, data_source='yfinance'):
        self.cache = ChineseStockCache()
        self.analyzer = ChineseStockAnalyzer(data_source)
        self.downloader = ChineseStockDownloader(data_source)
        
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
        # Add delay to avoid server resistance
        time.sleep(0.3)
        
        # Check if download failed recently
        if self.cache.is_failed_download(symbol, market):
            print(f"⏭️  Skipping {symbol} ({market}-shares) - recent download failed")
            return None
        
        # Check cache first
        cached_data = self.cache.get_cached_stock_data(symbol, market, period)
        if cached_data is not None:
            return cached_data
        
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
    
    def get_final_recommendations(self, strategy_type, top_n=5):
        """Get final recommendations with ML analysis"""
        print(f"\n🎯 Getting final recommendations for strategy: {self.strategies[strategy_type]['name']}")
        
        # Get top stocks from initial screening
        top_stocks = self.get_top_stocks_by_strategy(strategy_type, top_n=20)
        
        if not top_stocks:
            print("❌ No suitable stocks found for this strategy")
            return []
        
        print(f"\n🤖 Performing analysis on top {len(top_stocks)} stocks...")
        
        final_recommendations = []
        ml_success_count = 0
        
        for i, stock in enumerate(top_stocks, 1):
            print(f"📊 Analyzing {stock['symbol']} ({i}/{len(top_stocks)})", end='\r')
            
            # Add delay every 5 stocks to avoid server resistance
            if i % 5 == 0:
                time.sleep(0.5)
            
            # Try ML analysis first
            ml_result = self.analyze_stock_with_ml(stock['symbol'], stock['market'])
            
            if ml_result:
                # Use ML results
                ml_probability = ml_result.get('probability', 0.5)
                ml_prediction = ml_result.get('prediction', 'HOLD')
                
                # Calculate final score (70% technical + 30% ML)
                final_score = (stock['score'] * 0.7) + (ml_probability * 100 * 0.3)
                
                # Determine action based on combined score and ML probability
                if final_score >= 80 and ml_probability > 0.7:
                    action = "STRONG BUY"
                elif final_score >= 70 and ml_probability > 0.6:
                    action = "BUY"
                elif final_score >= 60 and ml_probability > 0.5:
                    action = "HOLD"
                else:
                    action = "HOLD"
                
                ml_success_count += 1
                stock_name = ml_result['stock_name']
            else:
                # Fallback to technical analysis only
                ml_probability = 0.5
                ml_prediction = 'HOLD'
                final_score = stock['score']
                stock_name = self.downloader.get_stock_name(stock['symbol'], stock['market'])
                
                # Determine action based on technical score only
                if final_score >= 80:
                    action = "STRONG BUY"
                elif final_score >= 70:
                    action = "BUY"
                elif final_score >= 60:
                    action = "HOLD"
                else:
                    action = "HOLD"
            
            # Create recommendation with all available data
            recommendation = {
                'symbol': stock['symbol'],
                'market': stock['market'],
                'stock_name': stock_name,
                'final_score': round(final_score, 2),
                'technical_score': stock['score'],
                'ml_probability': round(ml_probability, 3),
                'ml_prediction': ml_prediction,
                'action': action,
                'current_price': stock['current_price'],
                'reasons': stock['reasons'],
                'volume_ratio': stock['volume_ratio'],
                'momentum_5d': stock['momentum_5d'],
                'rsi': stock['rsi']
            }
            
            # Add price estimates and confidence if available from ML analysis
            if ml_result:
                if 'estimated_high_10d' in ml_result:
                    recommendation['estimated_high_10d'] = ml_result['estimated_high_10d']
                if 'estimated_low_10d' in ml_result:
                    recommendation['estimated_low_10d'] = ml_result['estimated_low_10d']
                if 'potential_gain_10d' in ml_result:
                    recommendation['potential_gain_10d'] = ml_result['potential_gain_10d']
                if 'potential_loss_10d' in ml_result:
                    recommendation['potential_loss_10d'] = ml_result['potential_loss_10d']
                if 'high_confidence' in ml_result:
                    recommendation['high_confidence'] = ml_result['high_confidence']
                if 'low_confidence' in ml_result:
                    recommendation['low_confidence'] = ml_result['low_confidence']
            
            final_recommendations.append(recommendation)
        
        print(f"\n✅ Analysis completed for {len(final_recommendations)} stocks")
        if ml_success_count > 0:
            print(f"🤖 ML integration successful for {ml_success_count}/{len(final_recommendations)} stocks")
        else:
            print(f"📊 Using technical analysis only (ML integration in development)")
        
        # Sort by final score and return top N
        final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        return final_recommendations[:top_n]
    
    def display_recommendations(self, recommendations):
        """Display final recommendations"""
        if not recommendations:
            print("❌ No recommendations available")
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 TOP {len(recommendations)} CHINESE STOCK RECOMMENDATIONS")
        print(f"{'='*80}")
        
        for i, rec in enumerate(recommendations, 1):
            stock_display = f"{rec['symbol']} ({rec['market']}-shares)"
            if 'stock_name' in rec and rec['stock_name'] != rec['symbol']:
                stock_display += f" - {rec['stock_name']}"
            
            print(f"\n{i}. {stock_display}")
            print(f"   📊 Final Score: {rec['final_score']:.2f}/100")
            print(f"   📈 Technical Score: {rec['technical_score']:.2f}/100")
            print(f"   🤖 ML Probability: {rec['ml_probability']:.3f}")
            print(f"   🤖 ML Prediction: {rec['ml_prediction']}")
            print(f"   💡 Action: {rec['action']}")
            print(f"   💰 Current Price: ¥{rec['current_price']:.2f}")
            
            # Show price estimates and confidence if available
            if 'estimated_high_10d' in rec and 'estimated_low_10d' in rec:
                print(f"   📈 10d Range: ¥{rec['estimated_low_10d']:.2f} - ¥{rec['estimated_high_10d']:.2f}")
                print(f"   📊 Gain: {rec['potential_gain_10d']:.1%} | Loss: {rec['potential_loss_10d']:.1%}")
                
                # Show confidence with emoji
                if 'high_confidence' in rec and 'low_confidence' in rec:
                    avg_confidence = (rec['high_confidence'] + rec['low_confidence']) / 2
                    if avg_confidence >= 80:
                        conf_emoji = "🟢"
                    elif avg_confidence >= 70:
                        conf_emoji = "🟡"
                    elif avg_confidence >= 60:
                        conf_emoji = "🟠"
                    else:
                        conf_emoji = "🔴"
                    print(f"   🎯 Price Confidence: {conf_emoji} {avg_confidence:.0f}%")
            
            print(f"   📊 Volume Ratio: {rec['volume_ratio']:.2f}")
            print(f"   📈 5-day Momentum: {rec['momentum_5d']:.2%}")
            print(f"   📊 RSI: {rec['rsi']:.1f}")
            
            if rec['reasons']:
                print(f"   ✅ Key Strengths:")
                for reason in rec['reasons'][:3]:  # Show top 3 reasons
                    print(f"      • {reason}")
            
            print(f"   {'-'*60}")
        
        print(f"\n📊 Summary:")
        strong_buy = len([r for r in recommendations if r['action'] == 'STRONG BUY'])
        buy = len([r for r in recommendations if r['action'] == 'BUY'])
        hold = len([r for r in recommendations if r['action'] == 'HOLD'])
        
        print(f"   🚀 Strong Buy: {strong_buy}")
        print(f"   📈 Buy: {buy}")
        print(f"   ⏸️  Hold: {hold}")
        
        if strong_buy > 0:
            print(f"\n🎯 RECOMMENDED ACTION: Focus on STRONG BUY stocks for best potential returns!")
        elif buy > 0:
            print(f"\n📈 RECOMMENDED ACTION: Consider BUY stocks with proper risk management!")
        else:
            print(f"\n⚠️  RECOMMENDED ACTION: Market conditions may not be optimal. Consider waiting for better opportunities.")
    
    def run_recommendation_analysis(self, strategy_type='1'):
        """Run complete recommendation analysis"""
        print(f"\n🚀 Starting Chinese Stock Recommendation Analysis")
        print(f"📊 Strategy: {self.strategies[strategy_type]['name']}")
        print(f"📝 Description: {self.strategies[strategy_type]['description']}")
        
        # Clear expired cache
        self.cache.clear_expired_cache()
        
        # Get final recommendations
        recommendations = self.get_final_recommendations(strategy_type, top_n=5)
        
        # Display results
        self.display_recommendations(recommendations)
        
        return recommendations
