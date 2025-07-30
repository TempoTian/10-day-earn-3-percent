#!/usr/bin/env python3
"""
持仓诊断 (Portfolio Diagnosis) System
Analyzes current stock holdings and provides sell/hold signals for A-shares
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from chinese_stock_analyzer import ChineseStockAnalyzer
from chinese_stock_downloader import ChineseStockDownloader

class PortfolioDiagnosis:
    def __init__(self, data_source='akshare'):
        """
        Initialize Portfolio Diagnosis System
        data_source: 'yfinance' or 'akshare'
        """
        self.analyzer = ChineseStockAnalyzer(data_source)
        self.downloader = ChineseStockDownloader(data_source)
        self.data_source = data_source
    
    def get_stock_name(self, symbol):
        """Get Chinese stock name for display"""
        # Then try to get from downloader
        try:
            return self.downloader.get_stock_name(symbol, 'A')
        except:
            return symbol
    
    def analyze_holding(self, symbol, buy_price=None, buy_date=None, shares=None):
        """
        Analyze a single stock holding
        Returns analysis result with sell/hold recommendation
        """
        print(f"📊 Analyzing {symbol}...")
        
        try:
            # Add delay to avoid server resistance
            time.sleep(0.5)
            
            # Download data and analyze
            download_success, stock_name = self.analyzer.download_chinese_stock_data(symbol, 'A')
            if not download_success:
                return {
                    'symbol': symbol,
                    'stock_name': symbol,
                    'status': 'ERROR',
                    'error': 'Failed to download data',
                    'recommendation': 'UNKNOWN',
                    'score': 0
                }
            
            # Calculate indicators
            self.analyzer.calculate_chinese_indicators()
            
            # Get current price and basic info
            current = self.analyzer.data.iloc[-1]
            current_price = current['close']
            
            # Calculate return if buy price provided
            current_return = None
            if buy_price:
                current_return = (current_price - buy_price) / buy_price
            
            # Get technical score
            technical_score = self.analyzer.calculate_chinese_technical_score()
            
            # Train or load ML model and get prediction
            ml_prediction, ml_probability = None, None
            ml_score = 0
            
            try:
                # Try to load existing model first
                model_loaded = self.analyzer.load_model(symbol, 'A')
                
                if not model_loaded:
                    # Train new model if loading fails
                    print(f"   🤖 Training ML model for {symbol}...")
                    model_trained = self.analyzer.train_ml_model(holding_period=10, profit_threshold=0.03)
                    if model_trained:
                        # Save the trained model
                        self.analyzer.save_model(symbol, 'A')
                
                # Get ML prediction
                if self.analyzer.model is not None:
                    ml_prediction, ml_probability = self.analyzer.get_ml_prediction()
                    if ml_probability is not None:
                        ml_score = int(ml_probability * 100)
                    else:
                        ml_score = 0
                else:
                    ml_score = 0
                    
            except Exception as ml_error:
                print(f"   ⚠️  ML analysis failed for {symbol}: {str(ml_error)}")
                ml_score = 0
                ml_probability = None
                ml_prediction = None
            
            # Calculate combined score (70% technical + 30% ML)
            if ml_probability is not None and ml_score > 0:
                combined_score = int(technical_score * 0.7 + ml_score * 0.3)
            else:
                combined_score = technical_score
                ml_score = 0
            
            # Determine recommendation based on score and return
            recommendation = self._determine_recommendation(combined_score, current_return, ml_probability)
            
            # Calculate price estimates
            estimated_high_10d = self.analyzer.estimate_highest_price_10_days(symbol, current_price, 'A')
            estimated_low_10d = self.analyzer.estimate_lowest_price_10_days(symbol, current_price, 'A')
            
            # Calculate confidence
            high_confidence, high_reasoning = self.analyzer.calculate_ml_price_confidence(estimated_high_10d, current_price, 'high')
            low_confidence, low_reasoning = self.analyzer.calculate_ml_price_confidence(estimated_low_10d, current_price, 'low')
            
            # Calculate potential gains/losses
            potential_gain_10d = (estimated_high_10d - current_price) / current_price
            potential_loss_10d = (estimated_low_10d - current_price) / current_price
            
            # Calculate position value if shares provided
            position_value = None
            if shares and current_price:
                position_value = shares * current_price
            
            return {
                'symbol': symbol,
                'stock_name': stock_name,
                'status': 'SUCCESS',
                'current_price': current_price,
                'buy_price': buy_price,
                'current_return': current_return,
                'shares': shares,
                'position_value': position_value,
                'technical_score': technical_score,
                'ml_score': ml_score,
                'ml_probability': ml_probability,
                'ml_prediction': ml_prediction,
                'combined_score': combined_score,
                'recommendation': recommendation,
                'estimated_high_10d': estimated_high_10d,
                'estimated_low_10d': estimated_low_10d,
                'potential_gain_10d': potential_gain_10d,
                'potential_loss_10d': potential_loss_10d,
                'high_confidence': high_confidence,
                'low_confidence': low_confidence,
                'volume_ratio': current['Volume_Ratio'],
                'momentum_5d': current['Price_Momentum_5'],
                'rsi': current['RSI'],
                'volatility': current['Volatility_20'],
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'stock_name': symbol,
                'status': 'ERROR',
                'error': str(e),
                'recommendation': 'UNKNOWN',
                'score': 0
            }
    
    def _determine_recommendation(self, combined_score, current_return, ml_probability):
        """
        Determine sell/hold recommendation based on score and return
        """
        # Base recommendation on combined score
        if combined_score >= 80:
            base_recommendation = "STRONG HOLD"
        elif combined_score >= 70:
            base_recommendation = "HOLD"
        elif combined_score >= 60:
            base_recommendation = "WEAK HOLD"
        elif combined_score >= 50:
            base_recommendation = "CONSIDER SELL"
        else:
            base_recommendation = "SELL"
        
        # Adjust based on current return
        if current_return is not None:
            if current_return >= 0.10:  # 10%+ profit
                if combined_score < 60:
                    return "TAKE PROFIT"
                else:
                    return "HOLD (High Profit)"
            elif current_return >= 0.05:  # 5%+ profit
                if combined_score < 50:
                    return "TAKE PROFIT"
                else:
                    return base_recommendation
            elif current_return <= -0.05:  # 5%+ loss
                if combined_score < 40:
                    return "STOP LOSS"
                else:
                    return "HOLD (Cut Loss)"
            elif current_return <= -0.03:  # 3%+ loss
                if combined_score < 30:
                    return "STOP LOSS"
                else:
                    return "HOLD (Monitor)"
        
        # Adjust based on ML probability
        if ml_probability is not None:
            if ml_probability < 0.3 and combined_score < 60:
                return "SELL (ML Bearish)"
            elif ml_probability > 0.7 and combined_score > 70:
                return "HOLD (ML Bullish)"
        
        return base_recommendation
    
    def analyze_portfolio(self, holdings_list):
        """
        Analyze entire portfolio
        holdings_list: List of dicts with keys: symbol, buy_price (optional), buy_date (optional), shares (optional)
        """
        print(f"\n{'='*80}")
        print(f"📊 持仓诊断 (PORTFOLIO DIAGNOSIS)")
        print(f"{'='*80}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 Data Source: {self.data_source}")
        print(f"🎯 Total Holdings: {len(holdings_list)}")
        print(f"{'='*80}")
        
        results = []
        total_position_value = 0
        total_profit_loss = 0
        
        for i, holding in enumerate(holdings_list, 1):
            symbol = holding['symbol']
            buy_price = holding.get('buy_price')
            buy_date = holding.get('buy_date')
            shares = holding.get('shares')
            
            print(f"\n📊 Analyzing {i}/{len(holdings_list)}: {symbol}")
            
            # Analyze the holding
            result = self.analyze_holding(symbol, buy_price, buy_date, shares)
            results.append(result)
            
            # Calculate portfolio totals
            if result['status'] == 'SUCCESS' and result['position_value']:
                total_position_value += result['position_value']
                if result['current_return']:
                    total_profit_loss += result['position_value'] * result['current_return']
        
        # Sort results by recommendation priority
        recommendation_priority = {
            'STOP LOSS': 1,
            'TAKE PROFIT': 2,
            'SELL': 3,
            'SELL (ML Bearish)': 4,
            'CONSIDER SELL': 5,
            'WEAK HOLD': 6,
            'HOLD': 7,
            'HOLD (High Profit)': 8,
            'HOLD (Cut Loss)': 9,
            'HOLD (Monitor)': 10,
            'HOLD (ML Bullish)': 11,
            'STRONG HOLD': 12,
            'UNKNOWN': 13,
            'ERROR': 14
        }
        
        results.sort(key=lambda x: recommendation_priority.get(x['recommendation'], 99))
        
        # Generate summary
        summary = self._generate_summary(results, total_position_value, total_profit_loss)
        
        # Save results to file
        self._save_results(results, summary)
        
        # Display results
        self._display_results(results, summary)
        
        return results, summary
    
    def _generate_summary(self, results, total_position_value, total_profit_loss):
        """Generate portfolio summary"""
        successful_analyses = [r for r in results if r['status'] == 'SUCCESS']
        error_analyses = [r for r in results if r['status'] == 'ERROR']
        
        # Count recommendations
        recommendation_counts = {}
        for result in successful_analyses:
            rec = result['recommendation']
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Calculate average scores
        avg_technical_score = np.mean([r['technical_score'] for r in successful_analyses]) if successful_analyses else 0
        avg_combined_score = np.mean([r['combined_score'] for r in successful_analyses]) if successful_analyses else 0
        
        # Calculate portfolio return
        portfolio_return = (total_profit_loss / total_position_value * 100) if total_position_value > 0 else 0
        
        return {
            'total_holdings': len(results),
            'successful_analyses': len(successful_analyses),
            'error_analyses': len(error_analyses),
            'total_position_value': total_position_value,
            'total_profit_loss': total_profit_loss,
            'portfolio_return': portfolio_return,
            'recommendation_counts': recommendation_counts,
            'avg_technical_score': avg_technical_score,
            'avg_combined_score': avg_combined_score,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _save_results(self, results, summary):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"portfolio_diagnosis_{timestamp}.json"
        
        output_data = {
            'summary': summary,
            'holdings': results,
            'analysis_info': {
                'data_source': self.data_source,
                'analysis_date': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n✅ Results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {str(e)}")
    
    def _display_results(self, results, summary):
        """Display analysis results"""
        print(f"\n{'='*80}")
        print(f"📊 PORTFOLIO DIAGNOSIS RESULTS")
        print(f"{'='*80}")
        
        # Display summary
        print(f"\n📈 PORTFOLIO SUMMARY:")
        print(f"   📊 Total Holdings: {summary['total_holdings']}")
        print(f"   ✅ Successful Analyses: {summary['successful_analyses']}")
        print(f"   ❌ Errors: {summary['error_analyses']}")
        print(f"   💰 Total Position Value: ¥{summary['total_position_value']:,.2f}")
        print(f"   📈 Total P&L: ¥{summary['total_profit_loss']:,.2f}")
        print(f"   📊 Portfolio Return: {summary['portfolio_return']:.2f}%")
        print(f"   📊 Average Technical Score: {summary['avg_technical_score']:.1f}/100")
        print(f"   📊 Average Combined Score: {summary['avg_combined_score']:.1f}/100")
        
        # Display recommendation counts
        print(f"\n🎯 RECOMMENDATION SUMMARY:")
        for rec, count in summary['recommendation_counts'].items():
            print(f"   {rec}: {count}")
        
        # Display individual holdings
        print(f"\n{'='*80}")
        print(f"📊 INDIVIDUAL HOLDINGS ANALYSIS")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            if result['status'] == 'SUCCESS':
                print(f"\n{i}. {result['symbol']} - {result['stock_name']}")
                print(f"   💰 Current Price: ¥{result['current_price']:.2f}")
                
                if result['buy_price']:
                    print(f"   📈 Buy Price: ¥{result['buy_price']:.2f}")
                    print(f"   📊 Return: {result['current_return']:.2%}")
                
                if result['position_value']:
                    print(f"   💼 Position Value: ¥{result['position_value']:,.2f}")
                
                print(f"   📊 Technical Score: {result['technical_score']:.1f}/100")
                print(f"   🤖 ML Score: {result['ml_score']:.1f}/100")
                print(f"   🎯 Combined Score: {result['combined_score']:.1f}/100")
                print(f"   💡 Recommendation: {result['recommendation']}")
                
                if result['estimated_high_10d'] and result['estimated_low_10d']:
                    print(f"   📈 10d Range: ¥{result['estimated_low_10d']:.2f} - ¥{result['estimated_high_10d']:.2f}")
                    print(f"   📊 Potential Gain: {result['potential_gain_10d']:.2%}")
                    print(f"   📊 Potential Loss: {result['potential_loss_10d']:.2%}")
                
                print(f"   📊 Volume Ratio: {result['volume_ratio']:.2f}")
                print(f"   📈 5-day Momentum: {result['momentum_5d']:.2%}")
                print(f"   📊 RSI: {result['rsi']:.1f}")
                
            else:
                print(f"\n{i}. {result['symbol']} - ERROR")
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
            
            print(f"   {'-'*60}")
        
        # Display action items
        print(f"\n{'='*80}")
        print(f"🎯 ACTION ITEMS")
        print(f"{'='*80}")
        
        sell_stocks = [r for r in results if 'SELL' in r.get('recommendation', '') and r['status'] == 'SUCCESS']
        take_profit_stocks = [r for r in results if 'TAKE PROFIT' in r.get('recommendation', '') and r['status'] == 'SUCCESS']
        stop_loss_stocks = [r for r in results if 'STOP LOSS' in r.get('recommendation', '') and r['status'] == 'SUCCESS']
        
        if sell_stocks:
            print(f"\n🚨 IMMEDIATE ACTION REQUIRED:")
            for stock in sell_stocks:
                print(f"   • SELL {stock['symbol']} ({stock['stock_name']}) - {stock['recommendation']}")
        
        if take_profit_stocks:
            print(f"\n💰 CONSIDER TAKING PROFITS:")
            for stock in take_profit_stocks:
                print(f"   • {stock['symbol']} ({stock['stock_name']}) - Current Return: {stock['current_return']:.2%}")
        
        if stop_loss_stocks:
            print(f"\n⚠️  STOP LOSS ALERTS:")
            for stock in stop_loss_stocks:
                print(f"   • {stock['symbol']} ({stock['stock_name']}) - Current Return: {stock['current_return']:.2%}")
        
        if not (sell_stocks or take_profit_stocks or stop_loss_stocks):
            print(f"\n✅ No immediate action required. Portfolio appears stable.")

def test_portfolio_diagnosis():
    """Test function for portfolio diagnosis with custom holdings"""
    print("🚀 持仓诊断 (Portfolio Diagnosis) Test")
    print("=" * 60)
    
    # Example holdings list - MODIFY THIS WITH YOUR ACTUAL HOLDINGS
    holdings_list = [
        {'symbol': '000001', 'buy_price': 12.5, 'shares': 500},   # 平安银行
        {'symbol': '600519', 'buy_price': 1800.0, 'shares': 10},  # 贵州茅台
        {'symbol': '000002', 'buy_price': 18.0, 'shares': 200},   # 万科A
        {'symbol': '000858', 'buy_price': 180.0, 'shares': 50},   # 五粮液
    ]
    
    print("📋 Current test holdings:")
    for i, holding in enumerate(holdings_list, 1):
        print(f"   {i}. {holding['symbol']} - {holding['shares']} shares @ ¥{holding['buy_price']:.2f}")
    
    print(f"\n💡 To use your own holdings, modify the 'holdings_list' in this function")
    print(f"💡 Format: {{'symbol': 'STOCK_CODE', 'buy_price': PRICE, 'shares': SHARES}}")
    print(f"💡 Optional: add 'buy_date': 'YYYY-MM-DD' if you want to track holding period")
    
    # Initialize diagnosis system
    diagnosis = PortfolioDiagnosis(data_source='akshare')  # or 'yfinance'
    
    # Run analysis
    results, summary = diagnosis.analyze_portfolio(holdings_list)
    
    print(f"\n✅ Portfolio diagnosis completed!")
    print(f"📊 Check the generated JSON file for detailed results.")

def create_holdings_from_array(stock_symbols, buy_prices=None, shares=None):
    """
    Create holdings list from arrays
    stock_symbols: list of stock symbols
    buy_prices: list of buy prices (optional, will use None if not provided)
    shares: list of share counts (optional, will use 100 if not provided)
    """
    holdings_list = []
    
    for i, symbol in enumerate(stock_symbols):
        holding = {'symbol': symbol}
        
        if buy_prices and i < len(buy_prices):
            holding['buy_price'] = buy_prices[i]
        
        if shares and i < len(shares):
            holding['shares'] = shares[i]
        else:
            holding['shares'] = 100  # Default to 100 shares
        
        holdings_list.append(holding)
    
    return holdings_list

def analyze_custom_holdings(stock_symbols, buy_prices=None, shares=None, data_source='akshare'):
    """
    Analyze custom holdings from arrays
    """
    print("🚀 持仓诊断 (Portfolio Diagnosis) - Custom Holdings")
    print("=" * 60)
    
    # Create holdings list from arrays
    holdings_list = create_holdings_from_array(stock_symbols, buy_prices, shares)
    
    print(f"📋 Analyzing {len(holdings_list)} holdings:")
    for i, holding in enumerate(holdings_list, 1):
        price_info = f" @ ¥{holding['buy_price']:.2f}" if holding.get('buy_price') else " (no buy price)"
        print(f"   {i}. {holding['symbol']} - {holding['shares']} shares{price_info}")
    
    # Initialize diagnosis system
    diagnosis = PortfolioDiagnosis(data_source=data_source)
    
    # Run analysis
    results, summary = diagnosis.analyze_portfolio(holdings_list)
    
    return results, summary

if __name__ == "__main__":
    # You can choose which function to run:
    
    # Option 1: Run with example holdings
    test_portfolio_diagnosis()
    
    # Option 2: Run with custom arrays (uncomment and modify as needed)
    # stock_symbols = ['688018', '000001', '600519', '000002', '000858']
    # buy_prices = [150.0, 12.5, 1800.0, 18.0, 180.0]
    # shares = [100, 500, 10, 200, 50]
    # results, summary = analyze_custom_holdings(stock_symbols, buy_prices, shares, 'akshare') 