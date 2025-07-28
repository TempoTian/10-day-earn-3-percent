#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Usage Example for 10Day-Earn-3%
=====================================

This example demonstrates how to use the stock analysis system
for basic stock analysis and recommendations.
"""

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from us_stock_analyzer import EnhancedStockAnalyzer
from chinese_stock_analyzer import ChineseStockAnalyzer

def analyze_us_stocks():
    """Example: Analyze US stocks"""
    print("🇺🇸 US Stock Analysis Example")
    print("=" * 50)

    analyzer = EnhancedStockAnalyzer()

    # Analyze 3 popular US stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT']

    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        result = analyzer.analyze_stock_with_backtest(symbol)

        if result:
            print(f"✅ {symbol}: {result['recommendation']}")
            print(f"   Success Rate: {result['backtest_results']['Success_Rate']:.1%}")
            print(f"   Potential Gain: {result['potential_gain_10d']:.2%}")
            print(f"   Current Price: ${result['current_price']:.2f}")
        else:
            print(f"❌ Failed to analyze {symbol}")

def analyze_chinese_stocks():
    """Example: Analyze Chinese stocks"""
    print("\n🇨🇳 Chinese Stock Analysis Example")
    print("=" * 50)

    analyzer = ChineseStockAnalyzer()

    # Analyze 3 Chinese A-shares
    stocks = [
        {'symbol': '000001', 'market': 'A'},  # Ping An Bank
        {'symbol': '000002', 'market': 'A'},  # China Vanke
        {'symbol': '000858', 'market': 'A'}   # Wuliangye
    ]

    try:
        results = analyzer.compare_chinese_stocks(stocks)

        if results:
            print(f"\n🎯 Best Recommendation: {results['symbol']}")
            print(f"   Score: {results['score']}/100")
            if results['ml_probability'] is not None:
                print(f"   ML Probability: {results['ml_probability']:.3f}")
            print(f"   Recommendation: {results['recommendation']}")
        else:
            print("❌ No results generated for Chinese stocks")
    except Exception as e:
        print(f"❌ Error analyzing Chinese stocks: {str(e)}")
        print("This might be due to data availability issues with Chinese stocks.")

def main():
    """Main example function"""
    print("🚀 10Day-Earn-3% - Basic Usage Examples")
    print("=" * 60)

    # US Stock Analysis
    analyze_us_stocks()

    # Chinese Stock Analysis
    analyze_chinese_stocks()

    print("\n✅ Examples completed!")
    print("For more advanced features, run: python main.py")

if __name__ == "__main__":
    main()