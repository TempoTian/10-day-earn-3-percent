#!/usr/bin/env python3
"""
Analyze Technical Score vs ML Probability Discrepancy
"""

from chinese_stock_analyzer import ChineseStockAnalyzer
import pandas as pd
import numpy as np

def analyze_score_discrepancy():
    print("🔍 ANALYZING TECHNICAL SCORE vs ML PROBABILITY DISCREPANCY")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ChineseStockAnalyzer()
    
    # Test with a stock that showed high technical score but low ML probability
    symbol = "002796"  # Example from your test
    market = "A"
    
    print(f"📊 Analyzing {symbol} ({market}-shares) for score discrepancy")
    
    # Download data
    success = analyzer.download_chinese_stock_data(symbol, market, "2y")
    if not success:
        print("❌ Failed to download data")
        return
    
    # Calculate indicators
    analyzer.calculate_chinese_indicators()
    
    # Load or train ML model
    if not analyzer.load_model(symbol, market):
        print("🔄 Training ML model...")
        analyzer.train_ml_model(holding_period=10, profit_threshold=0.03)
    
    # Get current data
    current = analyzer.data.iloc[-1]
    
    print(f"\n📈 CURRENT MARKET DATA:")
    print(f"   Current Price: ¥{current['Close']:.2f}")
    print(f"   5-day Momentum: {current['Price_Momentum_5']:.2%}")
    print(f"   Volume Ratio: {current['Volume_Ratio']:.2f}")
    print(f"   Volatility (20d): {current['Volatility_20']:.2%}")
    print(f"   Price vs SMA20: {((current['Close'] - current['SMA_20']) / current['SMA_20']):.2%}")
    print(f"   Price vs SMA50: {((current['Close'] - current['SMA_50']) / current['SMA_50']):.2%}")
    
    # Calculate technical score step by step
    print(f"\n🎯 TECHNICAL SCORE CALCULATION:")
    print("=" * 50)
    
    score = 50
    print(f"   Base Score: {score}")
    
    # 1. Momentum scoring
    print(f"\n   1️⃣ MOMENTUM ANALYSIS:")
    if current['Price_Momentum_5'] > 0.05:
        score += 15
        print(f"      ✅ Strong positive momentum (+15): {current['Price_Momentum_5']:.2%}")
    elif current['Price_Momentum_5'] > 0.02:
        score += 10
        print(f"      ✅ Moderate positive momentum (+10): {current['Price_Momentum_5']:.2%}")
    elif current['Price_Momentum_5'] < -0.05:
        score -= 15
        print(f"      ❌ Strong negative momentum (-15): {current['Price_Momentum_5']:.2%}")
    else:
        print(f"      ➖ Neutral momentum (0): {current['Price_Momentum_5']:.2%}")
    
    # 2. Volume scoring
    print(f"\n   2️⃣ VOLUME ANALYSIS:")
    if current['Volume_Ratio'] > 1.5:
        score += 10
        print(f"      ✅ High volume (+10): {current['Volume_Ratio']:.2f}")
    elif current['Volume_Ratio'] < 0.5:
        score -= 5
        print(f"      ❌ Low volume (-5): {current['Volume_Ratio']:.2f}")
    else:
        print(f"      ➖ Normal volume (0): {current['Volume_Ratio']:.2f}")
    
    # 3. Moving average scoring
    print(f"\n   3️⃣ MOVING AVERAGE ANALYSIS:")
    if current['Close'] > current['SMA_20'] > current['SMA_50']:
        score += 15
        print(f"      ✅ Strong uptrend (+15): Price > SMA20 > SMA50")
    elif current['Close'] < current['SMA_20'] < current['SMA_50']:
        score -= 15
        print(f"      ❌ Strong downtrend (-15): Price < SMA20 < SMA50")
    else:
        print(f"      ➖ Mixed signals (0): Price vs MA relationship unclear")
    
    # 4. Volatility scoring
    print(f"\n   4️⃣ VOLATILITY ANALYSIS:")
    avg_volatility = analyzer.data['Volatility_20'].mean()
    if current['Volatility_20'] > avg_volatility * 1.5:
        score -= 10
        print(f"      ❌ High volatility (-10): {current['Volatility_20']:.2%} vs avg {avg_volatility:.2%}")
    elif current['Volatility_20'] < avg_volatility * 0.5:
        score += 5
        print(f"      ✅ Low volatility (+5): {current['Volatility_20']:.2%} vs avg {avg_volatility:.2%}")
    else:
        print(f"      ➖ Normal volatility (0): {current['Volatility_20']:.2%} vs avg {avg_volatility:.2%}")
    
    score = max(0, min(100, score))
    print(f"\n   🎯 FINAL TECHNICAL SCORE: {score:.2f}/100")
    
    # Get ML prediction
    ml_prediction, ml_probability = analyzer.get_ml_prediction()
    
    print(f"\n🤖 ML ANALYSIS:")
    print("=" * 50)
    print(f"   ML Prediction: {ml_prediction} ({'RISE' if ml_prediction == 1 else 'DECLINE'})")
    print(f"   ML Probability: {ml_probability:.3f} ({ml_probability:.1%})")
    
    # Calculate combined score
    if ml_probability is not None:
        combined_score = 0.6 * (score / 100) + 0.4 * ml_probability
        final_score = int(combined_score * 100)
        print(f"   Combined Score: {final_score:.2f}/100 (60% tech + 40% ML)")
    else:
        final_score = score
        print(f"   Combined Score: {final_score:.2f}/100 (technical only)")
    
    print(f"\n🔍 DISCREPANCY ANALYSIS:")
    print("=" * 50)
    
    if ml_probability is not None:
        discrepancy = abs(score - (ml_probability * 100))
        print(f"   Score Difference: {discrepancy:.1f} points")
        
        if discrepancy > 30:
            print(f"   ⚠️  HIGH DISCREPANCY - Technical and ML signals conflict significantly")
        elif discrepancy > 20:
            print(f"   ⚠️  MODERATE DISCREPANCY - Some conflict between signals")
        else:
            print(f"   ✅ LOW DISCREPANCY - Signals are relatively aligned")
        
        print(f"\n💡 WHY THIS HAPPENS:")
        print("   1. Technical Analysis: Based on current market conditions and patterns")
        print("   2. ML Model: Based on historical patterns and probability of future outcomes")
        print("   3. Different Timeframes: Technical = immediate, ML = 10-day prediction")
        print("   4. Different Data Sources: Technical = price/volume, ML = 30+ features")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print("=" * 50)
        
        if score >= 80 and ml_probability < 0.3:
            print(f"   🚨 HIGH TECHNICAL SCORE + LOW ML = CAUTION")
            print(f"      • Technical indicators show strong buy signals")
            print(f"      • ML predicts potential decline")
            print(f"      • RECOMMENDATION: Wait for ML confirmation or use smaller position")
            
        elif score >= 80 and ml_probability > 0.6:
            print(f"   ✅ HIGH TECHNICAL SCORE + HIGH ML = STRONG BUY")
            print(f"      • Both technical and ML agree on bullish outlook")
            print(f"      • RECOMMENDATION: Strong buy signal")
            
        elif score < 50 and ml_probability > 0.7:
            print(f"   ⚠️  LOW TECHNICAL SCORE + HIGH ML = MODERATE BUY")
            print(f"      • Technical indicators are weak")
            print(f"      • ML predicts strong rise")
            print(f"      • RECOMMENDATION: Consider buying with caution")
            
        elif score < 50 and ml_probability < 0.3:
            print(f"   ❌ LOW TECHNICAL SCORE + LOW ML = AVOID")
            print(f"      • Both signals indicate poor prospects")
            print(f"      • RECOMMENDATION: Avoid this stock")
            
        else:
            print(f"   🔄 MIXED SIGNALS = HOLD")
            print(f"      • Technical and ML signals are mixed")
            print(f"      • RECOMMENDATION: Wait for clearer signals")
        
        print(f"\n📊 OPTIMIZED DECISION FRAMEWORK:")
        print("=" * 50)
        print(f"   🎯 STRONG BUY: Technical ≥80 AND ML ≥60%")
        print(f"   📈 BUY: Technical ≥65 AND ML ≥50%")
        print(f"   ⏸️  HOLD: Mixed signals or moderate scores")
        print(f"   ❌ AVOID: Technical <50 AND ML <30%")
        print(f"   🚨 CAUTION: High technical + Low ML (wait for confirmation)")
        
        print(f"\n💡 WHEN TO RELY MORE ON TECHNICAL SCORE:")
        print("   • Short-term trading (1-3 days)")
        print("   • High-volume, liquid stocks")
        print("   • Clear technical patterns")
        print("   • Market momentum is strong")
        
        print(f"\n💡 WHEN TO RELY MORE ON ML PROBABILITY:")
        print("   • Medium-term holding (1-2 weeks)")
        print("   • Volatile or illiquid stocks")
        print("   • Complex market conditions")
        print("   • ML model has high accuracy (>80%)")
        
    else:
        print(f"   ❌ ML prediction not available - relying on technical score only")
        print(f"   💡 RECOMMENDATION: Use technical score with caution")
    
    print(f"\n✅ Analysis completed!")

if __name__ == "__main__":
    analyze_score_discrepancy() 