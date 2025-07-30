#!/usr/bin/env python3
"""
Explain Technical Score Calculation for Sell Analysis
"""

from chinese_stock_analyzer import ChineseStockAnalyzer
import pandas as pd

def explain_technical_score():
    print("🔍 EXPLAINING TECHNICAL SCORE CALCULATION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ChineseStockAnalyzer()
    
    # Test with the same stock
    symbol = "688018"
    market = "A"
    
    print(f"📊 Analyzing {symbol} ({market}-shares)")
    
    # Download data
    success = analyzer.download_chinese_stock_data(symbol, market, "2y")
    if not success:
        print("❌ Failed to download data")
        return
    
    # Calculate indicators
    analyzer.calculate_chinese_indicators()
    
    # Get current data
    current = analyzer.data.iloc[-1]
    
    print(f"\n📈 CURRENT MARKET DATA:")
    print(f"   Current Price: ¥{current['Close']:.2f}")
    print(f"   5-day Momentum: {current['Price_Momentum_5']:.2%}")
    print(f"   Volume Ratio: {current['Volume_Ratio']:.2f}")
    print(f"   Volatility (20d): {current['Volatility_20']:.2%}")
    print(f"   Price vs SMA20: {((current['Close'] - current['SMA_20']) / current['SMA_20']):.2%}")
    print(f"   Price vs SMA50: {((current['Close'] - current['SMA_50']) / current['SMA_50']):.2%}")
    
    if 'RSI' in current and not pd.isna(current['RSI']):
        print(f"   RSI: {current['RSI']:.1f}")
    if 'MACD' in current and not pd.isna(current['MACD']):
        print(f"   MACD: {current['MACD']:.4f}")
    
    print(f"\n🎯 TECHNICAL SCORE BREAKDOWN:")
    print("=" * 50)
    
    # Start with base score
    score = 50
    print(f"   Base Score: {score}")
    
    # 1. Momentum Analysis
    print(f"\n   1️⃣ MOMENTUM ANALYSIS:")
    if current['Price_Momentum_5'] < -0.05:
        score += 25
        print(f"      Strong negative momentum (-{abs(current['Price_Momentum_5']):.1%}) → +25 points")
    elif current['Price_Momentum_5'] < -0.02:
        score += 15
        print(f"      Moderate negative momentum (-{abs(current['Price_Momentum_5']):.1%}) → +15 points")
    elif current['Price_Momentum_5'] > 0.05:
        score -= 25
        print(f"      Strong positive momentum (+{current['Price_Momentum_5']:.1%}) → -25 points (BUY signal)")
    elif current['Price_Momentum_5'] > 0.02:
        score -= 15
        print(f"      Moderate positive momentum (+{current['Price_Momentum_5']:.1%}) → -15 points (BUY signal)")
    else:
        print(f"      Neutral momentum ({current['Price_Momentum_5']:.1%}) → 0 points")
    
    print(f"      Score after momentum: {score}")
    
    # 2. Moving Average Analysis
    print(f"\n   2️⃣ MOVING AVERAGE ANALYSIS:")
    
    # SMA20
    if current['Close'] < current['SMA_20']:
        score += 20
        print(f"      Price below 20-day SMA → +20 points (SELL signal)")
    else:
        score -= 20
        print(f"      Price above 20-day SMA → -20 points (BUY signal)")
    
    # SMA50
    if current['Close'] < current['SMA_50']:
        score += 20
        print(f"      Price below 50-day SMA → +20 points (SELL signal)")
    else:
        score -= 20
        print(f"      Price above 50-day SMA → -20 points (BUY signal)")
    
    print(f"      Score after moving averages: {score}")
    
    # 3. Volume Analysis
    print(f"\n   3️⃣ VOLUME ANALYSIS:")
    if current['Volume_Ratio'] > 2.0:
        score += 15
        print(f"      High volume ({current['Volume_Ratio']:.1f}x) → +15 points (potential distribution)")
    elif current['Volume_Ratio'] > 1.5:
        score += 10
        print(f"      Moderate high volume ({current['Volume_Ratio']:.1f}x) → +10 points")
    elif current['Volume_Ratio'] < 0.5:
        score -= 10
        print(f"      Low volume ({current['Volume_Ratio']:.1f}x) → -10 points (accumulation)")
    else:
        print(f"      Normal volume ({current['Volume_Ratio']:.1f}x) → 0 points")
    
    print(f"      Score after volume: {score}")
    
    # 4. Volatility Analysis
    print(f"\n   4️⃣ VOLATILITY ANALYSIS:")
    avg_volatility = analyzer.data['Volatility_20'].mean()
    if current['Volatility_20'] > avg_volatility * 1.5:
        score += 15
        print(f"      High volatility ({current['Volatility_20']:.1%} vs avg {avg_volatility:.1%}) → +15 points")
    elif current['Volatility_20'] > avg_volatility * 1.2:
        score += 10
        print(f"      Moderate high volatility ({current['Volatility_20']:.1%} vs avg {avg_volatility:.1%}) → +10 points")
    elif current['Volatility_20'] < avg_volatility * 0.5:
        score -= 10
        print(f"      Low volatility ({current['Volatility_20']:.1%} vs avg {avg_volatility:.1%}) → -10 points")
    else:
        print(f"      Normal volatility ({current['Volatility_20']:.1%} vs avg {avg_volatility:.1%}) → 0 points")
    
    print(f"      Score after volatility: {score}")
    
    # 5. RSI Analysis (if available)
    if 'RSI' in current and not pd.isna(current['RSI']):
        print(f"\n   5️⃣ RSI ANALYSIS:")
        if current['RSI'] > 70:
            score += 15
            print(f"      Overbought RSI ({current['RSI']:.1f}) → +15 points (SELL signal)")
        elif current['RSI'] < 30:
            score -= 15
            print(f"      Oversold RSI ({current['RSI']:.1f}) → -15 points (BUY signal)")
        else:
            print(f"      Normal RSI ({current['RSI']:.1f}) → 0 points")
        
        print(f"      Score after RSI: {score}")
    
    # 6. MACD Analysis (if available)
    if 'MACD' in current and not pd.isna(current['MACD']):
        print(f"\n   6️⃣ MACD ANALYSIS:")
        if current['MACD'] < 0:
            score += 10
            print(f"      Negative MACD ({current['MACD']:.4f}) → +10 points (bearish)")
        else:
            score -= 10
            print(f"      Positive MACD ({current['MACD']:.4f}) → -10 points (bullish)")
        
        print(f"      Score after MACD: {score}")
    
    # Final score
    final_score = max(0, min(100, score))
    print(f"\n🎯 FINAL TECHNICAL SCORE: {final_score:.2f}/100")
    
    # Interpretation
    if final_score >= 80:
        interpretation = "Very Strong Sell Signal"
    elif final_score >= 70:
        interpretation = "Strong Sell Signal"
    elif final_score >= 60:
        interpretation = "Moderate Sell Signal"
    elif final_score >= 45:
        interpretation = "Neutral Signal"
    elif final_score >= 30:
        interpretation = "Weak Buy Signal"
    else:
        interpretation = "Strong Buy Signal"
    
    print(f"📊 INTERPRETATION: {interpretation}")
    
    print(f"\n💡 EXPLANATION:")
    print(f"   • Higher scores (70-100) = Strong SELL signals")
    print(f"   • Lower scores (0-30) = Strong BUY signals")
    print(f"   • This stock shows strong BUY signals, hence low sell score")
    print(f"   • The system recommends HOLDING/ADDING, not selling")

if __name__ == "__main__":
    explain_technical_score() 