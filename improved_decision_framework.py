#!/usr/bin/env python3
"""
Improved Decision Framework with Dynamic Weighting
"""

from chinese_stock_analyzer import ChineseStockAnalyzer
import pandas as pd
import numpy as np

def calculate_dynamic_weights(technical_score, ml_probability, market_conditions):
    """
    Calculate dynamic weights based on signal strength and market conditions
    """
    # Base weights
    tech_weight = 0.6
    ml_weight = 0.4
    
    # Adjust based on signal strength
    if technical_score >= 80:
        tech_weight += 0.1  # Increase technical weight for strong signals
        ml_weight -= 0.1
    elif technical_score <= 30:
        tech_weight -= 0.1  # Decrease technical weight for weak signals
        ml_weight += 0.1
    
    if ml_probability >= 0.7:
        ml_weight += 0.1  # Increase ML weight for strong predictions
        tech_weight -= 0.1
    elif ml_probability <= 0.3:
        ml_weight -= 0.1  # Decrease ML weight for weak predictions
        tech_weight += 0.1
    
    # Adjust based on market conditions
    if market_conditions == 'volatile':
        ml_weight += 0.1  # ML better in volatile markets
        tech_weight -= 0.1
    elif market_conditions == 'trending':
        tech_weight += 0.1  # Technical better in trending markets
        ml_weight -= 0.1
    
    # Ensure weights sum to 1
    total_weight = tech_weight + ml_weight
    tech_weight /= total_weight
    ml_weight /= total_weight
    
    return tech_weight, ml_weight

def assess_market_conditions(analyzer):
    """
    Assess current market conditions
    """
    if analyzer.data is None or len(analyzer.data) < 50:
        return 'normal'
    
    current = analyzer.data.iloc[-1]
    
    # Calculate volatility
    volatility = current['Volatility_20']
    avg_volatility = analyzer.data['Volatility_20'].mean()
    
    # Calculate trend strength
    trend_strength = abs(current['SMA_20'] - current['SMA_50']) / current['SMA_50']
    
    # Determine market condition
    if volatility > avg_volatility * 1.5:
        return 'volatile'
    elif trend_strength > 0.05 and abs(current['Price_Momentum_20']) > 0.1:
        return 'trending'
    else:
        return 'normal'

def improved_decision_framework():
    print("🎯 IMPROVED DECISION FRAMEWORK WITH DYNAMIC WEIGHTING")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ChineseStockAnalyzer()
    
    # Test stocks
    test_stocks = [
        {"symbol": "002796", "market": "A", "name": "High Tech Score, Low ML"},
        {"symbol": "600276", "market": "A", "name": "Balanced Signals"},
        {"symbol": "000063", "market": "A", "name": "High ML, Moderate Tech"}
    ]
    
    for stock in test_stocks:
        print(f"\n📊 ANALYZING: {stock['symbol']} - {stock['name']}")
        print("-" * 60)
        
        # Download data
        success = analyzer.download_chinese_stock_data(stock['symbol'], stock['market'], "2y")
        if not success:
            print("❌ Failed to download data")
            continue
        
        # Calculate indicators
        analyzer.calculate_chinese_indicators()
        
        # Load or train ML model
        if not analyzer.load_model(stock['symbol'], stock['market']):
            print("🔄 Training ML model...")
            analyzer.train_ml_model(holding_period=10, profit_threshold=0.03)
        
        # Get analysis result
        result = analyzer.analyze_chinese_stock(stock['symbol'], stock['market'])
        
        if result:
            technical_score = result['technical_score']
            ml_probability = result['ml_probability']
            
            # Assess market conditions
            market_conditions = assess_market_conditions(analyzer)
            
            print(f"📈 Market Conditions: {market_conditions.upper()}")
            print(f"📊 Technical Score: {technical_score:.2f}/100")
            print(f"🤖 ML Probability: {ml_probability:.3f} ({ml_probability:.1%})")
            
            if ml_probability is not None:
                # Calculate dynamic weights
                tech_weight, ml_weight = calculate_dynamic_weights(
                    technical_score, ml_probability, market_conditions
                )
                
                # Calculate improved combined score
                improved_score = tech_weight * (technical_score / 100) + ml_weight * ml_probability
                final_score = int(improved_score * 100)
                
                print(f"\n⚖️  DYNAMIC WEIGHTING:")
                print(f"   Technical Weight: {tech_weight:.1%}")
                print(f"   ML Weight: {ml_weight:.1%}")
                print(f"   Improved Score: {final_score:.2f}/100")
                
                # Generate recommendation based on improved framework
                print(f"\n🎯 IMPROVED RECOMMENDATION:")
                
                if final_score >= 75 and technical_score >= 70 and ml_probability >= 0.6:
                    print(f"   🚀 STRONG BUY - High confidence in both signals")
                    print(f"   💡 Action: Buy with full position")
                    
                elif final_score >= 65 and (technical_score >= 65 or ml_probability >= 0.6):
                    print(f"   📈 BUY - Good signal from at least one indicator")
                    print(f"   💡 Action: Buy with normal position")
                    
                elif final_score >= 55 and technical_score >= 60 and ml_probability >= 0.5:
                    print(f"   ⚠️  MODERATE BUY - Mixed but positive signals")
                    print(f"   💡 Action: Buy with smaller position")
                    
                elif technical_score >= 80 and ml_probability < 0.4:
                    print(f"   🚨 CAUTION - Strong technical but weak ML")
                    print(f"   💡 Action: Wait for ML confirmation or use small position")
                    
                elif technical_score < 50 and ml_probability > 0.7:
                    print(f"   ⚠️  ML OPPORTUNITY - Weak technical but strong ML")
                    print(f"   💡 Action: Consider buying with caution")
                    
                elif final_score < 45:
                    print(f"   ❌ AVOID - Poor signals from both indicators")
                    print(f"   💡 Action: Avoid this stock")
                    
                else:
                    print(f"   ⏸️  HOLD - Mixed or neutral signals")
                    print(f"   💡 Action: Wait for clearer signals")
                
                # Signal strength analysis
                print(f"\n📊 SIGNAL STRENGTH ANALYSIS:")
                tech_strength = "STRONG" if technical_score >= 70 else "MODERATE" if technical_score >= 50 else "WEAK"
                ml_strength = "STRONG" if ml_probability >= 0.6 else "MODERATE" if ml_probability >= 0.4 else "WEAK"
                
                print(f"   Technical Signal: {tech_strength} ({technical_score}/100)")
                print(f"   ML Signal: {ml_strength} ({ml_probability:.1%})")
                
                # Confidence level
                if tech_strength == "STRONG" and ml_strength == "STRONG":
                    confidence = "VERY HIGH"
                elif tech_strength == "STRONG" or ml_strength == "STRONG":
                    confidence = "HIGH"
                elif tech_strength == "MODERATE" and ml_strength == "MODERATE":
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                
                print(f"   Overall Confidence: {confidence}")
                
            else:
                print(f"   ❌ ML prediction not available")
                print(f"   💡 Recommendation: Use technical score with caution")
        
        print(f"\n" + "="*60)
    
    print(f"\n📋 SUMMARY OF IMPROVED FRAMEWORK:")
    print("=" * 60)
    print(f"🎯 Key Improvements:")
    print(f"   1. Dynamic weighting based on signal strength")
    print(f"   2. Market condition assessment")
    print(f"   3. Confidence-based position sizing")
    print(f"   4. Clear action recommendations")
    
    print(f"\n💡 WHEN TO RELY MORE ON TECHNICAL SCORE:")
    print(f"   • Strong technical signals (≥70/100)")
    print(f"   • Trending market conditions")
    print(f"   • Short-term trading (1-3 days)")
    print(f"   • High-volume, liquid stocks")
    
    print(f"\n💡 WHEN TO RELY MORE ON ML PROBABILITY:")
    print(f"   • Strong ML predictions (≥60%)")
    print(f"   • Volatile market conditions")
    print(f"   • Medium-term holding (1-2 weeks)")
    print(f"   • Complex or illiquid stocks")
    
    print(f"\n✅ Improved framework analysis completed!")

if __name__ == "__main__":
    improved_decision_framework() 