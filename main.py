#!/usr/bin/env python3
"""
Enhanced Stock Analysis Tool with Backtesting
Expert stock analyzer with ML-powered predictions and 2-year historical validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from us_stock_analyzer import EnhancedStockAnalyzer
from chinese_stock_analyzer import ChineseStockAnalyzer
from chinese_stock_recommender import ChineseStockRecommender
from dynamic_model_analyzer import DynamicModelAnalyzer
import sys

# Global verbose mode toggle
verbose_mode = True

def display_ml_and_confidence_info(rec, prefix="   "):
    """
    Reusable function to display ML information and confidence levels
    """
    # Show ML information if available
    if rec.get('ml_probability') is not None:
        print(f"{prefix}ML Probability: {rec['ml_probability']:.3f}")
        print(f"{prefix}ML Prediction: {rec['ml_prediction']}")
        if 'ml_score' in rec:
            print(f"{prefix}ML Score: {rec['ml_score']:.2f}/100")
        if 'combined_score' in rec:
            print(f"{prefix}Combined Score: {rec['combined_score']:.2f}/100")
        if 'ml_insights' in rec:
            print(f"{prefix}ML Insights: {rec['ml_insights']}")
    else:
        print(f"{prefix}ML Analysis: Not available")
        if 'combined_score' in rec:
            print(f"{prefix}Combined Score: {rec['combined_score']:.2f}/100 (Technical only)")
    
    # Show estimated price range for next 10 days
    if 'estimated_high_10d' in rec and 'estimated_low_10d' in rec:
        print(f"\n{prefix}📈 PRICE ESTIMATES (Next 10 Days):")
        print(f"{prefix}   Estimated High: ¥{rec['estimated_high_10d']:.2f}")
        print(f"{prefix}   Estimated Low: ¥{rec['estimated_low_10d']:.2f}")
        print(f"{prefix}   Potential Gain: {rec['potential_gain_10d']:.2%}")
        print(f"{prefix}   Potential Loss: {rec['potential_loss_10d']:.2%}")
        
        # Show confidence levels if available
        if 'high_confidence' in rec and 'low_confidence' in rec:
            print(f"\n{prefix}🎯 PRICE ESTIMATE CONFIDENCE:")
            print(f"{prefix}   High Price Confidence: {rec['high_confidence']:.0f}%")
            print(f"{prefix}   Low Price Confidence: {rec['low_confidence']:.0f}%")
            
            # Show confidence reasoning if available
            if 'high_reasoning' in rec and 'low_reasoning' in rec:
                print(f"{prefix}   High Price Reasoning: {rec['high_reasoning']}")
                print(f"{prefix}   Low Price Reasoning: {rec['low_reasoning']}")
            
            # Overall confidence assessment
            avg_confidence = (rec['high_confidence'] + rec['low_confidence']) / 2
            if avg_confidence >= 80:
                confidence_level = "VERY HIGH"
            elif avg_confidence >= 70:
                confidence_level = "HIGH"
            elif avg_confidence >= 60:
                confidence_level = "MODERATE"
            elif avg_confidence >= 50:
                confidence_level = "LOW"
            else:
                confidence_level = "VERY LOW"
            
            print(f"{prefix}   Overall Confidence: {confidence_level} ({avg_confidence:.0f}%)")

def display_concise_confidence(rec):
    """
    Reusable function to display concise confidence information
    """
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
        print(f"🎯 Price Confidence: {conf_emoji} {avg_confidence:.0f}%")

def main():
    global verbose_mode
    
    print("🚀 ENHANCED STOCK ANALYZER")
    print("🎯 Target: 3% profit in 10 days with stop-loss protection")
    print("=" * 80)
    print("📊 Advanced Technical Indicators: MACD, RSI, Bollinger Bands, ADX, Stochastic")
    print("🤖 Machine Learning: Random Forest with 70+ features")
    print("📈 Backtesting: 2-year historical validation for 80%+ success rate")
    print("🇨🇳 Chinese Stocks: A-shares, H-shares support")
    print("🔄 Dynamic Model: Self-improving ML model with continuous updates")
    print("🎯 Target: 3% profit in 10 days with stop-loss protection")
    print("=" * 80)
    
    enhanced_analyzer = EnhancedStockAnalyzer()
    chinese_analyzer = ChineseStockAnalyzer()
    dynamic_analyzer = DynamicModelAnalyzer()
    chinese_recommender = ChineseStockRecommender()
    
    while True:
        print("\nOptions:")
        print("1. 🎯 Analyze 3 US stocks with backtesting (RECOMMENDED)")
        print("2. 🇨🇳 Analyze 3 Chinese stocks (A-shares/H-shares)")
        print("3. 📊 Analyze single stock with backtesting")
        print("4. 🎯 Sell Point Estimation (RECOMMENDED)")
        print("5. 🔄 Dynamic Model Analysis (NEW!)")
        print("6. 🔧 Custom backtesting parameters")
        print("7. 📈 View detailed backtest results")
        print("8. 🤖 Model Management (NEW!)")
        print("9. 🎯 Chinese Stock Recommendations (NEW!)")
        print("10. ⚙️  Toggle Verbose Mode (Currently: {'ON' if verbose_mode else 'OFF'})")
        print("11. ❌ Exit")
        
        choice = input("\nEnter your choice (1-11): ").strip()
        
        if choice == '1':
            print("\n🎯 Enter 3 US stock symbols for enhanced analysis:")
            symbols = []
            for i in range(3):
                symbol = input(f"Stock {i+1}: ").strip().upper()
                if symbol:
                    symbols.append(symbol)
            
            if len(symbols) == 3:
                print(f"\n🚀 Analyzing US stocks with backtesting: {', '.join(symbols)}")
                print("⏳ This may take a few minutes for comprehensive analysis...")
                
                best = enhanced_analyzer.compare_stocks_enhanced(symbols, holding_period=10, profit_threshold=0.03)
                
                if best:
                    bt = best['backtest_results']
                    print(f"\n🎯 FINAL RECOMMENDATION: {best['symbol']}")
                    
                    if verbose_mode:
                        print(f"📊 Success Rate: {bt['Success_Rate']:.2%}")
                        print(f"💡 Action: {best['recommendation']}")
                        print(f"💰 Current Price: ${best['current_price']:.2f}")
                        print(f"📈 Estimated High (10d): ${best['estimated_high_10d']:.2f}")
                        print(f"📉 Estimated Low (10d): ${best['estimated_low_10d']:.2f}")
                        print(f"🚀 Potential Gain: {best['potential_gain_10d']:.2%}")
                        print(f"⚠️  Potential Loss: {best['potential_loss_10d']:.2%}")
                        print(f"📈 Total Return: {bt['Total_Return']:.2%}")
                        print(f"📊 Total Trades: {bt['Total_Trades']}")
                        print(f"⚠️  Risk Level: {'LOW' if bt['Success_Rate'] >= 0.8 else 'MEDIUM' if bt['Success_Rate'] >= 0.6 else 'HIGH'}")
                        
                        if bt['Success_Rate'] >= 0.8:
                            print(f"✅ EXCELLENT CHOICE - Success rate over 80%!")
                        elif bt['Success_Rate'] >= 0.6:
                            print(f"✅ GOOD CHOICE - Success rate over 60%")
                        else:
                            print(f"⚠️  CAUTION - Success rate below 60%")
                    else:
                        # Concise output
                        print(f"📊 Success Rate: {bt['Success_Rate']:.1%}")
                        print(f"💡 Action: {best['recommendation']}")
                        print(f"💰 Price: ${best['current_price']:.2f} | Gain: {best['potential_gain_10d']:.1%} | Loss: {best['potential_loss_10d']:.1%}")
                        print(f"📈 Total Return: {bt['Total_Return']:.1%} | Trades: {bt['Total_Trades']}")
                        
                        # Quick assessment
                        if bt['Success_Rate'] >= 0.8:
                            print(f"✅ EXCELLENT")
                        elif bt['Success_Rate'] >= 0.6:
                            print(f"✅ GOOD")
                        else:
                            print(f"⚠️  CAUTION")
            else:
                print("Please enter exactly 3 stock symbols.")
                
        elif choice == '2':
            print("\n🇨🇳 Enter 3 Chinese stock symbols for analysis:")
            print("Format: Stock code (e.g., 000001 for A-shares, 0700 for H-shares)")
            stocks_list = []
            
            for i in range(3):
                symbol = input(f"Stock {i+1} code: ").strip()
                if symbol:
                    market = input(f"Market for {symbol} (A/H, default A): ").strip().upper() or 'A'
                    stocks_list.append({'symbol': symbol, 'market': market})
            
            if len(stocks_list) == 3:
                stock_display = ', '.join([f"{s['symbol']}({s['market']})" for s in stocks_list])
                print(f"\n🇨🇳 Analyzing Chinese stocks: {stock_display}")
                
                best = chinese_analyzer.compare_chinese_stocks(stocks_list)
                
                if best:
                    print(f"\n🎯 FINAL CHINESE RECOMMENDATION: {best['symbol']} ({best['market']}-shares)")
                    
                    if verbose_mode:
                        print("=" * 60)
                        
                        print(f"📊 SCORE ANALYSIS:")
                        print(f"   Final Score: {best['score']:.2f}/100")
                        print(f"   Technical Score: {best['technical_score']:.2f}/100")
                        
                        if best['ml_probability'] is not None:
                            # Calculate ML score
                            ml_score = int(best['ml_probability'] * 100)
                            print(f"   ML Score: {ml_score}/100")
                            print(f"   ML Probability: {best['ml_probability']:.3f}")
                            print(f"   ML Prediction: {best['ml_prediction']} ({'RISE' if best['ml_prediction'] == 1 else 'DECLINE'})")
                            
                            # Show score calculation
                            tech_weight = 0.6
                            ml_weight = 0.4
                            calculated_score = int(tech_weight * best['technical_score'] + ml_weight * ml_score)
                            print(f"   Score Calculation: {calculated_score}/100 (60% tech + 40% ML)")
                            
                            # ML assessment
                            print(f"\n🤖 ML ASSESSMENT:")
                            if best['ml_probability'] > 0.7:
                                print(f"   ✅ ML strongly predicts rise ({best['ml_probability']:.1%} probability)")
                            elif best['ml_probability'] > 0.6:
                                print(f"   ✅ ML moderately predicts rise ({best['ml_probability']:.1%} probability)")
                            elif best['ml_probability'] < 0.3:
                                print(f"   ⚠️  ML strongly predicts decline ({best['ml_probability']:.1%} probability)")
                            elif best['ml_probability'] < 0.4:
                                print(f"   ⚠️  ML moderately predicts decline ({best['ml_probability']:.1%} probability)")
                            else:
                                print(f"   🔄 ML neutral ({best['ml_probability']:.1%} probability)")
                        else:
                            print(f"   ML Score: Not available")
                            print(f"   ML Probability: Not available")
                            print(f"   ML Prediction: Not available")
                        
                        print(f"🤖 ML Model Used: {best['ml_model_used']}")
                        
                        print(f"\n💡 RECOMMENDATION:")
                        print(f"   Action: {best['recommendation']}")
                        print(f"   Confidence: {best['confidence']}")
                        
                        print(f"\n💰 PRICE ANALYSIS:")
                        print(f"   Current Price: ¥{best['current_price']:.2f}")
                        print(f"   Estimated High (10d): ¥{best['estimated_high_10d']:.2f}")
                        print(f"   Estimated Low (10d): ¥{best['estimated_low_10d']:.2f}")
                        print(f"   Potential Gain: {best['potential_gain_10d']:.2%}")
                        print(f"   Potential Loss: {best['potential_loss_10d']:.2%}")
                        
                        # Show confidence levels if available
                        if 'high_confidence' in best and 'low_confidence' in best:
                            print(f"\n🎯 PRICE ESTIMATE CONFIDENCE:")
                            print(f"   High Price Confidence: {best['high_confidence']:.0f}%")
                            print(f"   Low Price Confidence: {best['low_confidence']:.0f}%")
                            
                            # Show confidence reasoning if available
                            if 'high_reasoning' in best and 'low_reasoning' in best:
                                print(f"   High Price Reasoning: {best['high_reasoning']}")
                                print(f"   Low Price Reasoning: {best['low_reasoning']}")
                            
                            # Overall confidence assessment
                            avg_confidence = (best['high_confidence'] + best['low_confidence']) / 2
                            if avg_confidence >= 80:
                                confidence_level = "VERY HIGH"
                            elif avg_confidence >= 70:
                                confidence_level = "HIGH"
                            elif avg_confidence >= 60:
                                confidence_level = "MODERATE"
                            elif avg_confidence >= 50:
                                confidence_level = "LOW"
                            else:
                                confidence_level = "VERY LOW"
                            
                            print(f"   Overall Confidence: {confidence_level} ({avg_confidence:.0f}%)")
                        
                        # Risk/Reward analysis
                        risk_reward_ratio = best['potential_gain_10d'] / abs(best['potential_loss_10d']) if best['potential_loss_10d'] != 0 else float('inf')
                        print(f"   Risk/Reward Ratio: {risk_reward_ratio:.2f}")
                        
                        print(f"\n📈 TECHNICAL INDICATORS:")
                        print(f"   5-day Momentum: {best['momentum_5d']:.2%}")
                        print(f"   Volume Ratio: {best['volume_ratio']:.2f}")
                        
                        print(f"\n✅ QUALITY ASSESSMENT:")
                        if best['score'] >= 80:
                            print(f"   ✅ EXCELLENT CHOICE - Score over 80!")
                        elif best['score'] >= 65:
                            print(f"   ✅ GOOD CHOICE - Score over 65")
                        else:
                            print(f"   ⚠️  CAUTION - Score below 65")
                        
                        # Risk/Reward assessment
                        if risk_reward_ratio >= 3:
                            print(f"   ✅ Excellent risk/reward ratio (≥3:1)")
                        elif risk_reward_ratio >= 2:
                            print(f"   ✅ Good risk/reward ratio (≥2:1)")
                        elif risk_reward_ratio >= 1:
                            print(f"   ⚠️  Moderate risk/reward ratio (≥1:1)")
                        else:
                            print(f"   ❌ Poor risk/reward ratio (<1:1)")
                    else:
                        print(f"📊 Score: {best['score']:.2f}/100")
                        if best['ml_probability'] is not None:
                            print(f"🤖 ML: {best['ml_probability']:.1%} probability {'RISE' if best['ml_prediction'] == 1 else 'DECLINE'}")
                        print(f"💡 Action: {best['recommendation']}")
                        print(f"💰 Price: ¥{best['current_price']:.2f} | Gain: {best['potential_gain_10d']:.1%} | Loss: {best['potential_loss_10d']:.1%}")
                        print(f"📈 10d Range: ¥{best['estimated_low_10d']:.2f} - ¥{best['estimated_high_10d']:.2f}")
                        
                        # Show confidence in concise mode
                        if 'high_confidence' in best and 'low_confidence' in best:
                            avg_confidence = (best['high_confidence'] + best['low_confidence']) / 2
                            if avg_confidence >= 80:
                                conf_emoji = "🟢"
                            elif avg_confidence >= 70:
                                conf_emoji = "🟡"
                            elif avg_confidence >= 60:
                                conf_emoji = "🟠"
                            else:
                                conf_emoji = "🔴"
                            print(f"🎯 Price Confidence: {conf_emoji} {avg_confidence:.0f}%")
                        
                        print(f"⚠️  Confidence: {best['confidence']}")
                        
                        # Quick assessment
                        if best['score'] >= 80:
                            print(f"✅ EXCELLENT CHOICE")
                        elif best['score'] >= 65:
                            print(f"✅ GOOD CHOICE")
                        else:
                            print(f"⚠️  CAUTION")
                
        elif choice == '3':
            market_type = input("Market type (US/CN, default US): ").strip().upper() or 'US'
            
            if market_type == 'US':
                symbol = input("📊 Enter US stock symbol: ").strip().upper()
                if symbol:
                    print(f"⏳ Analyzing {symbol} with backtesting...")
                    result = enhanced_analyzer.analyze_stock_with_backtest(symbol, holding_period=10, profit_threshold=0.03)
                    
                    if result and result['backtest_results']:
                        bt = result['backtest_results']
                        print(f"\n📈 ENHANCED ANALYSIS RESULTS FOR {symbol}")
                        
                        if verbose_mode:
                            print(f"Current Price: ${result['current_price']:.2f}")
                            print(f"Estimated High (10d): ${result['estimated_high_10d']:.2f}")
                            print(f"Potential Gain: {result['potential_gain_10d']:.2%}")
                            print(f"Success Rate: {bt['Success_Rate']:.2%}")
                            print(f"Total Trades: {bt['Total_Trades']}")
                            print(f"Average Return: {bt['Average_Return']:.2%}")
                            print(f"Total Return: {bt['Total_Return']:.2%}")
                            print(f"Recommendation: {result['recommendation']}")
                            
                            if bt['Success_Rate'] >= 0.8:
                                print(f"✅ EXCELLENT - Success rate over 80%!")
                            elif bt['Success_Rate'] >= 0.6:
                                print(f"✅ GOOD - Success rate over 60%")
                            else:
                                print(f"⚠️  CAUTION - Success rate below 60%")
                        else:
                            # Concise output
                            print(f"📊 Success Rate: {bt['Success_Rate']:.1%}")
                            print(f"💡 Action: {result['recommendation']}")
                            print(f"💰 Price: ${result['current_price']:.2f} | Gain: {result['potential_gain_10d']:.1%}")
                            print(f"📈 Total Return: {bt['Total_Return']:.1%} | Trades: {bt['Total_Trades']}")
                            
                            # Quick assessment
                            if bt['Success_Rate'] >= 0.8:
                                print(f"✅ EXCELLENT")
                            elif bt['Success_Rate'] >= 0.6:
                                print(f"✅ GOOD")
                            else:
                                print(f"⚠️  CAUTION")
            
            elif market_type == 'CN':
                symbol = input("📊 Enter Chinese stock code: ").strip()
                market = input("Market (A/H, default A): ").strip().upper() or 'A'
                
                if symbol:
                    print(f"⏳ Analyzing {symbol} ({market}-shares)...")
                    result = chinese_analyzer.analyze_chinese_stock(symbol, market)
                    
                    if result:
                        print(f"\n📈 CHINESE STOCK ANALYSIS RESULTS FOR {symbol}")
                        
                        if verbose_mode:
                            print("=" * 60)
                            
                            print(f"📊 SCORE ANALYSIS:")
                            print(f"   Final Score: {result['score']:.2f}/100")
                            print(f"   Technical Score: {result['technical_score']:.2f}/100")
                            
                            if result['ml_probability'] is not None:
                                # Calculate ML score
                                ml_score = int(result['ml_probability'] * 100)
                                print(f"   ML Score: {ml_score}/100")
                                print(f"   ML Probability: {result['ml_probability']:.3f}")
                                print(f"   ML Prediction: {result['ml_prediction']} ({'RISE' if result['ml_prediction'] == 1 else 'DECLINE'})")
                                
                                # Show score calculation
                                tech_weight = 0.6
                                ml_weight = 0.4
                                calculated_score = int(tech_weight * result['technical_score'] + ml_weight * ml_score)
                                print(f"   Score Calculation: {calculated_score}/100 (60% tech + 40% ML)")
                                
                                # ML assessment
                                print(f"\n🤖 ML ASSESSMENT:")
                                if result['ml_probability'] > 0.7:
                                    print(f"   ✅ ML strongly predicts rise ({result['ml_probability']:.1%} probability)")
                                elif result['ml_probability'] > 0.6:
                                    print(f"   ✅ ML moderately predicts rise ({result['ml_probability']:.1%} probability)")
                                elif result['ml_probability'] < 0.3:
                                    print(f"   ⚠️  ML strongly predicts decline ({result['ml_probability']:.1%} probability)")
                                elif result['ml_probability'] < 0.4:
                                    print(f"   ⚠️  ML moderately predicts decline ({result['ml_probability']:.1%} probability)")
                                else:
                                    print(f"   🔄 ML neutral ({result['ml_probability']:.1%} probability)")
                            else:
                                print(f"   ML Score: Not available")
                                print(f"   ML Probability: Not available")
                                print(f"   ML Prediction: Not available")
                            
                            print(f"🤖 ML Model Used: {result['ml_model_used']}")
                            
                            print(f"\n💡 RECOMMENDATION:")
                            print(f"   Action: {result['recommendation']}")
                            print(f"   Confidence: {result['confidence']}")
                            
                            print(f"\n💰 PRICE ANALYSIS:")
                            print(f"   Current Price: ¥{result['current_price']:.2f}")
                            print(f"   Estimated High (10d): ¥{result['estimated_high_10d']:.2f}")
                            print(f"   Estimated Low (10d): ¥{result['estimated_low_10d']:.2f}")
                            print(f"   Potential Gain: {result['potential_gain_10d']:.2%}")
                            print(f"   Potential Loss: {result['potential_loss_10d']:.2%}")
                            
                            # Use reusable function to display confidence levels
                            display_ml_and_confidence_info(result, "   ")
                            
                            # Risk/Reward analysis
                            risk_reward_ratio = result['potential_gain_10d'] / abs(result['potential_loss_10d']) if result['potential_loss_10d'] != 0 else float('inf')
                            print(f"   Risk/Reward Ratio: {risk_reward_ratio:.2f}")
                            
                            print(f"\n📈 TECHNICAL INDICATORS:")
                            print(f"   5-day Momentum: {result['momentum_5d']:.2%}")
                            print(f"   Volume Ratio: {result['volume_ratio']:.2f}")
                            print(f"   Volatility: {result['volatility']:.4f}")
                            
                            print(f"\n✅ QUALITY ASSESSMENT:")
                            if result['score'] >= 80:
                                print(f"   ✅ EXCELLENT - Score over 80!")
                            elif result['score'] >= 65:
                                print(f"   ✅ GOOD - Score over 65")
                            else:
                                print(f"   ⚠️  CAUTION - Score below 65")
                            
                            # Risk/Reward assessment
                            if risk_reward_ratio >= 3:
                                print(f"   ✅ Excellent risk/reward ratio (≥3:1)")
                            elif risk_reward_ratio >= 2:
                                print(f"   ✅ Good risk/reward ratio (≥2:1)")
                            elif risk_reward_ratio >= 1:
                                print(f"   ⚠️  Moderate risk/reward ratio (≥1:1)")
                            else:
                                print(f"   ❌ Poor risk/reward ratio (<1:1)")
                        else:
                            print(f"📊 Score: {result['score']:.2f}/100")
                            if result['ml_probability'] is not None:
                                print(f"🤖 ML: {result['ml_probability']:.1%} probability {'RISE' if result['ml_prediction'] == 1 else 'DECLINE'}")
                            print(f"💡 Action: {result['recommendation']}")
                            print(f"💰 Price: ¥{result['current_price']:.2f} | Gain: {result['potential_gain_10d']:.1%} | Loss: {result['potential_loss_10d']:.1%}")
                            print(f"📈 10d Range: ¥{result['estimated_low_10d']:.2f} - ¥{result['estimated_high_10d']:.2f}")
                            
                            # Use reusable function for confidence display
                            display_concise_confidence(result)
                            
                            print(f"⚠️  Confidence: {result['confidence']}")
                            
                            # Quick assessment
                            if result['score'] >= 80:
                                print(f"✅ EXCELLENT")
                            elif result['score'] >= 65:
                                print(f"✅ GOOD")
                            else:
                                print(f"⚠️  CAUTION")
                
        elif choice == '4':
            print("\n🎯 SELL POINT ESTIMATION")
            print("Enter your stock position details:")
            
            market_type = input("Market type (US/CN, default US): ").strip().upper() or 'US'
            
            if market_type == 'US':
                symbol = input("Stock symbol: ").strip().upper()
                try:
                    buy_price = float(input("Buy price ($): "))
                    buy_date = input("Buy date (YYYY-MM-DD, optional): ").strip() or None
                    holding_period = int(input("Target holding period (days, default 10): ") or "10")
                    
                    if symbol and buy_price > 0:
                        print(f"⏳ Analyzing sell point for {symbol}...")
                        result = enhanced_analyzer.estimate_sell_point(symbol, buy_price, buy_date, holding_period)
                        
                        if result and result['recommendation']:
                            rec = result['recommendation']
                            print(f"\n🎯 SELL POINT ANALYSIS RESULTS:")
                            print(f"Symbol: {symbol}")
                            
                            if verbose_mode:
                                print(f"Buy Price: ${result['buy_price']:.2f}")
                                print(f"Current Price: ${result['current_price']:.2f}")
                                print(f"Current Return: {result['current_return']:.2%}")
                                print(f"Action: {rec['action']}")
                                print(f"Urgency: {rec['urgency']}")
                                print(f"Target Price: ${rec['target_price']:.2f}")
                                print(f"Risk Level: {rec['risk_level']}")
                                print(f"Technical Score: {rec['technical_score']:.1f}/100")
                                
                                # Use reusable function to display ML and confidence info
                                display_ml_and_confidence_info(rec)
                                
                                print(f"Reasoning: {rec['reasoning']}")
                                
                                if rec['sell_signals']:
                                    print(f"\n📉 SELL SIGNALS:")
                                    for signal in rec['sell_signals']:
                                        print(f"   • {signal}")
                                
                                if rec['hold_signals']:
                                    print(f"\n📈 HOLD SIGNALS:")
                                    for signal in rec['hold_signals']:
                                        print(f"   • {signal}")
                                
                                if rec['risk_factors']:
                                    print(f"\n⚠️  RISK FACTORS:")
                                    for risk in rec['risk_factors']:
                                        print(f"   • {risk}")
                                
                                if rec['limit_up_near']:
                                    print(f"\n🚨 LIMIT UP NEAR - HIGH RISK OF REVERSAL")
                                elif rec['limit_down_near']:
                                    print(f"\n📈 LIMIT DOWN NEAR - POTENTIAL BOUNCE")
                                
                                if rec['action'] == "SELL NOW":
                                    print(f"\n🚨 IMMEDIATE ACTION REQUIRED: {rec['action']}")
                                elif rec['action'] == "SELL SOON":
                                    print(f"\n⚠️  CONSIDER SELLING: {rec['action']}")
                                else:
                                    print(f"\n✅ HOLDING RECOMMENDED: {rec['action']}")
                            else:
                                # Concise output
                                print(f"💰 Buy: {result['buy_price']:.2f} | Current: {result['current_price']:.2f} | Return: {result['current_return']:.1%}")
                                print(f"💡 Action: {rec['action']} | Urgency: {rec['urgency']}")
                                print(f"🎯 Target: {rec['target_price']:.2f} | Risk: {rec['risk_level']}")
                                print(f"📊 Technical Score: {rec['technical_score']:.2f}/100")
                                
                                # Show ML info in concise mode
                                if rec['ml_probability'] is not None:
                                    print(f"🤖 ML: {rec['ml_probability']:.1%} probability {'RISE' if rec['ml_prediction'] == 1 else 'DECLINE'}")
                                else:
                                    print(f"🤖 ML: Not available")
                                
                                # Show price estimates in concise mode
                                if 'estimated_high_10d' in rec and 'estimated_low_10d' in rec:
                                    print(f"📈 10d Range: ¥{rec['estimated_low_10d']:.2f} - ¥{rec['estimated_high_10d']:.2f} | Gain: {rec['potential_gain_10d']:.1%} | Loss: {rec['potential_loss_10d']:.1%}")
                                    
                                    # Use reusable function for confidence display
                                    display_concise_confidence(rec)
                                
                                # Quick action summary
                                if rec['action'] == "SELL NOW":
                                    print(f"🚨 IMMEDIATE ACTION REQUIRED")
                                elif rec['action'] == "SELL SOON":
                                    print(f"⚠️  CONSIDER SELLING")
                                else:
                                    print(f"✅ HOLDING RECOMMENDED")
                
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")
            
            elif market_type == 'CN':
                symbol = input("Stock code: ").strip()
                market = input("Market (A/H, default A): ").strip().upper() or 'A'
                try:
                    buy_price = float(input("Buy price: "))
                    buy_date = input("Buy date (YYYY-MM-DD, optional): ").strip() or None
                    holding_period = int(input("Target holding period (days, default 10): ") or "10")
                    
                    if symbol and buy_price > 0:
                        print(f"⏳ Analyzing sell point for {symbol} ({market}-shares)...")
                        result = chinese_analyzer.estimate_sell_point(symbol, buy_price, market, buy_date, holding_period)
                        
                        if result and result['recommendation']:
                            rec = result['recommendation']
                            print(f"\n🎯 CHINESE STOCK SELL POINT ANALYSIS:")
                            print(f"Symbol: {symbol} ({market}-shares)")
                            
                            if verbose_mode:
                                print(f"Buy Price: {result['buy_price']:.2f}")
                                print(f"Current Price: {result['current_price']:.2f}")
                                print(f"Current Return: {result['current_return']:.2%}")
                                print(f"Action: {rec['action']}")
                                print(f"Urgency: {rec['urgency']}")
                                print(f"Target Price: {rec['target_price']:.2f}")
                                print(f"Risk Level: {rec['risk_level']}")
                                print(f"Technical Score: {rec['technical_score']:.1f}/100")
                                print(f"ML Probability: {rec['ml_probability']:.3f}")
                                print(f"ML Prediction: {rec['ml_prediction']}")
                                print(f"ML Score: {rec['ml_score']:.2f}/100")
                                print(f"Combined Score: {rec['combined_score']:.2f}/100")
                                print(f"ML Insights: {rec.get('ml_insights', 'Not available')}")
                                
                                # Use reusable function to display ML and confidence info
                                display_ml_and_confidence_info(rec)
                                
                                print(f"Reasoning: {rec['reasoning']}")
                                
                                if rec['sell_signals']:
                                    print(f"\n📉 SELL SIGNALS:")
                                    for signal in rec['sell_signals']:
                                        print(f"   • {signal}")
                                
                                if rec['hold_signals']:
                                    print(f"\n📈 HOLD SIGNALS:")
                                    for signal in rec['hold_signals']:
                                        print(f"   • {signal}")
                                
                                if rec['risk_factors']:
                                    print(f"\n⚠️  RISK FACTORS:")
                                    for risk in rec['risk_factors']:
                                        print(f"   • {risk}")
                                
                                if rec['limit_up_near']:
                                    print(f"\n🚨 LIMIT UP NEAR - HIGH RISK OF REVERSAL")
                                elif rec['limit_down_near']:
                                    print(f"\n📈 LIMIT DOWN NEAR - POTENTIAL BOUNCE")
                                
                                if rec['action'] == "SELL NOW":
                                    print(f"\n🚨 IMMEDIATE ACTION REQUIRED: {rec['action']}")
                                elif rec['action'] == "SELL SOON":
                                    print(f"\n⚠️  CONSIDER SELLING: {rec['action']}")
                                else:
                                    print(f"\n✅ HOLDING RECOMMENDED: {rec['action']}")
                            else:
                                # Concise output
                                print(f"💰 Buy: {result['buy_price']:.2f} | Current: {result['current_price']:.2f} | Return: {result['current_return']:.1%}")
                                print(f"💡 Action: {rec['action']} | Urgency: {rec['urgency']}")
                                print(f"🎯 Target: {rec['target_price']:.2f} | Risk: {rec['risk_level']}")
                                print(f"📊 Technical Score: {rec['technical_score']:.2f}/100")
                                
                                # Quick action summary
                                if rec['action'] == "SELL NOW":
                                    print(f"🚨 IMMEDIATE ACTION REQUIRED")
                                elif rec['action'] == "SELL SOON":
                                    print(f"⚠️  CONSIDER SELLING")
                                else:
                                    print(f"✅ HOLDING RECOMMENDED")
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")
                        
        elif choice == '5':
            print("\n🔄 DYNAMIC MODEL ANALYSIS")
            print("This feature uses a self-improving ML model that gets more accurate over time!")
            
            sub_choice = input("\nChoose action:\n1. Initialize new dynamic model\n2. Analyze stock with dynamic model\n3. Update existing model\n4. View model status\nEnter choice (1-4): ").strip()
            
            if sub_choice == '1':
                symbol = input("Enter stock symbol to initialize dynamic model: ").strip().upper()
                if symbol:
                    print(f"⏳ Initializing dynamic model for {symbol}...")
                    success = dynamic_analyzer.initialize_model(symbol)
                    
                    if success:
                        print(f"✅ Dynamic model initialized successfully for {symbol}!")
                        print("The model will now continuously improve as new data becomes available.")
                    else:
                        print(f"❌ Failed to initialize dynamic model for {symbol}")
            
            elif sub_choice == '2':
                symbol = input("Enter stock symbol to analyze: ").strip().upper()
                market = input("Market (A/H/US, default A): ").strip().upper() or 'A'
                if symbol:
                    print(f"⏳ Analyzing {symbol} with dynamic model...")
                    result = dynamic_analyzer.analyze_stock(symbol, market=market)
                    
                    if result:
                        print(f"\n🔄 DYNAMIC MODEL ANALYSIS RESULTS:")
                        print(f"Symbol: {result['symbol']}")
                        print(f"Current Price: ${result['current_price']:.2f}")
                        print(f"Estimated High (10d): ${result['estimated_high_10d']:.2f}")
                        print(f"Potential Gain: {result['potential_gain_10d']:.2%}")
                        print(f"ML Prediction: {result['prediction']}")
                        print(f"ML Probability: {result['probability']:.3f}")
                        print(f"Technical Score: {result['technical_score']:.3f}")
                        print(f"Combined Score: {result['combined_score']:.3f}")
                        print(f"Recommendation: {result['recommendation']}")
                        print(f"Confidence: {result['confidence']}")
                        print(f"Model Version: {result['model_version']}")
                        print(f"Data Points Used: {result['data_points']}")
                        
                        if result['combined_score'] >= 0.8:
                            print(f"✅ EXCELLENT - Strong buy signal!")
                        elif result['combined_score'] >= 0.6:
                            print(f"✅ GOOD - Buy signal")
                        else:
                            print(f"⚠️  CAUTION - Weak or negative signal")
                    else:
                        print(f"❌ Analysis failed for {symbol}")
            
            elif sub_choice == '3':
                symbol = input("Enter stock symbol to update model: ").strip().upper()
                if symbol:
                    print(f"⏳ Updating dynamic model for {symbol}...")
                    success = dynamic_analyzer.update_model()
                    
                    if success:
                        print(f"✅ Dynamic model updated successfully for {symbol}!")
                    else:
                        print(f"❌ Failed to update dynamic model for {symbol}")
            
            elif sub_choice == '4':
                status = dynamic_analyzer.get_model_status()
                print(f"\n📊 DYNAMIC MODEL STATUS:")
                print(f"Status: {status['status']}")
                
                if status['status'] == 'ACTIVE':
                    print(f"Symbol: {status['symbol']}")
                    print(f"Model Version: {status['model_version']}")
                    print(f"Last Update: {status['last_update']}")
                    print(f"Data Points: {status['data_points']}")
                    print(f"Current Accuracy: {status.get('current_accuracy', 'N/A')}")
                    
                    # Show performance summary
                    summary = dynamic_analyzer.get_performance_summary()
                    print(summary)
                else:
                    print(f"Message: {status['message']}")
                    print("To get started, choose option 1 to initialize a dynamic model.")
            
            else:
                print("Invalid choice. Please enter 1-4.")
                
        elif choice == '6':
            print("\n🔧 Custom Backtesting Parameters:")
            try:
                holding_period = int(input("Holding period (days, default 10): ") or "10")
                profit_threshold = float(input("Profit threshold (%, default 3.0): ") or "3.0") / 100
                stop_loss = float(input("Stop loss (%, default 2.0): ") or "2.0") / 100
                
                market_type = input("Market type (US/CN, default US): ").strip().upper() or 'US'
                
                if market_type == 'US':
                    symbol = input("Enter US stock symbol: ").strip().upper()
                    if symbol:
                        print(f"⏳ Running custom backtest for {symbol}...")
                        result = enhanced_analyzer.analyze_stock_with_backtest(symbol, holding_period, profit_threshold)
                        
                        if result and result['backtest_results']:
                            bt = result['backtest_results']
                            print(f"\n🔧 CUSTOM BACKTEST RESULTS:")
                            print(f"Symbol: {symbol}")
                            print(f"Holding Period: {holding_period} days")
                            print(f"Profit Target: {profit_threshold:.1%}")
                            print(f"Stop Loss: {stop_loss:.1%}")
                            print(f"Success Rate: {bt['Success_Rate']:.2%}")
                            print(f"Total Trades: {bt['Total_Trades']}")
                            print(f"Average Return: {bt['Average_Return']:.2%}")
                
                elif market_type == 'CN':
                    symbol = input("Enter Chinese stock code: ").strip()
                    market = input("Market (A/H, default A): ").strip().upper() or 'A'
                    
                    if symbol:
                        print(f"⏳ Running custom analysis for {symbol} ({market}-shares)...")
                        result = chinese_analyzer.analyze_chinese_stock(symbol, market)
                        
                        if result:
                            print(f"\n🔧 CUSTOM CHINESE ANALYSIS RESULTS:")
                            print(f"Symbol: {symbol} ({market}-shares)")
                            print(f"Holding Period: {holding_period} days")
                            print(f"Profit Target: {profit_threshold:.1%}")
                            print(f"Stop Loss: {stop_loss:.1%}")
                            print(f"Score: {result['score']:.2f}/100")
                            print(f"Recommendation: {result['recommendation']}")
                            
            except ValueError:
                print("Invalid input. Using default parameters.")
                
        elif choice == '7':
            if enhanced_analyzer.backtest_results:
                bt = enhanced_analyzer.backtest_results
                print(f"\n📈 DETAILED BACKTEST RESULTS FOR {enhanced_analyzer.symbol}")
                print(f"{'='*60}")
                print(f"Total Trades: {bt['Total_Trades']}")
                print(f"Successful Trades: {bt['Successful_Trades']}")
                print(f"Success Rate: {bt['Success_Rate']:.2%}")
                print(f"Average Return: {bt['Average_Return']:.2%}")
                print(f"Total Return: {bt['Total_Return']:.2%}")
                
                if bt['Trades']:
                    print(f"\n📊 RECENT TRADES:")
                    for i, trade in enumerate(bt['Trades'][-5:], 1):
                        print(f"{i}. {trade['Entry_Date'].date()} → {trade['Exit_Date'].date()}")
                        print(f"   ${trade['Entry_Price']:.2f} → ${trade['Exit_Price']:.2f}")
                        print(f"   Return: {trade['Returns']:.2%} ({'✅' if trade['Success'] else '❌'})")
                        print(f"   Exit: {trade['Exit_Reason']}")
            else:
                print("No backtest results available. Run an analysis first.")
                    
        elif choice == '8':
            print("\n🤖 Model Management:")
            print("1. View all saved models")
            print("2. View status of a specific model")
            print("3. Delete a specific model")
            print("4. Exit")
            
            model_choice = input("\nEnter your choice (1-4): ").strip()
            
            if model_choice == '1':
                saved_models = dynamic_analyzer.list_saved_models()
                if saved_models:
                    print("\n💾 SAVED MODELS:")
                    for i, model in enumerate(saved_models, 1):
                        print(f"{i}. {model}")
                else:
                    print("No saved models found.")
            
            elif model_choice == '2':
                model_name = input("Enter model name to view status: ").strip()
                if model_name:
                    status = dynamic_analyzer.get_model_status(model_name)
                    if status:
                        print(f"\n📊 STATUS FOR MODEL '{model_name}':")
                        print(f"Status: {status['status']}")
                        if status['status'] == 'ACTIVE':
                            print(f"Symbol: {status['symbol']}")
                            print(f"Model Version: {status['model_version']}")
                            print(f"Last Update: {status['last_update']}")
                            print(f"Data Points: {status['data_points']}")
                            print(f"Current Accuracy: {status.get('current_accuracy', 'N/A')}")
                            print(f"Model File: {status['model_file']}")
                        else:
                            print(f"Message: {status['message']}")
                    else:
                        print(f"Model '{model_name}' not found.")
            
            elif model_choice == '3':
                model_name = input("Enter model name to delete: ").strip()
                if model_name:
                    success = dynamic_analyzer.delete_model(model_name)
                    if success:
                        print(f"✅ Model '{model_name}' deleted successfully.")
                    else:
                        print(f"❌ Failed to delete model '{model_name}'.")
            
            elif model_choice == '4':
                print("Returning to main menu.")
            
            else:
                print("Invalid choice. Please enter 1-4.")
                
        elif choice == '9':
            print("\n🎯 Chinese Stock Recommendations (A500 Analysis):")
            print("Analyze top Chinese A500 stocks using multiple strategies and ML predictions")
            print("1. All strategies")
            print("2. 强中选强 (Strong Among Strong)")
            print("3. 中位破局 (Mid-range Breakout)")
            print("4. 低位反弹 (Low Position Rebound)")
            print("5. 技术突破 (Technical Breakout)")
            print("6. 价值回归 (Value Reversion)")
            print("7. 成长加速 (Growth Acceleration)")
            print("8. View cache statistics")
            print("9. Clear cache")
            print("10. Exit")
            
            strategy_choice = input("\nEnter your choice (1-10): ").strip()
            
            if strategy_choice in ['1', '2', '3', '4', '5', '6', '7']:
                print(f"\n🚀 Starting Chinese Stock Recommendation Analysis...")
                print(f"📊 Strategy: {chinese_recommender.strategies[strategy_choice]['name']}")
                print(f"📝 Description: {chinese_recommender.strategies[strategy_choice]['description']}")
                print(f"🎯 Total stocks to analyze: {len(chinese_recommender.a500_symbols)}")
                print(f"⏳ This may take 3-5 minutes for comprehensive analysis...")
                
                # Ask if user wants fresh analysis
                fresh_choice = input("\n🔄 Force fresh analysis (ignore cache)? (y/n): ").strip().lower()
                if fresh_choice == 'y':
                    print("🗑️  Clearing cache for fresh analysis...")
                    chinese_recommender.cache.clear_all_cache()
                    print("✅ Cache cleared, starting fresh analysis...")
                
                try:
                    recommendations = chinese_recommender.run_recommendation_analysis(strategy_choice)
                    
                    if recommendations:
                        print(f"\n✅ Analysis completed successfully!")
                        print(f"📊 Found {len(recommendations)} top recommendations")
                        print(f"🎯 Analyzed all {len(chinese_recommender.a500_symbols)} stocks")
                        
                        # Ask if user wants to save results
                        save_choice = input("\n💾 Save recommendations to file? (y/n): ").strip().lower()
                        if save_choice == 'y':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"chinese_recommendations_{strategy_choice}_{timestamp}.txt"
                            
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(f"Chinese Stock Recommendations - {chinese_recommender.strategies[strategy_choice]['name']}\n")
                                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"Strategy: {chinese_recommender.strategies[strategy_choice]['description']}\n")
                                f.write(f"Total stocks analyzed: {len(chinese_recommender.a500_symbols)}\n")
                                f.write("="*80 + "\n\n")
                                
                                for i, rec in enumerate(recommendations, 1):
                                    f.write(f"{i}. {rec['symbol']} ({rec['market']}-shares)\n")
                                    f.write(f"   Final Score: {rec['final_score']:.2f}/100\n")
                                    f.write(f"   Technical Score: {rec['technical_score']:.2f}/100\n")
                                    f.write(f"   ML Probability: {rec['ml_probability']:.3f}\n")
                                    f.write(f"   ML Prediction: {rec['ml_prediction']}\n")
                                    f.write(f"   Action: {rec['action']}\n")
                                    f.write(f"   Current Price: ¥{rec['current_price']:.2f}\n")
                                    f.write(f"   Volume Ratio: {rec['volume_ratio']:.2f}\n")
                                    f.write(f"   5-day Momentum: {rec['momentum_5d']:.2%}\n")
                                    f.write(f"   RSI: {rec['rsi']:.1f}\n")
                                    f.write(f"   Key Strengths: {', '.join(rec['reasons'][:3])}\n")
                                    f.write("-"*60 + "\n\n")
                            
                            print(f"✅ Recommendations saved to {filename}")
                    else:
                        print("❌ No recommendations found. Try a different strategy or check market conditions.")
                        
                except Exception as e:
                    print(f"❌ Error during analysis: {str(e)}")
                    print("Please try again or check your internet connection.")
            
            elif strategy_choice == '8':
                stats = chinese_recommender.cache.get_cache_stats()
                print(f"\n📊 Cache Statistics:")
                print(f"   📈 Cached stocks: {stats['total_cached_stocks']}")
                print(f"   ❌ Failed downloads: {stats['total_failed_downloads']}")
                print(f"   🎯 Cached recommendations: {stats['total_cached_recommendations']}")
                print(f"   📊 Total stocks in system: {len(chinese_recommender.a500_symbols)}")
                
                if stats['total_cached_recommendations'] > 0:
                    print(f"\n💡 Cache Status: {'Using cached results' if stats['total_cached_recommendations'] > 0 else 'Fresh analysis needed'}")
                    print(f"🔄 To force fresh analysis, choose option 9 (Clear cache)")
                else:
                    print(f"\n💡 Cache Status: Fresh analysis will be performed")
            
            elif strategy_choice == '9':
                confirm = input("🗑️  Clear all cache? This will force fresh downloads (y/n): ").strip().lower()
                if confirm == 'y':
                    chinese_recommender.cache.clear_all_cache()
                    print("✅ Cache cleared successfully")
                else:
                    print("Cache clearing cancelled")
            
            elif strategy_choice == '10':
                print("Returning to main menu.")
            
            else:
                print("Invalid choice. Please enter 1-10.")
                
        elif choice == '10':
            verbose_mode = not verbose_mode
            print(f"✅ Verbose mode {'ENABLED' if verbose_mode else 'DISABLED'}")
            print(f"   {'📊 Detailed analysis with technical indicators and ML scores' if verbose_mode else '🎯 Concise output with final decisions only'}")
            
        elif choice == '11':
            print("Thank you for using the Enhanced Stock Analyzer!")
            break
            
        else:
            print("Invalid choice. Please enter 1-10.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 