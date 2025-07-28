#!/usr/bin/env python3
"""
Enhanced Stock Analysis Tool with Backtesting
Expert stock analyzer with ML-powered predictions and 2-year historical validation
"""

from us_stock_analyzer import EnhancedStockAnalyzer
from chinese_stock_analyzer import ChineseStockAnalyzer
from dynamic_model_analyzer import DynamicModelAnalyzer
import sys

def main():
    print("=" * 80)
    print("🚀 ENHANCED STOCK ANALYZER - 短期持有 (1-2周) WITH BACKTESTING")
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
        print("9. ❌ Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
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
                    print(f"📊 Score: {best['score']}/100")
                    print(f"📊 Technical Score: {best['technical_score']}/100")
                    if best['ml_probability'] is not None:
                        print(f"🤖 ML Probability: {best['ml_probability']:.3f}")
                        print(f"🤖 ML Prediction: {best['ml_prediction']}")
                    print(f"🤖 ML Model Used: {best['ml_model_used']}")
                    print(f"💡 Action: {best['recommendation']}")
                    print(f"💰 Current Price: {best['current_price']:.2f}")
                    print(f"📈 Estimated High (10d): {best['estimated_high_10d']:.2f}")
                    print(f"📉 Estimated Low (10d): {best['estimated_low_10d']:.2f}")
                    print(f"🚀 Potential Gain: {best['potential_gain_10d']:.2%}")
                    print(f"⚠️  Potential Loss: {best['potential_loss_10d']:.2%}")
                    print(f"📈 5-day Momentum: {best['momentum_5d']:.2%}")
                    print(f"📊 Volume Ratio: {best['volume_ratio']:.2f}")
                    print(f"⚠️  Confidence: {best['confidence']}")
                    
                    if best['score'] >= 80:
                        print(f"✅ EXCELLENT CHOICE - Score over 80!")
                    elif best['score'] >= 65:
                        print(f"✅ GOOD CHOICE - Score over 65")
                    else:
                        print(f"⚠️  CAUTION - Score below 65")
            else:
                print("Please enter exactly 3 stock symbols.")
                
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
            
            elif market_type == 'CN':
                symbol = input("📊 Enter Chinese stock code: ").strip()
                market = input("Market (A/H, default A): ").strip().upper() or 'A'
                
                if symbol:
                    print(f"⏳ Analyzing {symbol} ({market}-shares)...")
                    result = chinese_analyzer.analyze_chinese_stock(symbol, market)
                    
                    if result:
                        print(f"\n📈 CHINESE STOCK ANALYSIS RESULTS FOR {symbol}")
                        print(f"Current Price: {result['current_price']:.2f}")
                        print(f"Estimated High (10d): {result['estimated_high_10d']:.2f}")
                        print(f"Potential Gain: {result['potential_gain_10d']:.2%}")
                        print(f"Score: {result['score']}/100")
                        print(f"Recommendation: {result['recommendation']}")
                        print(f"Confidence: {result['confidence']}")
                        print(f"5-day Momentum: {result['momentum_5d']:.2%}")
                        print(f"Volume Ratio: {result['volume_ratio']:.2f}")
                        print(f"Volatility: {result['volatility']:.4f}")
                        
                        if result['score'] >= 80:
                            print(f"✅ EXCELLENT - Score over 80!")
                        elif result['score'] >= 65:
                            print(f"✅ GOOD - Score over 65")
                        else:
                            print(f"⚠️  CAUTION - Score below 65")
                        
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
                            print(f"Buy Price: ${result['buy_price']:.2f}")
                            print(f"Current Price: ${result['current_price']:.2f}")
                            print(f"Current Return: {result['current_return']:.2%}")
                            print(f"Action: {rec['action']}")
                            print(f"Urgency: {rec['urgency']}")
                            print(f"Target Price: ${rec['target_price']:.2f}")
                            print(f"Risk Level: {rec['risk_level']}")
                            print(f"Technical Score: {rec['technical_score']:.1f}/100")
                            if rec['ml_score'] > 0:
                                print(f"ML Score: {rec['ml_score']:.3f}")
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
                            
                            if rec['action'] == "SELL NOW":
                                print(f"\n🚨 IMMEDIATE ACTION REQUIRED: {rec['action']}")
                            elif rec['action'] == "SELL SOON":
                                print(f"\n⚠️  CONSIDER SELLING: {rec['action']}")
                            else:
                                print(f"\n✅ HOLDING RECOMMENDED: {rec['action']}")
                
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
                            print(f"Buy Price: {result['buy_price']:.2f}")
                            print(f"Current Price: {result['current_price']:.2f}")
                            print(f"Current Return: {result['current_return']:.2%}")
                            print(f"Action: {rec['action']}")
                            print(f"Urgency: {rec['urgency']}")
                            print(f"Target Price: {rec['target_price']:.2f}")
                            print(f"Risk Level: {rec['risk_level']}")
                            print(f"Technical Score: {rec['technical_score']:.1f}/100")
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
                            print(f"Score: {result['score']}/100")
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
            print("Thank you for using the Enhanced Stock Analyzer!")
            break
            
        else:
            print("Invalid choice. Please enter 1-9.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 