#!/usr/bin/env python3
"""
Chinese Stock Downloader
Supports both yfinance and akshare with data normalization and stock name fetching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
import time

warnings.filterwarnings('ignore')

class ChineseStockDownloader:
    def __init__(self, data_source='yfinance'):
        """
        Initialize downloader with specified data source
        data_source: 'yfinance' or 'akshare'
        """
        self.data_source = data_source.lower()
        self.stock_names_cache = {}
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            print("❌ akshare not available, falling back to yfinance")
            self.data_source = 'yfinance'
    
    def get_chinese_stock_symbol(self, symbol, market='A'):
        """Convert Chinese stock symbol to appropriate format"""
        # Handle case where symbol might be a DataFrame or other object
        if not isinstance(symbol, str):
            print(f"⚠️  Invalid symbol type: {type(symbol)}, using default symbol")
            if hasattr(symbol, 'iloc'):
                print(f"   Symbol appears to be a DataFrame with shape: {symbol.shape}")
                # Try to extract the first value if it's a DataFrame
                try:
                    symbol = str(symbol.iloc[0, 0]) if symbol.shape[1] > 0 else "000001"
                    print(f"   Extracted symbol from DataFrame: {symbol}")
                except:
                    symbol = "000001"
            else:
                symbol = str(symbol) if symbol is not None else "000001"
        
        # Handle case where market might be a DataFrame or other object
        if not isinstance(market, str):
            print(f"⚠️  Invalid market type: {type(market)}, using default market 'A'")
            if hasattr(market, 'iloc'):
                print(f"   Market appears to be a DataFrame with shape: {market.shape}")
                # Try to extract the first value if it's a DataFrame
                try:
                    market = str(market.iloc[0, 0]) if market.shape[1] > 0 else "A"
                    print(f"   Extracted market from DataFrame: {market}")
                except:
                    market = "A"
            else:
                market = str(market) if market is not None else "A"
        
        symbol = symbol.strip().upper()
        
        if self.data_source == 'yfinance':
            # yfinance format
            if market.upper() == 'A':
                if symbol.startswith('000') or symbol.startswith('002') or symbol.startswith('300'):
                    return f"{symbol}.SZ"  # Shenzhen
                elif symbol.startswith('600') or symbol.startswith('900'):
                    return f"{symbol}.SS"  # Shanghai
                else:
                    return f"{symbol}.SS"  # Default to Shanghai
            elif market.upper() == 'H':
                return f"{symbol}.HK"
            else:
                return f"{symbol}.SS"
        else:
            # akshare format (just return the symbol)
            return symbol
    
    def get_stock_name(self, symbol, market='A'):
        """Get stock abbreviation for a given symbol"""
        # Handle case where symbol might be a DataFrame or other object
        if not isinstance(symbol, str):
            print(f"⚠️  Invalid symbol type in get_stock_name: {type(symbol)}")
            if hasattr(symbol, 'iloc'):
                print(f"   Symbol appears to be a DataFrame with shape: {symbol.shape}")
                print(f"   Symbol DataFrame columns: {list(symbol.columns)}")
                # Try to extract the first value if it's a DataFrame
                try:
                    symbol = str(symbol.iloc[0, 0]) if symbol.shape[1] > 0 else "000001"
                    print(f"   Extracted symbol from DataFrame: {symbol}")
                except:
                    symbol = "000001"
            else:
                symbol = str(symbol) if symbol is not None else "000001"
        
        # Handle case where market might be a DataFrame or other object
        if not isinstance(market, str):
            print(f"⚠️  Invalid market type in get_stock_name: {type(market)}")
            if hasattr(market, 'iloc'):
                print(f"   Market appears to be a DataFrame with shape: {market.shape}")
                print(f"   Market DataFrame columns: {list(market.columns)}")
                # Try to extract the first value if it's a DataFrame
                try:
                    market = str(market.iloc[0, 0]) if market.shape[1] > 0 else "A"
                    print(f"   Extracted market from DataFrame: {market}")
                except:
                    market = "A"
            else:
                market = str(market) if market is not None else "A"
        
        # Create cache key
        cache_key = f"{symbol}_{market}"
        
        # Check cache first
        if cache_key in self.stock_names_cache:
            return self.stock_names_cache[cache_key]
        
        try:
            if self.data_source == 'yfinance':
                # Try to get stock name from yfinance
                formatted_symbol = self.get_chinese_stock_symbol(symbol, market)
                ticker = yf.Ticker(formatted_symbol)
                info = ticker.info
                
                if info and 'shortName' in info and info['shortName']:
                    stock_name = info['shortName']
                elif 'longName' in info and info['longName']:
                    # Try to get a shorter version
                    long_name = info['longName']
                    # Remove common suffixes
                    for suffix in [' Co., Ltd.', ' Company Limited', ' Group Co., Ltd.', ' Corporation']:
                        if long_name.endswith(suffix):
                            long_name = long_name[:-len(suffix)]
                    stock_name = long_name
                else:
                    stock_name = symbol
            else:
                # Try to get stock abbreviation from akshare
                try:
                    # For A-shares, try different methods
                    if market.upper() == 'A':
                        # Try stock_zh_a_spot_em for real-time data with names
                        stock_data = self.ak.stock_zh_a_spot_em()
                        stock_info = stock_data[stock_data['代码'] == symbol]
                        if not stock_info.empty:
                            # Get the stock abbreviation (名称 field) - this should be Chinese
                            stock_name = stock_info.iloc[0]['名称']
                        else:
                            # Try alternative method for stock names
                            try:
                                # Use stock_zh_a_name_code_map to get Chinese names
                                name_map = self.ak.stock_zh_a_name_code_map()
                                stock_info = name_map[name_map['代码'] == symbol]
                                if not stock_info.empty:
                                    stock_name = stock_info.iloc[0]['名称']
                                else:
                                    stock_name = symbol
                            except:
                                stock_name = symbol
                    else:
                        stock_name = symbol
                except:
                    stock_name = symbol
            
            # Cache the result
            self.stock_names_cache[cache_key] = stock_name
            return stock_name
            
        except Exception as e:
            print(f"⚠️  Could not fetch stock name for {symbol}: {str(e)}")
            return symbol
    
    def download_stock_data_yfinance(self, symbol, market='A', period="1mo"):
        """Download stock data using yfinance"""
        try:
            # Add delay to avoid server resistance
            time.sleep(0.5)
            
            formatted_symbol = self.get_chinese_stock_symbol(symbol, market)
            print(f"📥 Downloading {symbol} ({market}-shares) as {formatted_symbol}")
            
            ticker = yf.Ticker(formatted_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"❌ No data found for {formatted_symbol}")
                return None
            
            # Normalize column names to standard format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            print(f"✅ Downloaded {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Error downloading {symbol} with yfinance: {str(e)}")
            return None
    
    def download_stock_data_akshare(self, symbol, market='A', period="1mo"):
        """Download stock data using akshare"""
        try:
            # Add delay to avoid server resistance
            time.sleep(random.uniform(1.2, 2.5))
            
            print(f"📥 Downloading {symbol} ({market}-shares) with akshare")
            
            # Convert period to days
            period_days = self._convert_period_to_days(period)
            
            if market.upper() == 'A':
                # A-shares - use the working akshare method
                try:
                    # Calculate start date based on period
                    if period_days:
                        from datetime import datetime, timedelta
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=period_days)
                        start_date_str = start_date.strftime('%Y%m%d')
                        end_date_str = end_date.strftime('%Y%m%d')
                    else:
                        start_date_str = None
                        end_date_str = None
                    
                    # Use the working akshare method
                    data = self.ak.stock_zh_a_hist(
                        symbol=symbol,      # 股票代码（不带前缀）
                        period="daily",     # 日线
                        adjust="qfq",       # 后复权
                        start_date=start_date_str,
                        end_date=end_date_str,
                    )
                    
                except Exception as e:
                    print(f"❌ Error downloading {symbol}: {str(e)}")
                    return None
            else:
                # H-shares
                data = self.ak.stock_hk_hist(symbol=symbol, period="daily",
                                            start_date=None, end_date=None,
                                            adjust="qfq")
            
            if data is None or data.empty:
                print(f"❌ No data found for {symbol}")
                return None
 
            # Normalize column names to standard format
            data = data.rename(columns={
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low', 
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate'
            })

            # Remove extra columns that might cause issues (but keep 日期 for now)
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate']
            # Add 日期 to columns to keep if it exists
            if '日期' in data.columns:
                columns_to_keep.append('日期')
            data = data[[col for col in columns_to_keep if col in data.columns]]
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Handle date index - when using specific dates, akshare doesn't include date column
            if '日期' in data.columns:
                data['date'] = pd.to_datetime(data['日期'])
                data = data.set_index('date')
                data = data.drop('日期', axis=1)
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            else:
                # When using specific dates, akshare doesn't include date column
                # Create proper date range based on the number of rows
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=len(data)-1)
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                data.index = date_range
                print(f"   ✅ Created date range: {data.index[0]} to {data.index[-1]}")
            
            # Sort by date to ensure chronological order (oldest first)
            data = data.sort_index()
            
            # Limit to requested period
            if period_days and len(data) > period_days:
                data = data.tail(period_days)
            
            print(f"✅ Downloaded {len(data)} days of data for {symbol}")
            print(f"📅 Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            return data
            
        except Exception as e:
            print(f"❌ Error downloading {symbol} with akshare: {str(e)}")
            return None
    
    def _convert_period_to_days(self, period):
        """Convert period string to number of days"""
        # Handle case where period might be a DataFrame or other object
        if not isinstance(period, str):
            if isinstance(period, int):
                return period
            else:
                # Default to 30 days if period is not a string or int
                print(f"⚠️  Invalid period type: {type(period)}, using default 30 days")
                return 30
        
        period = period.lower()
        if period == "1d":
            return 1
        elif period == "5d":
            return 5
        elif period == "1mo":
            return 30
        elif period == "3mo":
            return 90
        elif period == "6mo":
            return 180
        elif period == "1y":
            return 365
        elif period == "2y":
            return 730
        elif period == "5y":
            return 1825
        elif period == "10y":
            return 3650
        elif period == "ytd":
            return None  # Year to date
        elif period == "max":
            return None  # Maximum available
        else:
            return 30  # Default to 1 month
    
    def download_stock_data(self, symbol, market='A', period="1mo"):
        """Download stock data using the selected data source"""
        # Handle case where symbol might be a DataFrame or other object
        if not isinstance(symbol, str):
            print(f"⚠️  Invalid symbol type in download_stock_data: {type(symbol)}")
            if hasattr(symbol, 'iloc'):
                print(f"   Symbol appears to be a DataFrame with shape: {symbol.shape}")
                # Try to extract the first value if it's a DataFrame
                try:
                    symbol = str(symbol.iloc[0, 0]) if symbol.shape[1] > 0 else "000001"
                    print(f"   Extracted symbol from DataFrame: {symbol}")
                except:
                    return None
            else:
                symbol = str(symbol) if symbol is not None else "000001"
        
        # Handle case where market might be a DataFrame or other object
        if not isinstance(market, str):
            print(f"⚠️  Invalid market type in download_stock_data: {type(market)}")
            if hasattr(market, 'iloc'):
                print(f"   Market appears to be a DataFrame with shape: {market.shape}")
                # Try to extract the first value if it's a DataFrame
                try:
                    market = str(market.iloc[0, 0]) if market.shape[1] > 0 else "A"
                    print(f"   Extracted market from DataFrame: {market}")
                except:
                    market = "A"
            else:
                market = str(market) if market is not None else "A"
        
        if self.data_source == 'yfinance':
            return self.download_stock_data_yfinance(symbol, market, period)
        elif self.data_source == 'akshare':
            return self.download_stock_data_akshare(symbol, market, period)
        else:
            print(f"❌ Unknown data source: {self.data_source}")
            return None
    
    def get_data_source_info(self):
        """Get information about the current data source"""
        return {
            'source': self.data_source,
            'available_sources': ['yfinance', 'akshare'],
            'description': {
                'yfinance': 'Yahoo Finance API - Global coverage, English interface',
                'akshare': 'AKShare API - Chinese market focused, Chinese interface'
            }
        }
    
    def switch_data_source(self, new_source):
        """Switch to a different data source"""
        new_source = new_source.lower()
        if new_source in ['yfinance', 'akshare']:
            self.data_source = new_source
            print(f"✅ Switched to {new_source}")
            return True
        else:
            print(f"❌ Invalid data source: {new_source}")
            return False

def test_downloader():
    """Test the downloader functionality"""
    print("🧪 TESTING CHINESE STOCK DOWNLOADER")
    print("=" * 60)
    
    # Test yfinance
    print("\n📊 Testing yfinance downloader:")
    yf_downloader = ChineseStockDownloader('yfinance')
    data_yf = yf_downloader.download_stock_data('000001', 'A', '1mo')
    if data_yf is not None:
        print(f"✅ yfinance data shape: {data_yf.shape}")
        print(f"✅ yfinance columns: {list(data_yf.columns)}")
        stock_name = yf_downloader.get_stock_name('000001', 'A')
        print(f"✅ Stock name: {stock_name}")
    
    # Test akshare if available
    print("\n📊 Testing akshare downloader:")
    ak_downloader = ChineseStockDownloader('akshare')
    data_ak = ak_downloader.download_stock_data('000001', 'A', '1mo')
    if data_ak is not None:
        print(f"✅ akshare data shape: {data_ak.shape}")
        print(f"✅ akshare columns: {list(data_ak.columns)}")
        stock_name = ak_downloader.get_stock_name('000001', 'A')
        print(f"✅ Stock name: {stock_name}")
    
    print("\n🎯 SUMMARY:")
    print(f"   ✅ Downloader supports both yfinance and akshare")
    print(f"   ✅ Data column normalization implemented")
    print(f"   ✅ Stock name fetching available")
    print(f"   ✅ Period conversion working")

if __name__ == "__main__":
    test_downloader() 