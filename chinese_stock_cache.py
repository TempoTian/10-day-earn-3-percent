#!/usr/bin/env python3
"""
Chinese Stock Data Cache Manager
Handles caching of stock data with 24-hour expiration and failed download tracking
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

class ChineseStockCache:
    def __init__(self, cache_dir="chinese_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.data_cache_file = self.cache_dir / "stock_data_cache.json"
        self.failed_cache_file = self.cache_dir / "failed_downloads.json"
        self.recommendation_cache_file = self.cache_dir / "recommendations_cache.json"
        
        # Initialize cache files if they don't exist
        self._init_cache_files()
    
    def _init_cache_files(self):
        """Initialize cache files if they don't exist"""
        if not self.data_cache_file.exists():
            self._save_json_cache({}, self.data_cache_file)
        
        if not self.failed_cache_file.exists():
            self._save_json_cache({}, self.failed_cache_file)
        
        if not self.recommendation_cache_file.exists():
            self._save_json_cache({}, self.recommendation_cache_file)
    
    def _save_json_cache(self, data, file_path):
        """Save data to JSON cache file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving cache to {file_path}: {e}")
    
    def _load_json_cache(self, file_path):
        """Load data from JSON cache file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache from {file_path}: {e}")
            return {}
    
    def _save_pickle_cache(self, data, file_path):
        """Save data to pickle cache file"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving pickle cache to {file_path}: {e}")
    
    def _load_pickle_cache(self, file_path):
        """Load data from pickle cache file"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle cache from {file_path}: {e}")
            return None
    
    def get_cache_key(self, symbol, market='A', period="2y"):
        """Generate cache key for stock data"""
        return f"{symbol}_{market}_{period}"
    
    def is_cache_valid(self, cache_time, max_age_hours=24):
        """Check if cache is still valid (within 24 hours)"""
        if not cache_time:
            return False
        
        try:
            if isinstance(cache_time, str):
                cache_time = datetime.fromisoformat(cache_time.replace('Z', '+00:00'))
            
            age = datetime.now() - cache_time
            return age.total_seconds() < (max_age_hours * 3600)
        except Exception:
            return False
    
    def get_cached_stock_data(self, symbol, market='A', period="2y"):
        """Get cached stock data if valid"""
        cache_data = self._load_json_cache(self.data_cache_file)
        cache_key = self.get_cache_key(symbol, market, period)
        
        if cache_key in cache_data:
            entry = cache_data[cache_key]
            if self.is_cache_valid(entry.get('timestamp')):
                data_file = self.cache_dir / f"{cache_key}_data.pkl"
                if data_file.exists():
                    stock_data = self._load_pickle_cache(data_file)
                    if stock_data is not None:
                        print(f"✅ Using cached data for {symbol} ({market}-shares) - {len(stock_data)} days")
                        return stock_data
        
        return None
    
    def cache_stock_data(self, symbol, market='A', period="2y", stock_data=None):
        """Cache stock data with timestamp"""
        if stock_data is None or stock_data.empty:
            return False
        
        cache_data = self._load_json_cache(self.data_cache_file)
        cache_key = self.get_cache_key(symbol, market, period)
        
        # Save stock data to pickle file
        data_file = self.cache_dir / f"{cache_key}_data.pkl"
        self._save_pickle_cache(stock_data, data_file)
        
        # Update cache metadata
        cache_data[cache_key] = {
            'symbol': symbol,
            'market': market,
            'period': period,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(stock_data),
            'data_file': str(data_file)
        }
        
        self._save_json_cache(cache_data, self.data_cache_file)
        print(f"✅ Cached data for {symbol} ({market}-shares) - {len(stock_data)} days")
        return True
    
    def is_failed_download(self, symbol, market='A'):
        """Check if download failed recently (within 24 hours)"""
        failed_data = self._load_json_cache(self.failed_cache_file)
        cache_key = f"{symbol}_{market}"
        
        if cache_key in failed_data:
            entry = failed_data[cache_key]
            if self.is_cache_valid(entry.get('timestamp'), max_age_hours=24):
                return True
        
        return False
    
    def mark_download_failed(self, symbol, market='A', error_msg=""):
        """Mark download as failed"""
        failed_data = self._load_json_cache(self.failed_cache_file)
        cache_key = f"{symbol}_{market}"
        
        failed_data[cache_key] = {
            'symbol': symbol,
            'market': market,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        }
        
        self._save_json_cache(failed_data, self.failed_cache_file)
        print(f"❌ Marked {symbol} ({market}-shares) as failed download")
    
    def get_cached_recommendation(self, strategy_type):
        """Get cached recommendation results"""
        cache_data = self._load_json_cache(self.recommendation_cache_file)
        
        if strategy_type in cache_data:
            entry = cache_data[strategy_type]
            if self.is_cache_valid(entry.get('timestamp')):
                return entry.get('recommendations', [])
        
        return None
    
    def cache_recommendation(self, strategy_type, recommendations):
        """Cache recommendation results"""
        cache_data = self._load_json_cache(self.recommendation_cache_file)
        
        cache_data[strategy_type] = {
            'strategy_type': strategy_type,
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'count': len(recommendations)
        }
        
        self._save_json_cache(cache_data, self.recommendation_cache_file)
        print(f"✅ Cached recommendations for strategy: {strategy_type}")
    
    def clear_expired_cache(self):
        """Clear expired cache entries"""
        # Clear expired stock data cache
        cache_data = self._load_json_cache(self.data_cache_file)
        expired_keys = []
        
        for key, entry in cache_data.items():
            if not self.is_cache_valid(entry.get('timestamp')):
                expired_keys.append(key)
                # Remove data file
                data_file = self.cache_dir / f"{key}_data.pkl"
                if data_file.exists():
                    data_file.unlink()
        
        for key in expired_keys:
            del cache_data[key]
        
        if expired_keys:
            self._save_json_cache(cache_data, self.data_cache_file)
            print(f"🗑️  Cleared {len(expired_keys)} expired cache entries")
        
        # Clear expired failed downloads
        failed_data = self._load_json_cache(self.failed_cache_file)
        expired_failed = []
        
        for key, entry in failed_data.items():
            if not self.is_cache_valid(entry.get('timestamp'), max_age_hours=24):
                expired_failed.append(key)
        
        for key in expired_failed:
            del failed_data[key]
        
        if expired_failed:
            self._save_json_cache(failed_data, self.failed_cache_file)
            print(f"🗑️  Cleared {len(expired_failed)} expired failed download records")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        cache_data = self._load_json_cache(self.data_cache_file)
        failed_data = self._load_json_cache(self.failed_cache_file)
        rec_cache_data = self._load_json_cache(self.recommendation_cache_file)
        
        stats = {
            'total_cached_stocks': len(cache_data),
            'total_failed_downloads': len(failed_data),
            'total_cached_recommendations': len(rec_cache_data)
        }
        
        return stats
    
    def clear_all_cache(self):
        """Clear all cache data"""
        try:
            # Clear stock data cache
            self._save_json_cache({}, self.data_cache_file)
            
            # Clear failed downloads cache
            self._save_json_cache({}, self.failed_cache_file)
            
            # Clear recommendations cache
            self._save_json_cache({}, self.recommendation_cache_file)
            
            # Remove all pickle data files
            for file_path in self.cache_dir.glob("*_data.pkl"):
                try:
                    file_path.unlink()
                except Exception:
                    pass
            
            print("✅ All cache cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
