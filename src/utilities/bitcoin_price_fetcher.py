"""
Bitcoin Price Data Fetcher using CoinGecko API

This module fetches current and historical Bitcoin price data from CoinGecko.
CoinGecko is free to use and doesn't require an API key for basic endpoints.

Created by: Renesh Ravi
"""

import logging
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BitcoinPriceFetcher:
    """
    Fetches Bitcoin price data from CoinGecko API
    
    The CoinGecko API provides free cryptocurrency market data without
    requiring authentication. Rate limited to ~50 requests per minute.
    """
    
    def __init__(self):
        """Initialize the Bitcoin price fetcher"""
        
        # CoinGecko API base URL
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Create a requests session for connection pooling
        # This is more efficient than making individual requests
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'bitcoin-sentiment-analyzer/1.0'
        })
        
        # Rate limiting configuration
        # CoinGecko free tier: ~50 requests per minute
        self.request_delay = 1.2  # Wait 1.2 seconds between requests
        self.last_request_time = 0
        
        logger.info("BitcoinPriceFetcher initialized with CoinGecko API")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a rate-limited request to CoinGecko API
        
        This handles rate limiting, error handling, and retry logic.
        
        Args:
            endpoint: API endpoint path (e.g., "/simple/price")
            params: Query parameters as dictionary
        
        Returns:
            JSON response as dictionary, or None if request failed
        """
        
        # Enforce rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Build full URL
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Make the HTTP GET request
            response = self.session.get(url, params=params, timeout=10)
            self.last_request_time = time.time()
            
            # Check for HTTP errors (4xx, 5xx status codes)
            response.raise_for_status()
            
            # Parse and return JSON
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {url}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            return None
    
    def get_current_price(self) -> Optional[float]:
        """
        Get the current Bitcoin price in USD
        
        Uses CoinGecko's /simple/price endpoint which is fast and lightweight.
        
        Returns:
            Current Bitcoin price in USD, or None if request failed
        
        Example:
            >>> fetcher = BitcoinPriceFetcher()
            >>> price = fetcher.get_current_price()
            >>> print(f"Bitcoin: ${price:,.2f}")
            Bitcoin: $61,234.50
        """
        
        logger.info("Fetching current Bitcoin price")
        
        # Call CoinGecko simple price endpoint
        data = self._make_request("/simple/price", {
            'ids': 'bitcoin',           # Cryptocurrency ID
            'vs_currencies': 'usd'      # Compare to USD
        })
        
        # Extract price from response
        if data and 'bitcoin' in data and 'usd' in data['bitcoin']:
            price = float(data['bitcoin']['usd'])
            logger.info(f"Current Bitcoin price: ${price:,.2f}")
            return price
        else:
            logger.error("Failed to fetch current Bitcoin price")
            return None
    
    def get_historical_prices(self, days: int = 30) -> List[Dict]:
        """
        Get historical Bitcoin prices for the last N days
        
        Returns daily price points suitable for correlation analysis.
        The free CoinGecko API allows up to 365 days of daily data.
        
        Args:
            days: Number of days of historical data (1-365)
                 Values > 365 will be capped at 365
        
        Returns:
            List of dictionaries with daily price data:
            [
                {'date': '2025-09-01', 'price': 60500.00, 'timestamp': 1725148800000},
                {'date': '2025-09-02', 'price': 61200.00, 'timestamp': 1725235200000},
                ...
            ]
        
        Example:
            >>> fetcher = BitcoinPriceFetcher()
            >>> prices = fetcher.get_historical_prices(days=7)
            >>> print(f"Retrieved {len(prices)} days of price data")
            Retrieved 7 days of price data
        """
        
        # Cap at 365 days for free API tier
        if days > 365:
            logger.warning(f"Requested {days} days, capping at 365 for free API")
            days = 365
        
        logger.info(f"Fetching {days} days of historical Bitcoin prices")
        
        # Call CoinGecko market chart endpoint
        data = self._make_request("/coins/bitcoin/market_chart", {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'daily'  # Daily granularity
        })
        
        if not data or 'prices' not in data:
            logger.error("Failed to fetch historical Bitcoin prices")
            return []
        
        # Convert timestamp-price pairs to readable format
        price_data = []
        
        for timestamp_ms, price in data['prices']:
            # Convert Unix timestamp (milliseconds) to date
            date_obj = datetime.fromtimestamp(timestamp_ms / 1000)
            date_str = date_obj.strftime('%Y-%m-%d')
            
            price_data.append({
                'date': date_str,
                'price': round(price, 2),
                'timestamp': timestamp_ms
            })
        
        logger.info(f"Retrieved {len(price_data)} days of price data")
        
        if price_data:
            prices = [p['price'] for p in price_data]
            logger.info(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
        
        return price_data
    
    def get_price_statistics(self, days: int = 30) -> Dict:
        """
        Get comprehensive Bitcoin price statistics for a period
        
        Calculates various price metrics useful for analysis.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with price statistics:
            {
                'current_price': 61234.50,
                'period_change_percent': 5.2,
                'volatility_percent': 12.5,
                'min_price': 58000.00,
                'max_price': 63000.00,
                ...
            }
        """
        
        price_data = self.get_historical_prices(days)
        
        if len(price_data) < 2:
            logger.warning("Insufficient price data for statistics")
            return {}
        
        prices = [p['price'] for p in price_data]
        dates = [p['date'] for p in price_data]
        
        # Basic price statistics
        current_price = prices[-1]
        start_price = prices[0]
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        # Calculate price change over period
        total_change = current_price - start_price
        total_change_percent = (total_change / start_price) * 100
        
        # Calculate daily changes
        daily_changes = []
        for i in range(1, len(prices)):
            change_percent = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            daily_changes.append(change_percent)
        
        avg_daily_change = sum(daily_changes) / len(daily_changes) if daily_changes else 0
        
        # Calculate volatility (price range as percentage of average)
        volatility = ((max_price - min_price) / avg_price) * 100
        
        statistics = {
            'period_days': days,
            'price_data': price_data,
            'current_price': current_price,
            'start_price': start_price,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'total_change': total_change,
            'total_change_percent': total_change_percent,
            'avg_daily_change_percent': avg_daily_change,
            'volatility_percent': volatility,
            'date_range': f"{dates[0]} to {dates[-1]}"
        }
        
        logger.info(f"Price statistics calculated for {days} days")
        logger.info(f"Period change: {total_change_percent:+.2f}%, Volatility: {volatility:.1f}%")
        
        return statistics


# Test function
def test_bitcoin_price_fetcher():
    """
    Test the Bitcoin price fetcher to verify API connectivity
    """
    
    print("=" * 60)
    print("TESTING BITCOIN PRICE FETCHER")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create fetcher
    fetcher = BitcoinPriceFetcher()
    
    # Test 1: Current price
    print("\n1️⃣  Testing current price:")
    print("-" * 40)
    current = fetcher.get_current_price()
    
    if current:
        print(f"   ✅ SUCCESS: ${current:,.2f}")
    else:
        print("   ❌ FAILED")
    
    # Test 2: Historical prices
    print("\n2️⃣  Testing historical prices (7 days):")
    print("-" * 40)
    historical = fetcher.get_historical_prices(days=7)
    
    if historical:
        print(f"   ✅ SUCCESS: {len(historical)} days retrieved")
        print("   Last 3 days:")
        for day in historical[-3:]:
            print(f"      {day['date']}: ${day['price']:,.2f}")
    else:
        print("   ❌ FAILED")
    
    # Test 3: Price statistics
    print("\n3️⃣  Testing price statistics (30 days):")
    print("-" * 40)
    stats = fetcher.get_price_statistics(days=30)
    
    if stats:
        print(f"   ✅ SUCCESS")
        print(f"   Period: {stats['date_range']}")
        print(f"   Current: ${stats['current_price']:,.2f}")
        print(f"   30-day change: {stats['total_change_percent']:+.2f}%")
        print(f"   Volatility: {stats['volatility_percent']:.1f}%")
    else:
        print("   ❌ FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    tests_passed = sum([current is not None, bool(historical), bool(stats)])
    print(f"Tests passed: {tests_passed}/3")
    
    if tests_passed == 3:
        print("✅ All tests passed - ready for correlation analysis!")
    else:
        print("⚠️  Some tests failed - check network/API")
    
    print("=" * 60)


if __name__ == "__main__":
    test_bitcoin_price_fetcher()
