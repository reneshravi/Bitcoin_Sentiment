#!/usr/bin/env python3
"""
Flask Dashboard for Bitcoin Sentiment Analysis with Price Correlation
This creates a web server that displays sentiment analysis and price correlation results

Created by: Renesh Ravi
"""

import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import logging
import json
from datetime import datetime, timedelta
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from src.scrapers.coindesk_scraper import CoindeskScraper
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.utilities.bitcoin_price_fetcher import BitcoinPriceFetcher
from src.analysis.correlation_analyzer import CorrelationAnalyzer

# Create Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (created once when server starts)
scraper = CoindeskScraper()
analyzer = None  # Don't load yet
price_fetcher = BitcoinPriceFetcher()
correlation_analyzer = CorrelationAnalyzer()

def get_analyzer():
    """Lazy load sentiment analyzer only when needed"""
    global analyzer
    if analyzer is None:
        logger.info("Loading sentiment analyzer (first time)...")
        analyzer = SentimentAnalyzer()
    return analyzer

# Cache for storing recent results
cache = {
    'last_analysis': None,
    'last_update': None,
    'headlines_data': []
}


# ============================================================================
# DASHBOARD ROUTES
# ============================================================================

@app.route('/')
def dashboard():
    """
    Main dashboard page - serves the HTML template
    """
    return render_template('enhanced_dashboard.html')


# ============================================================================
# SENTIMENT ANALYSIS ROUTES
# ============================================================================

@app.route('/api/sentiment-analysis', methods=['POST'])
def run_sentiment_analysis():
    """
    Run complete sentiment analysis on Bitcoin headlines
    """
    try:
        data = request.get_json() or {}
        num_headlines = data.get('num_headlines', 50)
        days_back = data.get('days_back', 7)
        max_pages = data.get('max_pages', 3)

        logger.info(f"Starting analysis: {num_headlines} headlines, {days_back} days, {max_pages} pages")

        # Step 1: Scrape headlines
        logger.info("Scraping headlines...")
        headlines = scraper.get_bitcoin_headlines(
            days_back=days_back,
            max_pages_per_source=max_pages
        )

        if not headlines:
            return jsonify({
                'success': False,
                'error': 'No headlines found'
            })

        # Step 2: Analyze sentiment
        logger.info("Analyzing sentiment...")
        headline_texts = [h['title'] for h in headlines]
        sentiment_results = get_analyzer().analyze_batch(headline_texts)

        # Step 3: Combine data
        combined_data = []
        for i, headline in enumerate(headlines):
            if i < len(sentiment_results['results']):
                sentiment_result = sentiment_results['results'][i]
                combined_item = {
                    'title': headline['title'],
                    'url': headline['url'],
                    'published_at': headline['published_at'].isoformat(),
                    'source': headline['source'],
                    'sentiment_score': sentiment_result.sentiment_score,
                    'sentiment_label': sentiment_result.sentiment_label,
                    'confidence': sentiment_result.confidence,
                    'probabilities': sentiment_result.probabilities
                }
                combined_data.append(combined_item)

        # Update cache
        cache['last_analysis'] = sentiment_results['summary']
        cache['last_update'] = datetime.now().isoformat()
        cache['headlines_data'] = combined_data

        return jsonify({
            'success': True,
            'summary': sentiment_results['summary'],
            'headlines': combined_data,
            'analysis_date': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/summary')
def get_summary():
    """
    Get cached summary data
    """
    return jsonify({
        'summary': cache['last_analysis'],
        'last_update': cache['last_update'],
        'headlines_count': len(cache['headlines_data'])
    })


@app.route('/api/headlines')
def get_headlines():
    """
    Get detailed headline data
    """
    return jsonify({
        'headlines': cache['headlines_data'],
        'last_update': cache['last_update']
    })


@app.route('/api/sentiment-trend')
def get_sentiment_trend():
    """
    Get sentiment trend over time (daily averages)
    """
    if not cache['headlines_data']:
        return jsonify({'dates': [], 'sentiment_scores': []})

    date_sentiments = {}
    for headline in cache['headlines_data']:
        date_str = headline['published_at'][:10]

        if date_str not in date_sentiments:
            date_sentiments[date_str] = []

        date_sentiments[date_str].append(headline['sentiment_score'])

    dates = []
    sentiment_scores = []

    for date_str in sorted(date_sentiments.keys()):
        dates.append(date_str)
        avg_sentiment = sum(date_sentiments[date_str]) / len(date_sentiments[date_str])
        sentiment_scores.append(avg_sentiment)

    return jsonify({
        'dates': dates,
        'sentiment_scores': sentiment_scores
    })


# ============================================================================
# BITCOIN PRICE ROUTES (NEW)
# ============================================================================

@app.route('/api/bitcoin-price')
def get_bitcoin_price():
    """
    Get current Bitcoin price from CoinGecko
    """
    try:
        current_price = price_fetcher.get_current_price()

        if current_price:
            return jsonify({
                'success': True,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch current Bitcoin price'
            })

    except Exception as e:
        logger.error(f"Error fetching Bitcoin price: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/historical-prices')
def get_historical_prices():
    """
    Get historical Bitcoin price data
    """
    try:
        days = request.args.get('days', default=30, type=int)

        # Cap at 365 for free API
        if days > 365:
            days = 365

        price_data = price_fetcher.get_historical_prices(days=days)

        if price_data:
            return jsonify({
                'success': True,
                'price_data': price_data,
                'days': days
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch historical prices'
            })

    except Exception as e:
        logger.error(f"Error fetching historical prices: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


# ============================================================================
# CORRELATION ANALYSIS ROUTES (NEW)
# ============================================================================

@app.route('/api/price-correlation', methods=['POST'])
def calculate_price_correlation():
    """
    Calculate correlation between sentiment and Bitcoin prices

    This performs three types of correlation analysis:
    1. Daily correlation (sentiment vs absolute price)
    2. Price change correlation (sentiment vs price movements)
    3. Leading indicator analysis (can sentiment predict future prices?)
    """
    try:
        # Check if we have sentiment data
        if not cache['headlines_data']:
            return jsonify({
                'success': False,
                'error': 'No sentiment data available. Run sentiment analysis first.'
            })

        data = request.get_json() or {}
        analysis_days = data.get('days', 30)

        # Fetch Bitcoin price data
        logger.info(f"Fetching {analysis_days} days of price data for correlation")
        price_data = price_fetcher.get_historical_prices(days=analysis_days)

        if not price_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch price data from CoinGecko API'
            })

        # Calculate different types of correlations
        logger.info("Calculating daily price correlation...")
        daily_correlation = correlation_analyzer.calculate_daily_correlation(
            cache['headlines_data'],
            price_data
        )

        logger.info("Calculating price change correlation...")
        price_change_correlation = correlation_analyzer.calculate_price_change_correlation(
            cache['headlines_data'],
            price_data
        )

        logger.info("Analyzing leading indicator...")
        leading_indicator = correlation_analyzer.analyze_leading_indicator(
            cache['headlines_data'],
            price_data,
            lag_days=3
        )

        # Calculate price statistics
        prices = [p['price'] for p in price_data]
        price_stats = {
            'current_price': prices[-1] if prices else 0,
            'period_min': min(prices) if prices else 0,
            'period_max': max(prices) if prices else 0,
            'period_avg': sum(prices) / len(prices) if prices else 0,
            'period_change_pct': ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 else 0,
            'volatility': float(np.std(prices) / np.mean(prices) * 100) if prices else 0
        }

        # Return comprehensive results
        return jsonify({
            'success': True,
            'analysis_period_days': int(analysis_days),
            'daily_correlation': {
                'correlation': float(daily_correlation.correlation_coefficient),
                'p_value': float(daily_correlation.p_value),
                'significant': bool(daily_correlation.is_significant),  # Convert to Python bool
                'sample_size': int(daily_correlation.sample_size),
                'interpretation': str(daily_correlation.interpretation)
            },
            'price_change_correlation': {
                'correlation': float(price_change_correlation.correlation_coefficient),
                'p_value': float(price_change_correlation.p_value),
                'significant': bool(price_change_correlation.is_significant),  # Convert to Python bool
                'sample_size': int(price_change_correlation.sample_size),
                'interpretation': str(price_change_correlation.interpretation)
            },
            'leading_indicator': {
                'lag_days': int(leading_indicator.get('lag_days', 3)),
                'correlation_coefficient': float(leading_indicator.get('correlation_coefficient', 0)),
                'p_value': float(leading_indicator.get('p_value', 1)),
                'is_significant': bool(leading_indicator.get('is_significant', False)),
                'sample_size': int(leading_indicator.get('sample_size', 0)),
                'prediction_accuracy': float(leading_indicator.get('prediction_accuracy', 0)),
                'predictions_correct': int(leading_indicator.get('predictions_correct', 0)),
                'total_predictions': int(leading_indicator.get('total_predictions', 0)),
                'interpretation': str(leading_indicator.get('interpretation', ''))
            },
            'price_statistics': {
                'current_price': float(price_stats['current_price']),
                'period_min': float(price_stats['period_min']),
                'period_max': float(price_stats['period_max']),
                'period_avg': float(price_stats['period_avg']),
                'period_change_pct': float(price_stats['period_change_pct']),
                'volatility': float(price_stats['volatility'])
            },
            'price_data': price_data[-30:],
            'analysis_timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/combined-chart-data')
def get_combined_chart_data():
    """
    Get data for combined sentiment/price visualization

    Only includes dates where BOTH sentiment and price data exist.
    This ensures accurate representation without interpolation.
    """
    try:
        if not cache['headlines_data']:
            return jsonify({
                'success': False,
                'error': 'No sentiment data available'
            })

        days = request.args.get('days', default=30, type=int)

        # Get price data
        logger.info(f"Fetching {days} days of price data for combined chart")
        price_data = price_fetcher.get_historical_prices(days=days)

        if not price_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch price data'
            })

        # Group sentiment by date
        sentiment_by_date = {}
        for headline in cache['headlines_data']:
            date_str = headline['published_at'][:10]  # Extract YYYY-MM-DD

            if date_str not in sentiment_by_date:
                sentiment_by_date[date_str] = []

            sentiment_by_date[date_str].append(headline['sentiment_score'])

        # Calculate daily average sentiment
        daily_sentiment = {}
        for date, scores in sentiment_by_date.items():
            daily_sentiment[date] = sum(scores) / len(scores)

        logger.info(f"Have sentiment data for {len(daily_sentiment)} unique days")

        # Combine with price data - ONLY include dates with both sentiment and price
        combined_data = []
        for price_point in price_data:
            date = price_point['date']
            sentiment_value = daily_sentiment.get(date)

            # Only include this date if we have sentiment data for it
            if sentiment_value is not None:
                combined_data.append({
                    'date': date,
                    'price': price_point['price'],
                    'sentiment': sentiment_value,
                    'headline_count': len(sentiment_by_date.get(date, []))
                })

        logger.info(f"Combined chart: {len(combined_data)} days with both sentiment and price data")

        return jsonify({
            'success': True,
            'combined_data': combined_data,
            'period_days': days,
            'overlapping_days': len(combined_data)
        })

    except Exception as e:
        logger.error(f"Combined chart data failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


# ============================================================================
# TESTING & STATUS ROUTES
# ============================================================================

@app.route('/api/test')
def test_endpoint():
    """
    Test endpoint to verify all components are working
    """
    return jsonify({
        'status': 'API is working!',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'scraper_ready': scraper is not None,
            'analyzer_ready': analyzer is not None,
            'price_fetcher_ready': price_fetcher is not None,
            'correlation_analyzer_ready': correlation_analyzer is not None
        },
        'cache_status': {
            'has_sentiment_data': len(cache['headlines_data']) > 0,
            'last_update': cache['last_update']
        }
    })


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import os

    # Get configuration from environment variables (production) or use defaults (development)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'

    print("=" * 60)
    print("Starting Bitcoin Sentiment Dashboard")
    print("=" * 60)
    print(f"Environment: {'Development' if debug_mode else 'Production'}")
    print(f"Port: {port}")
    print("\nFeatures:")
    print("  • Sentiment Analysis with finBERT")
    print("  • Bitcoin Price Tracking")
    print("  • Statistical Correlation Analysis")
    print("\nPress CTRL+C to quit")
    print("=" * 60)

    app.run(
        debug=debug_mode,
        host='0.0.0.0',  # Allow external connections
        port=port
    )