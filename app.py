#!/usr/bin/env python3
"""
Flask Dashboard for Bitcoin Sentiment Analysis
This creates a web server that displays your sentiment analysis results
"""

import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import logging
import json
from datetime import datetime, timedelta


sys.path.append(str(Path(__file__).parent / "src"))


from src.scrapers.coindesk_scraper import CoindeskScraper
from src.analysis.sentiment_analyzer import SentimentAnalyzer


app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


scraper = CoindeskScraper()
analyzer = SentimentAnalyzer()


cache = {
    'last_analysis': None,
    'last_update': None,
    'headlines_data': []
}

@app.route('/')
def dashboard():
    """
    Main dashboard page - serves the HTML template
    This is what users see when they visit your website
    """
    return render_template('dashboard.html')

@app.route('/api/sentiment-analysis', methods=['POST'])
def run_sentiment_analysis():
    """
    API endpoint that runs the complete sentiment analysis
    This is called by JavaScript when user clicks "Analyze" button
    """
    try:
        data = request.get_json() or {}
        num_headlines = data.get('num_headlines', 50)
        days_back = data.get('days_back', 7)
        max_pages = data.get('max_pages', 3)
        
        logger.info(f"Starting analysis: {num_headlines} headlines, {days_back} days, {max_pages} pages")

        logger.info("Scraping headlines...")
        headlines = scraper.get_bitcoin_headlines(
            limit=num_headlines,
            days_back=days_back,
            max_pages_per_source=max_pages
        )
        
        if not headlines:
            return jsonify({
                'success': False,
                'error': 'No headlines found'
            })

        logger.info("Analyzing sentiment...")
        headline_texts = [h['title'] for h in headlines]
        sentiment_results = analyzer.analyze_batch(headline_texts)

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
    API endpoint to get cached summary data
    Used to quickly load dashboard without running full analysis
    """
    return jsonify({
        'summary': cache['last_analysis'],
        'last_update': cache['last_update'],
        'headlines_count': len(cache['headlines_data'])
    })

@app.route('/api/headlines')
def get_headlines():
    """
    API endpoint to get detailed headline data
    Used for displaying individual headlines and detailed charts
    """
    return jsonify({
        'headlines': cache['headlines_data'],
        'last_update': cache['last_update']
    })

@app.route('/api/sentiment-trend')
def get_sentiment_trend():
    """
    API endpoint for sentiment trend over time
    Groups headlines by date and calculates daily sentiment averages
    """
    if not cache['headlines_data']:
        return jsonify({'dates': [], 'sentiment_scores': []})

    date_sentiments = {}
    for headline in cache['headlines_data']:
        date_str = headline['published_at'][:10]  # Extract YYYY-MM-DD
        
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


if __name__ == '__main__':
    print("Starting Bitcoin Sentiment Dashboard")

    app.run(
        debug=True,
        host='localhost',
        port=5000
    )
