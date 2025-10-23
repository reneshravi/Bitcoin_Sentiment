"""
Bitcoin Price-Sentiment Correlation Analysis

Calculates statistical correlations between news sentiment and Bitcoin price
movements to identify predictive relationships.

Created by: Renesh Ravi
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """
    Stores results from a correlation analysis
    
    Attributes:
        correlation_coefficient: Pearson r value from -1 to 1
            -1 = perfect negative correlation
             0 = no correlation
            +1 = perfect positive correlation
        p_value: Statistical significance (p < 0.05 is significant)
        is_significant: Whether correlation is statistically meaningful
        sample_size: Number of data points used in calculation
        interpretation: Human-readable explanation
    """
    correlation_coefficient: float
    p_value: float
    is_significant: bool
    sample_size: int
    interpretation: str


class CorrelationAnalyzer:
    """
    Analyzes correlation between Bitcoin sentiment and price movements
    
    Uses Pearson correlation coefficient to measure linear relationships
    between two variables.
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize correlation analyzer
        
        Args:
            significance_threshold: P-value threshold for significance
                                   Default 0.05 = 95% confidence level
        """
        self.significance_threshold = significance_threshold
        logger.info(f"CorrelationAnalyzer initialized (p-value threshold: {significance_threshold})")
    
    def calculate_daily_correlation(self, sentiment_data: List[Dict], 
                                   price_data: List[Dict]) -> CorrelationResult:
        """
        Calculate correlation between daily sentiment and daily Bitcoin prices
        
        This answers the question: "Do bullish headlines occur on days when
        Bitcoin price is higher?"
        
        Process:
        1. Group headlines by date and average sentiment scores
        2. Match sentiment dates with price dates
        3. Calculate Pearson correlation coefficient
        4. Determine statistical significance
        
        Args:
            sentiment_data: List of headlines with sentiment_score and published_at
            price_data: List of price points with date and price
        
        Returns:
            CorrelationResult with statistical analysis
        """
        
        logger.info("Calculating daily sentiment-price correlation")
        
        # Prepare sentiment data (convert to daily averages)
        sentiment_df = self._prepare_sentiment_dataframe(sentiment_data)
        
        # Prepare price data
        price_df = pd.DataFrame(price_data)
        
        # Check if we have enough data
        if sentiment_df.empty or price_df.empty:
            logger.warning("Insufficient data for correlation analysis")
            return self._empty_result("Insufficient data provided")
        
        # Merge on matching dates (inner join = only days with both sentiment and price)
        merged_df = pd.merge(sentiment_df, price_df, on='date', how='inner')
        
        if len(merged_df) < 3:
            logger.warning(f"Only {len(merged_df)} overlapping days - need at least 3")
            return self._empty_result(f"Only {len(merged_df)} overlapping days found")
        
        # Calculate Pearson correlation
        # scipy.stats.pearsonr returns (correlation_coefficient, p_value)
        correlation_coef, p_value = stats.pearsonr(
            merged_df['daily_avg_sentiment'],  # X variable: sentiment
            merged_df['price']                  # Y variable: price
        )
        
        # Determine if statistically significant
        is_significant = p_value < self.significance_threshold
        
        # Generate human-readable interpretation
        interpretation = self._interpret_correlation(
            correlation_coef, p_value, is_significant, "daily_prices"
        )
        
        logger.info(f"Correlation: r={correlation_coef:.4f}, p={p_value:.4f}, significant={is_significant}")
        
        return CorrelationResult(
            correlation_coefficient=correlation_coef,
            p_value=p_value,
            is_significant=is_significant,
            sample_size=len(merged_df),
            interpretation=interpretation
        )
    
    def calculate_price_change_correlation(self, sentiment_data: List[Dict], 
                                          price_data: List[Dict]) -> CorrelationResult:
        """
        Calculate correlation between sentiment and daily price changes
        
        This often shows stronger relationships than absolute prices because
        sentiment may better predict price movements rather than absolute levels.
        
        Answers: "Does bullish sentiment occur when Bitcoin is rising?"
        
        Args:
            sentiment_data: List of headlines with sentiment scores
            price_data: List of price points
        
        Returns:
            CorrelationResult analyzing sentiment vs price movements
        """
        
        logger.info("Calculating sentiment vs price change correlation")
        
        sentiment_df = self._prepare_sentiment_dataframe(sentiment_data)
        price_df = pd.DataFrame(price_data)
        
        if len(price_df) < 2:
            return self._empty_result("Need at least 2 days of price data")
        
        # Sort by date and calculate daily price changes
        price_df = price_df.sort_values('date')
        price_df['price_change_pct'] = price_df['price'].pct_change() * 100
        
        # Remove first row (has NaN for price change)
        price_df = price_df.dropna()
        
        # Merge with sentiment data
        merged_df = pd.merge(sentiment_df, price_df, on='date', how='inner')
        
        if len(merged_df) < 3:
            return self._empty_result(f"Only {len(merged_df)} overlapping days")
        
        # Calculate correlation
        correlation_coef, p_value = stats.pearsonr(
            merged_df['daily_avg_sentiment'],
            merged_df['price_change_pct']
        )
        
        is_significant = p_value < self.significance_threshold
        interpretation = self._interpret_correlation(
            correlation_coef, p_value, is_significant, "price_changes"
        )
        
        logger.info(f"Price change correlation: r={correlation_coef:.4f}, p={p_value:.4f}")
        
        return CorrelationResult(
            correlation_coefficient=correlation_coef,
            p_value=p_value,
            is_significant=is_significant,
            sample_size=len(merged_df),
            interpretation=interpretation
        )
    
    def analyze_leading_indicator(self, sentiment_data: List[Dict], 
                                  price_data: List[Dict], 
                                  lag_days: int = 3) -> Dict:
        """
        Test if sentiment predicts future price movements
        
        This analyzes whether today's sentiment correlates with price changes
        N days in the future, testing sentiment as a leading indicator.
        
        Args:
            sentiment_data: Headlines with sentiment scores
            price_data: Price points with dates
            lag_days: How many days ahead to check (default 3)
        
        Returns:
            Dictionary with predictive analysis:
            {
                'correlation_coefficient': 0.45,
                'prediction_accuracy': 65.2,
                'is_significant': True,
                ...
            }
        """
        
        logger.info(f"Analyzing sentiment as {lag_days}-day leading indicator")
        
        sentiment_df = self._prepare_sentiment_dataframe(sentiment_data)
        price_df = pd.DataFrame(price_data).sort_values('date')
        
        if len(price_df) < lag_days + 2:
            return {'error': 'Insufficient data for leading indicator analysis'}
        
        # Calculate future price changes
        # shift(-lag_days) moves values up, comparing today with N days later
        price_df['future_price'] = price_df['price'].shift(-lag_days)
        price_df['future_change_pct'] = (
            (price_df['future_price'] - price_df['price']) / price_df['price'] * 100
        )
        
        # Merge with sentiment
        merged_df = pd.merge(sentiment_df, price_df, on='date', how='inner')
        merged_df = merged_df.dropna()
        
        if len(merged_df) < 3:
            return {'error': f'Only {len(merged_df)} data points available'}
        
        # Calculate correlation between today's sentiment and future price change
        correlation_coef, p_value = stats.pearsonr(
            merged_df['daily_avg_sentiment'],
            merged_df['future_change_pct']
        )
        
        is_significant = p_value < self.significance_threshold
        
        # Calculate prediction accuracy
        # If sentiment > 0.1 (bullish) and price goes up, that's correct
        # If sentiment < -0.1 (bearish) and price goes down, that's correct
        predictions_correct = 0
        total_predictions = len(merged_df)
        
        for _, row in merged_df.iterrows():
            sentiment_bullish = row['daily_avg_sentiment'] > 0.1
            price_increased = row['future_change_pct'] > 0
            
            # Correct prediction if both agree
            if (sentiment_bullish and price_increased) or (not sentiment_bullish and not price_increased):
                predictions_correct += 1
        
        prediction_accuracy = (predictions_correct / total_predictions) * 100 if total_predictions > 0 else 0
        
        logger.info(f"Leading indicator: r={correlation_coef:.4f}, accuracy={prediction_accuracy:.1f}%")
        
        return {
            'lag_days': lag_days,
            'correlation_coefficient': correlation_coef,
            'p_value': p_value,
            'is_significant': is_significant,
            'sample_size': len(merged_df),
            'prediction_accuracy': prediction_accuracy,
            'predictions_correct': predictions_correct,
            'total_predictions': total_predictions,
            'interpretation': (
                f"Sentiment {'significantly predicts' if is_significant else 'does not significantly predict'} "
                f"price movements {lag_days} days ahead (accuracy: {prediction_accuracy:.1f}%)"
            )
        }
    
    def _prepare_sentiment_dataframe(self, sentiment_data: List[Dict]) -> pd.DataFrame:
        """
        Convert raw sentiment data to DataFrame with daily averages
        
        Multiple headlines per day are averaged into a single daily sentiment score.
        This is necessary because we have many headlines per day but only one
        price point per day.
        """
        
        if not sentiment_data:
            return pd.DataFrame()
        
        records = []
        
        for item in sentiment_data:
            if isinstance(item, dict):
                # Extract date
                date_str = item.get('published_at', '')
                
                if isinstance(date_str, str):
                    # Handle ISO format timestamps
                    if 'T' in date_str:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        except:
                            continue
                else:
                    date_obj = date_str
                
                records.append({
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'sentiment_score': item.get('sentiment_score', 0)
                })
        
        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # Group by date and calculate daily average sentiment
        daily_sentiment = df.groupby('date')['sentiment_score'].agg([
            ('daily_avg_sentiment', 'mean'),
            ('headline_count', 'count')
        ]).reset_index()
        
        logger.debug(f"Prepared {len(daily_sentiment)} unique days from {len(records)} headlines")
        
        return daily_sentiment
    
    def _interpret_correlation(self, correlation: float, p_value: float, 
                              is_significant: bool, analysis_type: str) -> str:
        """
        Generate human-readable interpretation of correlation
        
        Args:
            correlation: Correlation coefficient
            p_value: P-value from statistical test
            is_significant: Whether correlation is significant
            analysis_type: "daily_prices" or "price_changes"
        """
        
        # Classify strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if correlation > 0 else "negative"
        
        # Context-specific explanation
        if analysis_type == "price_changes":
            if correlation > 0:
                meaning = "Higher sentiment scores coincide with positive price movements"
            else:
                meaning = "Higher sentiment scores coincide with negative price movements"
        else:  # daily_prices
            if correlation > 0:
                meaning = "Higher sentiment occurs on days with higher Bitcoin prices"
            else:
                meaning = "Higher sentiment occurs on days with lower Bitcoin prices"
        
        # Build final interpretation
        if is_significant:
            interpretation = (
                f"Statistically significant {strength} {direction} correlation "
                f"(r={correlation:.3f}, p={p_value:.4f}). {meaning}. "
                f"This relationship is unlikely due to random chance."
            )
        else:
            interpretation = (
                f"{strength.capitalize()} {direction} correlation (r={correlation:.3f}), "
                f"but not statistically significant (p={p_value:.4f}). "
                f"Could be due to random chance."
            )
        
        return interpretation
    
    def _empty_result(self, reason: str) -> CorrelationResult:
        """Return empty result when analysis cannot be performed"""
        return CorrelationResult(
            correlation_coefficient=0.0,
            p_value=1.0,
            is_significant=False,
            sample_size=0,
            interpretation=f"Analysis not possible: {reason}"
        )


# Test function
def test_correlation_analyzer():
    """
    Test correlation analyzer with synthetic data
    """
    
    print("=" * 60)
    print("TESTING CORRELATION ANALYZER")
    print("=" * 60)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create synthetic test data (10 days)
    print("\nCreating synthetic test data...")
    
    # Sentiment data: varies from -0.5 to +0.5
    sentiment_data = []
    for i in range(10):
        date = datetime.now() - timedelta(days=i)
        sentiment_data.append({
            'published_at': date.isoformat(),
            'sentiment_score': 0.5 - (i * 0.1)  # Decreasing sentiment
        })
    
    # Price data: correlates with sentiment
    price_data = []
    for i in range(10):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        # Price follows sentiment trend
        base_price = 60000
        sentiment_effect = (0.5 - (i * 0.1)) * 5000
        price_data.append({
            'date': date,
            'price': base_price + sentiment_effect
        })
    
    print(f"   Created {len(sentiment_data)} days of sentiment data")
    print(f"   Created {len(price_data)} days of price data")
    
    # Create analyzer
    analyzer = CorrelationAnalyzer()
    
    # Test 1: Daily correlation
    print("\n1️⃣  Testing daily price correlation:")
    print("-" * 40)
    result = analyzer.calculate_daily_correlation(sentiment_data, price_data)
    
    print(f"   Correlation: {result.correlation_coefficient:.4f}")
    print(f"   P-value: {result.p_value:.4f}")
    print(f"   Significant: {result.is_significant}")
    print(f"   Sample size: {result.sample_size} days")
    print(f"   Interpretation:")
    print(f"      {result.interpretation}")
    
    # Test 2: Price change correlation
    print("\n2️⃣  Testing price change correlation:")
    print("-" * 40)
    result2 = analyzer.calculate_price_change_correlation(sentiment_data, price_data)
    
    print(f"   Correlation: {result2.correlation_coefficient:.4f}")
    print(f"   P-value: {result2.p_value:.4f}")
    print(f"   Significant: {result2.is_significant}")
    
    # Test 3: Leading indicator
    print("\n3️⃣  Testing leading indicator (3-day):")
    print("-" * 40)
    leading = analyzer.analyze_leading_indicator(sentiment_data, price_data, lag_days=3)
    
    if 'error' in leading:
        print(f"   Error: {leading['error']}")
    else:
        print(f"   Correlation: {leading['correlation_coefficient']:.4f}")
        print(f"   Prediction accuracy: {leading['prediction_accuracy']:.1f}%")
        print(f"   Correct: {leading['predictions_correct']}/{leading['total_predictions']}")
    
    print("\n" + "=" * 60)
    print("✅ Correlation analyzer test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_correlation_analyzer()
