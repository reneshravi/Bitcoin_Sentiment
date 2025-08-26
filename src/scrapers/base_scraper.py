"""
File: base_scraper.py
Description: Base scraper class for scraping news sources
Created by: Renesh Ravi
"""

import time
import logging
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from config.settings import (
    USER_AGENT, REQUEST_DELAY, TIMEOUT, MAX_RETRIES
)


logger = logging.getLogger(__name__)

class BaseScrapper(ABC):
    """
    Base class for all news scrapers
    """

    def __init__(self, base_url: str, source_name: str):
        """
        Initialize the scraper with a base URL and source identifier.
        :param base_url: The root URL for the news source to be scraped.
        :param source_name: Human-readable name of the news source.
        """
        self.base_url = base_url
        self.source_name = source_name
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def _make_request(self, url: str, retries: int = 0) -> Optional[
        requests.Response]:
        """
        Perform an HTTP GET request with retry logic and a built-in rate limiter.
        :param url: The target URL to request.
        :param retries: Current retry count (used internally for
        recursion). Defaults to 0.
        :return: A valid Response object if successful or None if the request ultimately fails.
        """
        try:
            time.sleep(REQUEST_DELAY)
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")

            if retries < MAX_RETRIES:
                wait_time = (retries + 1) * 2
                logger.info(f"Retrying in {wait_time} seconds")
                time.sleep(wait_time)
                return self._make_request(url, retries + 1)
            else:
                logger.error(f"Max retries exceeded for {url}")
                return None

    def _parse_html(self, html_content: str) -> BeautifulSoup:
        """
        Parse raw HTML content into a BeautifulSoup object.
        :param html_content: The raw HTML content as a string.
        :return: A BeautifulSoup object parsed with the 'html.parser'
        """
        return BeautifulSoup(html_content, 'html.parser')

    def _clean_text(self, text: str) -> str:
        """
        Cleans and normalizes the text from the HTML extraction.
        :param text: The raw text string to be cleaned.
        :return: The normalized text with extra whitespace removed and common
        Unicode characters (non-breaking spaces, smart quotes, curly apostrophes) replaced with plain ASCII equivalents.
        """
        if not text:
            return ""

        cleaned = ' '.join(text.split())

        cleaned = cleaned.replace('\u00a0', ' ')
        cleaned = cleaned.replace('\u2019', "'")
        cleaned = cleaned.replace('\u201c', '"')
        cleaned = cleaned.replace('\u201d', '"')

        return cleaned.strip()

    def _build_absolute_url(self, relative_url: str) -> str:
        """
        Construct an absolute URL from a relative path using the scraper's base URL.
        :param relative_url: A relative path or partial URL (e.g., '/news/article123')
                            extracted from the HTML.
        :return: Absolute URL that combines the scraper's base URL with the provided relative path.
        """
        return urljoin(self.base_url, relative_url)

    def _is_bitcoin_related(self, text: str) -> bool:
        """
        Determine whether a given text is related to Bitcoin or cryptocurrency.
        :param text: The text string (e.g., headline or article snippet) to check for Bitcoin-related content.
        :return: True if the text contains any Bitcoin or crypto-related keywords, otherwise False.
        """
        bitcoin_keywords = {
            'bitcoin', 'btc', 'blockchain', 'cryptocurrency', 'crypto',
            'hodl', 'mining', 'halving', 'whale', 'pump', 'dump', 'moon',
            'bearish', 'bullish', 'bear', ' bull'
        }

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bitcoin_keywords)

    def _extract_publish_date(self, article_element) -> Optional[datetime]:
        """
        Extract the date from an article element.
        :param article_element:  The HTML element or parsed object
        representing the article container
        :return:  The extracted publish date if available or a
        default placeholder value.
        """
        return datetime.now()

    @abstractmethod
    def get_bitcoin_headlines(self, limit: int = 50, days_back: int = 7) ->\
            List[Dict]:
            pass

    @abstractmethod
    def _parse_article_list(self, soup: BeautifulSoup) -> List[Dict]:
        pass




