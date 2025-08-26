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