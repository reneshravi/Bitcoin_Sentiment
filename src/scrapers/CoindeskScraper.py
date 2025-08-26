"""
File: CoindeskScraper.py
Description: Extends base_scraper to serve as a web scraper for CoinDesk
Bitcoin news
Created by: Renesh Ravi
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

from base_scraper import BaseScrapper

logger = logging.getLogger(__name__)
class CoindeskScraper(BaseScrapper):
    """
    Coindesk Scrapper class
    """

    def __init__(self):
        """
        Initializes the CoinDesk scraper by passing in the appropriate
        arguments
        """
        super().__init__(
            base_url="https://www,coindesk.com",
            source_name="CoinDesk"
        )
        self.bitcoin_section_url = f"{self.base_url}/tag/bitcoin"


    def get_bitcoin_headlines(self, limit: int = 50, days_back: int = 7) -> \
            List[Dict]:
        """
        Scrape bitcoin headlines from CoinDesk.
        :param limit: Maximum number of headlines to return.
        :param days_back: How many days back to search
        :return: List of headline dictionaries.
        """
        logger.info(f"Scraping Bitcoin headlines from {self.source_name}")

        headlines = []
        page = 1
        while len(headlines) < limit:
            page_url = f"{self.bitcoin_section_url}?page={page}"
            response = self._make_request(page_url)

            if not response:
                logger.warning(f"Failed to fetch page {page}")
                break

            soup = self._parse_html(response.content)
            page_headlines = self._parse_article_list(soup)

            if not page_headlines:
                logger.info(f"No more headlines found on page {page}")
                break

            cutoff_date = datetime.now() - timedelta(days=days_back)
            valid_headlines = [
                headline for headline in page_headlines if headline.get(
                    'published_at', datetime.now()) >= cutoff_date
            ]

            headlines.extend(valid_headlines)

            if len(valid_headlines) < len(page_headlines):
                break

            page += 1

            headlines = headlines[:limit]
            headlines.sort(key=lambda x: x.get('published_at',
                                               datetime.min), reverse=True)

            logger.info(f"Successfully scraped {len(headlines)} headlines "
                        f"from {self.source_name}")

            return headlines