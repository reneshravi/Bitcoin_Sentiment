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
