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

from base_scraper import BaseScraper

logger = logging.getLogger(__name__)
class CoindeskScraper(BaseScraper):
    """
    Coindesk Scrapper class
    """

    def __init__(self):
        """
        Initializes the CoinDesk scraper by passing in the appropriate
        arguments
        """
        super().__init__(
            base_url="https://www.coindesk.com",
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

    def _parse_article_list(self, soup: BeautifulSoup) -> List[Dict]:

        headlines = []
        article_selectors = [
            'article[data-module="ContentCard"]',
            '.card-content',
            '.article-card',
            '[data-testid="Card"]'
        ]

        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                break

        if not articles:
            articles = soup.find_all('article')

        for article in articles:
            try:
                headline_data = self._extract_headline_data(article)
                if headline_data:
                    headlines.append(headline_data)
            except Exception as e:
                logger.debug(f"Error parsin article: {e}")
                continue
        return headlines

    def _extract_headline_data(self, article_element) -> Optional[Dict]:
        try:
            title_selectors = [
                'h2 a', 'h3 a', 'h4 a',  # Title in heading with link
                '.card-title a', '.article-title a',  # Common class names
                'a[data-module="Headline"]',  # CoinDesk specific
                '.headline a'
            ]
            title_element = None
            title = ""
            url = ""

            for selector in title_selectors:
                title_element = article_element.select_one(selector)
                if title_element:
                    title = self._clean_text(title_element.get_text())
                    url = title_element.get('href','')
                    break

            if not title:
                for selector in ['h2', 'h3', 'h4', '.card-title',
                                 '.article-title']:
                    title_element = article_element.select_one(selector)
                    if title_element:
                        title = self._clean_text(title_element.get_text())
                        break

            if not title:
                return None

            if url:
                url = self._build_absolute_url(url)
                published_at = self._extract_publish_date(article_element)
                summary_selectors = [
                    '.card-description', '.article-summary', '.excerpt',
                    '.summary', 'p'
                ]

                summary = ""
                for selector in summary_selectors:
                    summary_element = article_element.select_one(selector)
                    if summary_element:
                        summary = self._clean_text(summary_element.get_text())
                        if len(summary) > 50:
                            break

                return {
                    'title': title,
                    'url': url,
                    'summary': summary,
                    'source': self.source_name,
                    'published_at': published_at,
                    'scraped_at': datetime.now(),
                    'bitcoin_related': self._is_bitcoin_related(
                        title + " " + summary)
                }

        except Exception as e:
            logger.debug(f"Error extracting headline data: {e}")
            return None

    def _extract_publish_date(self, article_element) -> datetime:
        """Extract publish date from article element"""
        date_selectors = [
            'time', '[datetime]', '.date', '.publish-date',
            '[data-testid="PublishDate"]', '.card-date'
            ]

        for selector in date_selectors:
            date_element = article_element.select_one(selector)
            if date_element:
                # Try to get datetime attribute first
                datetime_attr = date_element.get('datetime')
                if datetime_attr:
                        try:
                            return datetime.fromisoformat(
                                datetime_attr.replace('Z', '+00:00'))
                        except:
                            pass

                    # Try to parse text content
                date_text = date_element.get_text().strip()
                if date_text:
                        try:
                            # Handle common formats
                            for fmt in ['%Y-%m-%d', '%B %d, %Y', '%b %d, %Y',
                                        '%m/%d/%Y']:
                                try:
                                    return datetime.strptime(date_text, fmt)
                                except:
                                    continue
                        except:
                            pass

            # Default to current time if no date found
        return datetime.now()

    # Quick test function
def test_coindesk_scraper():
        """Test the CoinDesk scraper"""
        scraper = CoindeskScraper()
        headlines = scraper.get_bitcoin_headlines(limit=10)

        print(f"Scraped {len(headlines)} headlines:")
        for i, headline in enumerate(headlines, 1):
            print(f"{i}. {headline['title']}")
            print(f"   Source: {headline['source']}")
            print(f"   Bitcoin related: {headline['bitcoin_related']}")
            print(f"   Published: {headline['published_at']}")
            print()

if __name__ == "__main__":
    test_coindesk_scraper()