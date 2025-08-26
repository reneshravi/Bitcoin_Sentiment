"""
File: CoindeskScraper.py
Description: Extends base_scraper to serve as a web scraper for CoinDesk
Bitcoin news
Created by: Renesh Ravi
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

from base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class CoindeskScraper(BaseScraper):
    """Headline scraper for CoinDesk Bitcoin news"""

    def __init__(self):
        """
        Initializes the CoinDesk scraper by passing in the appropriate
        arguments.
        """
        super().__init__(
            base_url="https://www.coindesk.com",
            source_name="CoinDesk"
        )
        self.potential_urls = [
            f"{self.base_url}/tag/bitcoin/",  # Bitcoin-specific tag page
            f"{self.base_url}/",  # Homepage (lots of content)
            f"{self.base_url}/markets/",  # Markets section
        ]

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

        for url in self.potential_urls:
            logger.info(f"Trying URL: {url}")

            try:
                response = self._make_request(url)
                if not response:
                    continue

                soup = self._parse_html(response.content)
                page_headlines = self._parse_article_list(soup)

                if page_headlines:
                    logger.info(
                        f"Found {len(page_headlines)} headlines from {url}")
                    headlines.extend(page_headlines)
                    break
                else:
                    logger.info(f"No headlines found from {url}")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

        if not headlines:
            logger.warning("No headlines found from any URL")
            return []


        cutoff_date = datetime.now() - timedelta(days=days_back)
        bitcoin_headlines = []

        for headline in headlines:
            try:
                title_text = headline.get('title', '')
                summary_text = headline.get('summary', '')
                combined_text = f"{title_text} {summary_text}"

                if self._is_bitcoin_related(combined_text):
                    pub_date = headline.get('published_at', datetime.now())
                    if pub_date >= cutoff_date:
                        bitcoin_headlines.append(headline)
            except Exception as e:
                logger.debug(f"Error filtering headline: {e}")
                continue

        # Sort by date and limit results
        try:
            bitcoin_headlines.sort(
                key=lambda x: x.get('published_at', datetime.min),
                reverse=True)
        except Exception as e:
            logger.debug(f"Error sorting headlines: {e}")

        final_headlines = bitcoin_headlines[:limit]
        logger.info(
            f"Successfully filtered to {len(final_headlines)} Bitcoin headlines")
        return final_headlines

    def _parse_article_list(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse articles using the link-based strategy that works reliably"""
        headlines = []

        try:
            headlines = self._strategy_link_based(soup)
            if headlines:
                logger.info(
                    f"Link-based strategy found {len(headlines)} headlines")
                return headlines
        except Exception as e:
            logger.error(f"Link-based strategy failed: {e}")

        # Fallback to generic articles
        try:
            headlines = self._strategy_generic_articles(soup)
            if headlines:
                logger.info(
                    f"Generic articles strategy found {len(headlines)} headlines")
                return headlines
        except Exception as e:
            logger.error(f"Generic articles strategy failed: {e}")

        try:
            headlines = self._strategy_heading_based(soup)
            if headlines:
                logger.info(
                    f"Heading-based strategy found {len(headlines)} headlines")
                return headlines
        except Exception as e:
            logger.error(f"Heading-based strategy failed: {e}")

        logger.warning("All parsing strategies failed")
        return []

    def _strategy_link_based(self, soup: BeautifulSoup) -> List[Dict]:
        """Link-based extraction - proven to work with CoinDesk"""
        headlines = []

        news_patterns = [
            r'/\d{4}/\d{2}/\d{2}/',  # Date-based URLs
            r'/markets/',
            r'/policy/',
            r'/tech/',
            r'/business/',
            r'/news/'
        ]

        all_links = soup.find_all('a', href=True)

        for link in all_links:
            try:
                href = link.get('href', '')
                text = self._clean_text(link.get_text())

                # Skip empty text or very short text
                if not text or len(text) < 10:
                    continue

                if any(re.search(pattern, href) for pattern in news_patterns):
                    pub_date = self._safe_extract_date_from_url(href)

                    time_enhanced_date = self._safe_extract_time_from_context(
                        link, pub_date)

                    headline_data = {
                        'title': text,
                        'url': self._build_absolute_url(href),
                        'summary': '',
                        'source': self.source_name,
                        'published_at': time_enhanced_date,
                        'scraped_at': datetime.now(),
                        'bitcoin_related': self._is_bitcoin_related(text)
                    }

                    headlines.append(headline_data)
            except Exception as e:
                logger.debug(f"Error processing link: {e}")
                continue

        return headlines

    def _safe_extract_date_from_url(self, url: str) -> datetime:
        """Safely extract date from URL with fallback"""
        try:
            date_pattern = r'/(\d{4})/(\d{2})/(\d{2})/'
            match = re.search(date_pattern, url)

            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
        except Exception as e:
            logger.debug(f"Error extracting date from URL {url}: {e}")

        return datetime.now()

    def _safe_extract_time_from_context(self, link_element,
                                        base_date: datetime) -> datetime:
        """Safely try to extract time from surrounding context"""
        try:

            parent = link_element.parent
            if parent:

                time_spans = parent.find_all('span')

                for span in time_spans:
                    text = span.get_text().strip()


                    if self._looks_like_coindesk_timestamp(text):
                        parsed_time = self._safe_parse_coindesk_timestamp(
                            text)
                        if parsed_time:

                            return datetime.combine(base_date.date(),
                                                    parsed_time.time())
        except Exception as e:
            logger.debug(f"Error extracting time from context: {e}")

        return base_date

    def _looks_like_coindesk_timestamp(self, text: str) -> bool:
        """Check if text looks like CoinDesk timestamp - with safe error handling"""
        try:
            if not text or len(text.strip()) < 10:
                return False


            coindesk_pattern = r'[A-Za-z]{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+[ap]\.m\.'
            return bool(re.search(coindesk_pattern, text))
        except Exception as e:
            logger.debug(f"Error checking timestamp pattern: {e}")
            return False

    def _safe_parse_coindesk_timestamp(self, datetime_str: str) -> Optional[
        datetime]:
        """Safely parse CoinDesk timestamp with extensive error handling"""
        try:
            if not datetime_str:
                return None

            datetime_str = datetime_str.strip()


            if 'a.m.' in datetime_str or 'p.m.' in datetime_str:
                datetime_str = datetime_str.replace('a.m.', 'AM').replace(
                    'p.m.', 'PM')


            try:
                return datetime.strptime(datetime_str, '%b %d, %Y, %I:%M %p')
            except ValueError:
                pass

            # Try date-only format
            try:
                return datetime.strptime(datetime_str, '%b %d, %Y')
            except ValueError:
                pass

        except Exception as e:
            logger.debug(f"Error parsing timestamp '{datetime_str}': {e}")

        return None

    def _strategy_generic_articles(self, soup: BeautifulSoup) -> List[Dict]:
        """Generic article extraction"""
        headlines = []
        articles = soup.find_all('article')

        for article in articles:
            try:

                links = article.find_all('a', href=True)

                best_link = None
                best_text = ""

                for link in links:
                    text = self._clean_text(link.get_text())
                    if len(text) > len(best_text) and len(text) > 15:
                        best_link = link
                        best_text = text

                if best_link:
                    href = best_link.get('href', '')
                    pub_date = self._safe_extract_date_from_url(href)

                    headlines.append({
                        'title': best_text,
                        'url': self._build_absolute_url(href),
                        'summary': '',
                        'source': self.source_name,
                        'published_at': pub_date,
                        'scraped_at': datetime.now(),
                        'bitcoin_related': self._is_bitcoin_related(best_text)
                    })
            except Exception as e:
                logger.debug(f"Error processing article: {e}")
                continue

        return headlines

    def _strategy_heading_based(self, soup: BeautifulSoup) -> List[Dict]:
        """Heading-based extraction"""
        headlines = []
        heading_tags = soup.find_all(['h1', 'h2', 'h3', 'h4'])

        for heading in heading_tags:
            try:
                text = self._clean_text(heading.get_text())

                if not text or len(text) < 15:
                    continue

                url = ""
                link = heading.find('a') or heading.find_parent('a')
                if link:
                    href = link.get('href', '')
                    url = self._build_absolute_url(href)
                    pub_date = self._safe_extract_date_from_url(href)
                else:
                    pub_date = datetime.now()

                headlines.append({
                    'title': text,
                    'url': url,
                    'summary': '',
                    'source': self.source_name,
                    'published_at': pub_date,
                    'scraped_at': datetime.now(),
                    'bitcoin_related': self._is_bitcoin_related(text)
                })
            except Exception as e:
                logger.debug(f"Error processing heading: {e}")
                continue

        return headlines


def test_coindesk_scraper():
        """Test the updated CoinDesk scraper"""
        import logging

        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s: %(message)s')

        print(" TESTING COINDESK SCRAPER")
        print("=" * 50)

        scraper = CoindeskScraper()
        headlines = scraper.get_bitcoin_headlines(limit=10)

        print(f"\n RESULTS: Scraped {len(headlines)} headlines")
        print("=" * 60)

        if headlines:
            times_with_hours = 0
            for i, headline in enumerate(headlines, 1):
                pub_time = headline['published_at']

                # Check if we got actual time (not just midnight)
                has_time = pub_time.hour != 0 or pub_time.minute != 0
                if has_time:
                    times_with_hours += 1
                    time_marker = ""
                else:
                    time_marker = ""

                print(
                    f"{i:2d}. {'' if headline['bitcoin_related'] else ''} {headline['title']}")
                if headline['url']:
                    print(f"     {headline['url']}")
                print(
                    f"    {time_marker} {headline['published_at'].strftime('%Y-%m-%d %H:%M')}")
                print()

            print(f"Results:")
            print(f"   • Total headlines: {len(headlines)}")
            print(
                f"   • Bitcoin-related: {sum(1 for h in headlines if h['bitcoin_related'])}")
            print(f"   • With full timestamps: {times_with_hours}")
            print(f"   • With date only: {len(headlines) - times_with_hours}")
            print("     CoinDesk scraper is working")

        else:
            print("No headlines found")

if __name__ == "__main__":
    test_coindesk_scraper()
