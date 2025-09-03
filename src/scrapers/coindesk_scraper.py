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

from .base_scraper import BaseScraper

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
        self.source_configs = [
            {
                'url': f"{self.base_url}/tag/bitcoin/",
                'supports_pagination': True,
                'description': 'Bitcoin Tag Page'
            },
            {
                'url': f"{self.base_url}/tag/cryptocurrency/",
                'supports_pagination': True,
                'description': 'Cryptocurrency Tag Page'
            },
            {
                'url': f"{self.base_url}/",
                'supports_pagination': False,  # Homepage page 2 = 404
                'description': 'Homepage'
            },
            {
                'url': f"{self.base_url}/markets/",
                'supports_pagination': False,  # Markets page 2 = 404
                'description': 'Markets Section'
            },
            {
                'url': f"{self.base_url}/policy/",
                'supports_pagination': False,  # Single page
                'description': 'Policy Section'
            },
            {
                'url': f"{self.base_url}/tech/",
                'supports_pagination': False,  # Single page
                'description': 'Tech Section'
            }
        ]

    def get_bitcoin_headlines(self, days_back: int = 7,
                              max_pages_per_source: int = 3) -> List[Dict]:
        """
        Scrapes Bitcoin headlines from multiple CoinDesk sections.
        :param days_back: Only include articles published within the last
        'days_back' days. Defaults to 7.
        :param max_pages_per_source: Maximum paginated pages to fetch per
        section URL. Defaults to 3.
        :return:
        """
        logger.info(f"Scraping Bitcoin headlines from {self.source_name}")
        headlines = []

        for config in self.source_configs:
            base_url = config['url']
            supports_pagination = config['supports_pagination']
            description = config['description']

            logger.info(f"Processing {description}: {base_url}")

            if supports_pagination:
                logger.info(f"  Using pagination: up to {max_pages_per_source} pages")

                for page_num in range(1, max_pages_per_source + 1):
                    page_url = self._build_paginated_url(base_url, page_num)
                    logger.info(f"    Trying page {page_num}: {page_url}")

                    try:
                        response = self._make_request(page_url)
                        if not response:
                            logger.info(f"      Page {page_num}: No response - stopping pagination")
                            break

                        soup = self._parse_html(response.content)
                        page_headlines = self._parse_article_list(soup)

                        if page_headlines:
                            logger.info(f"      Page {page_num}: Found {len(page_headlines)} headlines")
                            headlines.extend(page_headlines)
                        else:
                            logger.info(f"      Page {page_num}: No headlines - end of pages")
                            break

                    except Exception as e:
                        logger.error(f"      Page {page_num}: Error - {e}")
                        break


            else:
                # Single page for sources that don't support pagination
                logger.info(f"  Single page source")
                try:
                    response = self._make_request(base_url)
                    if response:
                        soup = self._parse_html(response.content)
                        page_headlines = self._parse_article_list(soup)
                        if page_headlines:
                            logger.info(f"    Found {len(page_headlines)} headlines")
                            headlines.extend(page_headlines)
                except Exception as e:
                    logger.error(f"    Error: {e}")

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

        try:
            bitcoin_headlines.sort(key=lambda x: x.get('published_at', datetime.min), reverse=True)
        except Exception as e:
            logger.debug(f"Error sorting headlines: {e}")

        final_headlines = bitcoin_headlines
        logger.info(f"Successfully filtered to {len(final_headlines)} Bitcoin headlines")
        return final_headlines

    def _build_paginated_url(self, base_url: str, page_num: int) -> str:
        """
        Constructs a page-specific URL for a given CoinDesk section.
        :param base_url: The section URL to paginate.
        :param page_num: The page index.
        :return: The URL to fetch for the requested page.
        """
        if page_num == 1:
            return base_url

        if base_url.endswith('/'):
            return f"{base_url}{page_num}"
        elif '/tag/' in base_url:
            return f"{base_url}/{page_num}"
        elif base_url == self.base_url or base_url == f"{self.base_url}/":
            return f"{self.base_url}/?page={page_num}"
        else:
            separator = "?" if "?" not in base_url else "&"
            return f"{base_url}{separator}page={page_num}"

    def _parse_article_list(self, soup: BeautifulSoup) -> List[Dict]:
        """
        PArse a page into the headline dictionary using strategy fallbacks.
        :param soup: Parsed HTML document for a CoinDesk page.
        :return: Zero or more headline dictionaries extracted from the page.
        """
        headlines = []

        try:
            headlines = self._strategy_link_based(soup)
            if headlines:
                logger.info(
                    f"Link-based strategy found {len(headlines)} headlines")
                return headlines
        except Exception as e:
            logger.error(f"Link-based strategy failed: {e}")

        # Fallbacks
        try:
            headlines = self._strategy_generic_articles(soup)
            if headlines:
                logger.info(
                    f"Generic articles strategy found {len(headlines)} headlines")
                return headlines
        except Exception as e:
            logger.error(f"Generic articles strategy failed: {e}")

        # Heading-based
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
        """
        Extract headlines by scanning anchor tags and matching URLs.
        :param soup: Parsed HTML document.
        :return: Headline dictionaries derived from qualifying links.
        """
        headlines = []

        # Look for links that look like news articles
        news_patterns = [
            r'/\d{4}/\d{2}/\d{2}/',  # Date-based URLs (most reliable)
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
        """
        Extract a publication date from a URL, falling back to current date.
        :param url: URL to inspect.
        :return: date inferred from the URL or 'datetime.now() if no date
        is recognizable.
        """
        try:
            date_pattern = r'/(\d{4})/(\d{2})/(\d{2})/'
            match = re.search(date_pattern, url)

            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
        except Exception as e:
            logger.debug(f"Error extracting date from URL {url}: {e}")

        # Fallback to current date
        return datetime.now()

    def _safe_extract_time_from_context(self, link_element,
                                        base_date: datetime) -> datetime:
        """
        Potentially parse time from nearby context.
        :param link_element: the element associated with the headline.
        :param base_date: A date to which the parsed time component may be added to.
        :return: A 'datetime' that includes the parsed time if found,
        otherwise the original 'base_date'
        """
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

        return base_date  # Return the base date if time extraction fails

    def _looks_like_coindesk_timestamp(self, text: str) -> bool:
        """
        Checks whether 'text' matches CoinDesk timestamp stype for
        publication.
        :param text: Query to be checked.
        :return: True if the text appears to match the expected pattern,
        otherwise False.
        """
        try:
            if not text or len(text.strip()) < 10:
                return False

            # Pattern for: "Aug 25, 2025, 9:00 a.m."
            coindesk_pattern = r'[A-Za-z]{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+[ap]\.m\.'
            return bool(re.search(coindesk_pattern, text))
        except Exception as e:
            logger.debug(f"Error checking timestamp pattern: {e}")
            return False

    def _safe_parse_coindesk_timestamp(self, datetime_str: str) -> Optional[
        datetime]:
        """
        Parse CoinDesk-style publication timestamp safely.
        :param datetime_str: Raw timestamp text.
        :return: A parsed 'datetime' if recognized; otherwise 'None'
        """
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

            try:
                return datetime.strptime(datetime_str, '%b %d, %Y')
            except ValueError:
                pass

        except Exception as e:
            logger.debug(f"Error parsing timestamp '{datetime_str}': {e}")

        return None

    def _strategy_generic_articles(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract headlines from generic '<article>' blocks.
        :param soup: Parsed HTML document.
        :return: Headline dictionaries built from generic article nodes.
        """
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
        """
        Extract headlines by scanning heading tags
        :param soup: Parsed HTML document.
        :return: Headline dictionaries derived from headings.
        """
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


def test_coindesk_scraper_custom(days_back=14, max_pages=5):
    """
    Manual smoke test for CoinDesk scraper with pagination.
    :param days_back: How many days back to include.
    :param max_pages: Maximum pages per source to crawl.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print(f"TESTING COINDESK SCRAPER WITH PAGINATION")
    print(
        f" Getting headlines from last {days_back} days ({max_pages} pages per source)")
    print("=" * 70)

    scraper = CoindeskScraper()
    headlines = scraper.get_bitcoin_headlines(days_back=days_back,
                                              max_pages_per_source=max_pages)

    print(f"\n RESULTS: Scraped {len(headlines)} headlines")
    print("=" * 60)

    if headlines:
        times_with_hours = 0
        display_limit = min(15, len(headlines))  # Show max 15 for readability

        for i, headline in enumerate(headlines, 1):
            pub_time = headline['published_at']

            # Check if we got actual time (not just midnight)
            has_time = pub_time.hour != 0 or pub_time.minute != 0
            if has_time:
                times_with_hours += 1
                time_marker = ""
            else:
                time_marker = ""

            # Only display first few headlines to avoid spam
            if i <= display_limit:
                print(
                    f"{i:2d}. {'' if headline['bitcoin_related'] else ''} {headline['title']}")
                if headline['url']:
                    print(f"     {headline['url']}")
                print(
                    f"    {time_marker} {headline['published_at'].strftime('%Y-%m-%d %H:%M')}")
                print()
            elif i == display_limit + 1:
                print(f"... and {len(headlines) - display_limit} more headlines")
                print()

        # Show date distribution
        from collections import Counter

        dates = [h['published_at'].strftime('%Y-%m-%d') for h in headlines]
        date_counts = Counter(dates)

        print(f"  Results:")
        print(f"   • Total headlines: {len(headlines)}")
        print(
            f"   • Bitcoin-related: {sum(1 for h in headlines if h['bitcoin_related'])}")
        print(f"   • With full timestamps: {times_with_hours}")
        print(f"   • With date only: {len(headlines) - times_with_hours}")
        print(
            f"   • Sources configured: {len(scraper.source_configs)}")
        print(f"\n Headlines by date:")
        for date, count in sorted(date_counts.items(), reverse=True):
            print(f"   • {date}: {count} headlines")
        print("    CoinDesk scraper with pagination is working")

    else:
        print("No headlines found")

if __name__ == "__main__":
    test_coindesk_scraper_custom(days_back=20, max_pages=5)  #
