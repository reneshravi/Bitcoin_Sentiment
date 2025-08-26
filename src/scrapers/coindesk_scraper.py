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
            f"{self.base_url}/tag/bitcoin/",        # Bitcoin-specific tag page
            f"{self.base_url}/",                    # Homepage
            f"{self.base_url}/markets/",            # Markets
            f"{self.base_url}/tag/cryptocurrency/", # Broader crypto news
            f"{self.base_url}/policy/",             # Policy news
            f"{self.base_url}/tech/",               # Tech news
        ]

    def get_bitcoin_headlines(self, limit: int = 50, days_back: int = 7,
                              max_pages_per_source: int = 3) -> List[Dict]:
        """Scrape Bitcoin headlines from multiple sources with pagination"""
        logger.info(
            f"Scraping Bitcoin headlines from {self.source_name} (limit={limit}, max_pages={max_pages_per_source})")
        all_headlines = []

        # Try ALL URLs with pagination
        for base_url in self.potential_urls:
            logger.info(f"Processing source: {base_url}")

            # Try multiple pages for each source
            for page_num in range(1, max_pages_per_source + 1):
                page_url = self._build_paginated_url(base_url, page_num)
                logger.info(f"  Trying page {page_num}: {page_url}")

                try:
                    response = self._make_request(page_url)
                    if not response:
                        logger.info(f"    Page {page_num}: No response")
                        break  # Stop pagination for this source

                    soup = self._parse_html(response.content)
                    page_headlines = self._parse_article_list(soup)

                    if page_headlines:
                        logger.info(
                            f"    Page {page_num}: Found"
                            f" {len(page_headlines)} headlines")
                        all_headlines.extend(page_headlines)
                    else:
                        logger.info(
                            f"    Page {page_num}: No headlines found - end of pages")
                        break  # No more content, stop pagination for this source

                except Exception as e:
                    logger.error(f"    Page {page_num}: Error - {e}")
                    break  # Stop pagination on error

                # Early exit if we have enough headlines
                if len(all_headlines) >= limit * 2:  # Get extra for filtering
                    logger.info(
                        f"  📊 Collected enough raw headlines ({len(all_headlines)}), moving to next source")
                    break

        if not all_headlines:
            logger.warning("No headlines found from any URL/page")
            return []

        logger.info(f"📊 Total raw headlines collected: {len(all_headlines)}")

        # Remove duplicates based on URL (same article from different pages/sources)
        seen_urls = set()
        unique_headlines = []

        for headline in all_headlines:
            url = headline.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_headlines.append(headline)

        logger.info(
            f"After deduplication: {len(unique_headlines)} unique headlines")

        # Filter for Bitcoin-related content and by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        bitcoin_headlines = []

        for headline in unique_headlines:
            try:
                # Check if Bitcoin-related
                title_text = headline.get('title', '')
                summary_text = headline.get('summary', '')
                combined_text = f"{title_text} {summary_text}"

                if self._is_bitcoin_related(combined_text):
                    # Check date if available
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
            f"🎯 Final result: {len(final_headlines)} Bitcoin headlines (requested: {limit})")
        logger.info(
            f"📈 Sources crawled: {len(self.potential_urls)} URLs × {max_pages_per_source} pages each")

        return final_headlines

    def _build_paginated_url(self, base_url: str, page_num: int) -> str:
        """Build paginated URL for different CoinDesk sections"""

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
        """Parse articles using the link-based strategy that works reliably"""
        headlines = []

        # Use the link-based strategy that we know works
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

        # Last resort: heading-based
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

                # Skip empty text or very short text
                if not text or len(text) < 10:
                    continue

                # Check if URL looks like a news article
                if any(re.search(pattern, href) for pattern in news_patterns):
                    # Safe date extraction
                    pub_date = self._safe_extract_date_from_url(href)

                    # Try to extract time from the link's parent elements
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

        # Fallback to current date
        return datetime.now()

    def _safe_extract_time_from_context(self, link_element,
                                        base_date: datetime) -> datetime:
        """Safely try to extract time from surrounding context"""
        try:
            # Look for time info in the link's parent or nearby elements
            parent = link_element.parent
            if parent:
                # Look for spans that might contain timestamps
                time_spans = parent.find_all('span')

                for span in time_spans:
                    text = span.get_text().strip()

                    # Check if this looks like the CoinDesk timestamp format
                    if self._looks_like_coindesk_timestamp(text):
                        parsed_time = self._safe_parse_coindesk_timestamp(
                            text)
                        if parsed_time:
                            # Combine the date from URL with time from span
                            return datetime.combine(base_date.date(),
                                                    parsed_time.time())
        except Exception as e:
            logger.debug(f"Error extracting time from context: {e}")

        return base_date  # Return the base date if time extraction fails

    def _looks_like_coindesk_timestamp(self, text: str) -> bool:
        """Check if text looks like CoinDesk timestamp - with safe error handling"""
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
        """Safely parse CoinDesk timestamp with extensive error handling"""
        try:
            if not datetime_str:
                return None

            datetime_str = datetime_str.strip()

            # Handle CoinDesk's "a.m." and "p.m." format
            if 'a.m.' in datetime_str or 'p.m.' in datetime_str:
                datetime_str = datetime_str.replace('a.m.', 'AM').replace(
                    'p.m.', 'PM')

            # Try CoinDesk format
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
                # Look for any link with substantial text
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

                # Look for a nearby link
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


def test_coindesk_scraper_custom(limit=50, days_back=14, max_pages=5):
    import logging

# Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print(f"TESTING COINDESK SCRAPER WITH PAGINATION")
    print(
        f" Getting {limit} headlines from last {days_back} days ({max_pages} pages per source)")
    print("=" * 70)

    scraper = CoindeskScraper()
    headlines = scraper.get_bitcoin_headlines(limit=limit, days_back=days_back,
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

        print(f"📊 Results:")
        print(f"   • Total headlines: {len(headlines)}")
        print(
            f"   • Bitcoin-related: {sum(1 for h in headlines if h['bitcoin_related'])}")
        print(f"   • With full timestamps: {times_with_hours}")
        print(f"   • With date only: {len(headlines) - times_with_hours}")
        print(
            f"   • Sources × Pages: {len(scraper.potential_urls)} × {max_pages} = {len(scraper.potential_urls) * max_pages} total pages crawled")
        print(f"\n Headlines by date:")
        for date, count in sorted(date_counts.items(), reverse=True):
            print(f"   • {date}: {count} headlines")
        print("    CoinDesk scraper with pagination is working")

    else:
        print("No headlines found")

if __name__ == "__main__":
    test_coindesk_scraper_custom(limit=100, days_back=20, max_pages=5)  #
