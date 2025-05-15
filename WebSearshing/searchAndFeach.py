





import json
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup
import requests


class WebArticleSearcher:
    def __init__(self, max_results: int = 5,
                 result_found: int = 0,
                 delay_between_requests: float = 1.0):
        """
        Initialize the WebArticleSearcher.
        
        Args:
            max_results: Maximum number of articles to fetch
            result_found: Counter for results already found
            delay_between_requests: Time to wait between requests
        """
        self.max_results = max_results
        self.result_found = result_found
        self.delay = delay_between_requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Load or create reliability domains rating
        self.reliability_domains = self._load_reliability_domains()

    def _load_reliability_domains(self) -> Dict:
        """Load or create default reliability domains ratings."""
        try:
            with open("reliability_domains.json", 'r') as f:
                return json.load(f)
        except:
            # Default trusted domains with reliability scores (0-1)
            default_domains = {
                # Educational institutions
                "edu": 0.9,
                "ac.uk": 0.9,
                # Government sites
                "gov": 0.9,
                "gov.uk": 0.9,
                # Organizations
                "org": 0.7,
                # Specific high-reliability sources
                "wikipedia.org": 0.8,
                "arxiv.org": 0.9,
                "nih.gov": 0.95,
                "cdc.gov": 0.95,
                "who.int": 0.95,
                "nature.com": 0.95,
                "science.org": 0.95,
                "ieee.org": 0.9,
                "acm.org": 0.9,
                # Top academic institutions
                "mit.edu": 0.95,
                "stanford.edu": 0.95,
                "harvard.edu": 0.95,
                "berkeley.edu": 0.95,
                "ox.ac.uk": 0.95,
                "cam.ac.uk": 0.95,
                # Reliable news sources
                "reuters.com": 0.85,
                "apnews.com": 0.85,
                # Technical documentation
                "docs.python.org": 0.95,
                "developer.mozilla.org": 0.9,
                "stackoverflow.com": 0.75,
                "github.com": 0.75,
            }
            
            # Save for future use
            try:
                with open("reliability_domains.json", 'w') as f:
                    json.dump(default_domains, f, indent=2)
            except:
                pass
                
            return default_domains

    def _assess_domain_reliability(self, url: str) -> float:
        """
        Assess the reliability of a domain based on predefined trusted domains.
        Returns a score between 0 and 1, where 1 is most reliable.
        """
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check for exact domain match
            if domain in self.reliability_domains:
                return self.reliability_domains[domain]
            
            # Check for TLD or domain ending match
            for suffix, score in self.reliability_domains.items():
                if domain.endswith(f".{suffix}"):
                    return score
            
            # Check for partial domain match (for subdomains)
            for trusted_domain, score in self.reliability_domains.items():
                if trusted_domain in domain and "." in trusted_domain:
                    return score * 0.9  # Slightly lower score for subdomains
            
            # Default score for unknown domains
            return 0.5
        except:
            return 0.5

    def _is_paywall_content(self, html_content: str) -> bool:
        """
        Check if content is likely behind a paywall.
        Returns True if paywall indicators are detected.
        """
        paywall_indicators = [
            "subscribe to continue", "subscription required", "to continue reading",
            "create an account", "sign in to read", "premium content",
            "subscribe now", "members only", "paid subscribers only",
            "login to continue", "register to continue", "paywall",
            "subscribe for full access", "subscribe today", "subscribe for unlimited"
        ]
        
        lowercase_html = html_content.lower()
        
        # Check for paywall text indicators
        for indicator in paywall_indicators:
            if indicator in lowercase_html:
                return True
        
        # Check for login forms near the beginning of content
        if re.search(r'<form[^>]*login|<form[^>]*sign.?in', lowercase_html[:5000]):
            return True
                
        # Check content length - very short article body might indicate truncated content
        body_content = re.search(r'<body.*?>(.*?)</body>', lowercase_html, re.DOTALL)
        if body_content:
            text_content = BeautifulSoup(body_content.group(1), 'html.parser').get_text()
            if len(text_content.strip()) < 1500 and any(ind in lowercase_html for ind in paywall_indicators):
                return True
                
        return False

    def _search_duckduckgo(self, query: str, page: int = 1) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo.
        Returns list of article dictionaries with url and title.
        """
        try:
            # DuckDuckGo doesn't have traditional pagination, so we use the 's' parameter
            offset = (page - 1) * 10
            
            # Format URL with query and offset
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&s={offset}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.result'):
                title_elem = result.select_one('.result__title')
                link_elem = result.select_one('.result__url')
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    link = link_elem.get('href')
                    
                    # Clean up URL (DuckDuckGo uses redirects)
                    if 'uddg=' in link:
                        from urllib.parse import unquote
                        link = unquote(link.split('uddg=')[1].split('&')[0])
                    
                    results.append({
                        'title': title,
                        'url': link
                    })
            
            return results
        except Exception as e:
            print(f"Error with DuckDuckGo search: {e}")
            return []

    def _search_bing(self, query: str, page: int = 1) -> List[Dict[str, str]]:
        """
        Search using Bing.
        Returns list of article dictionaries with url and title.
        """
        try:
            # Calculate first result position for pagination
            first = (page - 1) * 10 + 1
            
            # Format URL with query and first result position
            url = f"https://www.bing.com/search?q={quote_plus(query)}&first={first}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.b_algo'):
                title_elem = result.select_one('h2 a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href')
                    
                    if link and title:
                        results.append({
                            'title': title,
                            'url': link
                        })
            
            return results
        except Exception as e:
            print(f"Error with Bing search: {e}")
            return []

    def search(self, query: str, page: int = 1, exclude_urls: Set[str] = None) -> List[Dict[str, str]]:
        """
        Search for articles related to the query using multiple search engines.
        
        Args:
            query: The search query text
            page: Current search page (for pagination)
            exclude_urls: Set of URLs to exclude from results
            
        Returns:
            List of dictionaries containing article URLs and titles
        """
        if exclude_urls is None:
            exclude_urls = set()
        
        all_results = []
        
        # Try multiple search engines for better coverage
        print(f"Searching for '{query}' using multiple search engines...")
        
        # Try DuckDuckGo
        try:
            ddg_results = self._search_duckduckgo(query, page)
            all_results.extend(ddg_results)
            print(f"Found {len(ddg_results)} results from DuckDuckGo")
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
        
        # Try Bing
        try:
            bing_results = self._search_bing(query, page)
            all_results.extend(bing_results)
            print(f"Found {len(bing_results)} results from Bing")
        except Exception as e:
            print(f"Bing search failed: {e}")
        
        # If all search engines failed, fall back to scholarly approach
        if not all_results:
            try:
                from scholarly import scholarly
                print("Falling back to scholarly API...")
                
                # Calculate offset for pagination
                offset = (page - 1) * self.max_results
                
                # Create a search query
                search_query = scholarly.search_pubs(query)
                
                # Skip results for pagination
                for _ in range(offset):
                    try:
                        next(search_query)
                    except StopIteration:
                        print(f"Not enough results to reach page {page}")
                        return []
                
                # Collect results
                count = 0
                while count < self.max_results:
                    try:
                        result = next(search_query)
                        
                        # Extract title
                        if 'bib' in result and 'title' in result['bib']:
                            title = result['bib']['title']
                        else:
                            title = "Unknown Title"
                        
                        # Try to get URL
                        url = None
                        if 'pub_url' in result and result['pub_url']:
                            url = result['pub_url']
                        elif 'cluster_id' in result and result['cluster_id']:
                            url = f"https://scholar.google.com/scholar?cluster={result['cluster_id']}"
                        else:
                            continue
                        
                        all_results.append({
                            'url': url,
                            'title': title
                        })
                        count += 1
                        
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error processing scholarly result: {e}")
                        continue
                        
                print(f"Found {count} results from scholarly API")
                
            except Exception as e:
                print(f"Scholarly API failed: {e}")
        
        # Remove duplicates and excluded URLs, and assess reliability
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            url = result['url']
            if url not in seen_urls and url not in exclude_urls:
                seen_urls.add(url)
                
                # Skip certain file types that aren't easily processable
                if url.lower().endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
                    continue
                
                # Assess reliability score based on domain
                result['reliability'] = self._assess_domain_reliability(url)
                unique_results.append(result)
        
        # Sort by reliability score
        sorted_results = sorted(unique_results, key=lambda x: x.get('reliability', 0), reverse=True)
        
        # Print the top few results with their reliability scores
        if sorted_results:
            print("\nTop search results by reliability:")
            for i, result in enumerate(sorted_results[:5]):
                print(f"{i+1}. {result['title']} - {result['url']} (Score: {result['reliability']:.2f})")
        
        return sorted_results[:min(len(sorted_results), self.max_results * 2)]  # Return more than needed to allow for filtering

    def fetch_article(self, url: str) -> Optional[str]:
        """
        Fetch the content of an article from the given URL with enhanced extraction.
        
        Args:
            url: URL of the article to fetch
                
        Returns:
            The article content as a string, or None if fetching failed
        """
        try:
            print(f"Fetching article from: {url}")
            
            # Fetch the page content with timeout
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Check for paywall
            if self._is_paywall_content(response.text):
                print(f"Paywall detected at {url}, skipping")
                return None
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.select("script, style, iframe, header, footer, nav, aside, .nav, .menu, .banner, .ads, .advertisement, .footer, .header, .sidebar, .comments"):
                element.extract()
            
            # Try multiple extraction strategies
            content = None
            min_content_length = 1000  # Minimum number of characters for valid content
            
            # Strategy 1: Find main article container
            for selector in ['article', 'main', '.article', '.content', '.post', '#content', '#main', 
                           '[role="main"]', '.article-content', '.post-content', '.entry-content']:
                container = soup.select_one(selector)
                if container and len(container.get_text(strip=True)) > min_content_length:
                    content = self._extract_structured_content(container)
                    if content and len(content) > min_content_length:
                        break
            
            # Strategy 2: Find div with most paragraph content
            if not content or len(content) < min_content_length:
                divs_with_content = []
                for div in soup.find_all('div'):
                    paragraphs = div.find_all('p')
                    if paragraphs and len(paragraphs) >= 3:
                        text_length = sum(len(p.get_text(strip=True)) for p in paragraphs)
                        if text_length > min_content_length:
                            divs_with_content.append((div, text_length))
                
                # Sort by content length
                divs_with_content.sort(key=lambda x: x[1], reverse=True)
                
                if divs_with_content:
                    content = self._extract_structured_content(divs_with_content[0][0])
            
            # Strategy 3: Collect all paragraphs
            if not content or len(content) < min_content_length:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    content = "\n\n".join([
                        p.get_text(strip=True) for p in paragraphs 
                        if len(p.get_text(strip=True)) > 40  # Skip very short paragraphs
                    ])
            
            # Final validation
            if content and len(content) > min_content_length:
                # Clean up the content
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = re.sub(r'\s*\n\s*', '\n\n', content)  # Clean up newlines
                content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
                
                print(f"Successfully extracted {len(content)} characters")
                return content
            else:
                print(f"Content too short or not found at {url}")
                return None
                
        except Exception as e:
            print(f"Error fetching article from {url}: {e}")
            return None

    def _extract_structured_content(self, container) -> str:
        """
        Extract structured content from a container, preserving headings.
        
        Args:
            container: BeautifulSoup element containing the content
            
        Returns:
            Formatted string with the content
        """
        # Extract headings and paragraphs
        elements = container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        
        # If not enough elements found, just get all text
        if len(elements) < 3:
            return container.get_text(strip=True)
        
        # Build the article text with proper structure
        article_parts = []
        for elem in elements:
            text = elem.get_text(strip=True)
            if not text:
                continue
                
            if elem.name.startswith('h'):
                article_parts.append(f"\n\n{text}\n")
            else:
                article_parts.append(text)
        
        return '\n\n'.join(article_parts)
    
    def search_and_format_results(self, query: str, time_limit_seconds: int = 60, page: int = 1) -> str:
        """
        Search for articles on a topic and return the results as a formatted string.
        
        Args:
            query (str): The search query
            time_limit_seconds (int): Maximum time to search for articles
            page (int): Initial page number to search
            
        Returns:
            str: Formatted string containing search results with full article content
        """
        # This function is kept for backward compatibility
        # Initialize variables
        start_time = time.time()
        processed_urls = set()
        results = []
        current_page = page
        
        # Initialize output string
        output_lines = [f"SEARCH RESULTS FOR: '{query}'", "=" * 50, ""]
        
        # Continue searching until we reach time limit or max results
        while (time.time() - start_time < time_limit_seconds and 
            len(results) < self.max_results):
            
            # Search for articles
            print(f"\nSearching page {current_page} for '{query}'...")
            articles = self.search(query, page=current_page, exclude_urls=processed_urls)
            
            if not articles:
                print("No more articles found")
                break
                
            # Process each article
            for article in articles:
                # Check time limit
                if time.time() - start_time >= time_limit_seconds:
                    output_lines.append("Time limit reached")
                    break
                    
                url = article['url']
                title = article['title']
                
                # Skip already processed URLs
                if url in processed_urls:
                    continue
                    
                processed_urls.add(url)
                output_lines.append(f"\nProcessing article: {title}")
                
                # Fetch article content
                content = self.fetch_article(url)
                
                if not content:
                    output_lines.append("No valid content found in article")
                    continue
                    
                # Add to results
                results.append({
                    'title': title,
                    'url': url,
                    'content': content,
                    'reliability': article.get('reliability', 0.5)
                })
                
                output_lines.append(f"Successfully fetched article ({len(content)} characters)")
                
                self.result_found = len(results)

                # Check if we've reached the maximum number of results
                if len(results) >= self.max_results:
                    output_lines.append(f"Reached maximum number of results ({self.max_results})")
                    break
                    
                # Add delay between requests
                time.sleep(self.delay)
                
            # Move to next page
            current_page += 1
        
        # Sort results by reliability
        results = sorted(results, key=lambda x: x.get('reliability', 0), reverse=True)
        
        # Print results summary
        output_lines.append(f"\nFetched {len(results)} articles in {time.time() - start_time:.2f} seconds")
        output_lines.append("")
        
        # Add full content for each article
        for i, result in enumerate(results):
            output_lines.append(f"ARTICLE {i+1}: {result['title']}")
            output_lines.append(f"URL: {result['url']}")
            output_lines.append(f"RELIABILITY SCORE: {result.get('reliability', 0.5):.2f}")
            output_lines.append("-" * 40)
            
            # Include the full article content
            output_lines.append(f"CONTENT:\n{result['content']}")
            output_lines.append("=" * 50)
            output_lines.append("")
        
        # Join all lines with newlines and return as a single string
        return "\n".join(output_lines)

