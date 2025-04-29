import time
import re
import requests
import traceback
import requests

from bs4 import BeautifulSoup
from typing import List, Dict, Union, Optional
from scholarly import scholarly

class WebArticleSearcher:
    def __init__(self, max_results: int = 5,
                 result_found: int = 0,
                 delay_between_requests: float = 1.0):
        """
        Initialize the WebArticleSearcher.
        
        Args:
            max_results: Maximum number of articles to fetch
            delay_between_requests: Time to wait between requests (to avoid rate limiting)
        """
        self.max_results = max_results
        self.result_found = result_found  # Initialize the number of results found
        self.delay = delay_between_requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }



    def search(self, query: str, page=1, exclude_urls=None) -> List[Dict[str, str]]:
        """
        Search for scientific articles related to the query using Google Scholar.
        
        Args:
            query: The search query text
            page: Current search page (for pagination)
            exclude_urls: Set of URLs to exclude from results
            
        Returns:
            List of dictionaries containing article URLs and titles
        """
        try:
            # Calculate offset for pagination
            offset = (page - 1) * self.max_results            
            # Create a search query
            search_query = scholarly.search_pubs(query)
            
            # Skip results for pagination
            skipped = 0
            while skipped < offset:
                try:
                    next(search_query)
                    skipped += 1
                except StopIteration:
                    print(f"Not enough results to reach page {page}")
                    return []
            
            # Get results
            search_results = []
            count = self.result_found  # Start counting from the number of results already found
            
            
            # Collect the requested number of results
            while count < self.max_results:
                try:
                    result = next(search_query)
                    
                    # Extract title
                    if 'bib' in result and 'title' in result['bib']:
                        title = result['bib']['title']
                    else:
                        title = "Unknown Title"
                    
                    # Try to get URL - first try pub_url
                    url = None
                    if 'pub_url' in result and result['pub_url']:
                        url = result['pub_url']
                    # If no pub_url, try citation URL from cluster_id
                    elif 'cluster_id' in result and result['cluster_id']:
                        url = f"https://scholar.google.com/scholar?cluster={result['cluster_id']}"
                    else:
                        # If no useful URL, skip this result
                        print(f"Skipping result with no URL: {title}")
                        continue
                    
                    # Skip if URL should be excluded
                    if exclude_urls and url in exclude_urls:
                        print(f"Skipping already processed URL: {url}")
                        continue
                    
                    # Add to results
                    search_results.append({
                        'url': url,
                        'title': title
                    })
                    count += 1
                    print(f"Found article {count}: {title} - {url}")
                    
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except StopIteration:
                    print("No more results available")
                    break
                except Exception as e:
                    print(f"Error processing result: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            traceback.print_exc()
            return []

    def fetch_article(self, url: str) -> Optional[str]:
        """
        Fetch the content of an article from the given URL by checking all tags for substantial content.
        
        Args:
            url: URL of the article to fetch
                
        Returns:
            The article content as a string, or None if fetching failed
        """
        try:
            print(f"Fetching article from: {url}")
            
            # Fetch the page content
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.select("script, style, header, footer, nav, aside, .nav, .menu, .banner, .ads, .advertisement, .footer, .header, .sidebar, .comments"):
                element.extract()
            
            # Collect all potential content containers
            content_containers = []
            
            # Function to check if element contains substantial content
            def has_substantial_content(element):
                text = element.get_text(strip=True)
                if not text:
                    return False
                    
                # Check for minimum content
                word_count = len(text.split())
                char_count = len(text)
                
                return word_count >= 50 or char_count >= 250
            
            # Look for containers with substantial content
            for tag in soup.find_all(['div', 'article', 'section', 'main', 'p', 'span']):
                if has_substantial_content(tag):
                    content_containers.append(tag)
            
            # Keep containers in their original document order - no sorting
            
            # If no substantial containers found, return None
            if not content_containers:
                print("No substantial content found in the article")
                return ""
            
            # Prioritize containers with article-like structure
            article_containers = []
            for container in content_containers:
                # Check if container has headings and paragraphs (article-like structure)
                headings = container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                paragraphs = container.find_all('p')
                
                if (headings and paragraphs) or len(paragraphs) >= 3:
                    article_containers.append(container)
            
            # Use the best container - prefer article-like structures, or fall back to the longest content
            target_container = article_containers[0] if article_containers else content_containers[0]
            
            # Extract all content elements from the target container
            content_elements = target_container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            
            # If few elements found, try to get all text-containing elements
            if len(content_elements) < 3:
                # Get all elements with text
                all_elements = target_container.find_all()
                content_elements = [elem for elem in all_elements if elem.get_text(strip=True) 
                                and not elem.name in ['script', 'style', 'meta', 'link']]
            
            # Build the article text
            article_parts = []
            for elem in content_elements:
                text = elem.get_text(strip=True)
                if not text:
                    continue
                    
                if elem.name and elem.name.startswith('h'):
                    article_parts.append(f"\n\n{text}\n")
                else:
                    article_parts.append(text)
            
            article_text = '\n\n'.join(article_parts)
            article_text = re.sub(r'\n{3,}', '\n\n', article_text)
            
            # Final check to ensure we have enough content
            if len(article_text) < 500:
                # If main approach failed, fallback: get text from any tag with substantial content
                all_content = []
                for tag in soup.find_all():
                    if tag.name not in ['script', 'style', 'meta', 'link', 'head'] and has_substantial_content(tag):
                        text = tag.get_text(strip=True)
                        all_content.append(text)
                
                if all_content:
                    article_text = '\n\n'.join(all_content)
                    article_text = re.sub(r'\n{3,}', '\n\n', article_text)
                else:
                    print("Extracted content too short, may be behind paywall or not a proper article")
                    return ""
                    
            print(f"Successfully extracted article: {len(article_text)} characters")
            return article_text
            
        except Exception as e:
            print(f"Error fetching article: {e}")
            traceback.print_exc()
            return ""
    

    def search_and_format_results(self, query, page=1):
        """
        Search for articles on a topic and return the results as a formatted string.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of articles to fetch
            delay (int): Delay between requests in seconds
            page (int): Page number to search
            
        Returns:
            str: Formatted string containing search results with full article content
        """
        
        # Search for articles
        article_dicts = self.search(query, page=page)
        
        # Initialize results list and output string
        results = []
        output_lines = [f"SEARCH RESULTS FOR: '{query}'", "=" * 50, ""]
        
        # Loop through each article dictionary and fetch content
        for article in article_dicts:
            url = article['url']
            title = article['title']
            
            output_lines.append(f"Fetching article: {title}")
            
            # Fetch the article content
            content = self.fetch_article(url)
            
            if content:
                output_lines.append(f"Successfully fetched article ({len(content)} characters)")
                # Store both the content and metadata
                results.append({
                    'title': title,
                    'url': url,
                    'content': content
                })
            else:
                output_lines.append(f"Failed to fetch content from {url}")
            
            output_lines.append("")  # Add blank line between articles
        
        # Print results summary
        output_lines.append(f"Fetched {len(results)} articles successfully")
        output_lines.append("")
        
        # Add full content for each article
        for i, result in enumerate(results):
            output_lines.append(f"ARTICLE {i+1}: {result['title']}")
            output_lines.append(f"URL: {result['url']}")
            output_lines.append("-" * 40)
            
            # Include the full article content
            output_lines.append(f"CONTENT:\n{result['content']}")
            output_lines.append("=" * 50)
            output_lines.append("")
        
        # Join all lines with newlines and return as a single string
        return "\n".join(output_lines)

