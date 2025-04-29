import os
import time
import re
import requests
import traceback
import requests

from bs4 import BeautifulSoup
from typing import List, Dict, Union, Optional
from scholarly import scholarly, ProxyGenerator
from newspaper import Article, ArticleException


class WebArticleFetcher:
    
    def __init__(self, save_directory: str = "articles", 
                 max_results: int = 5,
                 result_found: int = 0,
                 delay_between_requests: float = 1.0):
        """
        Initialize the WebArticleFetcher.
        
        Args:
            save_directory: Directory to save downloaded articles
            max_results: Maximum number of articles to fetch
            delay_between_requests: Time to wait between requests (to avoid rate limiting)
        """
        self.save_directory = save_directory
        self.max_results = max_results
        self.delay = delay_between_requests
        self.result_found = result_found  # Initialize the number of results found
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"Created directory: {save_directory}")   
    
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
   
    def save_article(self, title: str, content: str, url: str, format: str = "file") -> Union[str, None]:
        """
        Save the article content along with its source URL.

        Args:
            title: Article title
            content: Article content
            url: Source URL of the article
            format: 'file' to save as a file, 'string' to return as a string

        Returns:
            File path if format is 'file', content if format is 'string'
        """
        if format == "string":
            return f"Source: {url}\n\n{content}"

        # Create a valid filename from the title
        filename = re.sub(r'[\\/*?:"<>|]', "", title)
        filename = filename[:50].strip()  # Limit filename length and remove trailing whitespace
        if not filename:  # In case the title was just special characters
            filename = "article_" + str(int(time.time()))
        filepath = os.path.join(self.save_directory, f"{filename}.txt")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {url}\n\n")  # Add source URL at the top
                f.write(content)
            print(f"Successfully saved article")
            self.result_found += 1  # Increment the count of valid articles found
            return filepath
        except Exception as e:
            print(f"Error saving article: {e}")
            return None
    
    def fetch_and_save_related_articles(self, text: str, save_format: str = "file", time_limit_seconds: int = 300) -> List[Dict]:
        """
        Search for articles related to the given text, fetch their content, and save them.
        
        Args:
            text: The query text
            save_format: 'file' to save articles as files, 'string' to return content as strings
            time_limit_seconds: Maximum time (in seconds) to search for articles
            
        Returns:
            List of dictionaries containing article information
        """
        results = []
        articles_found = self.result_found  # To track the number of valid articles found
        processed_urls = set()  # Track URLs we've already processed
        search_page = 1  # To keep track of search result pages
        start_time = time.time()  # Record the start time
        
        while articles_found < self.max_results:
            # Check if we've exceeded the time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit_seconds:
                print(f"\nT ime limit of {time_limit_seconds} seconds reached. Stopping search.")
                break
                
            print(f"\nSearching for articles: {text} (Found {articles_found}/{self.max_results}) - Page {search_page}")
            print(f"Time elapsed: {elapsed_time:.1f} seconds / {time_limit_seconds} seconds")
            
            # Fetch search results with page number
            search_results = self.search(text, page=search_page, exclude_urls=processed_urls)
            
            # Check if we're getting new results
            if not search_results:
                print("No more search results available.")
                break
                    
            new_results_found = False
            
            # Process the search results
            for i, result in enumerate(search_results):
                # Check time limit again within the loop
                if time.time() - start_time > time_limit_seconds:
                    print(f"\nTime limit of {time_limit_seconds} seconds reached during processing. Stopping.")
                    break
                    
                if articles_found >= self.max_results:
                    break  # Exit the loop if we've reached the max number of articles
                
                # Skip URLs we've already processed
                if result['url'] in processed_urls:
                    print(f"Skipping already processed URL: {result['url']}")
                    continue
                    
                # Add URL to processed set
                processed_urls.add(result['url'])
                new_results_found = True
                
                print(f"\nProcessing article {i+1}/{len(search_results)}")
                
                # Fetch the article content
                article_content = self.fetch_article(result['url'])
                
                if article_content:  # Ensure there is substantial content
                    # Save the article if it has valid content
                    saved_result = self.save_article(result['title'], article_content, result['url'], save_format)
                    
                    results.append({
                        'title': result['title'],
                        'url': result['url'],
                        'content': article_content if save_format == 'string' else None,
                        'file_path': saved_result if save_format == 'file' else None
                    })
                    articles_found += 1  # Increment the count of valid articles
                    
                else:
                    pass
                
                # Delay to avoid overloading servers
                if i < len(search_results) - 1 and articles_found < self.max_results:
                    time.sleep(self.delay)
            
            # If there are no new results or we've tried all results, move to the next page
            if not new_results_found:
                print(f"No new results on page {search_page}, trying next page.")
            
            search_page += 1
            
            # Break if we've gone through too many pages
            if search_page > 10:  # Set a reasonable limit
                print("Reached maximum number of search pages.")
                break
        
        # Final report
        elapsed_time = time.time() - start_time
        print(f"\nSearch completed in {elapsed_time:.1f} seconds")
        print(f"Processed {len(results)} articles successfully")
        
        return results
 
# Initialize the fetcher
fetcher = WebArticleFetcher(
    max_results = 3,  # Limit to 3 articles
    save_directory="articles",
    delay_between_requests=2.0  # Be gentle with the servers
)

# Search for articles, fetch and save them
query = "Electrical components Life cycle"  # Example query
# ENd of life bla bla bla 
# Finding for recycling material 
results = fetcher.fetch_and_save_related_articles(query, time_limit_seconds=3000)
