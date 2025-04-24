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
            
            print(f"Searching Google Scholar for: {query} (Page {page})")
            
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
            count = 0
            
            print(f"Starting to collect results after skipping {offset} articles")
            
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
                    
                    print(f"Found article {count+1}: {title} - {url}")
                    count += 1
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except StopIteration:
                    print("No more results available")
                    break
                except Exception as e:
                    print(f"Error processing result: {e}")
                    continue
            
            print(f"Found {len(search_results)} articles from Google Scholar")
            return search_results
            
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            traceback.print_exc()
            return []
   
    def fetch_article(self, url: str) -> Optional[str]:
        """
        Fetch the content of an article from the given URL,
        focusing on extracting only the relevant content and handling paywalls.
        
        Args:
            url: URL of the article to fetch
            
        Returns:
            The article content as a string, or None if fetching failed or behind paywall
        """
        try:
            print(f"Fetching article from: {url}")
            
            # Try using newspaper3k for better content extraction
            try:                
                article = Article(url)
                article.download()
                article.parse()
                
                # If newspaper extracted decent content, use it
                if article.text and len(article.text) > 500:
                    print(f"Successfully extracted article using newspaper3k: {len(article.text)} characters")
                    return article.text
                    
                # Otherwise fall back to BeautifulSoup
                print("Content extraction with newspaper3k insufficient, falling back to BeautifulSoup")
                
            except (ImportError, ArticleException) as e:
                print(f"newspaper3k not available or failed: {e}")
            
            # Fallback: Use requests + BeautifulSoup
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Check if we hit a paywall - common indicators
            paywall_indicators = [
                'Subscribe to read', 'Sign in to access', 'Subscribe now',
                'Pay to access', 'Purchase this article', 'Login to view',
                'Access to this content requires a subscription', 'Access options',
                'Sign in for full access', 'This content is only available to subscribers',
                'To continue reading', 'Access this article'
            ]
            
            text_content = response.text.lower()
            if any(indicator.lower() in text_content for indicator in paywall_indicators):
                print(f"Detected potential paywall at {url}")
                
                # Check if we have enough content despite the paywall
                soup = BeautifulSoup(response.text, 'html.parser')
                main_text = soup.get_text(strip=True)
                
                # If content is very short, likely blocked by paywall
                if len(main_text) < 2000:  # Arbitrary threshold
                    print("Content too short, likely behind paywall. Skipping.")
                    return None
                
                print("Found enough content to proceed despite paywall indicators")
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside", 
                                "button", ".share", ".social", ".comments", ".related",
                                ".sidebar", ".ad", ".advertisement", ".popup", ".modal",
                                ".cookie", ".subscription", ".paywall", ".signin"]):
                element.extract()
            
            # Try to identify the abstract or introduction first
            abstract_sections = soup.find_all(['div', 'section', 'p'], 
                                            class_=lambda c: c and any(term in str(c).lower() 
                                                                    for term in ['abstract', 'summary', 'introduction']))
            
            # Try to find the main content container
            main_content = None
            
            # Check for scholarly article structure first
            paper_sections = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
            for section in paper_sections:
                elements = soup.find_all(['div', 'section'], 
                                        class_=lambda c: c and section.lower() in str(c).lower())
                # If we found any structured sections, we're likely in a scientific paper
                if elements:
                    all_sections = []
                    for section_name in paper_sections:
                        section_elements = soup.find_all(['div', 'section', 'h1', 'h2', 'h3'], 
                                                    class_=lambda c: c and section_name.lower() in str(c).lower())
                        for element in section_elements:
                            all_sections.append(element.get_text(strip=True))
                    
                    if all_sections:
                        main_content = '\n\n'.join(all_sections)
                        break
            
            # If we didn't find structured sections, try common content selectors
            if not main_content:
                selectors = [
                    'article', 'main', '.paper', '.publication', '.research-article',
                    '.post-content', '.entry-content', '.article-content', '.content-body',
                    '.content', '#content', '.document', '.paper-content'
                ]
                
                for selector in selectors:
                    content = soup.select_one(selector)
                    if content and len(content.get_text(strip=True)) > 200:
                        main_content = content
                        break
            
            # If still no specific content container found, use the body
            if not main_content:
                main_content = soup.body or soup
            
            # Get text content
            if isinstance(main_content, str):
                article_text = main_content
            else:
                article_text = main_content.get_text()
            
            # Clean up whitespace and formatting
            lines = (line.strip() for line in article_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            article_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Additional cleanup
            # Remove excessive newlines
            article_text = re.sub(r'\n{3,}', '\n\n', article_text)
            # Remove citation numbers and brackets often found in academic papers
            article_text = re.sub(r'\[\d+[,\s]*\d*\]', '', article_text)
            # Remove excessive spacing
            article_text = re.sub(r'\s{2,}', ' ', article_text)
            
            print(f"Extracted article text: {len(article_text)} characters")
            
            # If article is too short, it might be behind a paywall or extraction failed
            if len(article_text) < 500:
                print("Extracted content too short, may be behind paywall or not a proper article")
                return None
                
            return article_text
            
        except Exception as e:
            print(f"Error fetching article from {url}: {e}")
            return None
    
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
            print(f"Saving article to: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {url}\n\n")  # Add source URL at the top
                f.write(content)
            print(f"Successfully saved article: {len(content)} characters")
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
        articles_found = 0  # To track the number of valid articles found
        processed_urls = set()  # Track URLs we've already processed
        search_page = 1  # To keep track of search result pages
        start_time = time.time()  # Record the start time
        
        while articles_found < self.max_results:
            # Check if we've exceeded the time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit_seconds:
                print(f"\nTime limit of {time_limit_seconds} seconds reached. Stopping search.")
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
                
                print(f"\nProcessing article {i+1}/{len(search_results)}: {result['title']}")
                
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
                    print(f"Successfully saved article ({articles_found}/{self.max_results})")
                    
                else:
                    print(f"Insufficient content for: {result['title']}, skipping this article.")
                
                # Delay to avoid overloading servers
                if i < len(search_results) - 1 and articles_found < self.max_results:
                    print(f"Waiting {self.delay} seconds before next request...")
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
    max_results = 2,  # Limit to 3 articles
    save_directory="articles",
    delay_between_requests=2.0  # Be gentle with the servers
)

# Search for articles, fetch and save them
query = "social media and mental health"  # Example query
# ENd of life bla bla bla 
# Finding for recycling material 
results = fetcher.fetch_and_save_related_articles(query, time_limit_seconds=60)

# Print results
print("\nSummary of retrieved articles:")
for i, result in enumerate(results):
    print(f"\nArticle {i+1}:")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Saved to: {result['file_path']}")
