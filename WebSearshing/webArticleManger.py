
import time
import time
from WebSearshing.searchAndFeach import WebArticleSearcher
from WebSearshing.saveArticles import SaveArticles
from typing import List, Dict
from typing import List, Dict


class WebArticleManager:
    
    def __init__(self, max_results: int = 5,
                 save_directory: str = "articles",
                 delay_between_requests: float = 1.0,
                 ):
        """
        Initialize the WebArticleManager to coordinate searching and saving.
        
        Args:
            max_results: Maximum number of articles to fetch
            save_directory: Directory to save downloaded articles
            delay_between_requests: Time to wait between requests
            dwnload_articles: Whether to save articles to disk
        """
        self.searcher = WebArticleSearcher(
            max_results=max_results,
            delay_between_requests=delay_between_requests
        )
        
        self.saver = SaveArticles(
            save_directory=save_directory,
        )
    
    def fetch_and_save_related_articles(self, query: str, save_format: str = "file", 
                                        time_limit_seconds: int = 60) -> List[Dict]:
    
        """
        Search for articles related to the query, fetch and save them.
        
        Args:
            query: The search query text
            save_format: 'file' to save as files, 'string' to return as strings
            time_limit_seconds: Maximum time to search for articles
            
        Returns:
            List of article information dictionaries
        """
        start_time = time.time()
        processed_urls = set()
        results = []
        page = 1
        
        while (time.time() - start_time < time_limit_seconds and 
               len(results) < self.searcher.max_results):
            
            # Search for articles
            print(f"\nSearching page {page} for '{query}'...")
            articles = self.searcher.search(query, page=page, exclude_urls=processed_urls)
            
            if not articles:
                print("No more articles found")
                break
                
            # Process each article
            for article in articles:
                # Check time limit
                if time.time() - start_time >= time_limit_seconds:
                    print("Time limit reached")
                    break
                    
                url = article['url']
                title = article['title']
                
                # Skip already processed URLs
                if url in processed_urls:
                    continue
                    
                processed_urls.add(url)
                print(f"\nProcessing article: {title}")
                
                # Fetch article content
                content = self.searcher.fetch_article(url)
                
                if not content:
                    print("No valid content found in article")
                    continue
                    
                # Save the article
                filepath = self.saver.save_article(title, content, url, format=save_format)
                
                # Add to results
                result_info = {
                    'title': title,
                    'url': url,
                    'content_length': len(content) if content else 0
                }
                
                if save_format == 'file' and filepath:
                    result_info['filepath'] = filepath
                elif save_format == 'string':
                    result_info['content'] = content
                    
                results.append(result_info)

                self.searcher.result_found = len(results)
                
                # Check if we've reached the maximum number of results
                if len(results) >= self.searcher.max_results:
                    print(f"Reached maximum number of results ({self.searcher.max_results})")
                    break
                    
                # Add delay between requests
                time.sleep(self.searcher.delay)
                
            # Move to next page
            page += 1
            
        print(f"Found {len(results)} articles in {time.time() - start_time:.2f} seconds")
        return results