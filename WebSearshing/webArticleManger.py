import time
from typing import Dict, List

import concurrent
from WebSearshing.saveArticles import SaveArticles
from WebSearshing.searchAndFeach import WebArticleSearcher

class WebArticleManager:
    def __init__(self, max_results: int = 5,
                 save_directory: str = "articles",
                 delay_between_requests: float = 1.0):
        """
        Initialize the WebArticleManager to coordinate searching and saving.
        
        Args:
            max_results: Maximum number of articles to fetch
            save_directory: Directory to save downloaded articles
            delay_between_requests: Time to wait between requests
        """
        self.searcher = WebArticleSearcher(
            max_results=max_results,
            delay_between_requests=delay_between_requests
        )
        
        self.saver = SaveArticles(
            save_directory=save_directory,
        )
    
    def fetch_and_save_related_articles(self, query: str, save_format: str = "file", 
                                       time_limit_seconds: int = 300) -> List[Dict]:
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
        
        print(f"\n===== SEARCHING FOR RELIABLE INFORMATION: '{query}' =====\n")
        print(f"Time limit: {time_limit_seconds} seconds")
        print(f"Maximum results: {self.searcher.max_results}")
        print(f"Save directory: {self.saver.save_directory}")
        print("=" * 60 + "\n")
        
        while (time.time() - start_time < time_limit_seconds and 
               len(results) < self.searcher.max_results):
            
            # Search for articles
            print(f"\nSearching page {page} for '{query}'...")
            articles = self.searcher.search(query, page=page, exclude_urls=processed_urls)
            
            if not articles:
                print("No more articles found")
                break
                
            # Use concurrent.futures to process articles in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit fetch tasks for each article
                future_to_article = {
                    executor.submit(self.searcher.fetch_article, article['url']): article
                    for article in articles if article['url'] not in processed_urls
                }
                
                for future in concurrent.futures.as_completed(future_to_article):
                    article = future_to_article[future]
                    url = article['url']
                    title = article['title']
                    reliability = article.get('reliability', 0.5)
                    
                    # Check time limit
                    if time.time() - start_time >= time_limit_seconds:
                        print("Time limit reached")
                        break
                    
                    # Skip already processed URLs
                    if url in processed_urls:
                        continue
                        
                    processed_urls.add(url)
                    print(f"\nProcessing article: {title}")
                    
                    try:
                        content = future.result()
                        
                        if not content:
                            print("No valid content found in article")
                            continue
                            
                        # Save the article
                        filepath = self.saver.save_article(title, content, url, format=save_format)
                        
                        # Add to results
                        result_info = {
                            'title': title,
                            'url': url,
                            'content_length': len(content) if content else 0,
                            'reliability': reliability
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
                            
                    except Exception as e:
                        print(f"Error processing {url}: {e}")
                
                # Check if we've reached the maximum number of results
                if len(results) >= self.searcher.max_results:
                    break
                    
            # Move to next page
            page += 1
            
            # Add delay between pages
            time.sleep(self.searcher.delay)
            
        # Sort results by reliability
        results = sorted(results, key=lambda x: x.get('reliability', 0), reverse=True)
            
        print("\n" + "=" * 60)
        print(f"Found {len(results)} articles in {time.time() - start_time:.2f} seconds")
        print("=" * 60)
        
        # Print summary of results
        if results:
            print("\nSummary of retrieved articles:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   Reliability score: {result['reliability']:.2f}")
                print(f"   Content length: {result['content_length']} characters")
                if 'filepath' in result:
                    print(f"   Saved to: {result['filepath']}")
                print()
        
        return results