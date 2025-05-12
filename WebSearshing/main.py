from webArticleManger import WebArticleManager

manager = WebArticleManager(
    max_results=3,  # Limit to 3 articles
    delay_between_requests=2.0,  # Be gentle with servers
)

# 2. Search for articles, fetch and save them
query = "Social Media"
results = manager.fetch_and_save_related_articles(query, time_limit_seconds=300)

