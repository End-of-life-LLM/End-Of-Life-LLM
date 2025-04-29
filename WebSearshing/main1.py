
from searchAndFeach import WebArticleSearcher

# Initialize the searcher
searcher = WebArticleSearcher(
    max_results=3,
    delay_between_requests=1
)

# Define your query
query = "Electrical components Life cycle"

# Search for articles
article_dicts = searcher.search_and_format_results(query)


print (article_dicts)