
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

with open("articles.txt", "w", encoding="utf-8") as file:
    file.write(article_dicts)

print("Saved to articles.txt")
#print (article_dicts)