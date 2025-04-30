import os
import time
import re

from typing import Union, List, Dict

class SaveArticles:

    def __init__(self, save_directory: str = "articles"):
        """
        Initialize the article saving functionality.
        
        Args:
            save_directory: Directory to save downloaded articles
        """
        # Create save directory if it doesn't exist
        self.save_directory = save_directory
        self.result_saved = 0  # Track how many articles we've saved
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)            
            print(f"Created directory: {save_directory}")   

    def save_article(self, title: str, content: str, url: str, format: str = "file") -> Union[str, None]:
        """
        Save the article content along with its source URL.

        Args:
            title: Article title
            content: Article content
            url: Source URL of the article
            format: 'file' to save as a file, 'string' to return as a string

        Returns:
            File path if format is 'file', content string if format is 'string'
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
            print(f"Successfully saved article to {filepath}\n")
            self.result_saved += 1  # Increment the count of saved articles
            return filepath
        except Exception as e:
            print(f"Error saving article: {e}")
            return None
            
    