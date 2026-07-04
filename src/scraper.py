import requests
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HelpCenterScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v2/help_center/en-us/articles.json"

    def fetch_articles(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetches articles using the Zendesk Help Center API."""
        articles = []
        page_url = self.api_url
        
        while len(articles) < limit and page_url:
            logger.info(f"Fetching from: {page_url}")
            try:
                response = requests.get(page_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                articles.extend(data.get("articles", []))
                page_url = data.get("next_page")  # Zendesk API provides pagination links
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching articles: {e}")
                break
                
        return articles[:limit]