from firecrawl import FirecrawlApp
from pydantic import PrivateAttr
from typing import List
import os

class CorpusBuilderAgent:
    _firecrawl: FirecrawlApp = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[CorpusBuilderAgent] Initializing...")
        if not os.getenv("FIRECRAWL_API_KEY"):
            raise ValueError("FIRECRAWL_API_KEY environment variable not set.")
        self._firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    def run(self, urls: List[str]) -> List[str]:
        """
        Crawls a list of base URLs to build a corpus of documents.

        Args:
            urls: A list of base URLs of fact-checking websites.

        Returns:
            A list of all scraped document contents.
        """
        print(f"Starting crawl for {len(urls)} base URLs...")
        all_documents = []
        for url in urls:
            print(f"Crawling {url}...")
            try:
                # We can add more params here to control the crawl, 
                # e.g., max_depth, limit, etc.
                scraped_data = self._firecrawl.crawl_url(url=url, params={'crawlerOptions': {'limit': 10}}) # Limit to 10 pages per site for now
                
                for item in scraped_data:
                    if 'markdown' in item and item['markdown']:
                        all_documents.append(item['markdown'])

                print(f"Found {len(scraped_data)} documents at {url}.")
            except Exception as e:
                print(f"An error occurred while crawling {url}: {e}")
        
        print(f"Finished crawling. Total documents found: {len(all_documents)}")
        return all_documents 