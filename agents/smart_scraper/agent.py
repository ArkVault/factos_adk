import os
from pydantic import HttpUrl, PrivateAttr
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from typing import Dict


load_dotenv()

class SmartScraperAgent:
    _firecrawl: FirecrawlApp = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[SmartScraperAgent] Initializing...")
        if not os.getenv("FIRECRAWL_API_KEY"):
            raise ValueError("FIRECRAWL_API_KEY environment variable not set.")
        self._firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    def run(self, url: str) -> str:
        """
        Scrapes a single URL for its main content and returns it as a string.
        This function is intended to be used as a tool by an ADK LlmAgent.
        """
        print(f"Scraping {url} with Firecrawl...")
        try:
            http_url = str(HttpUrl(url=url))
            scraped_data_obj = self._firecrawl.scrape_url(url=http_url, params={'pageOptions': {'onlyMainContent': True}})
            
            scraped_data: Dict = dict(scraped_data_obj)

            if scraped_data and 'markdown' in scraped_data:
                return scraped_data['markdown']
            else:
                print(f"Warning: No markdown content found for {url}. Full response: {scraped_data}")
                return ""

        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")
            return f"Error scraping {url}: {e}"
