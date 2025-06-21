import requests
import json
from typing import Dict, Any
import os

class SmartScraperAgent:
    def run(self, url: str) -> Dict[str, Any]:
        """
        Uses a direct REST API call to scrape the content of a single URL.
        """
        print(f"Scraping {url} with Firecrawl REST API...")
        api_key = os.getenv("FIRECRAWL_API_KEY")

        if not api_key:
            return {"error": "FIRECRAWL_API_KEY environment variable not set."}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        scrape_url_endpoint = "https://api.firecrawl.dev/v1/scrape"
        payload = {
            "url": url,
            "onlyMainContent": True
        }

        try:
            response = requests.post(scrape_url_endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            scraped_data = response.json()
            if scraped_data.get("success"):
                return {"content": scraped_data["data"].get("markdown", "")}
            else:
                return {"error": f"API call failed: {scraped_data.get('error')}"}

        except requests.exceptions.HTTPError as e:
            error_details = ""
            try:
                error_details = e.response.json()
            except json.JSONDecodeError:
                error_details = e.response.text
            return {"error": f"HTTP error during scrape: {e}. Details: {error_details}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Unexpected error during scrape: {e}"}
