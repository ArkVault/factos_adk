import requests
import json
from typing import List
import os

class CorpusBuilderAgent:
    def run(self, urls: List[str]) -> List[str]:
        """
        Uses direct REST API calls to scrape fact-check articles.
        """
        print(f"Starting REST API-based scrape for {len(urls)} base URLs...")
        all_documents = []
        api_key = os.getenv("FIRECRAWL_API_KEY")

        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable not set.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        scrape_url = "https://api.firecrawl.dev/v1/scrape"

        for url in urls:
            print(f"Scraping {url} with Firecrawl REST API...")
            payload = {
                "url": url, 
                "onlyMainContent": True
            }
            
            try:
                response = requests.post(scrape_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                scraped_data = response.json()
                
                # The successful response is nested under a 'data' key
                if scraped_data.get("success") and "data" in scraped_data:
                    markdown = scraped_data["data"].get("markdown")
                    if markdown:
                        all_documents.append(markdown)
                        print(f"Successfully scraped {url}.")
                else:
                    print(f"API call succeeded but scrape failed for {url}: {scraped_data.get('error')}")

            except requests.exceptions.HTTPError as e:
                # Handle non-2xx responses
                print(f"HTTP error occurred while scraping {url}: {e}")
                # Try to print the error from the response body if it exists
                try:
                    print(f"API Response: {e.response.json()}")
                except json.JSONDecodeError:
                    print(f"API Response (non-JSON): {e.response.text}")
            except requests.exceptions.RequestException as e:
                # Handle other network errors (timeout, connection error, etc.)
                print(f"An error occurred while making the request for {url}: {e}")

        print(f"Finished scraping. Total documents found: {len(all_documents)}")
        return all_documents 