from transformers import pipeline as hf_pipeline
from pydantic import PrivateAttr
from typing import List, Dict, Any

from schemas.messages import Claim


class ClaimExtractorAgent:
    _summarizer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[ClaimExtractorAgent] Initializing...")
        print("[ClaimExtractorAgent] Loading summarization model...")
        self._summarizer = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device_map="auto"
        )
        print("[ClaimExtractorAgent] Model loaded.")

    def run(self, scraped_data: Dict[str, Any]) -> List[Claim]:
        """
        Extracts key claims from a raw text using a summarization model.
        It now expects a dictionary from the SmartScraperAgent.
        """
        print("Extracting claims from text...")

        if "error" in scraped_data and scraped_data["error"]:
            print(f"Skipping claim extraction due to scraper error: {scraped_data['error']}")
            return []

        text_content = scraped_data.get("content", "")
        if not text_content:
            print("No content to extract claims from.")
            return []

        # Chunk the text to avoid memory errors with large articles
        chunk_size = 4096  # A reasonable chunk size for a summarization model
        overlap = 200
        
        text_chunks = [
            text_content[i:i+chunk_size] 
            for i in range(0, len(text_content), chunk_size - overlap)
        ]

        print(f"Split content into {len(text_chunks)} chunks.")

        # The summarizer expects a list of texts
        summaries: List[Dict[str, str]] = self._summarizer(
            text_chunks,
            max_length=150,
            min_length=30,
            do_sample=False,
            truncation=True # Ensure chunks that are slightly too long are handled
        )
        
        claims = [Claim(claim_text=summary['summary_text']) for summary in summaries]
        # Remove duplicate claims that might arise from overlapping chunks
        unique_claims = list({claim.claim_text: claim for claim in claims}.values())
        
        print(f"Extracted {len(unique_claims)} unique claims.")
        return unique_claims
