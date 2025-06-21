from transformers.pipelines import pipeline
from pydantic import PrivateAttr
from typing import List

from schemas.messages import Claim


class ClaimExtractorAgent:
    _summarizer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[ClaimExtractorAgent] Initializing...")
        print("[ClaimExtractorAgent] Loading summarization model...")
        self._summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device_map="auto"
        )
        print("[ClaimExtractorAgent] Model loaded.")

    def run(self, text: str) -> List[Claim]:
        """
        Extracts factual claims from a text using a summarization model.
        This function is intended to be used as a tool by an ADK LlmAgent.
        """
        print("Extracting claims from text...")
        # The summarizer returns a list of dictionaries
        summaries = self._summarizer(
            text,
            max_length=100,
            min_length=10,
            do_sample=False
        )
        
        claims = [
            Claim(claim_text=summary['summary_text'])
            for summary in summaries
        ]
        
        print(f"Extracted {len(claims)} claims.")
        return claims
