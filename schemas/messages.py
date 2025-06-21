from typing import List
from pydantic import BaseModel, HttpUrl


class ScrapedData(BaseModel):
    """
    Data object for scraped content from a URL.
    """
    url: HttpUrl
    content: str


class Claim(BaseModel):
    """
    A single factual claim extracted from a text.
    """
    claim_text: str


class FactCheck(BaseModel):
    """
    A fact-check result corresponding to a claim.
    """
    claim: Claim
    match_document: str
    match_score: float


class ScoredClaim(BaseModel):
    """
    A claim that has been scored for truthfulness.
    """
    claim: Claim
    fact_check: FactCheck
    truth_score: float
    explanation: str
