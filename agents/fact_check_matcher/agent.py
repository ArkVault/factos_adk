import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import PrivateAttr
from typing import List, Dict, Any

from schemas.messages import Claim, FactCheck


class FactCheckMatcherAgent:
    _embedding_model = PrivateAttr()
    _db_client = PrivateAttr()
    _collection = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[FactCheckMatcherAgent] Initializing...")
        
        print("[FactCheckMatcherAgent] Loading embedding model...")
        self._embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='mps'
        )
        print("[FactCheckMatcherAgent] Model loaded.")

        # Use a persistent client to store the database on disk
        self._db_client = chromadb.PersistentClient(path="data/fact_checks_db")
        self._collection = self._db_client.get_or_create_collection("fact_checks")
        
        print(f"ChromaDB collection '{self._collection.name}' loaded with {self._collection.count()} documents.")

    def add_documents(self, documents: List[str]):
        """Adds a list of documents to the ChromaDB collection."""
        if not documents:
            return

        print(f"Adding {len(documents)} new documents to the collection...")
        # Add new documents. ChromaDB handles embedding via the model provided at collection creation.
        # However, since we use SentenceTransformer, we'll embed manually for consistency.
        embeddings = self._embedding_model.embed(documents).tolist()
        
        # Create unique IDs for each document
        # This is a simple way to create IDs, could be more robust
        start_id = self._collection.count()
        ids = [f"doc_{start_id + i}" for i, _ in enumerate(documents)]

        self._collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print("Finished adding documents.")

    def run(self, claims: List[Claim]) -> List[FactCheck]:
        """
        Finds relevant fact-checks for a list of claims.
        This function is intended to be used as a tool by an ADK LlmAgent.
        """
        print(f"Finding fact-checks for {len(claims)} claims...")
        if not claims:
            return []

        claim_texts = [claim.claim_text for claim in claims]
        
        results = self._collection.query(
            query_texts=claim_texts,
            n_results=1
        )

        matches = []
        if not results:
            return matches

        # Unpack results safely, providing default empty lists
        documents = results.get('documents', [])
        distances = results.get('distances', [])

        if not documents or not distances:
            return matches

        # Each query will have a list of documents and distances
        for i, claim in enumerate(claims):
            if len(documents) > i and documents[i]:
                best_match_doc = documents[i][0]
                best_match_dist = distances[i][0] if len(distances) > i and distances[i] else 0.0
                
                matches.append(FactCheck(
                    claim=claim,
                    match_document=best_match_doc,
                    match_score=1 - best_match_dist # Convert distance to similarity
                ))
        
        print(f"Found {len(matches)} potential fact-check matches.")
        return matches
