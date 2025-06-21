import os
import sys
from agents.corpus_builder.agent import CorpusBuilderAgent
from agents.fact_check_matcher.agent import FactCheckMatcherAgent

def load_env():
    """Manually loads environment variables from a .env file."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    print(f"Loaded {key}")
    except FileNotFoundError:
        print("Warning: .env file not found. Relying on system environment variables.")

def main():
    """
    Builds the fact-checking corpus by crawling specified websites
    and adding their content to the ChromaDB vector store.
    """
    print("--- Starting Corpus Build Process ---")

    # Manually load environment variables first
    load_env()

    # 1. Define the fact-checker sources
    fact_checker_urls = [
        "https://www.factcheck.org/",
        "https://reporterslab.org/fact-checking/",
        "https://apnews.com/ap-fact-check",
    ]

    # 2. Initialize the agents
    # Make sure FIRECRAWL_API_KEY is set in your .env file
    corpus_builder = CorpusBuilderAgent()
    fact_matcher = FactCheckMatcherAgent()

    # 3. Run the corpus builder to get the documents
    documents = corpus_builder.run(urls=fact_checker_urls)

    # 4. Add the scraped documents to the persistent vector database
    if documents:
        fact_matcher.add_documents(documents)
    else:
        print("No new documents found to add to the corpus.")

    print("--- Corpus Build Process Finished ---")


if __name__ == "__main__":
    main() 