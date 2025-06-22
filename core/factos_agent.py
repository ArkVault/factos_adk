from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from typing import Callable, AsyncIterator, Dict, Any, List

from agents.smart_scraper.agent import SmartScraperAgent
from agents.claim_extractor.agent import ClaimExtractorAgent
from agents.fact_check_matcher.agent import FactCheckMatcherAgent
from agents.truth_scorer.agent import TruthScorerAgent
from schemas.messages import ScoredClaim


class FunctionAgent(BaseAgent):
    """An agent that runs a simple Python function."""
    fn: Callable
    input_key: str
    output_key: str

    async def _run_async_impl(self, ctx: 'InvocationContext') -> AsyncIterator[Event]:
        fn_input = ctx.session.state.get(self.input_key)
        result = self.fn(fn_input)
        ctx.session.state[self.output_key] = result
        if False:
            yield

class FactosAgent(BaseAgent):
    """The main agent for the Factos pipeline."""

    def __init__(self, **kwargs):
        super().__init__(name="FactosAgent", **kwargs)

    async def _run_async_impl(
        self, ctx: 'InvocationContext'
    ) -> AsyncIterator[Event]:
        
        processing_pipeline = SequentialAgent(
            name="ProcessingPipeline",
            sub_agents=[
                FunctionAgent(
                    fn=SmartScraperAgent().run,
                    name="SmartScraper",
                    input_key="url",
                    output_key="scraped_content",
                ),
                FunctionAgent(
                    fn=ClaimExtractorAgent().run,
                    name="ClaimExtractor",
                    input_key="scraped_content",
                    output_key="claims",
                ),
                FunctionAgent(
                    fn=FactCheckMatcherAgent().run,
                    name="FactCheckMatcher",
                    input_key="claims",
                    output_key="fact_checks",
                ),
                FunctionAgent(
                    fn=TruthScorerAgent().run,
                    name="TruthScorer",
                    input_key="fact_checks",
                    output_key="scored_claims",
                ),
            ]
        )

        # Run the processing pipeline first
        async for event in processing_pipeline.run_async(ctx):
            # We don't yield these events, we just let the state update
            pass

        # Now, dynamically create the final formatting agent with the data
        scored_claims: List[ScoredClaim] = ctx.session.state.get("scored_claims", [])
        
        # Create a string representation of the scored claims
        claims_str = "\\n".join([f"- Claim: {sc.claim.claim_text}\\n  Score: {sc.truth_score}\\n  Explanation: {sc.explanation}" for sc in scored_claims])

        formatter_instruction = f"""You are a fact-checking analyst. Your task is to generate a clear, concise, and user-friendly Markdown report based on the provided data.
Your report should follow this structure exactly:

1.  **Overall Verdict**: Start with a single, conclusive verdict for the main claim (e.g., "False", "Misleading", "True").
2.  **Main Claim**: State the most important claim that was analyzed.
3.  **Detailed Analysis**: Provide a paragraph explaining *why* the claim received its verdict. Synthesize the explanations from the provided data.
4.  **Verified Sources**: List the sources that were used to verify the claims. You will have to infer these from the fact-check documents.
5.  **Our Recommendation**: Write a corrected, more nuanced version of the main claim.
6.  **Media Literacy Tip**: Provide a general, helpful tip for identifying similar misinformation in the future.

Here is the data you must use:
{claims_str}
"""
        
        response_formatter_agent = LlmAgent(
            name="ResponseFormatter",
            model="gemini-1.5-flash-latest",
            instruction=formatter_instruction,
        )

        # Run the final formatter and yield its events
        async for event in response_formatter_agent.run_async(ctx):
            yield event 