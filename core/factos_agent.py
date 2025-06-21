from google.adk.agents import LlmAgent, SequentialAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part
from typing import Callable, AsyncGenerator

from agents.smart_scraper.agent import SmartScraperAgent
from agents.claim_extractor.agent import ClaimExtractorAgent
from agents.fact_check_matcher.agent import FactCheckMatcherAgent
from agents.truth_scorer.agent import TruthScorerAgent
from agents.response_formatter.agent import ResponseFormatterAgent


class FunctionAgent(BaseAgent):
    """A simple agent that executes a Python function."""

    model_config = {"arbitrary_types_allowed": True}
    
    # The function to execute
    fn: Callable
    # The key to read from session state for the function's input
    input_key: str
    # The key to write the function's output to in session state
    output_key: str

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Runs the function and yields a final response event."""
        print(f"[{self.name}] Running function...")
        
        # Get input from session state
        fn_input = ctx.session.state.get(self.input_key)
        
        if fn_input is None:
            raise ValueError(f"Input key '{self.input_key}' not found in session state.")

        # Run the function
        result = self.fn(fn_input)

        # Store the output in session state
        ctx.session.state[self.output_key] = result
        
        # Yield a final event to signal completion
        yield Event(
            author=self.name, 
            content=Content(parts=[Part(text=str(result))])
        )


# 1. Define Tool-based Agents for each step
# Use the reliable FunctionAgent for deterministic steps
smart_scraper_agent = FunctionAgent(
    name="SmartScraper",
    fn=SmartScraperAgent().run,
    input_key="url",
    output_key="scraped_content",
)

claim_extractor_agent = FunctionAgent(
    name="ClaimExtractor",
    fn=ClaimExtractorAgent().run,
    input_key="scraped_content",
    output_key="claims",
)

fact_check_matcher_agent = FunctionAgent(
    name="FactCheckMatcher",
    fn=FactCheckMatcherAgent().run,
    input_key="claims",
    output_key="fact_checks",
)

truth_scorer_agent = FunctionAgent(
    name="TruthScorer",
    fn=TruthScorerAgent().run,
    input_key="fact_checks",
    output_key="scored_claims",
)

# Use an LlmAgent only for the final, generative step
response_formatter_agent = LlmAgent(
    name="ResponseFormatter",
    model="gemini-1.5-flash-latest",
    instruction="You will receive a list of scored claims. Format them into a clear, concise, and user-friendly Markdown report. If the list is empty, state that no claims could be verified.",
    output_key="final_report",
)

# 2. Define the top-level sequential agent
FactosAgent = SequentialAgent(
    name="FactosAgent",
    sub_agents=[
        smart_scraper_agent,
        claim_extractor_agent,
        fact_check_matcher_agent,
        truth_scorer_agent,
        response_formatter_agent,
    ],
) 