import asyncio
import os
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# Load environment variables from .env file
load_dotenv()

# Set the GOOGLE_API_KEY if it's not already set
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not os.getenv("GOOGLE_API_KEY"):
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    else:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in the environment.")

from core.factos_agent import FactosAgent


async def main():
    """
    Runs the Factos verification pipeline with a test URL.
    """
    test_url = "https://www.theguardian.com/world/2025/jun/11/uk-and-gibraltar-strike-deal-over-territorys-future-and-borders"
    print(f"--- Running ADK Agent for URL: {test_url} ---")

    session_service = InMemorySessionService()

    # Create the session before running the agent, providing the initial URL.
    initial_state = {"url": test_url}
    await session_service.create_session(
        app_name="factos_adk",
        user_id="test_user",
        session_id="test_session",
        state=initial_state
    )
    
    runner = Runner(
        app_name="factos_adk",
        agent=FactosAgent,
        session_service=session_service
    )

    # The runner's run method expects user_id and session_id
    # The new_message must contain at least one Part, even if it's empty.
    events = runner.run(
        user_id="test_user",
        session_id="test_session",
        new_message=Content(parts=[Part()])
    )

    final_response = "No final response captured."
    for event in events:
        if event.is_final_response():
            # Assuming the final response is in the text of the first part
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
            break # Exit after getting the final response

    print("\n--- Final Response ---")
    print(final_response)
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    asyncio.run(main()) 