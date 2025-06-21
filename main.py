from fastapi import FastAPI, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from core.pipeline import VerificationPipeline

load_dotenv()

app = FastAPI(
    title="Factos ADK",
    description="A multi-agent system for verifying news truth.",
    version="0.1.0",
)

pipeline = VerificationPipeline()


class VerifyRequest(BaseModel):
    url: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Factos ADK API"}


@app.post("/verify")
async def verify_article(request: VerifyRequest):
    final_response = pipeline.run(request.url)
    return Response(content=final_response, media_type="application/json")
