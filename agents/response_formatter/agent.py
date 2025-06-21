from typing import List
from schemas.messages import ScoredClaim


class ResponseFormatterAgent:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[ResponseFormatterAgent] Initializing...")

    def run(self, scored_claims: List[ScoredClaim]) -> str:
        """
        Formats the final verification results into a user-friendly Markdown report.
        This function is intended to be used as a tool by an ADK LlmAgent.
        """
        print(f"Formatting final response for {len(scored_claims)} claims...")
        report = "# Fact-Check Report\n\n"

        if not scored_claims:
            return report + "No claims were processed or verified."

        for sc in scored_claims:
            report += f"## Claim: \"{sc.claim.claim_text}\"\n"
            report += f"**Truth Score:** {sc.truth_score:.2f}/1.0\n"
            report += f"**Explanation:** {sc.explanation}\n"
            report += "**Closest Fact-Check:**\n"
            report += f"> {sc.fact_check.match_document}\n"
            report += f"_(Match Similarity: {sc.fact_check.match_score:.2f})_\n\n"
            report += "---\n\n"
            
        print("Finished formatting response.")
        return report
