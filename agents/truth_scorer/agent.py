from typing import List

from schemas.messages import FactCheck, ScoredClaim


class TruthScorerAgent:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[TruthScorerAgent] Initializing...")

    def run(self, fact_checks: List[FactCheck]) -> List[ScoredClaim]:
        """
        Scores the truthfulness of claims based on their fact-check matches.
        This function is intended to be used as a tool by an ADK LlmAgent.
        """
        print(f"Scoring {len(fact_checks)} fact-checked claims...")
        scored_claims = []
        for fc in fact_checks:
            # Simple rule-based scoring logic
            if fc.match_score > 0.9:
                truth_score = 1.0
                explanation = "The claim is well-supported by a very close fact-check."
            elif fc.match_score > 0.7:
                truth_score = 0.7
                explanation = "The claim is likely true, with a reasonably similar fact-check."
            elif fc.match_score > 0.5:
                truth_score = 0.5
                explanation = "The claim is uncertain, with a moderately similar fact-check."
            else:
                truth_score = 0.2
                explanation = "The claim is likely false or unsubstantiated, with a dissimilar fact-check."

            scored_claims.append(ScoredClaim(
                claim=fc.claim,
                fact_check=fc,
                truth_score=truth_score,
                explanation=explanation
            ))
            
        print(f"Finished scoring claims.")
        return scored_claims
