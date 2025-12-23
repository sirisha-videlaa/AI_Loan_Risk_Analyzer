from pydantic import BaseModel


class LoanAnalysisResponse(BaseModel):
    risk_score: float
    summary: str
    contradictions: list[str]
    hidden_fees: list[str]