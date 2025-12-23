import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# make genai import optional so the app can start even if the package is missing
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---------- CONFIG ----------
if GENAI_AVAILABLE:
    genai.configure(api_key=os.getenv("API_KEY") or os.getenv("GENAI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    class _DummyModel:
        def generate_content(self, prompt):
            raise RuntimeError(
                "google.generativeai is not installed or configured. "
                "Install with: pip install google-generative-ai and set API_KEY (or GENAI_API_KEY) env var."
            )
    model = _DummyModel()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LOAD RAG ----------
try:
    with open("rag_corpus.txt", "r", encoding="utf-8") as f:
        RAG_CONTEXT = f.read()
except Exception:
    RAG_CONTEXT = ""

# ---------- UTILS ----------
def build_prompt(loan_text: str) -> str:
    return f"""
You are a financial risk analyst.

Use the regulatory context below to analyze the loan agreement.

REGULATORY CONTEXT:
{RAG_CONTEXT}

LOAN DOCUMENT:
{loan_text}

TASK:
1. Detect contradictions
2. Detect hidden or unclear fees
3. Summarize risks in plain language
4. Assign a risk score between 0 and 1

Return ONLY valid JSON in this exact format:
{{
  "risk_score": number,
  "summary": string,
  "contradictions": [string],
  "hidden_fees": [string]
}}
"""

# ---------- API ----------
@app.post("/analyze")
async def analyze_loan(file: UploadFile = File(...)):
    # import response model lazily so module import doesn't fail if models.py has issues
    try:
        from models import LoanAnalysisResponse  # type: ignore
        have_model_class = True
    except Exception:
        LoanAnalysisResponse = None
        have_model_class = False

    text = (await file.read()).decode("utf-8")
    prompt = build_prompt(text)

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = getattr(response, "text", str(response)).strip()

    # Gemini sometimes wraps JSON in markdown
    if result.startswith("```"):
        parts = result.split("```")
        if len(parts) >= 2:
            result = parts[1].strip()

    try:
        if have_model_class and LoanAnalysisResponse is not None:
            return LoanAnalysisResponse.model_validate_json(result)
        else:
            import json
            return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model output: {e}")
