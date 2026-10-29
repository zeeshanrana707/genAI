"""Application settings and LLM configuration module."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# Retrieve API Keys (Gemini only — all free, zero-cost)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Configured Model Identifiers
# Priority models that have live free-tier and subscription quota on Google Gemini
GEMINI_MODELS_PRIORITY = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "gemini-3.5-flash")
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.5-flash")
SCHEDULER_MODEL = os.getenv("SCHEDULER_MODEL", "gemini-3.5-flash")

# Execution Mode: 'live', 'mock', or 'auto'
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "auto").lower()

# Output directory for deliverables
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Uploaded papers storage directory
PAPERS_DIR = BASE_DIR / "uploaded_papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


def is_live_key_configured() -> bool:
    """Check if a real Gemini API key is configured."""
    key = GEMINI_API_KEY.strip() if GEMINI_API_KEY else ""
    return bool(key and not key.startswith("your_") and len(key) > 10)


def get_llm(model_name: str):
    """Create a CrewAI LLM instance configured with the specified model identifier."""
    from crewai import LLM

    formatted_model = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"
    api_key = GEMINI_API_KEY if is_live_key_configured() else "mock_gemini_api_key"

    return LLM(
        model=formatted_model,
        api_key=api_key,
        temperature=0.7
    )
