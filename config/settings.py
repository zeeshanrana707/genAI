"""Application settings and LLM configuration module."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# Retrieve API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# Configured Model Identifiers
# Priority models that have live free-tier quota on Google Gemini
GEMINI_MODELS_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]

RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "gemini-3.1-flash-lite")
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.1-flash-lite")
SCHEDULER_MODEL = os.getenv("SCHEDULER_MODEL", "gemini-3.1-flash-lite")

# Execution Mode: 'live', 'mock', or 'auto'
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "auto").lower()

# Output directory for deliverables
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_live_key_configured() -> bool:
    """Check if a real Gemini API key is configured."""
    key = GEMINI_API_KEY.strip() if GEMINI_API_KEY else ""
    return bool(key and not key.startswith("your_") and len(key) > 10)


def is_deepseek_key_configured() -> bool:
    """Check if a real DeepSeek API key is configured."""
    key = DEEPSEEK_API_KEY.strip() if DEEPSEEK_API_KEY else ""
    return bool(key and not key.startswith("your_") and len(key) > 10)


def get_llm(model_name: str):
    """Create a CrewAI LLM instance configured with the specified model identifier.
    
    Args:
        model_name: Model identifier (e.g., 'gemini-4-argon', 'gemini-3.1-pro')
        
    Returns:
        crewai.LLM configured for the specified model.
    """
    from crewai import LLM
    
    # Ensure prefix format expected by provider backend
    formatted_model = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"
    api_key = GEMINI_API_KEY if is_live_key_configured() else "mock_gemini_api_key"
    
    return LLM(
        model=formatted_model,
        api_key=api_key,
        temperature=0.7
    )
