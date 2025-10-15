import os
import json

# For deployment, use Streamlit secrets if available
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

FREE_TIER_MODELS = {"gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash-latest"}

def _is_free_tier_model(name: str) -> bool:
    if not name:
        return False
    short = name.split("/")[-1]
    return short in FREE_TIER_MODELS or short.startswith("gemini-1.5-flash")


def _resolve_supported_model(desired_model: str) -> str:
    """Return a supported model name. Falls back by listing available models.

    The google-generativeai API returns names like "models/gemini-1.5-flash".
    We normalize to the short form when constructing GenerativeModel.
    """
    if not genai:
        raise RuntimeError("google-generativeai not installed")
    try:
        # List available models for the account/key
        models = list(getattr(genai, "list_models")())
        available = []
        for m in models:
            name = getattr(m, "name", "")
            # keep only models that support generateContent
            supported = getattr(m, "supported_generation_methods", [])
            if "generateContent" in supported:
                short = name.split("/")[-1] if name else ""
                if short:
                    available.append(short)
        # If desired is available, use it
        if desired_model in available and _is_free_tier_model(desired_model):
            return desired_model
        # Restrict fallback strictly to free-tier friendly models
        free_tier_candidates = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash-latest"]
        for candidate in free_tier_candidates:
            if candidate in available:
                return candidate
        # As a last attempt, find any model name that starts with gemini-1.5-flash
        for a in available:
            if a.startswith("gemini-1.5-flash"):
                return a
        # Do not fall back to paid/experimental models
        return "gemini-1.5-flash"
    except Exception:
        # Silent fallback if list_models not allowed
        return "gemini-1.5-flash"
    # Last resort default to free-tier model
    return "gemini-1.5-flash"


def generate_plan_from_llm(goal: str):
    prompt = (
        "You are a smart task planner. Given the user's goal, create a detailed, step-by-step plan as structured JSON.\n"
        "Return ONLY JSON, not any explanation.\n"
        "JSON Format:\n"
        "{\n"
        "  \"goal\": \"<user's goal>\",\n"
        "  \"tasks\": [\n"
        "    {\"step\": 1, \"description\": \"...\"},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        f"Goal: {goal}\n"
    )
    if genai and GOOGLE_API_KEY:
        # Configure client with API key (no api_version option supported)
        genai.configure(api_key=GOOGLE_API_KEY)
        # Enforce allowlist even if provided secret/env is different
        desired = GEMINI_MODEL if _is_free_tier_model(GEMINI_MODEL) else "gemini-1.5-flash"
        model_name = _resolve_supported_model(desired)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        output = response.text if hasattr(response, "text") else response.candidates[0].text
        try:
            return json.loads(output)
        except Exception:
            raise ValueError("Failed to parse LLM response as JSON.")
    raise RuntimeError("No valid LLM API key/configured or google-generativeai not installed.")
