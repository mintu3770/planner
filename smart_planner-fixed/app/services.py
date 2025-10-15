import os
import json

# For deployment, use Streamlit secrets if available
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "")  # optional hint only
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")  # optional hint only

try:
    import google.generativeai as genai
except ImportError:
    genai = None

FREE_TIER_MODELS = {"gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash-latest"}

def _is_free_tier_model(name: str) -> bool:
    if not name:
        return False
    short = name.split("/")[-1]
    # Treat any flash variant as free-tier capable
    return short in FREE_TIER_MODELS or short.startswith("gemini-1.5-flash") or "flash" in short


def _list_free_tier_candidates(desired_model: str) -> list:
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
        # Start with hint if valid and free-tier
        candidates = []
        if desired_model and desired_model in available and _is_free_tier_model(desired_model):
            candidates.append(desired_model)
        # Add all available free-tier-capable models, prefer flash-8b then flash
        flash8b = [m for m in available if m.startswith("gemini-1.5-flash-8b")]
        flash = [m for m in available if _is_free_tier_model(m) and m not in flash8b]
        candidates.extend(flash8b + flash)
        # Deduplicate preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        # Fallback candidates if listing worked but found none
        if not ordered:
            ordered = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash-latest"]
        return ordered
    except Exception:
        # Silent fallback if list_models not allowed
        return ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash-latest"]


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
        # Build candidate list (auto-select free-tier), using GEMINI_MODEL only as a hint
        candidates = _list_free_tier_candidates(GEMINI_MODEL if _is_free_tier_model(GEMINI_MODEL) else "")
        # Bias to lighter models for latency
        candidates = sorted(set(candidates), key=lambda m: (0 if "flash-8b" in m else 1, m))
        last_error = None
        for model_name in candidates:
            try:
                model = genai.GenerativeModel(model_name)
                # Retry transient timeouts with increasing deadline
                for attempt in range(3):
                    try:
                        response = model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "top_k": 40,
                                "max_output_tokens": 512,
                            },
                            request_options={"timeout": 25 + attempt * 15},
                        )
                        output = response.text if hasattr(response, "text") else response.candidates[0].text
                        try:
                            return json.loads(output)
                        except Exception:
                            raise ValueError("Failed to parse LLM response as JSON.")
                    except Exception as inner:
                        # Bubble up quota errors immediately
                        if hasattr(inner, "status") and getattr(inner, "status") == 429:
                            raise
                        message = str(inner).lower()
                        if "timeout" in message or "deadline" in message or "504" in message:
                            # try next retry for same model
                            last_error = inner
                            continue
                        # non-timeout error: break to try next candidate
                        last_error = inner
                        break
            except Exception as e:
                # Try next candidate on 404/not supported; propagate 429 quota errors immediately
                if hasattr(e, "status") and getattr(e, "status") == 429:
                    raise
                # Keep last error to raise if all candidates fail
                last_error = e
                continue
        # If all candidates failed, raise the last error
        if last_error:
            raise last_error
    raise RuntimeError("No valid LLM API key/configured or google-generativeai not installed.")
