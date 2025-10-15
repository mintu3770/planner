import os
import json

# For deployment, use Streamlit secrets if available
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        output = response.text if hasattr(response, "text") else response.candidates[0].text
        try:
            return json.loads(output)
        except Exception:
            raise ValueError("Failed to parse LLM response as JSON.")
    raise RuntimeError("No valid LLM API key/configured or google-generativeai not installed.")
