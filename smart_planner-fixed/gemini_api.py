import streamlit as st
import google.generativeai as genai
import json

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

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
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # List models to ensure model availability
        models = genai.list_models()
        # Use gemini-pro or fallback to available model
        model_name = "gemini-pro"
        available_models = [m.name for m in models]
        if model_name not in available_models:
            model_name = available_models[0]  # fallback to first available
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        output = response.text if hasattr(response, "text") else response.candidates[0].text
        return json.loads(output)
    except Exception as e:
        return {"goal": goal, "tasks": [], "error": str(e)}
