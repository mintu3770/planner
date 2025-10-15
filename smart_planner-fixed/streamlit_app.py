import streamlit as st
from app.services import generate_plan_from_llm

st.set_page_config(page_title="Smart Task Planner", layout="centered")
st.title("Smart Task Planner")
st.write("Enter your goal below to receive a step-by-step plan:")

goal = st.text_area("Your Goal", placeholder="Describe your goal...")

if st.button("Generate Plan") and goal.strip():
    with st.spinner("Generating..."):
        try:
            plan = generate_plan_from_llm(goal)
            if plan and "tasks" in plan:
                st.success(f"Goal: {plan['goal']}")
                for task in plan["tasks"]:
                    st.info(f"**Step {task['step']}:** {task['description']}")
            else:
                st.error("No tasks generated.")
        except Exception as e:
            st.error(f"Error: {e}")