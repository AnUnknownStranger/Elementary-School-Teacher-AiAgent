import streamlit as st
import re
from logic import Route_Question

st.set_page_config(page_title="Elementary School Agent", layout="centered")
st.title("Elementary School Teaching Assistant")
st.caption("Grades 1-5 | Math, Science, English, Social Studies")

def render_math(text):
    parts = re.split(r'(\$.*?\$)', text)
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            st.latex(part.strip('$'))
        else:
            st.markdown(part)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_list" not in st.session_state:
    st.session_state.history_list = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_math(message["content"])

if prompt := st.chat_input("Ask me a question!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render_math(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = Route_Question(prompt, st.session_state.history_list)

        response_text = response_data["response"]
        detected_subject = response_data.get("detected_subject", "General").capitalize()
        detected_grade = response_data.get("detected_grade", "Any")

        st.info(f"**Subject:** {detected_subject} | **Grade Level:** {detected_grade}")
        render_math(response_text)
        if response_data["sources"]:
            with st.expander("View Referenced Sources"):
                for source in response_data["sources"]:
                    st.caption(f"- {source}")

    st.session_state.messages.append({"role": "assistant", "content": response_text})