import streamlit as st

st.set_page_config(page_title="Personalized RAG Application", layout="wide")

st.title("Welcome to the Personalized RAG Application")
st.write(
    """
    Use the sidebar to navigate between pages:
    - **Personalization:** Set your profile.
    - **Main Application:** Upload PDFs, choose RAG modes, and evaluate answers.
    """
)

