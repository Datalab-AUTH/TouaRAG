import streamlit as st

st.set_page_config(page_title="Personalized TouaRAG Application", layout="wide")

st.title("Welcome to the Personalized TouaRAG Application")
st.write(
    """
    Use the sidebar to navigate between pages:
    - **Personalization:** Set your profile.
    - **Main Application:** Upload PDFs, choose RAG modes, and evaluate answers.
    """
)

