import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Query Page", layout="wide")

# Initialize session state if not already set
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None
if "pdf_files" not in st.session_state:
    st.session_state["pdf_files"] = []


uploaded_files = st.session_state.get("pdf_files", [])
chat_history = st.session_state.get("messages", [])

st.title("Touareg Assistant")

# --- Model Selection ---
st.sidebar.subheader("Select Model")
model_option = st.sidebar.radio("Choose Model: (Local Default)", ["Local", "OpenAI"])

# Add load model button 
if st.sidebar.button("Load Model"):
    if model_option == "Local":
        try:
            response_model = requests.post(
                'http://127.0.0.1:8000/model/local',
                headers={'accept': 'application/json'}
            )
            if response_model.status_code == 200:
                st.sidebar.success("Local model loaded successfully")
            else:
                st.sidebar.error(f"Local model loading failed with status code {response_model.status_code}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error loading local model: {str(e)}")
    else:
        try:
            response_model = requests.post(
                'http://127.0.0.1:8000/model/openai',
                headers={'accept': 'application/json'}
            )
            if response_model.status_code == 200:
                st.sidebar.success("OpenAI model loaded successfully")
            else:
                st.sidebar.error(f"OpenAI model loading failed with status code {response_model.status_code}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error loading OpenAI model: {str(e)}")

# -----------------------
# Sidebar: File Upload & RAG Modes
# -----------------------

st.sidebar.subheader("Upload PDF Files")
st.sidebar.info("Upload your PDFs here. (Only .pdf files are accepted)")
uploaded_files = st.sidebar.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)
# Add top_k parameter in the sidebar
top_k = st.sidebar.slider("Select Top K Retrieved Results", min_value=1, max_value=10, value=5)

if uploaded_files:
    if st.sidebar.button("Process Uploaded Files"):
        status_container = st.sidebar.empty()
        
        for file in uploaded_files:
            with status_container:
                with st.spinner(f'Processing {file.name}...'):
                    # Prepare the files parameter for the POST request
                    files = {'files': (file.name, file, 'application/pdf')}
                    
                    try:
                        response = requests.post(
                            f'http://127.0.0.1:8000/api/upload/{top_k}',
                            headers={'accept': 'application/json'},
                            files=files
                        )
                        
                        if response.status_code == 200:
                            st.session_state["pdf_files"].append(file)
                            st.sidebar.success(f"✅ Successfully uploaded: {file.name}")
                        else:
                            st.sidebar.error(f"Failed to upload {file.name}. Status code: {response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        st.sidebar.error(f"Error uploading {file.name}: {str(e)}")

        status_container.success("✅ All files processed!")

st.sidebar.subheader("RAG Modes")
mode = st.sidebar.radio("Choose a mode:", ["Basic", "Advanced"])
if mode == "Basic":
    st.sidebar.write("Using baseline methodology.")
    # For basic mode, set default values for methodology 
    methodology = "Baseline"
else:
    st.sidebar.write("Configure advanced settings:")
    methodology = st.sidebar.selectbox("Select Methodology", ["Hybrid", "Automerge", "HyDE", "GraphRAG"])

# ------------------------------
# Chat Interface (Main Area)
# ------------------------------

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------
# API Call Functions
# ------------------------------

def call_query_api(query, method):
    """
    Call the query API endpoint with the provided query and method.
    """
    try:
        response = requests.post(
            'http://127.0.0.1:8000/api/query',
            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
            json={'query': query, 'method': method}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"role": "assistant", "content": f"Error: API call failed with status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}

# ------------------------------
# Accept User Input & Process Chat
# ------------------------------

# Accept user input from the chat input box
prompt = st.chat_input("What is up?")
if prompt:
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine the method based on selected mode and methodology
    method = "baseline" if mode == "Basic" else methodology.lower()

    # Show thinking message while processing
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking...")

        # Make API call
        response = call_query_api(prompt, method)

        # Format the API response for chat
        chat_response = {
            "role": "assistant",
            "content": response.get("response", "Sorry, I couldn't process your request.")
        }

        # Replace thinking message with actual response
        thinking_placeholder.markdown(chat_response["content"])

    # Append the response to the chat history
    st.session_state.messages.append(chat_response)
    
    # Display the retrieved context in a separate area if available
    context_data = response.get("context")
    if context_data:
        # Join list items into a single string if needed
        context_str = "\n".join(context_data) if isinstance(context_data, list) else context_data
        # Process escape characters
        context_str = context_str.encode('utf-8').decode('unicode_escape')
        
        # Create a new message for the retrieved context
        context_message = {
            "role": "assistant",
            "content": f"**Retrieved Context:**\n\n{context_str}"
        }
        
        # Append the context message to the chat history
        st.session_state.messages.append(context_message)
        
        # Display the context immediately in the chat (optional)
        with st.chat_message("assistant"):
            st.markdown(context_message["content"])
