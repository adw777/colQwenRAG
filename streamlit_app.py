import streamlit as st
import requests
import json

# Configure the page
st.set_page_config(page_title="ColPali Legal-AI K-Hub", layout="wide")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def query_api(user_input):
    """Send query to API and return response"""
    url = "http://localhost:8001/query"
    
    try:
        response = requests.post(
            url,
            json={"query": user_input, "limit": 10},
            timeout=90
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None

# Main title
st.title("ColPali Legal-AI K-Hub")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                st.text(message["sources"])

# Chat input
if prompt := st.chat_input("What would you like to know about?"):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get API response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_api(prompt)
            
            if response:
                st.write(response["ai_response"])
                with st.expander("View Sources"):
                    st.text(response["extracted_text"])
                
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["ai_response"],
                    "sources": response["extracted_text"]
                })
            else:
                st.error("Failed to get response from the API")

# Add a clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()