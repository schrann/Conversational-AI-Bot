import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("api")

# Hugging Face Inference API URL
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Function to query the Hugging Face API
def query_huggingface(prompt):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.json()}"

# Streamlit UI
st.title("🤖 AI Chatbot with Hugging Face")
st.write("Chat with an AI-powered bot using Mistral-7B-Instruct.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get bot response
    bot_response = query_huggingface(user_input)

    # Append bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display bot response
    with st.chat_message("assistant"):
        st.write(bot_response)
