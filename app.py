# app.py
import streamlit as st
from transformers import pipeline
from huggingface_hub import login

# Authenticate with Hugging Face
login(token="TOKEN_KEY")  # Replace 'your_token' with your Hugging Face token

# Load the model pipeline
qa_model = pipeline("text-generation", model="distilgpt2")

# Streamlit app UI
st.title("AI Assistant")

st.write("Ask a question, and the assistant will respond.")

# User input
user_input = st.text_input("Enter your question here:")

# Generate and display the response
if user_input:
    with st.spinner("Generating response..."):
        prompt = f"Assistant, answer the following question: {user_input}"
        response = qa_model(prompt, max_length=50, num_return_sequences=1)
        st.write("Response:", response[0]['generated_text'])
