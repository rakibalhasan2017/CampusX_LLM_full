from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
import os
from template import prompt_template

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    temperature=1.5,        # creativity
    max_new_tokens=100
)
chat_model = ChatHuggingFace(llm=llm)

st.title("summarize to")

paper = st.selectbox(
    "Choose the paper:",
    ("word2vec", "BERT", "GPT-3")
)

explanation_type = st.selectbox(
    "Choose the type of explanation:",
    ("Beginner Friendly", "Code Extensive", "Technical")
)

length_type = st.selectbox(
    "Choose the length of explanation:",
    ("Short(1- 2 paragraph)", "Medium", "Long(in detailed explanation)")
)

final_prompt = prompt_template.format(
    paper=paper,
    explanation_type=explanation_type,
    length_type=length_type
)

if st.button("Summarize"):
    with st.spinner("Generating summary... ⏳"):  # Loading indicator
        response = chat_model.invoke(final_prompt)
    st.success("Done!")  # Optional success message
    st.write(response.content)