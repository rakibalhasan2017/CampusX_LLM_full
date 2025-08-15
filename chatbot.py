import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=1.5,
    max_new_tokens=100
)
chat_model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content="You are a helpful assistant. Please answer the user's questions to the best of your ability."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    response = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Bot:", response.content)

print("Chat ended.")
print(chat_history)
