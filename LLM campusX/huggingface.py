from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face endpoint with temperature and max tokens
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    temperature=1.5,        # creativity
    max_new_tokens=100
)

# Create chat model
chat_model = ChatHuggingFace(llm=llm)

# Use the model
result = chat_model.invoke("write a poem of 5 line about my girlfrind name Rahmaa")

print(result.content)
