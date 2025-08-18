from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=1.5,
    max_new_tokens=100
)
chat_model = ChatHuggingFace(llm=llm)

loader = TextLoader('girlfriend.txt', encoding='utf-8')

documents = loader.load()

print(documents)

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["context"],
    template="what is my girlfriend name and age given in the following context: {context}"
)

chain = prompt | chat_model | parser

result = chain.invoke({"context": documents[0].page_content})

print(result)