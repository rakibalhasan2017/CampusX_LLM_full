from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain
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
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

loader = PyPDFLoader("Deviance and social control.pdf")
documents = loader.load()

prompt = PromptTemplate(
    input_variables=["context"],
    template="what is the main topic of the following document: {context}"
)

chain = prompt | model |  parser
context = " ".join([doc.page_content for doc in documents])

result = chain.invoke({'context' : context})

print(result)