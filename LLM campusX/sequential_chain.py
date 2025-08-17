from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=1.5,
    max_new_tokens=100
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="give me the  explanation of the  {topic}"
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template="give me a summary of the  {text}"
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.get_graph().draw_ascii())
