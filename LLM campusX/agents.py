from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(hf_token)

# Initialize Hugging Face endpoint with temperature and max tokens
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    huggingfacehub_api_token=hf_token,
    temperature=1.5,        # creativity
    max_new_tokens=100
)

chat_model = ChatHuggingFace(llm=llm)

search = DuckDuckGoSearchRun()

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n{text}\nSummary:"
)

translation_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Translate the following text to bangla:\n{summary}\nTranslation:"
)

parser = StrOutputParser()

summary_chain = summarization_prompt | chat_model | parser
translation_chain = translation_prompt | chat_model | parser

@tool
def summarization_tool(text: str) -> str:
    """Summarize the given text"""
    return summary_chain.invoke(text)

@tool
def translation_to_bangla(text: str) -> str:
    """Translate the given text to Bangla"""
    return translation_chain.invoke(text)

tools = [search, summarization_tool, translation_to_bangla]

agents = initialize_agent(
    tools=tools,
    llm=chat_model,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

result = agents.invoke("Who is the president of Bangladesh right now? give me the details of the president.")

print(result)