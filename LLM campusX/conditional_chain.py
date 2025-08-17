from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableBranch

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2-1.5B-Instruct",
    huggingfacehub_api_token=api_token
)
model = ChatHuggingFace(llm=llm)

# Sentiment classification chain
prompt1 = PromptTemplate(
    template='classify the sentiment of the following text: {text} in either positive or negative',
    input_variables=["text"]
)
parser = StrOutputParser()
classify_chain = prompt1 | model | parser

# Lambda to extract sentiment from full sentence
extract_sentiment = RunnableLambda(lambda x: {"sentiment": "positive" if "positive" in x.lower() else "negative", "text": x})

# Positive chain
positive_prompt = PromptTemplate(
    template="The sentiment is positive. Write a motivational message based on the text: {text}",
    input_variables=["text"]
)
positive_chain = positive_prompt | model | parser

# Negative chain
negative_prompt = PromptTemplate(
    template="The sentiment is negative. Give a suggestion to improve the mood based on the text: {text}",
    input_variables=["text"]
)
negative_chain = negative_prompt | model | parser

# Branch chain
# Default branch
default_branch = RunnableLambda(lambda x: "Sentiment not recognized")

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_chain),
    (lambda x: x["sentiment"] == "negative", negative_chain),
    default_branch  # <-- must provide default
)
# Full chain: classify -> extract -> branch
full_chain = classify_chain | extract_sentiment | branch_chain

# Test
result = full_chain.invoke({"text": "I love programming!"})
print(result)
