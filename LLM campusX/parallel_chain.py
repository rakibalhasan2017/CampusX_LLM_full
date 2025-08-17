from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain.chains import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2-1.5B-Instruct",
    huggingfacehub_api_token=api_token
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="give me a short notes of the following text: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="give me a question answer from the following text: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="merge the short note and the question answer/n notes = {notes}, questions = {questions}",
    input_variables=["notes", "questions"]
)

parser = StrOutputParser()

chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'questions': prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

final_chain = chain | merge_chain

print(final_chain.get_graph().draw_ascii())

result = final_chain.invoke({'text':  "Artificial Intelligence is transforming industries worldwide. "
    "It helps in automation, decision-making, and improving efficiency. "
    "However, it also raises concerns about job displacement and ethical use." })

print(result)