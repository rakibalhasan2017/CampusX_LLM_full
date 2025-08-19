from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Deviance and social control.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=8,   separator="\n")
chunks = text_splitter.split_text(documents[0].page_content)

print(chunks)  # Print the first chunk to verify
