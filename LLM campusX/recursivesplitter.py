from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NotebookLoader

loader = NotebookLoader("RSA_Algo.ipynb", include_outputs=False)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=500,
    chunk_overlap=50
)

splits = splitter.split_documents(docs)

for i, chunk in enumerate(splits):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)
