from langchain_community.document_loaders import WebBaseLoader


url = 'https://www.applegadgetsbd.com/product/macbook-air-m4-13-inch-24gb512gb-10-core-cpu-10-core-gpu'
loader = WebBaseLoader(url)

docs = loader.load()

print(docs)