from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
documents = [
    "i have a girlfriend name rahmaa",
    "rahmaa is 18 years old who live in dhanmondi dhaka ",
    "rahmaa loves to play badminton",
    "my friend ashique have a girlfriend named jerin"
]

embeddings = embedding_model.embed_documents(documents)

query = "what is the name of my girlfriend?"

query_embedding = embedding_model.embed_query(query)

similarities = cosine_similarity([query_embedding], embeddings)[0]

print(similarities)

most_similar_idx = similarities.argmax()

print(documents[most_similar_idx])


querys = [
    "where does rahmaa live?",
    "what sport does rahmaa love?",
]

query_embeddings = embedding_model.embed_documents(querys)

similaritiess = cosine_similarity(query_embeddings, embeddings)

print(similaritiess)

for i, query in enumerate(querys):
    most_similar_idx = similaritiess[i].argmax()
    print(documents[most_similar_idx])
    print(similaritiess[i][most_similar_idx])
    print()
    
