import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit UI
st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎬", layout="wide")
st.title("🎬 YouTube RAG Chatbot")

video_id = st.text_input("Enter YouTube Video ID", placeholder="dQw4w9WgXcQ")
query = st.text_input("Ask a question about the video", placeholder="Summarize this video")

if st.button("Get Answer") and video_id and query:
    with st.spinner("Fetching transcript & processing..."):
        try:
            ytt_api = YouTubeTranscriptApi()
            Transcript = ytt_api.fetch(video_id)
            full_text = " ".join([i.text for i in Transcript])
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
             chunk_overlap=200,
             )
            chunks = text_splitter.create_documents([full_text])
            texts = [chunk.page_content for chunk in chunks]  
            metadatas = [{"chunk_id": i} for i in range(len(texts))]
            ids = [str(i) for i in range(len(texts))]
            # Embeddings + VectorDB
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = Chroma(
            collection_name="youtube_collection",
            persist_directory="./chroma_db",
            embedding_function=embedding_model
            )
            vectordb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
             )
            retriever = vectordb.as_retriever(
              search_type="similarity",
               search_kwargs={"k": 4}
                )
            related_docs = retriever.invoke(query)

            context = "\n".join(doc.page_content for doc in related_docs)
            prompts = PromptTemplate(
             template="""
              You are a helpful assistant.
               Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

               {context}
               Question: {question}
                   """,
                     input_variables = ['context', 'question']
                    )
            # Define LLM
            llm = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen3-4B-Instruct-2507",
                task="text-generation",
                huggingfacehub_api_token=HF_TOKEN
            )
            chat_model =  ChatHuggingFace(llm=llm)
            final_prompt = prompts.invoke({"context": context, "question": query})
            # Run
            result = chat_model.invoke(final_prompt)
            # Show Answer
            st.subheader("💡 Answer")
            st.write(result.content)

            with st.expander("Sources"):
               for doc in related_docs:
                  st.write(doc.page_content[:200] + "...")
                  st.caption(doc.metadata)

        except Exception as e:
            st.error(f"❌ Error: {e}")