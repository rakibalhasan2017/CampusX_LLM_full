# Project Overview

## HuggingFace File
In this file, I call the **Qwen/Qwen3-4B-Instruct-2507** model, provide a prompt, and then print the result.

## Embedding File
In this file, I:
1. Embed a set of documents.
2. Provide single or multiple queries.
3. Perform a **semantic search** to calculate similarity scores.
4. Print the line(s) from the documents achieving the **highest similarity score**.
## chatbot file
In this file, I:
1. Create a chatbot
2. Save the history of the chat
3. message is saved as an aimessage, humanmessage, and systemmessage
## prompts file
learn to make a prompts template

## sequential_chain, parallel_chain, conditional_chain files
In this file, I:
1. learn how to use the chain method.
2. Explore the sequential, parallel, and conditional chain

## TextLoader, PdfLoader, WebsiteURL Loader file
In these files:
1. I explore how all the loaders work
2. Load the content and then create a prompt based on the content
3. Ask LLM based on the content

## Charactersplitter, recursivesplitter, semantic-meaning splitter
In this file:
1. we explore different cases of splitter
2. split code, text, document based on different method

## YouTube_ChatBot file
It's a project while I implement the full RAG pipeline. In this project:
1. First, I take a YouTube video ID as an input
2. Then I convert it to a Transcript
3. I divide the transcript into chunks
4. embedding to each chunk
5. Save the embedding to ChromaDb
6. Then add retrieval
7. Find the related_docs using the retrieval
8. Then take a prompt template
9. Give the related_docs and query to an LLM and print the output
10. Last i give it a UI using Streamlit. now, it looks like a website

## Agents.py file
1. Explore custom tools and built-in tools
2. Develop an agents that search on the web by DuckDuckGo and then get a result
3. Then the result will be summarized, and then translated the summary to Bangla
4. all the work is done by agents

## sending_email_agents.py
1. Build an agent to send an email
2. Use SendGrid to send the email
