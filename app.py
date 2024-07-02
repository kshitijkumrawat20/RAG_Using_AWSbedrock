# Building RAG application using AWS Bedrock
import json
import boto3
import os
import sys
from langchain_community.embeddings import BedrockEmbeddings 
from langchain_community.llms import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

# setting client
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# data ingestion 
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # splitting document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Creating vector embedding and storing in vector store 
def vector_store(docs):
    vectorStore = FAISS.from_documents(docs, bedrock_embeddings)
    vectorStore.save_local("faiss_index")

def get_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock_client, 
                  model_kwargs={'max_gen_len': 512, 'temperature': 0.5, 'top_p': 0.9})
    return llm

Prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but at least summarize with 250 words with detailed explanation. If you don't know the answer just say it is out of the context question please ask questions regarding the context.
<context>
{context}
</context>

Question: {question}
Assistant: 
"""
PROMPT = PromptTemplate(
    template=Prompt_template, input_variables=["context", "question"]
)

def get_response(llm, vectorStore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorStore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using AWS Bedrock")
    user_question = st.text_input("Ask question!!")
    with st.sidebar:
        st.title("Create Vector Store:")
        if st.button("Vector Update"):
            with st.spinner("Processing...."):
                docs = data_ingestion()
                vector_store(docs)
                st.success("Done")
                
    if st.button("Generate Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm()
            st.write(get_response(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
