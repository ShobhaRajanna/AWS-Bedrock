import json
import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#bedrock client
bedrock = boto3.client(service_name ="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

#data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

#vector embedding
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs,
                                             bedrock_embeddings)
    vectorstore_faiss.save_local("vectorstore_faiss")
    

def get_mistral_llm():
    llm = Bedrock("mistral.mixtral-8x7b-instruct-v0:1", client=bedrock,
                  model_kwargs={"max_tokens": 512, "temperature": 0.5, "top_p": 0.98})
    
    return llm

def get_amazon_llm():
    llm = Bedrock("amazon.titan-embed-text-v1", client=bedrock,
                  model_kwargs={"maxTokenCount": 512, "temperature": 0.5, "top_p": 0.98})
    
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a consise answer to the question at the end 
but use atleast summarize with 250 words with detailed explanations. If you don't know the answer,
just say you don't know,don't make up anything.
<context>
{context}
</context>
Question: {question}

Assistant:
"""

Prompt = PromptTemplate(template = prompt_template,input_variables = ["context","question"])

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type ='similarity',search_kwargs={"k":5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": Prompt})
    answer = qa({'query': query})
    return answer['result']
    
def main():
    st.set_page_config("Chat PDF")
    #add emoji
    st.header("Chat with PDF using AWS Bedrock 📚")
    
    USER_QUESTION = st.text_input("Enter your question")
    
    with st.sidebar:
        st.title("Update or Create Vector Store")
        
        if st.button("Vector Update"):
            with st.spinner('processing...'):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success('Vector store updated')
            
    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("vectorstore_faiss", bedrock_embeddings)
            llm = get_mistral_llm()
            st.write(get_response_llm(llm,faiss_index,USER_QUESTION))
            st.success("Done")
            
    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("vectorstore_faiss", bedrock_embeddings)
            llm = get_amazon_llm()
            st.write(get_response_llm(llm,faiss_index,USER_QUESTION))
            st.success("Done")
            
if __name__ == "__main__":
    main()