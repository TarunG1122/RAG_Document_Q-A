import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.

    Please provide the most accurate response possible.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Create Vector Embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = OpenAIEmbeddings()

        st.session_state.loader = PyPDFDirectoryLoader("Documents")

        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# UI
st.title("RAG Document Q&A using OpenAI")

user_prompt = st.text_input("Enter your question from the documents")

if st.button("Create Document Embeddings"):
    create_vector_embedding()
    st.success("Vector Database is Ready!")

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first.")

    else:

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        retriever = st.session_state.vectors.as_retriever()

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        start = time.time()

        response = retrieval_chain.invoke(
            {"input": user_prompt}
        )

        end = time.time()

        st.write(f"Response Time: {end - start:.2f} seconds")

        st.subheader("Answer")
        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"], 1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)
                st.divider()