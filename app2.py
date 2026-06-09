import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================================
# ENV VARIABLES
# ==========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Conversational PDF RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Conversational RAG with PDF & Chat History")
st.markdown(
    "Upload one or more PDFs and ask questions about them."
)

# ==========================================================
# EMBEDDINGS
# ==========================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ==========================================================
# GROQ API KEY
# ==========================================================

groq_api_key = st.text_input(
    "Enter Groq API Key",
    type="password"
)

if not groq_api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()

# ==========================================================
# LLM
# ==========================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ==========================================================
# SESSION ID
# ==========================================================

session_id = st.text_input(
    "Session ID",
    value="default_session"
)

# ==========================================================
# CHAT HISTORY STORE
# ==========================================================

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(
    session: str
) -> BaseChatMessageHistory:

    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]

# ==========================================================
# PDF UPLOAD
# ==========================================================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Processing PDFs..."):

        documents = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)

            docs = loader.load()

            documents.extend(docs)

        # ==================================================
        # CHUNKING
        # ==================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        # ==================================================
        # VECTOR STORE
        # ==================================================

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # ==================================================
        # HISTORY AWARE RETRIEVER
        # ==================================================

        contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question that can be understood
        without the chat history.

        Do NOT answer the question.
        Only rewrite it if necessary.
        """

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = (
            create_history_aware_retriever(
                llm,
                retriever,
                contextualize_q_prompt
            )
        )

        # ==================================================
        # QA PROMPT
        # ==================================================

        system_prompt = """
        You are an assistant for question-answering tasks.

        Use the retrieved context below to answer.

        If the answer is not present in the context,
        simply say you don't know.

        Keep answers concise and accurate.

        Context:
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    st.success("PDFs processed successfully!")

    # ======================================================
    # USER QUESTION
    # ======================================================

    user_input = st.text_input(
        "Ask a question about your PDFs:"
    )

    if user_input:

        try:

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            st.subheader("🤖 Assistant")
            st.write(response["answer"])

            # ==============================================
            # DISPLAY CHAT HISTORY
            # ==============================================

            history = get_session_history(session_id)

            with st.expander("Chat History"):

                for msg in history.messages:
                    st.write(
                        f"**{msg.type.capitalize()}**: {msg.content}"
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("Upload at least one PDF to begin.")