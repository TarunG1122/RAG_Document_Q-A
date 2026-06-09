import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="LangChain Search Chatbot",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 LangChain - Chat with Search")


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

st.sidebar.markdown("---")
st.sidebar.write(
    "Ask questions about current events, weather, technology, or anything that requires web search."
)


# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm a search-enabled chatbot. Ask me anything."
        }
    ]


# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------
# User Input
# ---------------------------
prompt = st.chat_input("Ask me anything...")


if prompt:

    if not api_key:
        st.warning("Please enter your Groq API Key.")
        st.stop()

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    # ---------------------------
    # LLM
    # ---------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        streaming=True
    )

    # ---------------------------
    # Search Tool
    # ---------------------------
    search = DuckDuckGoSearchRun(
        name="Search"
    )

    tools = [search]

    # ---------------------------
    # Agent
    # ---------------------------
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # ---------------------------
    # Assistant Response
    # ---------------------------
    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        try:

            response = agent.invoke(
                {"input": prompt},
                {"callbacks": [st_cb]}
            )

            answer = response["output"]

            st.write(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            error_message = f"Error: {str(e)}"

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )