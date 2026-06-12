import streamlit as st
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader


# ------------------------------------
# Helper Functions
# ------------------------------------

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_youtube_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    ]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    return None


# ------------------------------------
# Streamlit UI
# ------------------------------------

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize YouTube Videos and Web Pages")


# ------------------------------------
# Sidebar
# ------------------------------------

with st.sidebar:
    st.header("Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    model_name = st.selectbox(
        "Select Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b"
        ]
    )


# ------------------------------------
# URL Input
# ------------------------------------

generic_url = st.text_input(
    "Enter YouTube URL or Website URL"
)


# ------------------------------------
# Prompt Template
# ------------------------------------

prompt_template = """
Provide a concise summary of the following content in about 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# ------------------------------------
# Summarize Button
# ------------------------------------

if st.button("Summarize Content"):

    if not groq_api_key.strip():
        st.error("Please enter your Groq API Key.")
        st.stop()

    if not generic_url.strip():
        st.error("Please enter a URL.")
        st.stop()

    if not is_valid_url(generic_url):
        st.error("Please enter a valid URL.")
        st.stop()

    try:

        with st.spinner("Loading content and generating summary..."):

            llm = ChatGroq(
                model=model_name,
                groq_api_key=groq_api_key,
                temperature=0
            )

            # ------------------------------------
            # YouTube URL Handling
            # ------------------------------------
            if "youtube.com" in generic_url or "youtu.be" in generic_url:

                video_id = get_youtube_video_id(generic_url)

                if not video_id:
                    st.error("Invalid YouTube URL.")
                    st.stop()

                ytt_api = YouTubeTranscriptApi()

                transcript = ytt_api.fetch(video_id)

                transcript_text = " ".join(
                    snippet.text for snippet in transcript
                )

                docs = [
                    Document(page_content=transcript_text)
                ]

            # ------------------------------------
            # Website URL Handling
            # ------------------------------------
            else:

                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent":
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    }
                )

                docs = loader.load()

            if not docs:
                st.error("No content found.")
                st.stop()

            # ------------------------------------
            # Summarization Chain
            # ------------------------------------

            chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                prompt=prompt
            )

            summary = chain.run(docs)

            st.success("Summary Generated Successfully!")

            st.markdown("## Summary")
            st.write(summary)

    except Exception as e:
        st.error(f"Error: {e}")