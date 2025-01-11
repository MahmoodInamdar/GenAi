import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
import yt_dlp

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="\U0001f9dc")
st.title("\U0001f9dc LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YouTube or website) to summarize
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Function to fetch YouTube data using yt_dlp
def fetch_youtube_data(youtube_url):
    try:
        ydl_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            title = info.get('title', 'No Title Available')
            description = info.get('description', 'No Description Available')
            return title, description
    except Exception as e:
        raise RuntimeError(f"Failed to fetch YouTube data: {e}")

# Button to trigger summarization
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load content based on the URL type
                docs = None
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        title, description = fetch_youtube_data(generic_url)
                        st.write(f"**Video Title:** {title}")
                        docs = [{"text": description}]
                    except RuntimeError as yt_error:
                        st.error(f"Failed to process YouTube URL: {yt_error}")
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    loaded_docs = loader.load()

                    # Ensure loader returned valid documents
                    if isinstance(loaded_docs, list) and loaded_docs:
                        if isinstance(loaded_docs[0], dict) and "text" in loaded_docs[0]:
                            docs = loaded_docs
                        elif isinstance(loaded_docs[0], str):
                            docs = [{"text": " ".join(loaded_docs)}]
                        else:
                            raise ValueError("Unexpected format in loaded documents.")
                    else:
                        raise ValueError("Loader returned empty or invalid document list.")

                # Perform summarization if docs are available
                if docs:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)
                else:
                    st.error("No valid content to summarize.")
        except Exception as e:
            st.exception(f"Exception: {e}")
