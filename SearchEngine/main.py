import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from cachetools import TTLCache
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cache to avoid redundant API calls
cache = TTLCache(maxsize=100, ttl=600)  # Cache for 10 minutes

# Rate limit delay (in seconds)
RATE_LIMIT_DELAY = 2

def fetch_response(agent, messages, callback_handler):
    """
    Function to fetch response without using unhashable objects in caching.
    """
    try:
        time.sleep(RATE_LIMIT_DELAY)  # Throttle requests to avoid rate-limiting
        return agent.run(messages, callbacks=[callback_handler])
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize wrappers and tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = DuckDuckGoSearchRun(name="Search")

def main():
    st.title("🔎 LangChain - Chat with Search")

    """
    In this application, you can chat with an agent that uses tools like DuckDuckGo, Arxiv, and Wikipedia to answer your queries interactively.
    """

    # Sidebar for API key and settings
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi! I'm a chatbot that can search the web. How can I assist you today?"}
        ]

    # Display chat history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if prompt := st.chat_input(placeholder="Ask me anything..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not api_key:
            st.error("Please provide your Groq API Key in the sidebar.")
            return

        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [search_tool, arxiv_tool, wiki_tool]

        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = fetch_response(search_agent, st.session_state["messages"], st_cb)

            if "Error" in response:
                st.warning(response)
            elif "Ratelimit" in response:
                st.error("Rate limit exceeded. Please wait a moment and try again.")
            else:
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.write(response)

if __name__ == "__main__":
    main()
