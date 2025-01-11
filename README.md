# GenAi
This repository contains a collection of projects and experiments in Generative AI, showcasing the potential of machine learning models to create, transform, and innovate. These projects highlight cutting-edge techniques and real-world applications of generative models.


Text Summarization App with LangChain and Groq
This repository features a Streamlit-based web application for summarizing text content from YouTube videos or web pages. The app leverages LangChain, Groq API, and yt_dlp to process and summarize input content efficiently.

Features
YouTube Summarization: Extracts and summarizes video descriptions from YouTube URLs.
Web Page Summarization: Summarizes text content from any web page URL.
Custom Language Model: Uses the Gemma-7b-It model via Groq API for text processing and summarization.
User-Friendly Interface: A Streamlit-based app that allows users to input URLs, fetch data, and display summaries seamlessly.
Key Technologies
LangChain: For building the summarization pipeline.
Groq API: For language model inference using Gemma-7b-It.
yt_dlp: To fetch video metadata and descriptions from YouTube.
Streamlit: For building an interactive and easy-to-use web interface.
UnstructuredURLLoader: To scrape and process content from websites.
How It Works
Input: Enter a YouTube video URL or website URL in the app interface.
Validation: URLs are validated for correctness and type (YouTube or website).
Content Extraction:
For YouTube: Fetches the title and description using yt_dlp.
For Websites: Scrapes the content using UnstructuredURLLoader.
Summarization: The extracted content is summarized into 300 words using a custom prompt and the Gemma-7b-It model via LangChain.
Output: Displays the generated summary in the app interface.
Requirements
Python 3.8+
Required Python Libraries: streamlit, langchain, langchain_groq, validators, yt_dlp, langchain_community
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-repo/text-summarization-app.git  
cd text-summarization-app  
Install dependencies:
bash
Copy code
pip install -r requirements.txt  
Run the Streamlit app:
bash
Copy code
streamlit run app.py  
Provide the Groq API key and a valid YouTube or website URL to get a summarized output.
Contribution
Contributions are welcome! Feel free to open issues, submit pull requests, or suggest enhancements to make this app better.

License
This project is licensed under the MIT License.

