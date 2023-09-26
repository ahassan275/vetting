import streamlit as st
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import openai
from langchain.utilities import SerpAPIWrapper
import os
import re
from pydantic import BaseModel, Field

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

web = DuckDuckGoSearchAPIWrapper()

# SERP_API_KEY = os.environ["SERPAPI_API_KEY"]
#
# search = SerpAPIWrapper()

st.set_page_config(page_title="Vetting Assistant Chatbot")

# Chat with agent function
def chat_with_agent(input_text):
    response = agent.run(input_text)
    return response['output']

class DocumentInput(BaseModel):
    question: str = Field()

# Document Setup
@st.cache_data
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    return retriever


# Agent Initialization
llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-16k")
tools = [
    Tool(
        name="Document Retriever",
        args_schema=DocumentInput,
        description="Retrieve answers from the uploaded document",
        func=RetrievalQA.from_chain_type(llm=llm, retriever=load_document("TERMS OF SERVICE.pdf"))
    ),
    Tool(
        name="Web Search",
        description="Search the web for answers to user questions",
        func=web.run
    )
]
agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, verbose=True)

# Streamlit UI Setup
st.title("Vetting Assistant Chatbot")
st.write("Ask any question related to the vetting process:")

user_input = st.text_input("Your Question:")

if user_input:
    try:
        response = chat_with_agent(user_input)
        st.write(f"Answer: {response}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write(
    "Note: The chatbot retrieves answers from the uploaded document and can also search the web for relevant information.")
