import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import format_document
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
import streamlit as st
import tempfile
import getpass
import base64
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import getpass
import os
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentType, initialize_agent, Tool
from vetting_questions import extracted_questions
from langchain.schema import SystemMessage
import base64
from docx import Document
from langchain.chains import RetrievalQA
import requests
from langchain.text_splitter import CharacterTextSplitter


# Streamlit UI setup for multi-page application
st.set_page_config(page_title="RAG Demonstration APP", layout="wide", initial_sidebar_state="expanded")


# Set your OpenAI API key here
openai.api_key = os.environ["OPENAI_API_KEY"]
# Set OpenAI API key
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

# # Prompt for the API key
# api_key = getpass.getpass("Enter your Tavily API Key: ")

# # Set the environment variable
# os.environ["TAVILY_API_KEY"] = api_key

# # Now you can use the environment variable in your application
# print("API Key set successfully!")

api_key = st.secrets["TAVILY_API_KEY"]

# Setting the environment variable
os.environ["TAVILY_API_KEY"] = api_key


# Define functions for document processing
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200, add_start_index=True):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
    return text_splitter.split_documents(docs)

def embed_and_index(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import tempfile


# Function to read file content as string
def get_file_content_as_string(file_path):
    with open(file_path, 'rb') as f:
        binary_file_data = f.read()
    return base64.b64encode(binary_file_data).decode('utf-8')

# model: str = "text-embedding-ada-002"

def create_download_link(file_path, file_name):
    file_content = get_file_content_as_string(file_path)
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{file_content}" download="{file_name}">Download the responses</a>'
    return href

def google_search(query):
    try:
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query
        }
        response = requests.get(endpoint, params=params)
        results = response.json().get("items", [])
        return [result["link"] for result in results]
    except Exception as e:
        st.error(f"Error during web search: {e}")
        return []


import tempfile

# Function to process the uploaded document
@st.cache_resource
def process_document(file_paths):
    all_docs = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

        pages = loader.load_and_split()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = splitter.split_documents(pages)
        all_docs.extend(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    retriever = FAISS.from_documents(docs, embeddings)
    return retriever

@st.cache_resource
def process_documents(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    return retriever

def handle_uploaded_files(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

# Function to handle the uploaded files and return paths
def handle_uploaded_file(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            file_paths.append(tmp.name)
    return file_paths

def resume_cover_letter_page():
    st.subheader("Document Generator")
# Prompt Template
    prompt_template = """
    {chat_history}
    As a Multifaceted Writer, your task is to create a document that reflects the requirements and nuances of the provided information. Utilize the retrieved documents {context} to inform the substance, style, and tone of your output. Your creation should directly address the user query {message}  and incorporate any specific instructions or additional details {additional_context} provided.

    - Context: {context}
    (This section contains documents related to your query, offering background and examples relevant to your task.)

    - Message: {message}
    (Your query or the main topic around which the document should be centered.)

    - Additional Context: {additional_context}
    (Any specific instructions, details, or extra information that should be considered in the document creation process.)

    Based on these inputs, your goal is to synthesize the information and generate a document that is both informative and tailored to the specific requirements of the task. Ensure that your output is aligned with the themes and specifics mentioned in the context and message, while also adhering to any guidelines or directives provided in the additional context.

    """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["chat_history","context", "message", "additional_context"]
    )

    # Class for Prompt Input
    class PromptInput(BaseModel):
        context: str
        message: str
        additional_context: str
        chat_history: str

    memory = ConversationBufferMemory(memory_key="chat_history")

    # Initialize the LLM and LLMChain 
    llm = ChatOpenAI(model_name='gpt-4')
    chain = LLMChain(llm=llm, prompt=PROMPT, verbose=True, memory=memory)

    # Streamlit UI setup
    st.title("VectorDB Document Generator")

    # Upload PDF and process it
    uploaded_file = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_file:
        file_path = handle_uploaded_file(uploaded_file)
        retriever = process_document(file_path)
        st.success("Document processed. Please enter additional context and custom instructions.")

        # Input fields for job description and custom instructions
        message = st.text_area("Additional context", "Enter information here...")
        additional_context = st.text_area("Custom instructions", "Enter any specific instructions or additional context here...")

        if st.button("Generate Document"):
            with st.spinner('Generating your document...'):
                docs = retriever.similarity_search(query=message, k=3)
                inputs = [{"context": doc.page_content, "message": message, "additional_context": additional_context, "chat_history": memory} for doc in docs]
                results = chain.apply(inputs)
                text_results = []
                for d in results:
                    text = d["text"]
                    formatted_text = f"{text}"
                    if formatted_text not in text_results:
                        text_results.append(formatted_text)

                st.write("Generated Document", " ".join(text_results))
    else:
        st.write("Please upload a Document to start.")


def document_search_retrieval_page():
    st.subheader("Document Search and Retrieval")

    uploaded_file = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_file:
        file_path = handle_uploaded_file(uploaded_file)
        vectorstore = process_document(file_path)
        # file_path = handle_uploaded_file(uploaded_file)
        # docs = load_documents(file_path)
        # all_splits = split_documents(docs)
        # vectorstore = embed_and_index(all_splits)
        pdf_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # Create retriever tool with the vectorstore
        pdf_retriever_tool = create_retriever_tool(
            pdf_retriever,
            "document_question_and_answer_and_generation",
            "Useful for when you need to answer questions about the uploaeded document. Input should be a search query or a given action for information retrieval and generation.\
            '"
        )

        memory = ConversationBufferMemory()

        tavily_search_tool = TavilySearchResults()
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        tools = [pdf_retriever_tool, tavily_search_tool]
        agent_prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

        chat_container = st.container()
        with chat_container:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_input = st.text_input("Ask a question about the uploaded document:")

            if st.button('Query Document') and user_input:
                with st.spinner('Processing your question...'):
                    try:
                        response = agent_executor.invoke({"input": user_input})
                        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                        with st.chat_message("assistant"):
                            st.markdown(response["output"])
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

            if st.button('Search Web') and user_input:
                with st.spinner('Searching the web...'):
                    try:
                        tavily_search_result = agent_executor.invoke({
                            "input": user_input
                        })
                        st.session_state.messages.append({"role": "assistant", "content": tavily_search_result["output"]})
                        with st.chat_message("assistant"):
                            st.markdown(tavily_search_result["output"])
                    except Exception as e:
                        st.error(f"An error occurred during web search: {e}")
    else:
        st.write("Please upload a document to start.")

def vetting_assistant_page():
    st.title("Vetting Assistant Chatbot")

    if "uploaded_pdf_path" not in st.session_state or "retriever" not in st.session_state:
        # uploaded_file = st.file_uploader("Upload a PDF containing the terms of service", type=["pdf"])
        uploaded_file = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        if uploaded_file:
            file_path = handle_uploaded_file(uploaded_file)
            st.session_state.uploaded_pdf_path = file_path
            st.session_state.retriever = process_documents(st.session_state.uploaded_pdf_path)
    else:
        st.write("Using previously uploaded PDF. If you want to use a different PDF, please refresh the page.")

    app_name = st.text_input("Enter the name of the app:")

    if "retriever" in st.session_state:
        llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k")
        tools = [
            Tool(
                name="vetting_tool",
                description="Tool for retrieving infomration related to security and privacy",
                func=RetrievalQA.from_llm(llm=llm, retriever=st.session_state.retriever, return_source_documents=True)
            )
        ]
        # tool = create_retriever_tool(
        #     st.session_state.retriever,
        #     "search_terms_service",
        #     "Searches and returns an application's privacy and data policies and terms of use.",
        # )
        # tools = [tool]
        agent_kwargs = {
            "system_message": SystemMessage(content="You are an intelligent Vetting Assistant, "
                                                    "expertly designed to analyze and extract key "
                                                    "information from terms of service documents. "
                                                    "Your goal is to assist users in understanding "
                                                    "complex legal documents and provide clear, "
                                                    "concise answers to their queries. Use only the retrieved data from the pdf when responding to questions."
                                                    "Include citations including section and page number with your response")
        }
        agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, agent_kwargs=agent_kwargs,
                                 verbose=True)
        # agent = create_conversational_retrieval_agent(llm, tools)

        st.write("Ask any question related to the vetting process:")
        query_option = st.selectbox("Choose a predefined query:", extracted_questions)
        user_input = st.text_input("Your Question:", value=query_option)

        if st.button('Start Vetting') and user_input:
            with st.spinner('Processing your question...'):
                try:
                    response = agent.run(user_input)
                    # response = agent({"input": "user_input"})
                    st.write(f"Answer: {response}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        st.write("Note: The chatbot retrieves answers from the uploaded document.")

        if 'running_queries' not in st.session_state:
            st.session_state.running_queries = False

        placeholder_message = f"{app_name} is being vetted for compliance and its policies provided in context. Does {app_name} meet this criteria?"
        all_queries = [f"{question} {placeholder_message}" for question in extracted_questions]

        if st.button('Run All Queries'):
            with st.spinner('Processing all queries...'):
                st.session_state.running_queries = True
                doc = Document()
                doc.add_heading('Vetting Assistant Responses', 0)

                for question in all_queries:
                    if not st.session_state.running_queries:
                        break
                    try:
                        response = agent.run(question)
                        doc.add_heading('Q:', level=1)
                        doc.add_paragraph(question)
                        doc.add_heading('A:', level=1)
                        doc.add_paragraph(response)
                    except Exception as e:
                        doc.add_paragraph(f"Error for question '{question}': {e}")

                doc_path = "vetting_responses.docx"
                doc.save(doc_path)
                st.markdown(create_download_link(doc_path, "vetting_responses.docx"), unsafe_allow_html=True)

        if st.button('Stop Queries'):
            st.session_state.running_queries = False

        if st.button('Search Web'):
            with st.spinner('Searching the web...'):
                links = google_search(user_input)
                st.write("Top search results:")
                for link in links:
                    st.write(link)

# Streamlit UI setup for multi-page application
st.title("Document Processing and Retrieval Application")
page = st.sidebar.selectbox("Choose a Tool:", ["VectorDB Document Generator", "Document Search and Retrieval", "Vetting Assistant"])

# Load respective pages
if page == "VectorDB Document Generator":
    resume_cover_letter_page()
elif page == "Document Search and Retrieval":
    document_search_retrieval_page()
elif page == "Vetting Assistant":
    vetting_assistant_page()






