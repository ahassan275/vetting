import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import format_document
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Streamlit UI setup for multi-page application
# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="RAG Demonstration APP", layout="wide", initial_sidebar_state="expanded", page_icon="logo.png")


# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # lightmode
# Design hide "made with streamlit" footer menu area
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)



# Set your OpenAI API key here
# openai.api_key = os.environ["OPENAI_API_KEY"]
# Set OpenAI API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
# GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

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
    # st.subheader("Document Generator")
    # Prompt Template
    prompt_template = """
    {chat_history}
    As a proficient content creator, your task is to synthesize information from various sources to produce a tailored output that aligns with the user's needs. Your output should be informed by the background documents provided {context}, address the central topic{message}, and integrate any additional details as specified {additional_context}.

    - Context: {context}
    (Here, you'll find background information and relevant examples that will inform the substance, style, and tone of your output. This material is essential for understanding the nuances of the topic at hand.)

    - Message: {message}
    (This is the core topic or task that your output must address. It defines the primary focus of your content and sets the direction for your synthesis.)

    - Additional Context: {additional_context}
    (Additional details and instructions will be provided here to complement the main context. These should be woven into your output to ensure a comprehensive and nuanced response to the task outlined in the message.)

    Your output should be crafted with the following considerations:

    1. Relevance: Ensure all information is directly related to the topic and user requirements.
    2. Clarity: Articulate your ideas clearly and concisely, avoiding ambiguity.
    3. Structure: Organize your content logically, with a clear introduction, body, and conclusion.
    4. Tone: Match the tone to the context and purpose of the task—whether it's formal, informal, persuasive, descriptive, etc.
    5. Detail: Provide sufficient detail to convey a thorough understanding of the topic.

    Please format your output as follows:

    - If the task is informational or analytical, present your findings in a structured essay or report.
    - If the task is creative, produce content in the form of a narrative, dialogue, or other creative formats.
    - For instructional content, use a step-by-step format with actionable guidance.

    Example:

    If the task is to create an informative article on renewable energy sources:

    Introduction: Begin with an overview of the importance of renewable energy in today's world.
    Body: Discuss various renewable energy sources such as solar, wind, hydro, and geothermal power, including their benefits and challenges.
    Conclusion: Summarize the potential impact of renewable energy on the environment and economy.

    Remember to incorporate any specific examples or case studies provided in the context to enrich your output. If further clarification or additional information is required, use  'chain of thought' before providing an answer to ensure a more reliable and reasoned approach to solving the query
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
    llm = ChatOpenAI()
    chain = LLMChain(llm=llm, prompt=PROMPT, verbose=True, memory=memory)

    # Streamlit UI setup
    st.subheader("VectorDB Document Generator")

    

    # Upload PDF and process it
    uploaded_file = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_file:
        file_path = handle_uploaded_file(uploaded_file)
        retriever = process_document(file_path)
        st.success("Document processed. Please enter additional context and custom instructions.")

        # Input fields for job description and custom instructions
        message = st.text_area("Message", "Input specific instructions, tasks or directions expected within the generatated output")
        additional_context = st.text_area("Custom instructions", "Enter any specific instructions. If providing ")

        input_container = st.container()
        with input_container:
            temperature = st.slider("Adjust chatbot specificity:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            llm.temperature = temperature
        
        input_container = st.container()
        with input_container:
            chat_model = st.selectbox('What model would you live to choose',('gpt-4-0125-preview', 'gpt-3.5-turbo-0125'))
            llm.model_name = chat_model

        if st.button("Generate Document"):
            with st.spinner('Generating your document...'):
                docs = retriever.similarity_search(query=message, k=3, fetch_k=10)
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
            "Useful for when you need to answer questions about the uploaeded document. Input should be a search query or a given action for information retrieval and generation. Please answer thoroghly and thoughfully."
        )

        memory = ConversationBufferMemory()

        tavily_search_tool = TavilySearchResults()
        llm = ChatOpenAI()
        tools = [pdf_retriever_tool, tavily_search_tool]
        agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Generate a comprehensive and informative answer for a given question solely based on the provided web Search Results (URL and Summary). You must only use information from the provided search results. Use an unbiased and journalistic tone. Use this current date and time: Wednesday, December 07, 2022 22:50:56 UTC Combine search results together into a coherent answer. Do not repeat text. If different results refer to different entities with the same name, write separate answers for each entity."),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_openai_tools_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

        input_container = st.container()
        with input_container:
            temperature = st.slider("Adjust chatbot specificity:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            llm.temperature = temperature
        
        input_container = st.container()
        with input_container:
            chat_model = st.selectbox('What model would you like to choose',('gpt-4-0125-preview', 'gpt-3.5-turbo-0125'))
            llm.model_name = chat_model

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
        uploaded_file = st.file_uploader("Upload a PDF containing the terms of service", type=["pdf"])
        # uploaded_file = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        if uploaded_file:
            file_path = handle_uploaded_files(uploaded_file)
            st.session_state.uploaded_pdf_path = file_path
            st.session_state.retriever = process_documents(st.session_state.uploaded_pdf_path)
    else:
        st.write("Using previously uploaded PDF. If you want to use a different PDF, please refresh the page.")

    app_name = st.text_input("Enter the name of the app:")

    if "retriever" in st.session_state:
        llm = ChatOpenAI()
        tools = [
            Tool(
                name="vetting_tool",
                description="Tool for retrieving infomration related to security and privacy. Input should be a search query or a given action for information retrieval and generation.",
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
                                                    "information from terms of service and privacy policy documents. "
                                                    "Your goal is to assist users in understanding "
                                                    "complex legal documents and provide clear, "
                                                    "concise answers to their queries. Use only the retrieved data from the pdf when responding to questions."
                                                    "Include citations including section and page number with your response. ")
        }
        agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, agent_kwargs=agent_kwargs,
                                 verbose=True)
        # agent = create_conversational_retrieval_agent(llm, tools)

        input_container = st.container()
        with input_container:
            temperature = st.slider("Adjust chatbot specificity:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            llm.temperature = temperature
        
        input_container = st.container()
        with input_container:
            chat_model = st.selectbox('What model would you live to choose',('gpt-4-0125-preview', 'gpt-3.5-turbo-0125'))
            llm.model_name = chat_model

        st.write("Ask any question related to the vetting process:")
        query_option = st.selectbox("Choose a predefined query:", extracted_questions)
        user_input = st.text_input("Your Question:")

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
st.image('img/image.png')
st.title(":blue[Document Processing and Retrieval Application]")
st.markdown('Generate engaging documents and chat over your files based on your direct quries - powered by Artificial Intelligence (OpenAI GPT-4 and GPT-3.5) Implemented by PCM')
        # '[stefanrmmr](https://www.linkedin.com/in/stefanrmmr/) - '
        # 'view project source code on '
        # '[GitHub](https://github.com/stefanrmmr/gpt3_email_generator)'
st.write('\n')  # add spacing
page = st.sidebar.selectbox("Choose a Tool:", ["VectorDB Document Generator", "Document Search and Retrieval", "Vetting Assistant"])

# Load respective pages
if page == "VectorDB Document Generator":
    resume_cover_letter_page()
elif page == "Document Search and Retrieval":
    document_search_retrieval_page()
elif page == "Vetting Assistant":
    vetting_assistant_page()






