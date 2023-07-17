import streamlit as st
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pathlib
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import openai
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from vetting_questions import extracted_dict_list
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os

# from streamlit_agent.callbacks.capturing_callback_handler import playback_callbacks
# from streamlit_agent.clear_results import with_clear_container


openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_with_agent(input_text):
    response = agent({"input": input_text})
    return response['output']


class DocumentInput(BaseModel):
    question: str = Field()


llm = ChatOpenAI(temperature=0.5, model="gpt-4")

tools = []
files = [
    {
        "name": "dedoose-terms-of-service",
        "path": "TERMS OF SERVICE.pdf",
    },
]

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"useful when you want to answer questions about {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        )
    )

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
)

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

st.set_page_config(page_title="Vetting Assistant")

st.title("Vetting Assistant")

for question_dict in extracted_dict_list:
    user_input = question_dict['question']
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_input, callbacks=[st_callback])
        st.write(response)

# for question in extracted_dict_list:
#     input_text = question['question']
#     response = chat_with_agent(input_text)
#     print(f"Question: {input_text}")
#     print(f"Response: {response}")
#     print()
