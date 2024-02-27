from rag_qa import load_documents, split_documents, embed_and_index
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import openai
from langchain_community.chat_models import ChatOpenAI
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import format_document
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnableLambda


# Initialize the RAG components
docs = load_documents("TERMS OF SERVICE.pdf")
all_splits = split_documents(docs)
vectorstore = embed_and_index(all_splits)
retriever = vectorstore.as_retriever()

# # # Create a new prompt template for the RAG chain
template = """
Craft a cover letter that synthesizes the key qualifications and experiences from a provided resume with the requirements and responsibilities outlined in a job description. 

### Instructions:
1. Analyze the job description to identify essential qualifications, responsibilities, and preferred attributes.
2. Review the resume to extract relevant skills, experiences, and achievements that match the job description.
3. Compose a cover letter that effectively showcases how the candidate's background aligns with the job requirements.
4. Ensure the cover letter is concise, free from repetition, and clear of ambiguity.
5. Reflect the candidate's unique tone and voice, while maintaining an elegant and professional style.
6. The cover letter can be multiple pages, you must ensure all context and instructions are well incorporated in the cover letter.
7. Do not fabricate any roles or experiences; use only the information provided in the context.
8. The final product should read as a cohesive narrative that tells the story of the candidate's professional journey in relation to the target position.


### Desired Outcome:
A cover letter tailored to the job description, highlighting the candidate's most pertinent skills and experiences, and articulating why they are the ideal fit for the position.

{context} 
{message}
{custom_instructions}


REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

prompt = ChatPromptTemplate.from_template(template)

# # Initialize the model
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Define the RAG chain
chain = (
    {"context": itemgetter("message") | retriever,
     "message": RunnablePassthrough(),
     "custom_instructions": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# # Execute the chain with a given question
# result = chain.invoke({
#     "message": "What is dedoose's privacy policy?", 
#     "custom_instructions": "Provide concise information citing relevant sources."
# })

# # Load, split, embed, and index the document to create a retriever tool
# docs = load_documents("TERMS OF SERVICE.pdf")
# all_splits = split_documents(docs)
# vectorstore = embed_and_index(all_splits)
# pdf_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# # Create retriever tool with the vectorstore
# pdf_retriever_tool = create_retriever_tool(
#     pdf_retriever,
#     "search_terms_of_service",
#     "Searches and returns excerpts from the 'TERMS OF SERVICE' document."
# )

# # Create Tavily search tool
# tavily_search_tool = TavilySearchResults()

# # Create tools list to be used by agent
# tools = [pdf_retriever_tool, tavily_search_tool]

# # Retrieve the agent prompt from LangChain hub
# agent_prompt = hub.pull("hwchase17/openai-tools-agent")

# # Initialize the OpenAI Chat model
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Create an OpenAI agent with tools
# agent = create_openai_tools_agent(llm, tools, agent_prompt)

# # Set up AgentExecutor to execute the agent with provided tools
# agent_executor = AgentExecutor(agent=agent, tools=tools)

# # Example call to the agent to invoke Tavily Search
# tavily_search_result = agent_executor.invoke({
#     "input": "What are dedoose's privacy statement"
# })

# # Output results
# print("Tavily Search Result:", tavily_search_result["output"])

# # Example call to the agent to invoke PDF Retriever
# pdf_search_result = agent_executor.invoke({
#     "input": "What is Dedoose's privacy statement?"
# })

# # Output results
# print("PDF Retriever Result:", pdf_search_result["output"])

# Output the result
# print("Generated Answer:", result)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()

result_2 = conversational_qa_chain.invoke(
    {
        "question": "How does dedoose store user data?",
        "chat_history": [],
    }
)

# result_3 = conversational_qa_chain.invoke(
#     {
#         "question": "how does dedoose protect user privacy?",
#         "chat_history": [
#             HumanMessage(content="Who wrote this notebook?"),
#             AIMessage(content="Harrison"),
#         ],
#     }
# )
print(result_2)
# print(result_3)

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# # Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}
# # Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# # Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# # And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs"),
}
# # And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

inputs = {"question": "where does dedoose store user data?"}
result_4 = final_chain.invoke(inputs)
print(result_4)

# # Note that the memory does not save automatically
# # This will be improved in the future
# # For now you need to save it yourself
memory.save_context(inputs, {"answer": result_4["answer"].content})

memory.load_memory_variables({})

inputs = {"question": "is there anywhere else they can store data?"}
result_5 = final_chain.invoke(inputs)
print(result_5)

# # Existing functions from the previous script
# # ...

# # Function to convert a list of Documents into formatted text for the prompt
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Custom RAG prompt template
# custom_prompt_template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Use three sentences maximum and keep the answer as concise as possible. 
# Always say "thanks for asking!" at the end of the answer.
# {context} Question: {question} Helpful Answer:"""

# # Define the chain that generates an answer using the formatted context and question
# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
#     | PromptTemplate.from_template(custom_prompt_template)
#     | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     | StrOutputParser()
# )

# # Define a parallel chain that will return both the retrieved documents and the generated answer
# def setup_rag_chain_with_source(file_path, question):
#     # Load, split, embed, and index documents
#     docs = load_documents(file_path)
#     all_splits = split_documents(docs)
#     vectorstore = embed_and_index(all_splits)
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
  
#     # Create a parallel chain that uses retriever to fetch documents and rag_chain_from_docs to generate the answer
#     rag_chain_with_source = RunnableParallel(
#         {"context": retriever, "question": RunnablePassthrough()}
#     ).assign(answer=rag_chain_from_docs)
  
#     # Invoke the parallel chain with the question
#     result_with_source = rag_chain_with_source.invoke(question)
#     return result_with_source

# # Example file and question
# pdf_file_path = "TERMS OF SERVICE.pdf"  # Replace with the actual PDF file path
# example_question = "What is Dedoose's privacy statement?"  # Replace with the actual question

# # Execute the RAG Chain with source return
# result_with_source = setup_rag_chain_with_source(pdf_file_path, example_question)

# # Output the result with source documents
# for key, value in result_with_source.items():
#     if key == "context":
#         print("Retrieved Documents:")
#         for doc in value:
#             print(doc.page_content)  # Or any other relevant document content
#     else:
#         print(f"{key.capitalize()}: {value}")