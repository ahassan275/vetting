from openai import OpenAI
import streamlit as st
import time
from openai import OpenAI
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Function to upload a file and create an assistant
def upload_file_for_assistant(file_path):
    # Uploads a file and returns the file ID
    file = client.files.create(file=open(file_path, "rb"), purpose='assistants')
    return file.id


def setup_assistant(file_path):
    # Upload a file with an "assistants" purpose
    file_id = upload_file_for_assistant(file_path)

    # Create an assistant with the uploaded file
    assistant = client.beta.assistants.create(
        instructions="You are a customer support chatbot. Use your knowledge base to best respond to customer queries.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file_id]
    )
    return assistant

# Function to interact with the user
def interact_with_user(user_input, assistant):
    # Create a thread with the initial user message
    threads = client.beta.threads.create()
    thread_id = threads.id

    # Send user input as a message, referencing the uploaded file
    thread_message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input,
        file_ids=assistant.file_ids  # Reference the uploaded file from the assistant
    )

    # Start a run to process the user input
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant.id,
    )

    # Poll for the run to complete
    run_status = None
    while run_status is None or run_status.status != 'completed':
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        # No need to print the run status in the final version, this is for debugging
        # print(f"Run Status: {run_status.status}")

    # Retrieve the message object after the run is completed
    messages_page = client.beta.threads.messages.list(
        thread_id=thread_id
    )

    # Extract the assistant's response
    assistant_response = None
    for message in messages_page.data:
        if message.role == 'assistant':
            # Assuming message.content is a list containing a MessageContentText object
            for content_piece in message.content:
                if content_piece.type == 'text':
                    assistant_response = content_piece.text.value
                    # Break after finding the assistant's response to avoid duplication
                    break
            if assistant_response:
                # Break the outer loop if we've found the assistant's response
                break

    # Return the assistant's response without printing here
    return assistant_response

# Usage
if __name__ == "__main__":
    file_path = 'TERMS OF SERVICE.pdf'
    assistant = setup_assistant(file_path)
    user_question = "How does dedoose protect user privacy?"
    response = interact_with_user(user_question, assistant)
    # Print the assistant's response only once
    print(f"Assistant's response: {response}")



