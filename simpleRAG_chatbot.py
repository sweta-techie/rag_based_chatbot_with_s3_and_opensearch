import os
import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
import faiss as faiss_lib  # Alias to avoid naming conflicts
import numpy as np
import time
from botocore.exceptions import ClientError
import json

# Utility functions for chat history
CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load chat history from a JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as file:
                history = json.load(file)
            return history
        except json.JSONDecodeError:
            # If file is empty or corrupted
            return []
    else:
        return []

def save_chat_history(history):
    """Save chat history to a JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Initialize Streamlit session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Initialize Bedrock Runtime client
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Initialize Embedding Model with the correct model_id
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",  # Correct model_id for embeddings
    client=bedrock_client
)

# Initialize LLM with the correct model_id
def get_llm():
    try:
        llm = Bedrock(
            model_id="amazon.titan-text-premier-v1:0",  # Correct LLM model_id
            client=bedrock_client,
            model_kwargs={
                'maxTokenCount': 512,
                'temperature': 0.5,
                'topP': 0.9
            }
        )
        return llm
    except ClientError as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

# Define the prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know; don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to process uploaded PDFs
def process_uploaded_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            try:
                # Use NamedTemporaryFile to handle the uploaded file in memory
                with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file.flush()
                    loader = PyPDFLoader(temp_file.name)
                    loaded_docs = loader.load()
                    if not loaded_docs:
                        st.warning(f"No content found in {uploaded_file.name}.")
                    else:
                        st.write(f"Loaded {len(loaded_docs)} pages from {uploaded_file.name}.")
                    documents.extend(loaded_docs)
                os.remove(temp_file.name)  # Clean up the temporary file
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")

    if not documents:
        st.error("No valid documents were loaded from the uploaded PDFs.")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500
        )
        docs = text_splitter.split_documents(documents)
        if not docs:
            st.error("No documents were generated after splitting.")
        else:
            st.write(f"Total of {len(docs)} document chunks after splitting.")
        return docs
    except Exception as e:
        st.error(f"Failed to split documents: {e}")
        return []

# Function to generate embeddings and create FAISS vector store
def get_vector_store(docs):
    texts = [doc.page_content for doc in docs]
    try:
        # Use LangChain's FAISS.from_texts to create the vector store
        vectorstore_faiss = FAISS.from_texts(texts, bedrock_embeddings)
        
        # Save FAISS index
        faiss_lib.write_index(vectorstore_faiss.index, "faiss_index.index")
        st.success("Vector store created and saved locally as 'faiss_index.index'.")
        
        return vectorstore_faiss

    except Exception as e:
        st.error(f"An error occurred while creating the FAISS vector store: {e}")
        st.write("Debugging Information:", str(e))
        return None

# Function to load FAISS vector store
def load_vector_store():
    if os.path.exists("faiss_index.index"):
        try:
            # Load FAISS index using FAISS directly
            index = faiss_lib.read_index("faiss_index.index")
            # Reinitialize BedrockEmbeddings with the correct model_id
            bedrock_embeddings_loaded = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v2:0",  # Ensure consistency
                client=bedrock_client
            )
            vectorstore_faiss = FAISS(embedding_function=bedrock_embeddings_loaded, index=index)
            st.success("Vector store loaded successfully.")
            return vectorstore_faiss
        except Exception as e:
            st.error(f"Failed to load the vector store: {e}")
            return None
    else:
        st.error("FAISS index not found. Please upload and process PDF files first.")
        return None

# Function to get response from LLM
def get_response_llm(llm, vectorstore_faiss, query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        return answer['result']
    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
        st.write("Detailed Bedrock error: ", str(e))
        return None

# Callback function to handle question submission
def handle_submit():
    user_question = st.session_state.user_question.strip()
    if not user_question:
        st.error("Please enter a question.")
        return

    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Save updated chat history
    save_chat_history(st.session_state.chat_history)

    if st.session_state.vectorstore_faiss is None:
        # Attempt to load the FAISS index if not already created in this session
        if os.path.exists("faiss_index.index"):
            with st.spinner("Loading vector store..."):
                vectorstore_faiss = load_vector_store()
                if vectorstore_faiss is None:
                    return
                st.session_state.vectorstore_faiss = vectorstore_faiss
        else:
            st.error("FAISS index not found. Please upload and process PDF files first.")
            return

    with st.spinner("Processing your question..."):
        llm = get_llm()
        if llm is None:
            st.error("LLM initialization failed. Cannot process the question.")
            return

        response = get_response_llm(llm, st.session_state.vectorstore_faiss, user_question)
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.write(response)
            # Save updated chat history
            save_chat_history(st.session_state.chat_history)
            # Clear the input field by resetting the session state
            st.session_state.user_question = ""

# Function to clear chat history
def clear_chat_history():
    """Clear the chat history from session state and delete the history file."""
    # Clear the session state chat history
    st.session_state.chat_history = []
    
    # Delete the chat history file if it exists
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            os.remove(CHAT_HISTORY_FILE)
            st.sidebar.success("Chat history has been cleared.")
        except Exception as e:
            st.sidebar.error(f"Failed to delete chat history file: {e}")
    else:
        st.sidebar.info("No chat history to clear.")

def main():
    st.header("Upload and Process PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing uploaded PDFs..."):
            docs = process_uploaded_pdfs(uploaded_files)
            if docs:
                st.success(f"Processed {len(docs)} document chunks from uploaded PDFs.")

        if docs:
            with st.spinner("Generating embeddings and creating vector store..."):
                vectorstore_faiss = get_vector_store(docs)
                if vectorstore_faiss:
                    st.success("Vector store created from uploaded PDFs!")
                    # Store the vectorstore in session_state for later use
                    st.session_state.vectorstore_faiss = vectorstore_faiss

    st.markdown("---")
    st.header("Ask a Question")

    # Create a form for question submission
    with st.form(key='question_form'):
        user_question = st.text_input("Ask a Question from the PDF Files", key='user_question')
        submit_button = st.form_submit_button(label='Send', on_click=handle_submit)

    # Display chat history and the "Clear Chat History" button in the sidebar
    st.sidebar.header("Chat History")

    # "Clear Chat History" button
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history()

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.sidebar.markdown(f"**You:** {chat['content']}")
            elif chat["role"] == "assistant":
                st.sidebar.markdown(f"**Bot:** {chat['content']}")
    else:
        st.sidebar.write("No conversation yet.")

if __name__ == "__main__":
    main()
