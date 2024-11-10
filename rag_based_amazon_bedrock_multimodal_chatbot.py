import os
import time
import hashlib
import logging
import boto3
import botocore.exceptions
import streamlit as st
import numpy as np
from io import BytesIO
from tempfile import NamedTemporaryFile
from PIL import Image
import easyocr
from pdfminer.high_level import extract_text
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from functools import wraps

# Updated import statements
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM

# Set up Streamlit and logging configurations
st.set_page_config(layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS Bedrock client
try:
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
except Exception as e:
    logger.error(f"Error initializing Bedrock client: {str(e)}")
    st.error("Failed to initialize Bedrock client. Please check your AWS credentials and network connection.")
    st.stop()

# Initialize EasyOCR reader and cache it
@st.cache_resource
def initialize_reader():
    try:
        return easyocr.Reader(['en'])
    except Exception as e:
        logger.error(f"Error initializing EasyOCR Reader: {str(e)}")
        st.error("Failed to initialize EasyOCR Reader.")
        st.stop()

reader = initialize_reader()

# CSS for better text wrapping
def local_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

local_css('''
.chat-message {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
''')

# Functions to extract text from images and PDFs
def extract_text_from_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        results = reader.readtext(image_np)
        extracted_text = "\n".join([result[1] for result in results])
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        st.error("An error occurred while extracting text from the image.")
        return ""

def extract_text_from_pdf(pdf_bytes):
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            text = extract_text(temp_pdf.name)
        os.remove(temp_pdf.name)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        st.error("An error occurred while extracting text from the PDF.")
        return ""

def process_uploaded_files(uploaded_file):
    file_type = uploaded_file.type
    file_bytes = uploaded_file.read()
    extracted_text = ""

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        st.session_state['uploaded_file'] = uploaded_file  # Store in session state
        st.image(uploaded_file, caption='Uploaded Image.', width=400)  # Set width to 400 pixels
        st.write("Processing the image...")
        with st.spinner('Extracting text from the image...'):
            extracted_text = extract_text_from_image(file_bytes)
    elif file_type == "application/pdf":
        st.write("Uploaded PDF.")
        st.write("Processing the PDF...")
        with st.spinner('Extracting text from the PDF...'):
            extracted_text = extract_text_from_pdf(file_bytes)
    else:
        st.error("Unsupported file type.")
        return ""

    if not extracted_text.strip():
        st.error("No text found in the file.")
        return ""

    st.success("Text extracted from the file.")
    return extracted_text

# Caching Bedrock Embeddings
@st.cache_resource
def get_bedrock_embeddings():
    try:
        bedrock_embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v1"
        )
        return bedrock_embeddings
    except Exception as e:
        logger.error(f"Error initializing Bedrock Embeddings: {str(e)}")
        st.error("Failed to initialize Bedrock Embeddings.")
        st.stop()

bedrock_embeddings = get_bedrock_embeddings()

# Caching Bedrock LLM
@st.cache_resource
def get_bedrock_llm():
    try:
        llm = BedrockLLM(
            client=bedrock_client,
            model_id="amazon.titan-tg1-large",  # Updated model ID
            model_kwargs={
                'maxTokenCount': 1024,
                'temperature': 0.7,
                'topP': 0.9
            }
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing Bedrock LLM: {str(e)}")
        st.error("Failed to initialize Bedrock LLM.")
        st.stop()

# Adjusted rate limit values
CALLS = 1  # Maximum number of calls per period
PERIOD = 1  # Time period in seconds

def rate_limiter(max_calls, period):
    min_interval = period / max_calls
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            if elapsed < min_interval:
                time_to_wait = min_interval - elapsed
                logger.info(f"Rate limiter active. Sleeping for {time_to_wait} seconds.")
                time.sleep(time_to_wait)
            last_time_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(botocore.exceptions.ClientError)
)
@rate_limiter(max_calls=CALLS, period=PERIOD)
def invoke_bedrock_model_with_retry(qa_chain, user_query):
    st.session_state['llm_requests'] += 1
    logger.info(f"LLM requests made: {st.session_state['llm_requests']}")
    return qa_chain.invoke({"query": user_query})

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(botocore.exceptions.ClientError)
)
@rate_limiter(max_calls=CALLS, period=PERIOD)
def invoke_llm_with_retry(llm, user_query):
    st.session_state['llm_requests'] += 1
    logger.info(f"LLM requests made: {st.session_state['llm_requests']}")
    return llm.invoke(user_query)

def main():
    st.title("Multimodal Chatbot with Image and PDF Support")
    st.write("Upload an image or PDF containing text and ask questions about it or any general knowledge question.")
    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

    # Initialize request counters
    if 'embedding_requests' not in st.session_state:
        st.session_state['embedding_requests'] = 0
    if 'llm_requests' not in st.session_state:
        st.session_state['llm_requests'] = 0

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display the uploaded image if it exists in session state
    if 'uploaded_file' in st.session_state and st.session_state['uploaded_file'] is not None:
        st.image(
            st.session_state['uploaded_file'],
            caption='Uploaded Image.',
            width=400  # Set width to 400 pixels
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if 'uploaded_file_hash' not in st.session_state or file_hash != st.session_state['uploaded_file_hash']:
            st.session_state['uploaded_file_hash'] = file_hash
            extracted_text = process_uploaded_files(uploaded_file)
            if not extracted_text:
                return

            st.session_state['extracted_text'] = extracted_text

            # Adjusted text splitter parameters to reduce the number of chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Increased chunk size
                chunk_overlap=100  # Decreased overlap
            )
            docs = text_splitter.create_documents([extracted_text])

            try:
                # Create vectorstore using from_documents
                vectorstore = FAISS.from_documents(
                    documents=docs,
                    embedding=bedrock_embeddings
                )
                st.session_state['vectorstore'] = vectorstore
            except botocore.exceptions.ClientError as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                st.error("Failed to generate embeddings due to service throttling. Please try again later.")
                return
            except Exception as e:
                logger.error(f"Unexpected error during embeddings generation: {str(e)}")
                st.error("An unexpected error occurred during embeddings generation. Please try again.")
                return

            # Adjust retriever to improve relevance
            retriever = st.session_state['vectorstore'].as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
            st.session_state['retriever'] = retriever

            # Modified prompt template
            prompt_template = """
You are an AI assistant that provides concise and accurate answers to user questions.
Use the information provided in the context and your general knowledge to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:
"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            st.session_state['prompt'] = PROMPT

            # Get cached LLM
            st.session_state['llm'] = get_bedrock_llm()

            try:
                qa = RetrievalQA.from_chain_type(
                    llm=st.session_state['llm'],
                    chain_type="stuff",
                    retriever=st.session_state['retriever'],
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": st.session_state['prompt']}
                )
                st.session_state['qa'] = qa
            except botocore.exceptions.ClientError as e:
                logger.error(f"Bedrock ClientError during QA chain initialization: {str(e)}")
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException':
                    st.error("Service is currently busy. Please wait a moment and try again.")
                elif error_code == 'ValidationException':
                    st.error("Invalid model identifier provided in QA chain. Please verify the model name and try again.")
                else:
                    st.error("An error occurred while setting up the QA system. Please try again later.")
                return
            except Exception as e:
                logger.error(f"Unexpected error during QA chain initialization: {str(e)}")
                st.error("An unexpected error occurred while setting up the QA system. Please try again.")
                return
        else:
            extracted_text = st.session_state.get('extracted_text', '')
            qa = st.session_state.get('qa', None)
            if not qa:
                st.error("QA system is not initialized.")
                return

    # Initialize last request time for debouncing
    if 'last_request_time' not in st.session_state:
        st.session_state['last_request_time'] = 0

    REQUEST_COOLDOWN = 5  # seconds

    if 'qa' in st.session_state:
        with st.form(key='question_form', clear_on_submit=True):
            st.write("You can now ask questions about the extracted text or any general knowledge question.")
            user_input = st.text_input("Please ask a question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            current_time = time.time()
            if current_time - st.session_state.get('last_request_time', 0) < REQUEST_COOLDOWN:
                st.warning(f"Please wait {int(REQUEST_COOLDOWN - (current_time - st.session_state.get('last_request_time', 0)))} seconds before submitting another request.")
            else:
                st.session_state['last_request_time'] = current_time
                with st.spinner('Generating response...'):
                    try:
                        # Invoke Bedrock model with rate limiting and retry
                        response = invoke_bedrock_model_with_retry(st.session_state['qa'], user_input)
                        answer = response['result']
                        # Check if the answer indicates the model doesn't know
                        if "I don't know" in answer or "cannot find sufficient information" in answer.lower():
                            # Use LLM directly without context
                            answer = invoke_llm_with_retry(st.session_state['llm'], user_input)
                        st.session_state.chat_history.append({"question": user_input, "answer": answer})
                    except botocore.exceptions.ClientError as e:
                        error_code = e.response['Error']['Code']
                        if error_code == 'ThrottlingException':
                            st.error("Service is currently busy due to high demand. Please wait a moment and try again.")
                        elif error_code == 'ValidationException':
                            st.error("Invalid model identifier provided. Please verify the model name and try again.")
                        else:
                            st.error("Failed to generate a response. Please try again later.")
                        logger.error(f"Bedrock ClientError: {str(e)}")
                    except Exception as e:
                        st.error("An unexpected error occurred. Please try again.")
                        logger.error(f"Unexpected error: {str(e)}")

    if st.session_state.get('chat_history'):
        st.write("### Chat History")
        for index, chat in enumerate(st.session_state.chat_history):
            st.write(f"**You:** {chat['question']}")
            st.markdown(f'<div class="chat-message"><strong>Bot:</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload an image or PDF to get started.")

if __name__ == "__main__":
    main()
