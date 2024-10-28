import os
import streamlit as st
import boto3
import logging
from PIL import Image
import easyocr
import numpy as np
from io import BytesIO
from pdfminer.high_level import extract_text
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile
import hashlib

# Set up Streamlit and logging configurations
st.set_page_config(layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Bedrock client
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

def main():
    st.title("Multimodal Chatbot with Image and PDF Support")
    st.write("Upload an image or PDF containing text and ask questions about it or any general knowledge question.")
    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.create_documents([extracted_text])

            try:
                bedrock_embeddings = BedrockEmbeddings(
                    model_id="amazon.titan-embed-text-v1",
                    client=bedrock_client
                )
                st.session_state['bedrock_embeddings'] = bedrock_embeddings
            except Exception as e:
                logger.error(f"Error initializing Bedrock Embeddings: {str(e)}")
                st.error("Failed to initialize Bedrock Embeddings.")
                return

            try:
                vectorstore = FAISS.from_documents(
                    docs,
                    st.session_state['bedrock_embeddings']
                )
                st.session_state['vectorstore'] = vectorstore
            except Exception as e:
                logger.error(f"Error creating FAISS vector store: {str(e)}")
                st.error("Failed to create vector store.")
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

            try:
                llm = BedrockLLM(
                    model_id="amazon.titan-text-express-v1",
                    client=bedrock_client,
                    model_kwargs={
                        'maxTokenCount': 1024,
                        'temperature': 0.7,
                        'topP': 0.9
                    }
                )
                st.session_state['llm'] = llm
            except Exception as e:
                logger.error(f"Error initializing BedrockLLM: {str(e)}")
                st.error("Failed to initialize the language model.")
                return

            try:
                qa = RetrievalQA.from_chain_type(
                    llm=st.session_state['llm'],
                    chain_type="stuff",
                    retriever=st.session_state['retriever'],
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": st.session_state['prompt']}
                )
                st.session_state['qa'] = qa
            except Exception as e:
                logger.error(f"Error initializing RetrievalQA chain: {str(e)}")
                st.error("An error occurred while setting up the QA system.")
                return
        else:
            extracted_text = st.session_state.get('extracted_text', '')
            qa = st.session_state.get('qa', None)
            if not qa:
                st.error("QA system is not initialized.")
                return

    if 'qa' in st.session_state:
        with st.form(key='question_form', clear_on_submit=True):
            st.write("You can now ask questions about the extracted text or any general knowledge question.")
            user_input = st.text_input("Please ask a question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                try:
                    response = st.session_state['qa'].invoke({"query": user_input})
                    answer = response['result']
                    # Check if the answer indicates the model doesn't know
                    if "I don't know" in answer or "cannot find sufficient information" in answer.lower():
                        # Use LLM directly without context
                        answer = st.session_state['llm'](user_input)
                    st.session_state.chat_history.append({"question": user_input, "answer": answer})
                except Exception as e:
                    logger.error(f"Failed to generate a response: {str(e)}")
                    st.error("Failed to generate a response. Please try again.")

        if st.session_state.get('chat_history'):
            st.write("### Chat History")
            for index, chat in enumerate(st.session_state.chat_history):
                st.write(f"**You:** {chat['question']}")
                st.markdown(f'<div class="chat-message"><strong>Bot:</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload an image or PDF to get started.")

if __name__ == "__main__":
    main()
