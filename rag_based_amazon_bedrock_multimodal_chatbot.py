import os
import streamlit as st
import boto3
import json
import logging
from PIL import Image
import pytesseract
from io import BytesIO
from pdfminer.high_level import extract_text
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Configure the page layout to "wide"
st.set_page_config(layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Function to inject local CSS
def local_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Apply custom CSS for word wrapping
local_css('''
.chat-message {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
''')

# Function to extract text from images using pytesseract
def extract_text_from_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return ""

# Function to extract text from PDFs using pdfminer.six
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
        return ""

# Function to process uploaded files (images or PDFs)
def process_uploaded_files(uploaded_file):
    file_type = uploaded_file.type
    file_bytes = uploaded_file.read()
    extracted_text = ""

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
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
    st.write("Upload an image or PDF containing text and ask questions about it.")

    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        extracted_text = process_uploaded_files(uploaded_file)

        if not extracted_text:
            return

        # Initialize or retrieve session state variables
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Split the text into documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.create_documents([extracted_text])

        # Initialize Bedrock embeddings
        bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1", client=bedrock
        )

        # Create vector store
        vectorstore = FAISS.from_documents(
            docs,
            bedrock_embeddings
        )

        # Define the prompt template
        prompt_template = """
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Initialize the LLM
        llm = Bedrock(
            model_id="amazon.titan-text-express-v1",
            client=bedrock,
            model_kwargs={
                'maxTokenCount': 512,
                'temperature': 0.5,
                'topP': 0.9
            }
        )

        with st.form(key='question_form', clear_on_submit=True):
            st.write("You can now ask questions about the extracted text.")
            user_input = st.text_input("Your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            with st.spinner('Generating response...'):
                try:
                    answer = qa({"query": user_input})
                    response = answer['result']
                    st.session_state.chat_history.append({"question": user_input, "answer": response})
                except Exception as e:
                    st.error(f"Failed to generate a response: {str(e)}")

        if st.session_state.get('chat_history'):
            st.write("### Chat History")
            for index, chat in enumerate(st.session_state.chat_history):
                st.write(f"**You:** {chat['question']}")
                # Use st.markdown with custom CSS class for the bot's response
                st.markdown(f'<div class="chat-message"><strong>Bot:</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload an image or PDF to get started.")

if __name__ == "__main__":
    main()
