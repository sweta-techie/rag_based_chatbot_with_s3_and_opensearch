# rag_generate_ques_semantic_chunking_opensearch_corrected_aws.py
from opensearchpy import OpenSearch, RequestsHttpConnection
import os
import re
import requests
import json
from typing import List
from tempfile import NamedTemporaryFile
import time
import uuid

import streamlit as st
import fitz  # PyMuPDF for PDF processing

import boto3
from botocore.exceptions import BotoCoreError, ClientError


from requests_aws4auth import AWS4Auth

from dotenv import load_dotenv
import secrets

load_dotenv()

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="RAG PDF Q&A with Amazon Bedrock and OpenSearch", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) PDF Q&A Application with Amazon Bedrock and OpenSearch")

# -----------------------------
# AWS Configuration
# -----------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")  # e.g., "ai21/j2-jumbo-instruct"
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")  # e.g., 'search-rag-opensearch-domain.region.es.amazonaws.com'
OPENSEARCH_PORT = 443  # Default port for HTTPS

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME, BEDROCK_MODEL_ID, OPENSEARCH_ENDPOINT]):
    st.error("Missing AWS configuration. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, "
             "S3_BUCKET_NAME, BEDROCK_MODEL_ID, and OPENSEARCH_ENDPOINT environment variables.")
    st.stop()

# Initialize AWS clients
awsauth = AWS4Auth(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, 'es')

opensearch_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': OPENSEARCH_PORT}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

bedrock_client = boto3.client(
    'bedrock',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# -----------------------------
# Utility Functions
# -----------------------------

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing unwanted artifacts.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def download_pdf_from_url(url: str) -> NamedTemporaryFile:
    """
    Downloads a PDF from the given URL and saves it to a temporary file.

    Args:
        url (str): The URL pointing to the PDF.

    Returns:
        NamedTemporaryFile: The temporary file containing the downloaded PDF.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        temp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_pdf.write(chunk)
        temp_pdf.flush()
        return temp_pdf
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download PDF from URL: {e}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        cleaned_text = clean_text(text)
        return cleaned_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def bedrock_generate(prompt: str, model_id: str = BEDROCK_MODEL_ID) -> str:
    """
    Generates a response from Amazon Bedrock LLM.

    Args:
        prompt (str): The prompt to send to the model.
        model_id (str): The identifier for the Bedrock model.

    Returns:
        str: The generated text.
    """
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            })
        )
        response_body = response['body'].read().decode('utf-8')
        response_json = json.loads(response_body)
        generated_text = response_json.get('generated_text', '').strip()
        return generated_text
    except (BotoCoreError, ClientError) as error:
        st.error(f"Error interacting with Bedrock: {error}")
        return ""

def summarize_text(text: str) -> str:
    """
    Summarizes the given text using Amazon Bedrock.

    Args:
        text (str): The text to summarize.

    Returns:
        str: Summarized text.
    """
    try:
        prompt = (
            "You are a helpful assistant that summarizes academic documents.\n\n"
            f"Summarize the following text in a concise manner:\n\n{text}"
        )
        summary = bedrock_generate(prompt)
        return summary if summary else text
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return text

def extract_topics(text: str) -> List[str]:
    """
    Extracts key topics from the text using Amazon Bedrock.

    Args:
        text (str): The text from which to extract topics.

    Returns:
        List[str]: List of extracted topics.
    """
    try:
        prompt = (
            "You are an expert in extracting key topics from academic texts.\n\n"
            f"Extract the top 5 key topics from the following text:\n\n{text}"
        )
        topics_text = bedrock_generate(prompt)
        topics = re.split(r',|\n|\d+\.', topics_text)
        topics = [topic.strip().strip('-').strip() for topic in topics if topic.strip()]
        return topics[:5]
    except Exception as e:
        st.error(f"Error during topic extraction: {e}")
        return []

def generate_questions_batch(text: str, topics: List[str], num_questions: int) -> List[str]:
    """
    Generates multiple questions in a single API call based on provided topics.

    Args:
        text (str): The text to base questions on.
        topics (List[str]): List of topics to generate questions about.
        num_questions (int): Total number of questions to generate.

    Returns:
        List[str]: List of generated questions.
    """
    try:
        prompt = (
            f"You are an expert educator tasked with creating {num_questions} insightful and thought-provoking questions "
            f"to test comprehension of the following academic material. Each question should relate to one of the provided topics "
            f"and require an in-depth understanding of the material to answer.\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Topics:\n" + "\n".join([f"- {topic}" for topic in topics]) + "\n\n"
            f"Questions:\n1."
        )
        questions_text = bedrock_generate(prompt)
        questions = re.split(r'\n\d+\.', questions_text)
        questions = [q.strip().lstrip('. ').rstrip('?') + '?' for q in questions if q.strip()]
        return questions[:num_questions]
    except Exception as e:
        st.error(f"Error during question generation: {e}")
        return []

def search_documents_opensearch(query: str, index_name: str = 'rag-documents', size: int = 5) -> List[str]:
    """
    Searches for relevant documents in OpenSearch based on the query.

    Args:
        query (str): The search query.
        index_name (str): The name of the OpenSearch index.
        size (int): Number of documents to retrieve.

    Returns:
        List[str]: List of relevant document texts.
    """
    try:
        response = opensearch_client.search(
            body={
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                            "fuzziness": "AUTO"
                        }
                    }
                }
            },
            index=index_name,
            size=size
        )
        hits = response['hits']['hits']
        documents = [hit['_source']['text'] for hit in hits]
        return documents
    except Exception as e:
        st.error(f"Error searching documents in OpenSearch: {e}")
        return []

def fetch_answer(question: str, index_name: str, max_retrieved: int = 5) -> str:
    """
    Fetches an answer to the question using retrieved documents from OpenSearch.

    Args:
        question (str): The question to answer.
        index_name (str): The OpenSearch index name.
        max_retrieved (int): Number of documents to retrieve.

    Returns:
        str: The generated answer.
    """
    try:
        retrieved_docs = search_documents_opensearch(question, index_name, max_retrieved)
        context = "\n\n".join(retrieved_docs)
        prompt = (
            f"You are an expert in the following context.\n\n"
            f"Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        answer = bedrock_generate(prompt)
        return answer if answer else "An error occurred while generating the answer."
    except Exception as e:
        st.error(f"Error during answer fetching: {e}")
        return "An error occurred while generating the answer."

def fetch_answer_with_backoff(question: str, index_name: str, max_retries: int = 3) -> str:
    """
    Fetches an answer with exponential backoff in case of rate limits using OpenSearch retrieval.

    Args:
        question (str): The question to answer.
        index_name (str): The OpenSearch index name.
        max_retries (int): Maximum number of retries.

    Returns:
        str: The generated answer or error message.
    """
    for attempt in range(max_retries):
        try:
            return fetch_answer(question, index_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                wait_time = (2 ** attempt) + secrets.SystemRandom().uniform(0, 1)
                st.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Error during answer fetching: {e}")
                break
    st.error("Max retries exceeded. Unable to fetch answer at this time.")
    return "An error occurred while generating the answer."

def split_text_into_chunks(text: str, max_tokens: int) -> List[str]:
    """
    Splits the text into chunks that do not exceed the max_tokens limit.

    Args:
        text (str): The text to split.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = 0
        current_chunk += " " + sentence
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in the given text.
    This is a rough estimation using the ratio of characters to tokens.

    Args:
        text (str): The text to estimate.

    Returns:
        int: Estimated number of tokens.
    """
    # Average English word is ~4 characters, 1 token ~= 4 characters
    return len(text) // 4

def upload_pdf_to_s3(file, bucket_name: str, object_name: str = None) -> bool:
    """
    Uploads a PDF file to Amazon S3.

    Args:
        file: The file object to upload.
        bucket_name (str): The name of the S3 bucket.
        object_name (str, optional): S3 object name. Defaults to file's name.

    Returns:
        bool: True if file was uploaded, else False.
    """
    if object_name is None:
        object_name = file.name
    try:
        s3_client.upload_fileobj(file, bucket_name, object_name)
        return True
    except (BotoCoreError, ClientError) as error:
        st.error(f"Error uploading PDF to S3: {error}")
        return False

def index_document_to_opensearch(document_id: str, text: str):
    """
    Indexes a document into Amazon OpenSearch.

    Args:
        document_id (str): Unique identifier for the document.
        text (str): Extracted text from the document.
    """
    try:
        index_name = 'rag-documents'  # Choose an appropriate index name
        # Check if the index exists; if not, create it
        if not opensearch_client.indices.exists(index=index_name):
            opensearch_client.indices.create(
                index=index_name,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
            )
        # Index the document
        opensearch_client.index(
            index=index_name,
            id=document_id,
            body={
                "text": text,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )
    except (BotoCoreError, ClientError) as error:
        st.error(f"Error indexing document to OpenSearch: {error}")

# -----------------------------
# Session State Initialization
# -----------------------------

def initialize_session_state():
    """
    Initializes necessary session state variables.
    """
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = []  # List of dicts: {'key', 'name', 'text'}
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = ""
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'previous_num_questions' not in st.session_state:
        st.session_state.previous_num_questions = 5  # Default slider value
    if 'custom_question_input' not in st.session_state:
        st.session_state.custom_question_input = ""

initialize_session_state()

# -----------------------------
# Sidebar for Inputs
# -----------------------------

st.sidebar.header("Input Options")

# Function to generate unique key for each file
def generate_file_key(file) -> str:
    return f"{file.name}_{file.size}"

# Option to upload PDF files
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files is not None:
    # Generate keys for currently uploaded files
    current_file_keys = set(generate_file_key(file) for file in uploaded_files)
    # Generate keys for previously uploaded files
    previous_file_keys = set(pdf['key'] for pdf in st.session_state.uploaded_pdfs)

    # Identify removed files
    removed_keys = previous_file_keys - current_file_keys
    if removed_keys:
        st.session_state.uploaded_pdfs = [pdf for pdf in st.session_state.uploaded_pdfs if pdf['key'] not in removed_keys]
        st.success(f"Removed {len(removed_keys)} PDF(s).")
        # Clear previous questions and answers since PDFs have changed
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_summary = ""
        st.session_state.topics = []

    # Identify new uploads
    new_keys = current_file_keys - previous_file_keys
    if new_keys:
        for file in uploaded_files:
            key = generate_file_key(file)
            if key in new_keys:
                # Upload to S3
                upload_success = upload_pdf_to_s3(file, S3_BUCKET_NAME, object_name=key)
                if upload_success:
                    # Extract text from PDF
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(file.read())
                        temp_file.flush()
                        text = extract_text_from_pdf(temp_file.name)
                    os.remove(temp_file.name)
                    # Add the new PDF to session_state
                    st.session_state.uploaded_pdfs.append({'key': key, 'name': file.name, 'text': text})
                    # Index into OpenSearch
                    index_document_to_opensearch(document_id=key, text=text)
        st.success(f"Processed and uploaded {len(new_keys)} new PDF(s) to S3 and indexed to OpenSearch.")
        # Clear previous questions and answers since PDFs have changed
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_summary = ""
        st.session_state.topics = []
else:
    # Handle the case when all PDFs are removed
    if st.session_state.uploaded_pdfs:
        st.session_state.uploaded_pdfs = []
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_summary = ""
        st.session_state.topics = []
        st.info("All PDFs have been removed.")

# Option to input a PDF URL
pdf_url = st.sidebar.text_input("Or enter a PDF URL:", placeholder="https://example.com/paper.pdf")
if pdf_url:
    with st.spinner("Downloading and processing PDF from URL..."):
        temp_pdf = download_pdf_from_url(pdf_url)
        if temp_pdf:
            file_name = os.path.basename(temp_pdf.name)
            file_size = os.path.getsize(temp_pdf.name)
            key = f"{file_name}_{file_size}"
            # Upload to S3
            upload_success = upload_pdf_to_s3(temp_pdf, S3_BUCKET_NAME, object_name=key)
            if upload_success:
                text = extract_text_from_pdf(temp_pdf.name)
                # Add the new PDF to session_state
                st.session_state.uploaded_pdfs.append({'key': key, 'name': file_name, 'text': text})
                # Index into OpenSearch
                index_document_to_opensearch(document_id=key, text=text)
                st.success("PDF downloaded, processed, uploaded to S3, and indexed to OpenSearch successfully!")
            os.remove(temp_pdf.name)
    st.write("---")

# Slider for number of questions to generate
num_questions = st.sidebar.slider(
    "Number of questions to generate:",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="Select the number of questions you want to generate based on the document."
)

# Checkbox for summarization (unchecked by default)
summarize = st.sidebar.checkbox("Summarize Document", value=False)
st.session_state.summarize = summarize

# Detect changes in slider value to regenerate questions
if num_questions != st.session_state.previous_num_questions:
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.previous_num_questions = num_questions
    st.session_state.pdf_summary = ""  # Optionally reset summary
    st.session_state.topics = []

# -----------------------------
# Main Content
# -----------------------------

if st.session_state.uploaded_pdfs:
    # Define maximum tokens for input
    MAX_TOKENS = 7000  # Reserve some tokens for the completion

    # Combine all PDF texts
    combined_pdf_text = "\n\n".join([pdf['text'] for pdf in st.session_state.uploaded_pdfs])
    
    text_token_count = estimate_tokens(combined_pdf_text)
    if text_token_count > MAX_TOKENS:
        if not st.session_state.summarize:
            st.warning(
                "The combined text from the uploaded PDFs is too long to process without summarization. "
                "Please enable the 'Summarize Document' checkbox to reduce the text size."
            )
        else:
            # Summarize the combined text
            if not st.session_state.pdf_summary:
                with st.spinner("Summarizing the document..."):
                    summary = summarize_text(combined_pdf_text)
                    st.session_state.pdf_summary = summary
                st.write("**Summary of the Document:**")
                st.write(st.session_state.pdf_summary)
            
            # Split the summarized text into chunks
            chunks = split_text_into_chunks(st.session_state.pdf_summary, max_tokens=7000)
            all_questions = []
            for chunk in chunks:
                with st.spinner("Generating questions for a text chunk..."):
                    topics = extract_topics(chunk)
                    questions = generate_questions_batch(chunk, topics, num_questions)
                    all_questions.extend(questions)
            st.session_state.questions = all_questions
            st.success("Questions generated successfully from all chunks!")
    else:
        # Summarize if checkbox is selected and summary not already generated
        if st.session_state.summarize and not st.session_state.pdf_summary:
            with st.spinner("Summarizing the document..."):
                summary = summarize_text(combined_pdf_text)
                st.session_state.pdf_summary = summary
            st.write("**Summary of the Document:**")
            st.write(st.session_state.pdf_summary)
            text_for_topics = st.session_state.pdf_summary
        else:
            text_for_topics = combined_pdf_text

        # Extract Topics
        if not st.session_state.topics:
            with st.spinner("Extracting key topics from the document..."):
                topics = extract_topics(text_for_topics)
                st.session_state.topics = topics

        # Generate Questions
        if not st.session_state.questions:
            with st.spinner("Generating questions based on extracted topics..."):
                questions = generate_questions_batch(text_for_topics, st.session_state.topics, num_questions)
                st.session_state.questions = questions
            if st.session_state.questions:
                st.success("Questions generated successfully!")
            else:
                st.warning("No questions were generated. Please try with a different PDF or adjust the settings.")

    # Display Generated Questions as Clickable Buttons
    if st.session_state.questions:
        st.subheader("Generated Questions:")
        for idx, question in enumerate(st.session_state.questions, 1):
            # Each button has a unique key
            if st.button(f"Q{idx}: {question}", key=f"question_{idx}"):
                # Fetch the answer and store it in session state
                if question not in st.session_state.answers:
                    with st.spinner("Fetching answer..."):
                        answer = fetch_answer_with_backoff(question, index_name='rag-documents')
                        st.session_state.answers[question] = answer
                # Display the answer below the button
                st.write(f"**Answer:** {st.session_state.answers[question]}")
        st.write("---")

    # Custom Query Input Field using a Form
    st.subheader("Ask a Custom Question")

    with st.form(key='custom_query_form', clear_on_submit=True):
        user_question = st.text_input(
            "Enter your custom question here:",
            placeholder="Type your question and press Enter",
            key='custom_question_input'
        )
        submit_button = st.form_submit_button(label='Submit Custom Query')

    if submit_button:
        if not user_question.strip():
            st.error("Please enter a valid question.")
        else:
            with st.spinner("Fetching answer..."):
                answer = fetch_answer_with_backoff(user_question, index_name='rag-documents')
                st.session_state.answers[user_question] = answer
        st.write(f"**Your Question:** {user_question}")
        st.write(f"**Answer:** {st.session_state.answers[user_question]}")
        st.write("---")

else:
    st.info("Please upload a PDF file or enter a PDF URL to generate questions.")
