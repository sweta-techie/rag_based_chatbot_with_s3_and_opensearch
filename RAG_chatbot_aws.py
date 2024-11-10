import streamlit as st
import boto3
import json
import logging
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from ratelimit import limits, sleep_and_retry
from dataclasses import dataclass
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import os
 
# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
 
@dataclass
class Config:
    """Configuration settings."""
    aws_region: str = "us-east-1"
    model_id: str = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-premier-v1:0"
    temperature: float = 0.7
    top_p: float = 0.9
    max_token_count: int = 3072
    sagemaker_endpoint: str = "nl2sql-endpoint-1729993415"  # Replace with your SageMaker endpoint name
 
config = Config()
 
# Initialize SageMaker client
def generate_embeddings_sagemaker(texts, endpoint_name):
    """
    Generates embeddings for a list of texts using a SageMaker endpoint.
    """
    client = boto3.client('sagemaker-runtime', region_name=config.aws_region)
    payload = json.dumps({"instances": texts})
     
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
     
    result = json.loads(response['Body'].read().decode())
    embeddings = result['predictions']
    return embeddings
 
# Initialize FAISS index
def initialize_faiss(dim):
    index = faiss.IndexFlatL2(dim)  # Using L2 distance; you can switch to other indices based on needs
    return index
 
# In-memory storage for documents and embeddings
documents = []
embeddings_matrix = None
faiss_index = None
 
# Document processing functions
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text
 
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks
 
# LLM invocation
@sleep_and_retry
@limits(calls=2, period=60)
@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def invoke_bedrock_model(prompt):
    """Invoke the Bedrock model and return the generated text."""
    try:
        request_payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_token_count,
                "stopSequences": [],
                "temperature": config.temperature,
                "topP": config.top_p
            }
        }
 
        client_bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=config.aws_region
        )
 
        response = client_bedrock.invoke_model(
            modelId=config.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_payload)
        )
        response_body = response['body'].read().decode('utf-8')
        logger.info("Model Response:")
        logger.info(response_body)
        return parse_model_response(response_body)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        if error_code == 'ThrottlingException':
            logger.warning(f"ThrottlingException encountered: {error_message}")
            raise
        elif error_code == 'ValidationException':
            logger.error(f"ValidationException: {error_message}")
            raise
        else:
            logger.error(f"Unexpected ClientError: {error_message}")
            raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise
 
def parse_model_response(response_json):
    """
    Parses the model response and extracts the generated text.
    """
    response = json.loads(response_json)
    results = response.get("results", [])
    if results:
        return results[0].get("outputText", "No response generated.")
    return "No response generated."
 
# Streamlit App
def main():
    global documents, embeddings_matrix, faiss_index
    st.set_page_config(page_title="RAG Chatbot with Titan Text G1", layout="wide")
    st.title("📚 Retrieval-Augmented Generation (RAG) Chatbot")
     
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True
        )
         
        if uploaded_files:
            if st.button("Process Documents"):
                try:
                    with st.spinner("Processing documents..."):
                        all_chunks = []
                        for uploaded_file in uploaded_files:
                            text = extract_text_from_pdf(uploaded_file)
                            if not text.strip():
                                st.warning(f"No text extracted from {uploaded_file.name}. Skipping.")
                                continue
                            chunks = chunk_text(text)
                            embeddings = generate_embeddings_sagemaker(chunks, config.sagemaker_endpoint)
                            all_chunks.extend(zip(chunks, embeddings))
                        
                        if all_chunks:
                            # Separate chunks and embeddings
                            chunks, embeddings = zip(*all_chunks)
                            documents.extend(chunks)
                            
                            # Initialize FAISS index if not already
                            if faiss_index is None:
                                embedding_dim = len(embeddings[0])
                                faiss_index = initialize_faiss(embedding_dim)
                                embeddings_matrix = np.array(embeddings).astype('float32')
                                faiss_index.add(embeddings_matrix)
                            else:
                                new_embeddings = np.array(embeddings).astype('float32')
                                faiss_index.add(new_embeddings)
                                embeddings_matrix = np.vstack((embeddings_matrix, new_embeddings))
                            
                            st.success(f"Successfully processed and indexed {len(uploaded_files)} documents!")
                        else:
                            st.warning("No valid text chunks were extracted from the uploaded documents.")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
     
    st.markdown("### 🗨️ Chat with the Bot")
     
    user_input = st.text_input("You:", "")
     
    if st.button("Send"):
        if user_input:
            with st.spinner("Generating response..."):
                try:
                    if faiss_index is None or len(documents) == 0:
                        st.warning("No documents have been processed yet. Please upload and process documents first.")
                    else:
                        # Generate embedding for the user query
                        query_embedding = generate_embeddings_sagemaker([user_input], config.sagemaker_endpoint)[0]
                        query_embedding = np.array(query_embedding).astype('float32')
                        query_embedding = np.expand_dims(query_embedding, axis=0)
 
                        # Perform similarity search
                        D, I = faiss_index.search(query_embedding, k=5)  # Retrieve top 5
                        retrieved_chunks = [documents[i] for i in I[0]]
 
                        # Construct the prompt with context
                        prompt = construct_prompt(retrieved_chunks, user_input)
 
                        # Get response from LLM
                        generated_text = invoke_bedrock_model(prompt)
 
                        # Display the conversation
                        if 'conversation' not in st.session_state:
                            st.session_state.conversation = []
 
                        st.session_state.conversation.append(("You", user_input))
                        st.session_state.conversation.append(("Assistant", generated_text))
 
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ThrottlingException':
                        st.warning("The service is busy. Please wait 30-60 seconds before trying again.")
                    else:
                        st.error(f"An error occurred: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
     
    # Display chat history
    if 'conversation' in st.session_state:
        for sender, message in st.session_state.conversation:
            if sender == "You":
                st.markdown(f"🧑 **You:** {message}")
            else:
                st.markdown(f"🤖 **Assistant:** {message}")
 
def construct_prompt(retrieved_chunks, user_query):
    """Construct the prompt by combining retrieved chunks with the user query."""
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Use the following context to answer the question. Be specific and concise.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    return prompt
 
if __name__ == "__main__":
    main()
