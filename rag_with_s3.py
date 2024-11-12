import os
import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile
import faiss
from botocore.exceptions import ClientError

# ===========================
# Configuration
# ===========================

AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "your-s3-bucket-name"  # Replace with your bucket

# S3 Prefixes for different document sources
PREFIXES = {
    "knowledge_base": "knowledge_base/",
    "additional_docs": "additional_documents/",
    "uploaded_docs": "uploaded_documents/"
}

# ===========================
# Initialize AWS Clients
# ===========================

try:
    session = boto3.Session(region_name=AWS_REGION)
    s3_client = session.client('s3')
    bedrock_runtime = session.client('bedrock-runtime')
    st.sidebar.success("✅ AWS clients initialized successfully")
except Exception as e:
    st.error(f"❌ AWS Client Error: {str(e)}")
    st.stop()

# Initialize Bedrock embeddings
try:
    bedrock_embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1"
    )
    st.sidebar.success("✅ Bedrock embeddings initialized")
except Exception as e:
    st.error(f"❌ Bedrock Embeddings Error: {str(e)}")
    st.stop()
def list_s3_documents():
    """List all PDF and TXT documents in S3 bucket regardless of prefix."""
    try:
        all_documents = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # List all objects in the bucket
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if file is PDF or TXT
                    if key.lower().endswith(('.pdf', '.txt', '.csv')):
                        all_documents.append(key)
                        st.sidebar.info(f"📄 Found document: {key}")
        
        return all_documents
    except Exception as e:
        st.sidebar.error(f"❌ Error listing S3 documents: {str(e)}")
        return []

def load_s3_documents():
    """Load all documents from S3 bucket."""
    documents = []
    try:
        # Get list of all documents
        document_keys = list_s3_documents()
        
        if not document_keys:
            st.sidebar.warning("⚠️ No documents found in S3 bucket")
            return []
        
        # Process each document
        for key in document_keys:
            try:
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1]) as tmp_file:
                    # Download file from S3
                    s3_client.download_fileobj(S3_BUCKET_NAME, key, tmp_file)
                    tmp_file.flush()
                    
                    # Load based on file type
                    if key.lower().endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file.name)
                    else:  # txt or csv
                        loader = TextLoader(tmp_file.name, encoding='utf-8')
                        
                    # Load document
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    st.sidebar.success(f"✅ Loaded: {key}")
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error processing {key}: {str(e)}")
                continue
                
        st.sidebar.success(f"✅ Total documents loaded: {len(documents)}")
        return documents
        
    except Exception as e:
        st.sidebar.error(f"❌ Error in load_s3_documents: {str(e)}")
        return []

def initialize_vector_store():
    """Initialize vector store with all documents from S3."""
    try:
        # Load all documents
        all_documents = load_s3_documents()
        
        if not all_documents:
            st.sidebar.warning("⚠️ No documents found in S3")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        splits = text_splitter.split_documents(all_documents)
        st.sidebar.info(f"📄 Created {len(splits)} text chunks")
        
        # Create vector store
        vectorstore = FAISS.from_documents(splits, bedrock_embeddings)
        st.sidebar.success(f"✅ Vector store initialized with {len(splits)} chunks")
        
        return vectorstore
        
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing vector store: {str(e)}")
        st.error(f"Error details: {str(e)}")
        return None
def main():
    st.title("📚 RAG Chatbot with Amazon Bedrock & S3")
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    # System Status in Sidebar
    with st.sidebar:
        st.header("System Status")
        st.success("✅ AWS clients initialized successfully")
        st.success("✅ Bedrock embeddings initialized")
    
    # Initialize vector store if not already done
    if st.session_state.vectorstore is None:
        with st.spinner("🔄 Loading documents from S3..."):
            vectorstore = initialize_vector_store()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.sidebar.success("✅ Vector Store: Initialized")
            else:
                st.sidebar.warning("⚠️ Vector Store: Not Initialized")
    
    # Document Upload Section (Optional)
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload additional PDF or TXT files (optional):",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing new documents..."):
            for uploaded_file in uploaded_files:
                try:
                    # Upload directly to S3 root
                    s3_key = uploaded_file.name
                    s3_client.upload_fileobj(uploaded_file, S3_BUCKET_NAME, s3_key)
                    st.success(f"✅ Uploaded {uploaded_file.name}")
                    
                    # Reinitialize vector store
                    vectorstore = initialize_vector_store()
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        
                except Exception as e:
                    st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
    
    # Question Answering Section
    st.header("❓ Ask a Question")
    query = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        if not query:
            st.warning("⚠️ Please enter a question")
        elif st.session_state.vectorstore is None:
            st.error("❌ Knowledge base is not initialized")
        else:
            with st.spinner("🤔 Generating answer..."):
                response = generate_response(st.session_state.vectorstore, query)
                st.markdown("### Answer:")
                st.write(response)
                
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")