import os
import boto3
import streamlit as st
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile
import faiss
import json
from botocore.exceptions import ClientError

# ===========================
# Set Page Configuration
# ===========================

st.set_page_config(page_title="RAG Chatbot with Bedrock & S3", layout="wide")

# ===========================
# Configuration Variables
# ===========================

# AWS Configuration
AWS_REGION = "us-east-1"  # Replace with your AWS region
S3_BUCKET_NAME = "terra-help-bucket"  # Replace with your S3 bucket name
FAISS_INDEX_FILE = "faiss_index.index"  # Local path for FAISS index
FAISS_INDEX_S3_KEY = "faiss_index.index"  # S3 key for FAISS index file

# S3 Prefixes (Folders)
KNOWLEDGE_BASE_PREFIX = "knowledge_base/"  # Prefix for Knowledge Base documents
UPLOADED_DOCUMENTS_PREFIX = "uploaded_documents/"  # Prefix for uploaded documents

# Path to store chat history locally
CHAT_HISTORY_FILE = "chat_history.json"

# Knowledge Base Configuration
KNOWLEDGE_BASE_ID = "T1DTMLLRUP"  # Replace with your Knowledge Base ID
BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Embedding model ID
BEDROCK_LLM_MODEL_ID = "amazon.titan-text-premier-v1:0"  # LLM model ID

# Initialize AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ===========================
# Utility Functions
# ===========================

def load_chat_history():
    """Load chat history from a local JSON file."""
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
    """Save chat history to a local JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

def upload_faiss_index(index_path):
    """Upload FAISS index to S3."""
    try:
        s3_client.upload_file(index_path, S3_BUCKET_NAME, FAISS_INDEX_S3_KEY)
        st.sidebar.success("✅ FAISS index uploaded to S3 successfully.")
    except ClientError as e:
        st.sidebar.error(f"❌ Error uploading FAISS index to S3: {e}")

def download_faiss_index():
    """Download FAISS index from S3."""
    try:
        with open(FAISS_INDEX_FILE, 'wb') as f:
            s3_client.download_fileobj(S3_BUCKET_NAME, FAISS_INDEX_S3_KEY, f)
        st.sidebar.success("✅ FAISS index downloaded from S3 successfully.")
        return True
    except ClientError as e:
        st.sidebar.error(f"❌ Error downloading FAISS index from S3: {e}")
        return False

def load_s3_documents(prefix=None):
    """
    Load documents from S3 Bucket stored under a specific prefix.
    If prefix is None or empty, it will load documents from the root of the bucket.
    """
    documents = []
    try:
        if prefix:
            list_kwargs = {'Bucket': S3_BUCKET_NAME, 'Prefix': prefix}
        else:
            list_kwargs = {'Bucket': S3_BUCKET_NAME}

        response = s3_client.list_objects_v2(**list_kwargs)
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if prefix and not key.startswith(prefix):
                    continue  # Skip keys not under the current prefix
                if key.endswith(".pdf") or key.endswith(".txt"):
                    tmp_file = NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1])
                    s3_client.download_fileobj(S3_BUCKET_NAME, key, tmp_file)
                    tmp_file.close()
                    if key.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file.name)
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                    elif key.endswith(".txt"):
                        loader = TextLoader(tmp_file.name, encoding='utf-8')
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                    os.remove(tmp_file.name)
        else:
            location = f"prefix: {prefix}" if prefix else "root of the bucket"
            st.sidebar.warning(f"⚠️ No documents found under {location}")
        return documents
    except ClientError as e:
        st.sidebar.error(f"❌ Error loading documents from S3: {e}")
        return []

# ===========================
# Initialize Streamlit State
# ===========================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if 'vectorstore_faiss' not in st.session_state:
    # Initialize the FAISS vector store as None
    st.session_state.vectorstore_faiss = None

if 'user_question' not in st.session_state:
    st.session_state.user_question = ""  # Initialize user_question

# ===========================
# Initialize Bedrock Components
# ===========================

# Initialize Embedding Model
bedrock_embeddings = BedrockEmbeddings(
    model_id=BEDROCK_EMBEDDING_MODEL_ID,
    client=bedrock_runtime
)

# Function to Initialize LLM
def get_llm():
    try:
        llm = BedrockLLM(
            model_id=BEDROCK_LLM_MODEL_ID,
            client=bedrock_runtime,
            model_kwargs={
                'maxTokenCount': 512,
                'temperature': 0.5,
                'topP': 0.9
            }
        )
        return llm
    except ClientError as e:
        st.error(f"❌ Failed to initialize LLM: {e}")
        return None

# Define the prompt template
prompt_template = """
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# ===========================
# Core Functionalities
# ===========================

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF and TXT files and return document chunks."""
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type in ["application/pdf", "text/plain"]:
            try:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file.flush()
                    if uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(temp_file.name)
                    else:
                        loader = TextLoader(temp_file.name, encoding='utf-8')
                    loaded_docs = loader.load()
                    if not loaded_docs:
                        st.sidebar.warning(f"⚠️ No content found in {uploaded_file.name}.")
                    else:
                        st.sidebar.write(f"📄 Loaded {len(loaded_docs)} pages from {uploaded_file.name}.")
                    documents.extend(loaded_docs)
                os.remove(temp_file.name)
            except Exception as e:
                st.sidebar.error(f"❌ Failed to process {uploaded_file.name}: {e}")
    if not documents:
        st.sidebar.error("❌ No valid documents were loaded from the uploaded files.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        if not docs:
            st.sidebar.error("❌ No documents were generated after splitting.")
        else:
            st.sidebar.write(f"📄 Total of {len(docs)} document chunks after splitting.")
        return docs
    except Exception as e:
        st.sidebar.error(f"❌ Failed to split documents: {e}")
        return []

def build_vector_store(docs):
    """Build the vector store from documents and save the FAISS index."""
    texts = [doc.page_content for doc in docs]
    try:
        # Initialize FAISS vector store with texts
        vectorstore_faiss = FAISS.from_texts(texts, bedrock_embeddings)
        st.session_state.vectorstore_faiss = vectorstore_faiss
        st.sidebar.success("✅ FAISS vector store built successfully.")

        # Save the FAISS index locally
        faiss.write_index(vectorstore_faiss.index, FAISS_INDEX_FILE)
        st.sidebar.success("✅ FAISS index saved locally.")

        # Optionally, upload the FAISS index to S3 for backup
        upload_faiss_index(FAISS_INDEX_FILE)

        st.sidebar.write(f"📊 FAISS vector store contains {vectorstore_faiss.index.ntotal} vectors.")
        return vectorstore_faiss
    except Exception as e:
        st.sidebar.error(f"❌ An error occurred while building the FAISS vector store: {e}")
        return None

def load_or_build_vector_store():
    """Load the FAISS vector store from file or build it if not available."""
    # Attempt to load the FAISS index from local file
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            st.session_state.vectorstore_faiss = FAISS(embedding_function=bedrock_embeddings, index=index)
            st.sidebar.success("✅ FAISS index loaded from local file.")
            st.sidebar.write(f"📊 FAISS vector store contains {st.session_state.vectorstore_faiss.index.ntotal} vectors.")
            return st.session_state.vectorstore_faiss
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load FAISS index from local file: {e}")

    # If local file not found or failed to load, try downloading from S3
    try:
        with open(FAISS_INDEX_FILE, 'wb') as f:
            s3_client.download_fileobj(S3_BUCKET_NAME, FAISS_INDEX_S3_KEY, f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        st.session_state.vectorstore_faiss = FAISS(embedding_function=bedrock_embeddings, index=index)
        st.sidebar.success("✅ FAISS index downloaded and loaded from S3.")
        st.sidebar.write(f"📊 FAISS vector store contains {st.session_state.vectorstore_faiss.index.ntotal} vectors.")
        return st.session_state.vectorstore_faiss
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load FAISS index from S3: {e}")

    # If FAISS index not found, need to build it
    st.sidebar.info("ℹ️ FAISS index not found. Building a new vector store.")
    all_docs = load_all_documents()
    if all_docs:
        return build_vector_store(all_docs)
    else:
        st.sidebar.warning("⚠️ No documents found to build vector store.")
        return None

def update_vector_store(new_docs):
    """Update the vector store with new documents."""
    if st.session_state.vectorstore_faiss is None:
        st.error("❌ Vector store is not initialized.")
        return None
    texts = [doc.page_content for doc in new_docs]
    try:
        st.session_state.vectorstore_faiss.add_texts(texts)
        st.sidebar.success("✅ Added new documents to FAISS vector store.")

        # Save the updated FAISS index locally
        faiss.write_index(st.session_state.vectorstore_faiss.index, FAISS_INDEX_FILE)
        st.sidebar.success("✅ Updated FAISS index saved locally.")

        # Optionally, upload the updated FAISS index to S3
        upload_faiss_index(FAISS_INDEX_FILE)
        st.sidebar.write(f"📊 FAISS vector store now contains {st.session_state.vectorstore_faiss.index.ntotal} vectors.")
        return st.session_state.vectorstore_faiss
    except Exception as e:
        st.sidebar.error(f"❌ Failed to update the FAISS vector store: {e}")
        return None

def retrieve_and_generate_local(query, number_of_results=5):
    """Retrieve documents locally and generate a response."""
    if st.session_state.vectorstore_faiss is None:
        st.error("❌ FAISS vector store is not initialized.")
        return {'answer': "FAISS vector store is not initialized.", 'context': ""}

    # Retrieve similar documents using local vector store
    similar_docs = st.session_state.vectorstore_faiss.similarity_search(query, k=number_of_results)
    if not similar_docs:
        st.warning("⚠️ No similar documents found.")
        return {'answer': "No relevant documents found.", 'context': ""}

    # Generate response using LLM
    context = "\n\n".join([doc.page_content for doc in similar_docs])
    llm = get_llm()
    if llm:
        prompt = PROMPT.format(context=context, question=query)
        answer = llm(prompt)
        return {'answer': answer, 'context': context}
    else:
        return {'answer': "LLM initialization failed.", 'context': ""}

def handle_submit():
    """Handle the submission of a user question."""
    user_question = st.session_state.user_question.strip()
    if not user_question:
        st.error("❌ Please enter a question.")
        return
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    # Save updated chat history locally
    save_chat_history(st.session_state.chat_history)
    with st.spinner("⏳ Processing your question..."):
        response = retrieve_and_generate_local(user_question)
        if response['answer']:
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
            # Clear the input field
            st.session_state.user_question = ""
            # Save updated chat history locally
            save_chat_history(st.session_state.chat_history)

def clear_chat_history():
    """Clear the chat history from session state and local storage."""
    st.session_state.chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            os.remove(CHAT_HISTORY_FILE)
            st.sidebar.success("✅ Chat history has been cleared.")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to delete chat history file: {e}")
    else:
        st.sidebar.info("ℹ️ No chat history to clear.")
    st.experimental_rerun()  # Force a rerun to update the interface

# ===========================
# Main Application
# ===========================

def main():
    st.title("📚 Retrieval-Augmented Generation (RAG) Chatbot with Amazon Bedrock & S3")

    st.markdown("---")

    # ---------------------------
    # Upload Document Section
    # ---------------------------

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents to add to the Knowledge Base:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("📤 Uploading documents to S3..."):
            for uploaded_file in uploaded_files:
                original_filename = uploaded_file.name
                renamed_filename = f"{UPLOADED_DOCUMENTS_PREFIX}{original_filename}"

                # Save the uploaded file temporarily
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_filename)[1]) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Upload to S3
                try:
                    s3_client.upload_file(temp_file_path, S3_BUCKET_NAME, renamed_filename)
                    st.success(f"✅ Successfully uploaded `{original_filename}` to the Knowledge Base.")
                except ClientError as e:
                    st.error(f"❌ Error uploading `{original_filename}`: {e}")
                finally:
                    # Remove the temporary file
                    os.remove(temp_file_path)

        st.markdown("---")
        st.info("""
        📢 **AutoSync is Enabled**

        The uploaded documents will automatically trigger a Lambda function to sync with the Knowledge Base. 
        Please wait a few moments for the ingestion process to complete before querying the new documents.
        """)

        # Process and index uploaded documents
        with st.spinner("🔄 Processing and indexing uploaded documents..."):
            new_docs = process_uploaded_files(uploaded_files)
            if new_docs:
                vector_store = update_vector_store(new_docs)
                if vector_store:
                    st.success("✅ Uploaded documents have been processed and indexed successfully.")
            else:
                st.warning("⚠️ No valid documents were processed for indexing.")

    st.markdown("---")

    # ---------------------------
    # Load or Build Vector Store
    # ---------------------------

    if st.session_state.vectorstore_faiss is None:
        with st.spinner("🔄 Loading or building vector store..."):
            vector_store = load_or_build_vector_store()
            if vector_store:
                st.success("✅ Vector store is ready.")
            else:
                st.error("❌ Failed to initialize vector store.")

    st.markdown("---")

    # ---------------------------
    # Q&A Section
    # ---------------------------

    st.header("❓ Ask a Question")

    with st.form(key='qa_form'):
        st.text_input("Enter your question here:", key='user_question')
        submit_button = st.form_submit_button(label='Ask', on_click=handle_submit)

    # Display latest Q&A right below the input field
    if st.session_state.chat_history:
        # Fetch the latest interaction
        latest_interaction = st.session_state.chat_history[-1]
        if latest_interaction['role'] == 'assistant':
            st.markdown("### 🗨️ Latest Interaction")
            # Ensure there's at least one user message before the assistant's reply
            if len(st.session_state.chat_history) >= 2:
                user_question = st.session_state.chat_history[-2]['content']
            else:
                user_question = "N/A"
            bot_answer = latest_interaction['content']
            st.markdown(f"**You:** {user_question}")
            st.markdown(f"**Bot:** {bot_answer}")

    st.markdown("---")

    # ---------------------------
    # Chat History and Controls in Sidebar
    # ---------------------------

    st.sidebar.header("💬 Chat History")
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history, 1):
            if chat["role"] == "user":
                st.sidebar.markdown(f"**{idx}. You:** {chat['content']}")
            elif chat["role"] == "assistant":
                st.sidebar.markdown(f"**{idx}. Bot:** {chat['content']}")
    else:
        st.sidebar.write("ℹ️ No conversation yet.")

    # Move "Clear Chat History" button to sidebar
    st.sidebar.header("🛠️ Controls")
    if st.sidebar.button("🧹 Clear Chat History"):
        clear_chat_history()

    # ===========================
    # Instructions
    # ===========================

    st.markdown("""
    ## 📌 Instructions:
    1. **Upload Documents:**
       - Use the **Upload Documents** section above to add `.pdf` or `.txt` files to your Knowledge Base.
       - After uploading, wait for a few moments to allow the ingestion process to complete.
    2. **Ask Questions:**
       - In the **Ask a Question** section, type your question and click **Ask**.
       - The chatbot will generate a response based on all available documents.
    3. **Manage Chat History:**
       - View your conversation history in the left sidebar.
       - Use the **Clear Chat History** button in the sidebar to reset your conversation.
    """)

def load_all_documents():
    """Load and combine all documents from S3, including root-level documents."""
    # Load documents from all prefixes and root
    root_docs = load_s3_documents(prefix=None)  # Load documents from the root
    bedrock_docs = load_s3_documents(KNOWLEDGE_BASE_PREFIX)
    uploaded_docs = load_s3_documents(UPLOADED_DOCUMENTS_PREFIX)
    # Log the number of documents loaded
    st.sidebar.write(f"📄 Loaded {len(root_docs)} documents from root level.")
    st.sidebar.write(f"📄 Loaded {len(bedrock_docs)} documents from Knowledge Base prefix.")
    st.sidebar.write(f"📄 Loaded {len(uploaded_docs)} documents from Uploaded Documents prefix.")
    # Combine all documents
    all_docs = root_docs + bedrock_docs + uploaded_docs
    st.sidebar.write(f"📄 Total documents loaded: {len(all_docs)}")
    return all_docs

if __name__ == "__main__":
    main()
