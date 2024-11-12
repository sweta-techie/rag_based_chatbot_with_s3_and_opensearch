import os
import time
import random
import boto3
import botocore.config
import streamlit as st
import json
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_community.vectorstores import FAISS
import faiss

# ===========================
# Set Page Configuration
# ===========================

st.set_page_config(page_title="RAG Chatbot with Amazon Bedrock", layout="wide")

# ===========================
# Configuration Variables
# ===========================

# AWS Configuration
AWS_REGION = "us-east-1"  # Replace with your AWS region, e.g., "us-west-2"

# Knowledge Base Configuration
KNOWLEDGE_BASE_ID = "T1DTMLLRUP"  # Replace with your Knowledge Base ID

# Path to store chat history locally
CHAT_HISTORY_FILE = "chat_history.json"

# Path to store FAISS index
FAISS_INDEX_FILE = "faiss_index.index"

# Initialize AWS clients with retry configuration
config = botocore.config.Config(
    retries={
        'max_attempts': 5,
        'mode': 'standard'
    }
)

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION, config=config)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)

# ===========================
# Initialize Bedrock Components
# ===========================

# Initialize Embedding Model
BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
bedrock_embeddings = BedrockEmbeddings(
    model_id=BEDROCK_EMBEDDING_MODEL_ID,
    client=bedrock_runtime
)

# Initialize LLM
BEDROCK_LLM_MODEL_ID = "amazon.titan-text-premier-v1:0"
llm = BedrockLLM(
    model_id=BEDROCK_LLM_MODEL_ID,
    client=bedrock_runtime,
    model_kwargs={
        'maxTokenCount': 512,
        'temperature': 0.5,
        'topP': 0.9
    }
)

# Define the prompt template
from langchain.prompts import PromptTemplate
prompt_template = """
Use the following context to answer the question. Provide references to the sources used in your answer.

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

def retrieve_from_knowledge_base(question, knowledge_base_id):
    """Retrieve relevant documents from the knowledge base along with source details."""
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={'text': question}
        )
        # Check if response contains results
        if not response.get('retrievalResults'):
            st.warning("⚠️ No results returned from knowledge base retrieval.")
        # Extract retrieved passages and sources
        retrieved_passages = []
        sources = []
        for hit in response.get('retrievalResults', []):
            passage = hit.get('content', {}).get('text', '')
            # Extract source information
            location = hit.get('location', {})
            source_uri = ''
            if location.get('type') == 'S3':
                source_uri = location.get('s3Location', {}).get('uri', '')
            elif location.get('type') == 'WEB':
                source_uri = location.get('webLocation', {}).get('url', '')
            else:
                source_uri = 'Knowledge Base Document'

            retrieved_passages.append(passage)
            sources.append({'content': passage, 'source': source_uri})
        return retrieved_passages, sources
    except Exception as e:
        st.error(f"❌ Failed to retrieve from knowledge base: {e}")
        return [], []

def retrieve_from_local_documents(question, vectorstore, k=5):
    """Retrieve relevant documents from local vector store along with source details."""
    if vectorstore is None:
        return [], []
    try:
        similar_docs = vectorstore.similarity_search_with_score(question, k=k)
        retrieved_passages = []
        sources = []
        for doc, score in similar_docs:
            retrieved_passages.append(doc.page_content)
            sources.append({'content': doc.page_content, 'source': doc.metadata.get('source', 'Uploaded Document')})
        return retrieved_passages, sources
    except Exception as e:
        st.error(f"❌ Failed to retrieve from local documents: {e}")
        return [], []

def invoke_with_retry(llm, prompt, max_attempts=5):
    """Invoke the LLM with retry logic in case of throttling."""
    for attempt in range(1, max_attempts + 1):
        try:
            answer = llm.invoke(prompt)
            return answer
        except Exception as e:
            if "ThrottlingException" in str(e):
                if attempt == max_attempts:
                    raise e
                else:
                    sleep_time = random.uniform(1, 2 ** attempt)
                    st.warning(f"⚠️ Throttling detected. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            else:
                raise e

def handle_submit():
    """Handle the submission of a user question."""
    user_question = st.session_state.user_question.strip()
    if not user_question:
        st.error("❌ Please enter a question.")
        return

    # Rate limiting: Add a delay to prevent rapid submissions
    if 'last_submission_time' in st.session_state:
        time_since_last = time.time() - st.session_state.last_submission_time
        if time_since_last < 2:  # Minimum 2 seconds between submissions
            st.warning("⚠️ Please wait a moment before submitting another question.")
            return

    st.session_state.last_submission_time = time.time()

    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    # Save updated chat history locally
    save_chat_history(st.session_state.chat_history)
    with st.spinner("⏳ Processing your question..."):
        # Retrieve from knowledge base
        kb_contexts, kb_sources = retrieve_from_knowledge_base(user_question, KNOWLEDGE_BASE_ID)
        # Retrieve from local documents
        local_contexts, local_sources = retrieve_from_local_documents(user_question, st.session_state.vectorstore_faiss)
        # Combine contexts and sources (local first)
        combined_contexts = local_contexts + kb_contexts  # Local contexts first
        combined_sources = local_sources + kb_sources     # Local sources first

        if not combined_contexts:
            st.warning("⚠️ No relevant information found to answer the question. The response may not be based on the provided documents.")
            combined_sources = []
            combined_context = ""  # Allow the model to answer without context, or handle differently
        else:
            combined_context = "\n\n".join(combined_contexts)

        # Generate response
        prompt = PROMPT.format(context=combined_context, question=user_question)
        try:
            answer = invoke_with_retry(llm, prompt)
            # Deduplicate sources and limit to two
            unique_sources = []
            seen_sources = set()
            for source in combined_sources:
                source_uri = source.get('source', '')
                if source_uri and source_uri not in seen_sources:
                    unique_sources.append(source)
                    seen_sources.add(source_uri)
                if len(unique_sources) == 2:
                    break

            # Append assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": unique_sources  # Store only unique sources up to two
            })
            # Save updated chat history locally
            save_chat_history(st.session_state.chat_history)
            # Set a flag to clear the input field
            st.session_state.clear_input = True
            # Rerun the app to clear the input field
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to get a response from the model: {e}")

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
    st.rerun()

# ===========================
# Initialize Streamlit State
# ===========================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if 'user_question' not in st.session_state:
    st.session_state.user_question = ""  # Initialize user_question

if 'vectorstore_faiss' not in st.session_state:
    st.session_state.vectorstore_faiss = None

# Initialize last_submission_time
if 'last_submission_time' not in st.session_state:
    st.session_state.last_submission_time = 0

# Initialize clear_input flag
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

# ===========================
# Main Application
# ===========================

def main():
    st.title("📚 Retrieval-Augmented Generation (RAG) Chatbot with Amazon Bedrock")

    st.markdown("---")

    # ---------------------------
    # Upload Document Section
    # ---------------------------

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents to include in your personal context:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("🔄 Processing uploaded documents..."):
            new_docs = process_uploaded_files(uploaded_files)
            if new_docs:
                vector_store = update_vector_store(new_docs)
                if vector_store:
                    st.success("✅ Uploaded documents have been processed and added to your context.")
            else:
                st.warning("⚠️ No valid documents were processed for inclusion.")

    st.markdown("---")

    # ---------------------------
    # Q&A Section
    # ---------------------------

    st.header("❓ Ask a Question")

    with st.form(key='qa_form'):
        # Check if the input field should be cleared
        if st.session_state.clear_input:
            st.session_state.user_question = ""
            st.session_state.clear_input = False

        st.text_input("Enter your question here:", key='user_question')
        submit_button = st.form_submit_button(label='Ask')
        if submit_button:
            handle_submit()

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

            # Display the sources if available
            sources = latest_interaction.get('sources', [])
            if sources:
                st.markdown("#### 📚 Sources:")
                seen_sources = set()
                for source in sources:
                    source_uri = source.get('source', '')
                    if source_uri and source_uri not in seen_sources:
                        st.markdown(f"- {source_uri}")
                        seen_sources.add(source_uri)

    st.markdown("---")

    # ---------------------------
    # Chat History and Controls in Sidebar
    # ---------------------------

    st.sidebar.header("💬 Chat History")
    if st.session_state.chat_history:
        # Group messages into pairs (user and assistant)
        messages = []
        i = 0
        while i < len(st.session_state.chat_history):
            user_message = st.session_state.chat_history[i] if i < len(st.session_state.chat_history) else None
            assistant_message = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
            messages.append({'user': user_message, 'assistant': assistant_message})
            i += 2

        # Reverse the list to show the latest interactions first
        messages = messages[::-1]

        for pair in messages:
            if pair['user']:
                st.sidebar.markdown("👤 **You:**")
                st.sidebar.markdown(f"{pair['user']['content']}")
            if pair['assistant']:
                st.sidebar.markdown("🤖 **Bot:**")
                st.sidebar.markdown(f"{pair['assistant']['content']}")
                # Display the sources if available
                sources = pair['assistant'].get('sources', [])
                if sources:
                    st.sidebar.markdown("**📚 Sources:**")
                    seen_sources = set()
                    for source in sources:
                        source_uri = source.get('source', '')
                        if source_uri and source_uri not in seen_sources:
                            st.sidebar.markdown(f"- {source_uri}")
                            seen_sources.add(source_uri)
            st.sidebar.markdown("---")  # Separator between messages
    else:
        st.sidebar.write("ℹ️ No conversation yet.")

    # Move "Clear Chat History" button to sidebar
    st.sidebar.header("🛠️ Controls")
    if st.sidebar.button("🧹 Clear Chat History"):
        clear_chat_history()

    # ===========================
    # Instructions (Removed)
    # ===========================
    # Instructions section has been removed as per your request.

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
                        st.warning(f"⚠️ No content found in {uploaded_file.name}.")
                    else:
                        # Add metadata with source information
                        for doc in loaded_docs:
                            doc.metadata['source'] = uploaded_file.name
                    documents.extend(loaded_docs)
                os.remove(temp_file.name)
            except Exception as e:
                st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
    if not documents:
        st.error("❌ No valid documents were loaded from the uploaded files.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        if not docs:
            st.error("❌ No documents were generated after splitting.")
        else:
            pass
        return docs
    except Exception as e:
        st.error(f"❌ Failed to split documents: {e}")
        return []

def update_vector_store(new_docs):
    """Update the vector store with new documents."""
    try:
        if st.session_state.vectorstore_faiss is None:
            # Initialize FAISS vector store with documents
            vectorstore_faiss = FAISS.from_documents(new_docs, bedrock_embeddings)
            st.session_state.vectorstore_faiss = vectorstore_faiss
            st.success("✅ Created new vector store with uploaded documents.")
        else:
            # Add documents to existing FAISS vector store
            st.session_state.vectorstore_faiss.add_documents(new_docs)
            st.success("✅ Added new documents to existing vector store.")

        # Save the FAISS index locally
        faiss.write_index(st.session_state.vectorstore_faiss.index, FAISS_INDEX_FILE)
        return st.session_state.vectorstore_faiss
    except Exception as e:
        st.error(f"❌ Failed to update the vector store: {e}")
        return None

if __name__ == "__main__":
    main()
