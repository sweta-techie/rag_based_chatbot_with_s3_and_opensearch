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

# Prompt template
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

## Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Get embedding model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

# Function to process uploaded PDFs
def process_uploaded_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # Use NamedTemporaryFile to handle the uploaded file in memory
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file.flush()
                loader = PyPDFLoader(temp_file.name)
                documents.extend(loader.load())
            os.remove(temp_file.name)  # Clean up the temporary file
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings
        )
        vectorstore_faiss.save_local("faiss_index")
    except ValueError as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        # Optionally, add more detailed logging
        st.write("Debugging Information: ", str(e))


def get_llm():
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={
            'maxTokenCount': 512,
            'temperature': 0.5,
            'topP': 0.9
        }
    )
    return llm

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

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

def main():
    st.set_page_config("RAG Demo",layout="wide")
    st.header("End-to-End RAG Application")

    # PDF uploader
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Flag to trigger vector storage automatically
    vector_store_trigger = False

    # Check if files have been uploaded
    if uploaded_files:
        with st.spinner("Processing..."):
            docs = process_uploaded_pdfs(uploaded_files)
            vector_store_trigger = True  # Set the trigger for storing vectors automatically

        # Automatically trigger the vector store creation
        if vector_store_trigger:
            with st.spinner("Creating vector store..."):
                get_vector_store(docs)
                st.success("Vector store created from uploaded PDFs!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Send button for asking questions
    if st.button("Send"):
        if not user_question:
            st.error("Please enter a question.")
            return
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_llm()
            try:
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
            except ValueError as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 
