import streamlit as st
import boto3
import json
import base64
import numpy as np
import time
import random
import botocore.exceptions

# Initialize the Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')  # Replace with your region if necessary

# Function to get embeddings from the Titan Multimodal Embeddings model
def get_embeddings(text_input=None, image_input=None):
    model_id = 'amazon.titan-embed-image-v1'  # Updated with the correct model ID
    input_body = {}

    if text_input:
        input_body['inputText'] = text_input  # Correct key for text input

    if image_input:
        # Encode the image in base64
        encoded_image = base64.b64encode(image_input).decode('utf-8')
        input_body['inputImage'] = encoded_image  # Correct key for image input

    # Convert the input body to JSON string
    body_str = json.dumps(input_body)

    max_retries = 5
    backoff_factor = 1  # Base backoff time in seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                accept='application/json',
                contentType='application/json',
                body=body_str
            )
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            return embedding
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                wait_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 1)
                st.warning(f"Request throttled. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    st.error("Maximum retries exceeded. Please try again later.")
    return None

# Function to generate responses using Amazon's text generation model
def generate_text(prompt):
    model_id = 'amazon.titan-tg1-large'  # Confirmed as the correct model ID
    input_body = {
        'inputText': prompt
    }
    body_str = json.dumps(input_body)
    
    max_retries = 5
    backoff_factor = 1  # Base backoff time in seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                accept='application/json',
                contentType='application/json',
                body=body_str
            )
            response_body = json.loads(response['body'].read())
            generated_text = response_body.get('results', [{}])[0].get('outputText', '')
            return generated_text
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                wait_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 1)
                st.warning(f"Request throttled. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    st.error("Maximum retries exceeded. Please try again later.")
    return None

def main():
    st.title("Multimodal Chatbot with Amazon Bedrock")
    st.write("Upload an image and ask questions about it.")

    # Upload image using Streamlit's file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Processing the image...")

        # Read image bytes
        image_bytes = uploaded_file.read()

        # Get embeddings for the image
        try:
            image_embedding = get_embeddings(image_input=image_bytes)
        except Exception as e:
            st.error(f"Failed to get embeddings for the image: {str(e)}")
            return

        if not image_embedding:
            st.error("Failed to get embeddings for the image.")
            return

        # Convert the embedding to a NumPy array
        image_embedding_vector = np.array(image_embedding).astype('float32')

        st.success("Image processed successfully. You can now ask questions about the image.")

        # Initialize or retrieve the chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Input for user's question
        user_input = st.text_input("Your question:", key='user_input')

        if user_input:
            # Get embeddings for the user input
            try:
                text_embedding = get_embeddings(text_input=user_input)
            except Exception as e:
                st.error(f"Failed to get embeddings for the question: {str(e)}")
                return

            if not text_embedding:
                st.error("I'm sorry, I couldn't process your question.")
            else:
                text_embedding_vector = np.array(text_embedding).astype('float32')

                # Combine the image and question embeddings (placeholder)
                combined_embedding = np.concatenate((image_embedding_vector, text_embedding_vector))

                # Prepare the prompt for text generation
                prompt = (
                    "Based on the image provided and the following question, generate a detailed response.\n"
                    f"Question: {user_input}\n"
                    "Answer:"
                )

                # Generate and display the chatbot's response
                try:
                    response = generate_text(prompt)
                except Exception as e:
                    st.error(f"Failed to generate response: {str(e)}")
                    return

                if response:
                    # Update chat history
                    st.session_state.chat_history.append({"question": user_input, "answer": response})
                    # Clear the input box after submitting
                    st.experimental_rerun()
                else:
                    st.error("Failed to generate a response after retries.")

        # Display the chat history
        if st.session_state.get('chat_history'):
            st.write("### Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**You:** {chat['question']}")
                st.write(f"**Bot:** {chat['answer']}")

    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
