import streamlit as st
import boto3
import json
import base64
import numpy as np
import logging

# Initialize the clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
textract = boto3.client('textract', region_name='us-east-1')

# Function to extract text from image using Amazon Textract
def extract_text_from_image(image_bytes):
    response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )
    extracted_text = ''
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text'] + '\n'
    return extracted_text

# Function to generate responses using Amazon's text generation model
def generate_text(prompt):
    model_id = 'amazon.titan-tg1-large'
    input_body = {
        'inputText': prompt,
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'temperature': 0.7,
            'topP': 0.9,
            'stopSequences': []
        }
    }
    body_str = json.dumps(input_body)

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            accept='application/json',
            contentType='application/json',
            body=body_str
        )
        response_body_raw = response['body'].read().decode('utf-8')
        response_body = json.loads(response_body_raw)
        generated_text = response_body.get('results', [{}])[0].get('outputText', '')
        return generated_text
    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return None

def main():
    st.set_page_config("Text-Based Chatbot with Image OCR",layout="wide")
    # st.title("Text-Based Chatbot with Image OCR")
    st.write("Upload an image containing text and ask questions about it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Processing the image...")

        image_bytes = uploaded_file.read()

        with st.spinner('Extracting text from image...'):
            extracted_text = extract_text_from_image(image_bytes)

        if not extracted_text:
            st.error("No text found in the image.")
            return

        st.success("Text extracted from the image.")

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        with st.form(key='question_form', clear_on_submit=True):
            st.write("You can now ask questions about the extracted text.")
            user_input = st.text_input("Your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_input:
            prompt = (
                f"The following text was extracted from an image:\n\n{extracted_text}\n\n"
                f"Question: {user_input}\n"
                "Answer:"
            )

            with st.spinner('Generating response...'):
                response = generate_text(prompt)

            if response:
                st.session_state.chat_history.append({"question": user_input, "answer": response})
            else:
                st.error("Failed to generate a response.")

        if st.session_state.get('chat_history'):
            st.write("### Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**You:** {chat['question']}")
                st.markdown(f"**Bot:**\n\n{chat['answer']}")
    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
