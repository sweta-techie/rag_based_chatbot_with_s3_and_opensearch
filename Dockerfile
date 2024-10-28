# Use a Python base image
FROM python:3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the app code to the container
COPY . /app

# Install necessary packages, including OpenCV and EasyOCR
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "rag_based_amazon_bedrock_multimodal_chatbot.py"]
