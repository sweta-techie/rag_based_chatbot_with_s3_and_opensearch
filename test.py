import boto3
import json

# Initialize the Bedrock client
bedrock = boto3.client('bedrock', region_name='us-east-1')  # Replace 'us-east-1' with your region if different

def list_models():
    response = bedrock.list_foundation_models()
    models = response.get('modelSummaries', [])
    for model in models:
        print(f"Model ID: {model['modelId']}")
        print(f"Model Name: {model['modelName']}")
        print(f"Provider: {model['providerName']}")
        print("---")

if __name__ == "__main__":
    list_models()
