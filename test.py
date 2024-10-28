import boto3
from langchain_community.llms import Bedrock

def test_bedrock():
    try:
        # Initialize the Bedrock client
        bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
        
        # Initialize the Bedrock LLM
        llm = Bedrock(
            model_id="amazon.titan-text-express-v1",
            client=bedrock_client,
            model_kwargs={
                'maxTokenCount': 512,
                'temperature': 0.5,
                'topP': 0.9
            }
        )
        
        # Test the LLM with a simple prompt
        response = llm("What is the capital of France?")
        print("Bedrock LLM Response:", response)
    except Exception as e:
        print("Error with Bedrock LLM:", e)

if __name__ == "__main__":
    test_bedrock()
