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

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
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
    model_id: str = "amazon.titan-text-premier-v1:0"  # Ensure this is correct
    temperature: float = 0.7
    top_p: float = 0.9
    max_token_count: int = 3072  # Adjust based on model's max token limit

config = Config()

# Configure Boto3 client with retry settings
boto_config = BotoConfig(
    retries={
        'max_attempts': 10,  # Increased max attempts
        'mode': 'standard'    # Use 'standard' retry mode
    },
    read_timeout=30,
    connect_timeout=30,
    max_pool_connections=10,
    parameter_validation=True
)

client = boto3.client(
    service_name="bedrock-runtime",
    region_name=config.aws_region,
    config=boto_config
)

# Define rate limits: e.g., 2 calls per minute
CALLS = 2
PERIOD = 60  # seconds

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=60),  # Exponential backoff with jitter
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def invoke_bedrock_model():
    """Invoke the Bedrock model with proper error handling and rate limiting."""
    try:
        request_payload = {
            "inputText": "this is where you place your input text",
            "textGenerationConfig": {
                "maxTokenCount": config.max_token_count,
                "stopSequences": [],
                "temperature": config.temperature,
                "topP": config.top_p
            }
        }

        response = client.invoke_model(
            modelId=config.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_payload)
        )
        response_body = response['body'].read().decode('utf-8')
        logger.info("Model Response:")
        logger.info(response_body)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        if error_code == 'ThrottlingException':
            logger.warning(f"ThrottlingException encountered: {error_message}")
            raise  # Trigger retry
        elif error_code == 'ValidationException':
            logger.error(f"ValidationException: {error_message}")
            raise
        else:
            logger.error(f"Unexpected ClientError: {error_message}")
            raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        invoke_bedrock_model()
    except Exception as e:
        logger.error(f"Failed to invoke Bedrock model after retries: {str(e)}")
