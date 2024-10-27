import streamlit as st
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import json
import time
from botocore.exceptions import ClientError

# ---------------------------- Configuration ---------------------------- #

# Replace the following with your specific configurations

# IAM Role ARN with SageMaker permissions
IAM_ROLE = 'arn:aws:iam::141676083140:role/service-role/AmazonSageMaker-ExecutionRole-20230626T114375'

# Hugging Face model ID
HF_MODEL_ID = 'Salesforce/codegen-350M-multi'  # You can choose a different model if preferred

# SageMaker instance type
INSTANCE_TYPE = 'ml.m5.large'  # Choose based on your model's requirements

# SageMaker transformers, pytorch, and python versions
TRANSFORMERS_VERSION = '4.26'
PYTORCH_VERSION = '1.13'
PYTHON_VERSION = 'py39'  # Updated Python version to 'py39'

# SageMaker task
HF_TASK = 'text2sql'  # Define the task your model is performing

# Maximum wait time for the endpoint to be in service (in seconds)
MAX_WAIT_TIME = 1800  # 30 minutes

# ---------------------------------------------------------------------------- #

# Initialize boto3 clients
sagemaker_client = boto3.client('sagemaker')
runtime_client = boto3.client('runtime.sagemaker')

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

def deploy_model():
    """
    Deploys the Hugging Face model to SageMaker and returns the endpoint name.
    """
    st.info("Deploying the Hugging Face model to SageMaker...")
    try:
        # Initialize HuggingFaceModel with env parameters
        huggingface_model = HuggingFaceModel(
            role=IAM_ROLE,
            transformers_version=TRANSFORMERS_VERSION,
            pytorch_version=PYTORCH_VERSION,
            py_version=PYTHON_VERSION,
            env={
                'HF_MODEL_ID': HF_MODEL_ID,  # Specify the Hugging Face model ID
                'HF_TASK': HF_TASK           # Specify the task (e.g., 'text2sql')
            }
        )

        # Define a unique endpoint name
        endpoint_name = f"nl2sql-endpoint-{int(time.time())}"

        # Deploy the model
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name
        )

        st.success(f"Model deployed at endpoint: {predictor.endpoint_name}")
        return predictor.endpoint_name

    except TypeError as te:
        st.error(f"TypeError deploying model: {te}")
        return None
    except ClientError as ce:
        st.error(f"ClientError deploying model: {ce}")
        return None
    except Exception as e:
        st.error(f"Unexpected error deploying model: {e}")
        return None

def wait_for_endpoint(endpoint_name):
    """
    Waits until the SageMaker endpoint is in service.
    """
    st.info(f"Waiting for endpoint '{endpoint_name}' to be in service...")
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            if status == 'InService':
                progress_bar.progress(100)
                status_text.success("Endpoint is in service.")
                break
            elif status == 'Failed':
                progress_bar.progress(0)
                status_text.error(f"Endpoint deployment failed with status: {status}")
                return False
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time > MAX_WAIT_TIME:
                    progress_bar.progress(0)
                    status_text.error("Timeout waiting for endpoint to be in service.")
                    return False
                progress = min(int((elapsed_time / MAX_WAIT_TIME) * 100), 99)
                progress_bar.progress(progress)
                status_text.text(f"Current status: {status}. Waiting...")
                time.sleep(30)  # Wait for 30 seconds before checking again
        except ClientError as e:
            st.error(f"Error checking endpoint status: {e}")
            return False

def invoke_endpoint(endpoint_name, natural_language_query):
    """
    Sends a natural language query to the SageMaker endpoint and returns the SQL query.
    """
    payload = {
        'inputs': natural_language_query
    }

    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read().decode())
        # Depending on the model's response format, adjust the parsing
        if isinstance(result, list) and 'generated_text' in result[0]:
            sql_query = result[0]['generated_text']
        elif 'generated_text' in result:
            sql_query = result['generated_text']
        else:
            sql_query = str(result)

        return sql_query

    except ClientError as e:
        st.error(f"ClientError invoking endpoint: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error invoking endpoint: {e}")
        return None

def delete_endpoint(endpoint_name):
    """
    Deletes the SageMaker endpoint to avoid incurring additional costs.
    """
    st.warning(f"Deleting endpoint '{endpoint_name}'...")
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        st.success(f"Endpoint '{endpoint_name}' deletion initiated.")
    except ClientError as e:
        st.error(f"ClientError deleting endpoint: {e}")
    except Exception as e:
        st.error(f"Unexpected error deleting endpoint: {e}")

def get_existing_endpoints():
    """
    Retrieves a list of existing SageMaker endpoints that match the naming pattern.
    """
    try:
        response = sagemaker_client.list_endpoints(
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        endpoints = response['Endpoints']
        # Filter endpoints that start with 'nl2sql-endpoint-'
        filtered_endpoints = [ep['EndpointName'] for ep in endpoints if ep['EndpointName'].startswith('nl2sql-endpoint-')]
        return filtered_endpoints
    except ClientError as e:
        st.error(f"ClientError retrieving endpoints: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error retrieving endpoints: {e}")
        return []

# ---------------------------- Streamlit UI ---------------------------- #

def main():
    st.set_page_config(page_title="Natural Language to SQL Converter", layout="wide")
    st.title("📝 Natural Language to SQL Converter using AWS SageMaker and Hugging Face")

    menu = ["Deploy Model", "Convert NL to SQL", "Manage Endpoint"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Deploy Model":
        st.header("📦 Deploy Hugging Face Model on SageMaker")
        st.write("Deploy the natural language to SQL model to AWS SageMaker.")

        if st.button("Deploy Model"):
            with st.spinner("Deploying the model. This may take several minutes..."):
                endpoint_name = deploy_model()
                if endpoint_name:
                    # Wait for the endpoint to be in service
                    is_ready = wait_for_endpoint(endpoint_name)
                    if not is_ready:
                        st.error("Endpoint failed to deploy.")
        st.markdown("---")

    elif choice == "Convert NL to SQL":
        st.header("🔍 Convert Natural Language Queries to SQL")
        st.write("Enter your natural language query below and convert it to an SQL statement.")

        # Select existing endpoints
        endpoints = get_existing_endpoints()
        if not endpoints:
            st.warning("No active endpoints found. Please deploy a model first.")
            return

        endpoint_selection = st.selectbox("Select SageMaker Endpoint", endpoints)

        natural_language_query = st.text_area("Enter Natural Language Query", height=150)
        if st.button("Convert to SQL"):
            if not natural_language_query.strip():
                st.error("Please enter a valid natural language query.")
            else:
                with st.spinner("Generating SQL query..."):
                    sql_query = invoke_endpoint(endpoint_selection, natural_language_query)
                    if sql_query:
                        st.success("✅ SQL Query Generated Successfully!")
                        st.code(sql_query, language='sql')
                    else:
                        st.error("Failed to generate SQL query.")

        st.markdown("---")

    elif choice == "Manage Endpoint":
        st.header("🛠️ Manage SageMaker Endpoints")
        st.write("View and delete existing SageMaker endpoints.")

        endpoints = get_existing_endpoints()
        if not endpoints:
            st.warning("No active endpoints found.")
            return

        endpoint_selection = st.selectbox("Select Endpoint to Delete", endpoints)

        if st.button("Delete Selected Endpoint"):
            confirm = st.checkbox("Are you sure you want to delete this endpoint?")
            if confirm:
                delete_endpoint(endpoint_selection)
            else:
                st.info("Endpoint deletion canceled.")

        st.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.info("Developed with ❤️ using Streamlit, AWS SageMaker, and Hugging Face.")

if __name__ == "__main__":
    main()
