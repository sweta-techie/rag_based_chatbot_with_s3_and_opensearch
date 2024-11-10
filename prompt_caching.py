# Prompt Caching with Anthropic

# Install Required Libraries
# !pip install anthropic -q

# Import Libraries
import anthropic
import base64

# Load Claude API Key
from google.colab import userdata
CLAUDE_API_KEY = userdata.get('CLAUDE_API_KEY')

# Upload and Load PDF
def upload_pdf(path_to_pdf):
    with open(path_to_pdf, "rb") as pdf_file:
        binary_data = pdf_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
    return base64_string

# Provide path to your PDF file
path_to_pdf = 'BlackRock_investment-directions-q4-24-np-1-13.pdf'
pdf_data = upload_pdf(path_to_pdf)

# Define Completion Method with Prompt Caching Feature
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
MODEL_NAME = "claude-3-5-sonnet-20241022"

def get_completion(messages, model=MODEL_NAME):
    completion = client.beta.messages.create(
        betas=["pdfs-2024-09-25", "prompt-caching-2024-07-31"],
        model=model,
        max_tokens=8192,
        messages=messages,
        temperature=0,
    )
    return completion

# Build Message with Prompt Caching Enabled
def build_message_prompt_caching(query, pdf_data):
    messages = [
        {
            "role": 'user',
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": query}
            ]
        }
    ]
    return messages

# Send Queries with Caching
queries = [
    "What are the main investment themes discussed in the Q4 2024 outlook?",
    "What are the key takeaways from BlackRock's investment strategies for Q4 2024?",
    "Which sectors are expected to benefit most from the AI build-out, according to BlackRock?",
    "How might the upcoming U.S. presidential election impact investment strategies?",
    "What are BlackRock's views on the impact of trade policies and economic fragmentation on inflation?",
    "How does BlackRock suggest positioning a portfolio to mitigate geopolitical risks?"
]

# Send queries with prompt caching
for idx, query in enumerate(queries[:1]):
    print(f"Query n° {idx+1}")
    messages = build_message_prompt_caching(query, pdf_data)
    completion = get_completion(messages)
    print("\n--------ANSWER---------\n")
    print(completion.content[0].text)
    print("\n--------TOKENS COUNT---------\n")
    print(completion.usage)
    print("\n--------STOP REASON---------\n")
    print(completion.stop_reason)

# Compare Costs with and without Caching
# Iterate through remaining queries
for idx, query in enumerate(queries[1:]):
    print(f"Query n° {idx+2}")
    messages = build_message_prompt_caching(query, pdf_data)
    completion = get_completion(messages)
    print("\n--------ANSWER---------\n")
    print(completion.content[0].text)
    print("\n--------TOKENS COUNT---------\n")
    print(completion.usage)
    print("\n--------STOP REASON---------\n")
    print(completion.stop_reason)
