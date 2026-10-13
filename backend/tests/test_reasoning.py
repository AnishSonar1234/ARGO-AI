import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API")
model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

if not api_key:
    print("Error: OPENROUTER_API_KEY is not set in .env")
    exit(1)

print(f"Testing OpenRouter API with model: {model}")

# Step 1: First API call with reasoning
print("\n--- Step 1: Sending first question ---")
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'?"
            }
        ],
        "reasoning": {"enabled": True}
    })
)

if response.status_code != 200:
    print(f"Error {response.status_code}: {response.text}")
    exit(1)

resp_json = response.json()
first_message = resp_json['choices'][0]['message']
print(f"Assistant Content:\n{first_message.get('content')}")
print(f"Reasoning Details present: {'Yes' if first_message.get('reasoning_details') is not None else 'No'}")

# Step 2: Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": first_message.get('content'),
        "reasoning_details": first_message.get('reasoning_details')  # Pass back unmodified
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

# Step 3: Second API call - model continues reasoning from where it left off
print("\n--- Step 2: Follow-up question with preserved reasoning_details ---")
response2 = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True}
    })
)

if response2.status_code != 200:
    print(f"Error {response2.status_code}: {response2.text}")
    exit(1)

resp2_json = response2.json()
second_message = resp2_json['choices'][0]['message']
print(f"Assistant Content:\n{second_message.get('content')}")
print(f"Reasoning Details present: {'Yes' if second_message.get('reasoning_details') is not None else 'No'}")
print("\nSuccess: Both reasoning turns completed successfully!")
