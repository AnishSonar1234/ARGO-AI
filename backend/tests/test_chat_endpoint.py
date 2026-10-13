import os
import sys
import requests
import json

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

client = app.test_client()

# 1. Test single question
print("--- Test 1: Testing /chat endpoint ---")
res = client.post("/chat", json={
    "message": "How many r's are in the word 'strawberry'?"
})
print("Status code:", res.status_code)
data = res.get_json()
print("Reply:", data.get("reply"))
print("Has reasoning_details:", data.get("reasoning_details") is not None)

# 2. Test multi-turn with reasoning_details preserved
print("\n--- Test 2: Testing /chat follow-up with preserved reasoning_details ---")
history = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": data.get("reply"),
        "reasoning_details": data.get("reasoning_details")
    }
]
res2 = client.post("/chat", json={
    "message": "Are you sure? Think carefully.",
    "history": history
})
print("Status code:", res2.status_code)
data2 = res2.get_json()
print("Reply:", data2.get("reply"))
print("Has reasoning_details:", data2.get("reasoning_details") is not None)
print("\n--- All tests passed! ---")
