# Backend Setup Instructions

## Environment Variables

Create a `.env` file in the backend directory with the following content:

```
# OpenRouter API Configuration
# Get your API key from https://openrouter.ai/
OPENROUTER_API_KEY=your_actual_api_key_here

# Server Configuration
PORT=5000
```

## Getting an OpenRouter API Key

1. Go to [https://openrouter.ai/](https://openrouter.ai/)
2. Sign up for an account
3. Get your API key from the dashboard
4. Replace `your_actual_api_key_here` in the `.env` file with your real API key

## Running the Server

```bash
cd backend
python main.py
```

The server will start on http://localhost:5000

## API Endpoints

- `POST /chat` - Chat with the AI assistant
  - Body: `{"message": "Your question here"}`
  - Response: `{"reply": "AI response"}`

## Troubleshooting

- **401 Error**: Invalid API key - check your OpenRouter API key
- **500 Error**: Server error - check the console for detailed error messages
- **Connection Error**: Make sure the server is running on port 5000
