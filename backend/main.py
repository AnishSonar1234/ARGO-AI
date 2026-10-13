"""
ARGO-AI Platform - Backend Service Entry Point
Streamlined Flask router delegating to app.agent, app.rag, and app.tools.
"""
import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from app.agent import ocean_agent
from app.rag.vectorstore import rag_store
from app.rag.erddap_retriever import (
    get_realtime_dashboard_data,
    get_float_profile,
    fetch_real_time_argo_data,
    fetch_argo_float_locations,
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return api_key


def call_openrouter_chat(messages, max_tokens=None, temperature=None) -> dict:
    """
    Call OpenRouter chat completions API preserving reasoning_details.
    """
    api_key = get_openrouter_api_key()
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://argo-ai.local",
        "X-Title": "ARGO-AI Platform"
    }

    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True}
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        url=OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        error_msg = response.text
        try:
            err_json = response.json()
            if "error" in err_json:
                error_msg = err_json["error"].get("message", error_msg)
        except Exception:
            pass
        print(f"[OpenRouter] API error ({response.status_code}): {error_msg}")
        raise RuntimeError(f"OpenRouter API error ({response.status_code}): {error_msg}")

    res_json = response.json()
    if "error" in res_json:
        err_detail = res_json["error"].get("message", str(res_json["error"]))
        raise RuntimeError(f"OpenRouter upstream error: {err_detail}")

    choices = res_json.get("choices")
    if not choices or len(choices) == 0:
        raise RuntimeError("OpenRouter returned empty response")

    return choices[0]["message"]


# ============================================================================
# API Routes
# ============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "ARGO-AI Backend"})


@app.post("/chat")
def chat():
    """Main conversational endpoint with NLP parsing, ChromaDB RAG, and live data reasoning"""
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        result = ocean_agent.process_query(
            user_message=user_message,
            history=history,
            call_llm_func=call_openrouter_chat,
        )
        return jsonify(result)

    except RuntimeError as e:
        return jsonify({"error": "API configuration error: " + str(e)}), 500
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "User not found" in error_msg:
            return jsonify({"error": "Invalid API key. Please check your OpenRouter API key."}), 401
        return jsonify({"error": "Server error: " + error_msg}), 500


@app.get("/api/rag/status")
def rag_status():
    """Returns ChromaDB and RAG vector store statistics"""
    return jsonify({
        "status": "ready",
        "vector_store": "ChromaDB",
        "collection": "argo_ocean_knowledge",
        "documents_indexed": rag_store.count(),
        "framework": "LangChain",
        "nlp_parser": "Pydantic & Rule-Based Intent Extraction"
    })


@app.get("/api/dashboard/overview")
def dashboard_overview():
    """Returns comprehensive real-time ARGO dashboard telemetry"""
    force_refresh = request.args.get("refresh", "false").lower() in ("true", "1", "yes")
    return jsonify(get_realtime_dashboard_data(force_refresh=force_refresh))


@app.get("/api/dashboard/floats")
def dashboard_floats():
    """Returns active ARGO float platforms with optional region filter"""
    region = request.args.get("region", "").lower()
    data = get_realtime_dashboard_data()
    floats = data.get("activeFloats", [])
    if region:
        floats = [f for f in floats if region in f.get("region", "").lower()]
    return jsonify({"floats": floats, "count": len(floats)})


@app.get("/api/dashboard/profile")
def dashboard_profile():
    """Returns specific vertical CTD depth profile for a selected float ID"""
    float_id = request.args.get("float_id")
    return jsonify(get_float_profile(float_id))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)