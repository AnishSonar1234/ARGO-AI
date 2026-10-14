"""
ARGO-AI Platform - Backend Service Entry Point
Streamlined Flask router delegating to app.agent, app.rag, and app.tools.
"""
import os
import sys
import json
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure backend root is on sys.path for direct module execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables before importing app modules
load_dotenv()

from app.agent import ocean_agent
from app.rag.vectorstore import rag_store
from app.rag.erddap_retriever import (
    get_realtime_dashboard_data,
    get_float_profile,
    fetch_real_time_argo_data,
    fetch_argo_float_locations,
)

app = Flask(__name__)
CORS(app)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3.5-lightning:free",
]


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return api_key


def get_candidate_models() -> list:
    """Return prioritized list of models to try for resilience against provider rate limits (429/503)."""
    env_model = os.getenv("OPENROUTER_MODEL")
    candidates = []
    if env_model:
        candidates.append(env_model.strip())
    for m in FALLBACK_MODELS:
        if m not in candidates:
            candidates.append(m)
    return candidates


def call_openrouter_chat(messages, max_tokens=None, temperature=None) -> dict:
    """
    Call OpenRouter chat completions API with automatic model fallback and exponential backoff.
    Guarantees resilient handling of 429 (Provider returned error) and 503 (Provider overloaded).
    """
    api_key = get_openrouter_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://argo-ai.local",
        "X-Title": "ARGO-AI Platform"
    }

    candidates = get_candidate_models()
    last_error = None

    for idx, model in enumerate(candidates):
        for attempt in range(2):
            payload = {
                "model": model,
                "messages": messages,
                "reasoning": {"enabled": True}
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature

            try:
                print(f"[OpenRouter] Requesting model '{model}' (attempt {attempt + 1}/2)...")
                response = requests.post(
                    url=OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=90
                )

                if response.status_code == 200:
                    res_json = response.json()
                    if "error" in res_json:
                        err_detail = res_json["error"].get("message", str(res_json["error"]))
                        print(f"[OpenRouter] Provider returned body error on {model}: {err_detail}")
                        last_error = err_detail
                        time.sleep(1.0)
                        continue

                    choices = res_json.get("choices")
                    if choices and len(choices) > 0:
                        choice_msg = choices[0]["message"]
                        print(f"[OpenRouter] Successfully generated response with '{model}'")
                        return choice_msg
                    else:
                        print(f"[OpenRouter] Empty choices from {model}")
                        last_error = "Empty response choices"
                        continue

                elif response.status_code in (429, 503, 502, 504):
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        err_json = response.json()
                        if "error" in err_json:
                            error_msg = err_json["error"].get("message", error_msg)
                    except Exception:
                        pass
                    print(f"[OpenRouter] Rate limited / provider overloaded ({response.status_code}) on {model}: {error_msg}")
                    last_error = f"{model} ({response.status_code}): {error_msg}"
                    time.sleep(1.2 * (attempt + 1))
                    continue

                else:
                    error_msg = response.text
                    try:
                        err_json = response.json()
                        if "error" in err_json:
                            error_msg = err_json["error"].get("message", error_msg)
                    except Exception:
                        pass
                    print(f"[OpenRouter] Error ({response.status_code}) on {model}: {error_msg}")
                    last_error = f"{model} ({response.status_code}): {error_msg}"
                    if response.status_code in (401, 403):
                        raise RuntimeError(f"OpenRouter authentication failed: {error_msg}")
                    break

            except requests.exceptions.RequestException as req_err:
                print(f"[OpenRouter] Network exception on {model}: {req_err}")
                last_error = str(req_err)
                time.sleep(1.0)
                continue

    raise RuntimeError(
        f"OpenRouter rate limit or provider error: {last_error}. "
        "The free-tier model capacity is temporarily overloaded. Please try again in a few moments."
    )


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
        err_str = str(e)
        if "401" in err_str or "authentication" in err_str.lower():
            return jsonify({"error": "Invalid API key. Please check your OpenRouter API key."}), 401
        if "429" in err_str or "rate limit" in err_str.lower() or "overloaded" in err_str.lower():
            return jsonify({
                "error": "Whale AI is experiencing high global traffic on OpenRouter free endpoints (Rate Limit 429). The system attempted fallback models. Please wait a few seconds and try again."
            }), 429
        return jsonify({"error": err_str}), 500
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "User not found" in error_msg:
            return jsonify({"error": "Invalid API key. Please check your OpenRouter API key."}), 401
        if "429" in error_msg or "rate limit" in error_msg.lower() or "overloaded" in error_msg.lower():
            return jsonify({
                "error": "Whale AI is experiencing high global traffic on OpenRouter (Rate Limit 429). Please wait a few seconds and try again."
            }), 429
        return jsonify({"error": "Server error: " + error_msg}), 500


@app.get("/api/rag/status")
def rag_status():
    """Returns ChromaDB and RAG vector store statistics"""
    return jsonify({
        "status": "ready",
        "vector_store": "ChromaDB",
        "mode": getattr(rag_store, "mode", "local"),
        "database": os.getenv("CHROMA_DATABASE", "local_sqlite"),
        "collection": "argo_ocean_knowledge",
        "documents_indexed": rag_store.count(),
        "framework": "LangChain & ChromaDB Cloud",
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
    host = os.environ.get("HOST", "0.0.0.0")
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")

    if debug_mode:
        print(f"[ARGO-AI] Starting Flask development server on http://{host}:{port} (debug=True)")
        app.run(host=host, port=port, debug=True)
    else:
        try:
            from waitress import serve
            threads = int(os.environ.get("WAITRESS_THREADS", "8"))
            print("=" * 60)
            print(f"[ARGO-AI] Production WSGI Server (Waitress) active")
            print(f"[ARGO-AI] Listening on http://{host}:{port} with {threads} worker threads")
            print(f"[ARGO-AI] Production telemetry, NLP parsing, and RAG ready")
            print("=" * 60)
            serve(app, host=host, port=port, threads=threads)
        except ImportError:
            print(f"[ARGO-AI] Waitress not found, starting threaded Flask runner on http://{host}:{port}")
            app.run(host=host, port=port, threaded=True)