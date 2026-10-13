"""
ARGO-AI LangChain Agent - app/agent.py
Orchestration entry point: NLP -> RAG (ChromaDB) -> Live Data -> LLM -> Visualization tools.
"""
import pandas as pd
from typing import Dict, Any, List, Optional

from app.nlp.parser import parse_ocean_query_nlp, get_region_bbox, OceanQueryIntent
from app.rag.vectorstore import rag_store
from app.tools.map_tool import create_map
from app.tools.chart_tool import create_graph


class OceanLangChainAgent:
    """
    Orchestration Agent combining LangChain patterns, ChromaDB RAG,
    NLP Intent Recognition, and OpenRouter Multi-Turn Reasoning.
    """
    def __init__(self):
        self.rag = rag_store

    def process_query(
        self,
        user_message: str,
        history: List[Dict[str, Any]],
        call_llm_func,
        fetch_locations_func,
        fetch_data_func,
        # Optional overrides allow main.py to pass custom tools if needed
        create_map_func=None,
        create_graph_func=None,
    ) -> Dict[str, Any]:
        """
        Executes the full NLP -> RAG -> Tool/Data -> LLM Reasoning pipeline.
        """
        _create_map = create_map_func or create_map
        _create_graph = create_graph_func or create_graph

        # 1. NLP Intent & Entity Extraction
        parsed: OceanQueryIntent = parse_ocean_query_nlp(user_message)
        print(f"[NLP] intent={parsed.intent}, region={parsed.region}, vars={parsed.variables}")

        # 2. ChromaDB RAG Retrieval
        rag_context = self.rag.format_context_for_prompt(user_message, n_results=2)
        print(f"[RAG] Retrieved ChromaDB context ({len(rag_context)} chars)")

        # 3. Live Data Retrieval (only when map/graph or region is detected)
        live_data_summary = ""
        live_df: Optional[pd.DataFrame] = None

        if parsed.intent in ("map_visualization", "graph_analytics") or parsed.region:
            try:
                if parsed.intent == "map_visualization":
                    live_df = fetch_locations_func(parsed.region)
                else:
                    live_df = fetch_data_func(parsed.region, days_back=7)

                if live_df is not None and not live_df.empty:
                    sample_rows = min(len(live_df), 10)
                    lines = []
                    for _, row in live_df.head(sample_rows).iterrows():
                        lat = row.get("latitude", 0.0)
                        lon = row.get("longitude", 0.0)
                        temp = row.get("temperature", 0.0)
                        sal = row.get("salinity", 0.0)
                        depth = row.get("depth", 0.0)
                        fid = row.get("float_id", "UNKNOWN")
                        lines.append(
                            f"Float {fid}: Lat {lat:.2f}, Lon {lon:.2f}, "
                            f"Depth {depth:.1f}m, Temp {temp:.2f}°C, Salinity {sal:.2f} PSU"
                        )
                    live_data_summary = "\n".join(lines)
            except Exception as data_err:
                print(f"[Data] Notice fetching live data: {data_err}")

        # 4. Build Augmented Prompt (RAG knowledge + real float observations)
        system_instruction = (
            "You are ARGO-AI, an expert oceanographic intelligence assistant powered by global ARGO float data.\n"
            "Use the provided verified domain knowledge and real-time float measurements to provide accurate, "
            "scientifically rigorous, and concise insights."
        )

        augmented_parts = []
        if rag_context:
            augmented_parts.append(f"=== Verified Oceanographic Knowledge (ChromaDB Vector Store) ===\n{rag_context}")
        if live_data_summary:
            augmented_parts.append(f"=== Live ARGO Float Observations (NOAA ERDDAP) ===\n{live_data_summary}")
        augmented_parts.append(f"User Query: {user_message}")
        final_user_prompt = "\n\n".join(augmented_parts)

        # 5. Format multi-turn conversation messages
        messages = [{"role": "system", "content": system_instruction}]
        for msg in history:
            role = msg.get("role")
            if role in ("user", "system"):
                messages.append({"role": role, "content": msg.get("content", "")})
            elif role == "assistant":
                asst = {"role": "assistant", "content": msg.get("content", "")}
                if msg.get("reasoning_details") is not None:
                    asst["reasoning_details"] = msg.get("reasoning_details")
                messages.append(asst)
        messages.append({"role": "user", "content": final_user_prompt})

        # 6. LLM Reasoning
        llm_response = call_llm_func(messages)
        ai_reply = llm_response.get("content") or ""
        reasoning_details = llm_response.get("reasoning_details")

        # 7. Execute Visualization Tools Based on NLP Intent
        map_html = None
        has_map = False
        graph_img = None
        has_graph = False

        if parsed.intent == "map_visualization":
            try:
                if live_df is None or live_df.empty:
                    live_df = fetch_locations_func(parsed.region)
                map_html = _create_map(user_message, live_df, ai_reply)
                has_map = True
            except Exception as map_err:
                print(f"[MapTool] Error: {map_err}")

        elif parsed.intent == "graph_analytics":
            try:
                if live_df is None or live_df.empty:
                    live_df = fetch_data_func(parsed.region, days_back=7)
                graph_img = _create_graph(user_message, live_df, ai_reply)
                has_graph = True
            except Exception as graph_err:
                print(f"[ChartTool] Error: {graph_err}")

        return {
            "reply": ai_reply,
            "reasoning_details": reasoning_details,
            "has_map": has_map,
            "map": map_html,
            "has_graph": has_graph,
            "graph": graph_img,
            "intent": parsed.intent,
            "rag_context_used": bool(rag_context),
            "assistant_message": {
                "role": "assistant",
                "content": ai_reply,
                "reasoning_details": reasoning_details,
            },
        }


# Singleton instance consumed by main.py
ocean_agent = OceanLangChainAgent()
