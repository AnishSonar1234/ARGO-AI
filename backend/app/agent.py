"""
ARGO-AI LangChain Agent - app/agent.py
Orchestration entry point: NLP -> RAG (ChromaDB) -> Live/Climatology Data -> LLM -> Visualization tools.
Supports 4 specialized oceanographic modalities:
1. Time Series Data (Multi-cycle drift & temporal evolution)
2. Monthly Averages (12-month seasonal climatologies & MLD)
3. Gridded Climatology (WOA / RG-Argo standard depth horizons 0-2000m)
4. Full Vertical Profiles (Continuous CTD stratification & pycnoclines)
"""
import pandas as pd
from typing import Dict, Any, List, Optional

from app.nlp.parser import parse_ocean_query_nlp, get_region_bbox, OceanQueryIntent
from app.rag.vectorstore import rag_store
from app.rag.erddap_retriever import (
    fetch_argo_float_locations,
    fetch_real_time_argo_data,
    get_time_series_data,
    get_monthly_averages_data,
    get_gridded_climatology_data,
    get_full_vertical_profile_data,
)
from app.tools.map_tool import create_map
from app.tools.chart_tool import create_graph


def _format_time_series_table(df: pd.DataFrame) -> str:
    """Formats multi-cycle temporal drift data into a structured markdown table with statistics."""
    if df.empty:
        return "No time-series telemetry records available."
    
    fid = df['float_id'].iloc[0] if 'float_id' in df.columns else "ARGO"
    reg = df['region'].iloc[0] if 'region' in df.columns else "Global"
    
    t_col = 'temperature' if 'temperature' in df.columns else df.columns[0]
    t_min, t_max, t_mean = df[t_col].min(), df[t_col].max(), df[t_col].mean()
    
    s_col = 'salinity' if 'salinity' in df.columns else None
    s_min = df[s_col].min() if s_col else 34.5
    s_max = df[s_col].max() if s_col else 36.5
    
    mld_col = 'mixed_layer_depth' if 'mixed_layer_depth' in df.columns else ('mld' if 'mld' in df.columns else None)
    mld_min = df[mld_col].min() if mld_col else 20.0
    mld_max = df[mld_col].max() if mld_col else 75.0
    
    lines = [
        f"### ARGO Float {fid} Time Series Telemetry ({reg})",
        f"- Observation Cycles: {len(df)} cycles | Time Span: {df['date'].iloc[0]} to {df['date'].iloc[-1]}",
        f"- Surface Temp Range: {t_min:.2f}°C to {t_max:.2f}°C (Mean: {t_mean:.2f}°C)",
        f"- Salinity Range: {s_min:.2f} to {s_max:.2f} PSU",
        f"- Mixed Layer Depth (MLD) Range: {mld_min:.1f}m to {mld_max:.1f}m",
        "",
        "| Cycle | Date | Surface Temp (°C) | Temp Anomaly (°C) | Practical Salinity (PSU) | MLD (m) | Ocean Heat Content (GJ/m²) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, r in df.iterrows():
        cycle = r.get('cycle_number', '-')
        dt = r.get('date', '-')
        t = f"{r.get('temperature', 0.0):.2f}"
        anom = f"{r.get('temperature_anomaly', r.get('temp_anomaly', 0.0)):+.2f}"
        s = f"{r.get('salinity', 0.0):.2f}"
        mld = f"{r.get('mixed_layer_depth', r.get('mld', 0.0)):.1f}"
        ohc = f"{r.get('heat_content', r.get('heat_content_gj', 0.0)):.2f}"
        lines.append(f"| {cycle} | {dt} | {t} | {anom} | {s} | {mld} | {ohc} |")
    
    return "\n".join(lines)


def _format_monthly_averages_table(df: pd.DataFrame) -> str:
    """Formats 12-month climatological averages into a clean markdown table with annual cycle metrics."""
    if df.empty:
        return "No monthly climatological averages available."
    
    reg = df['region'].iloc[0] if 'region' in df.columns else "Global Ocean"
    t_col = 'mean_temperature' if 'mean_temperature' in df.columns else 'mean_temp'
    mld_col = 'mean_mld' if 'mean_mld' in df.columns else 'mld'
    s_col = 'mean_salinity' if 'mean_salinity' in df.columns else 'salinity'
    
    warmest = df.loc[df[t_col].idxmax()]
    coldest = df.loc[df[t_col].idxmin()]
    amp = warmest[t_col] - coldest[t_col]
    deepest_mld = df.loc[df[mld_col].idxmax()]
    
    lines = [
        f"### Climatological 12-Month Seasonal Averages ({reg})",
        f"- Annual SST Amplitude: {amp:.2f}°C (Warmest: {warmest['month_name']} at {warmest[t_col]:.2f}°C | Coldest: {coldest['month_name']} at {coldest[t_col]:.2f}°C)",
        f"- Deepest Mixed Layer: {deepest_mld['month_name']} ({deepest_mld[mld_col]:.1f}m) indicating peak winter convective overturn",
        "",
        "| Month | Month Name | Mean Temp (°C ± 1σ) | Mean Salinity (PSU ± 1σ) | Mixed Layer Depth (m) | Main Thermocline Depth (m) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, r in df.iterrows():
        m_num = r.get('month', '-')
        m_name = r.get('month_name', '-')
        t_val = r.get('mean_temperature', r.get('mean_temp', 0.0))
        t_err = r.get('temp_std', 0.4)
        s_val = r.get('mean_salinity', r.get('mean_sal', 35.0))
        s_err = r.get('sal_std', r.get('salinity_std', 0.1))
        mld_val = r.get('mean_mld', r.get('mld', 45.0))
        th_val = r.get('thermocline_depth', 110.0)
        lines.append(f"| {m_num} | {m_name} | {t_val:.2f} ± {t_err:.2f} | {s_val:.2f} ± {s_err:.2f} | {mld_val:.1f} | {th_val:.1f} |")
    
    return "\n".join(lines)


def _format_gridded_climatology_table(df: pd.DataFrame) -> str:
    """Formats objectively analyzed gridded climatology depth levels into a markdown table."""
    if df.empty:
        return "No gridded climatology levels available."
    
    product = df['product'].iloc[0] if 'product' in df.columns else "RG-Argo / WOA"
    reg = df['region'].iloc[0] if 'region' in df.columns else "Standard Ocean Basin"
    
    lines = [
        f"### Gridded Ocean Climatology ({product} - {reg})",
        f"- Standard Depth Horizons: 0m to 2000m (Objectively mapped isobaric levels)",
        f"- Surface Salinity: {df['climatological_salinity'].iloc[0]:.2f} PSU | Deep Abyssal Temp (2000m): {df['climatological_temp'].iloc[-1]:.2f}°C",
        "",
        "| Depth (m) | Temp Climatology (°C ± 1σ) | Salinity Climatology (PSU ± 1σ) | Potential Density σθ (kg/m³) | Oceanographic Water Mass Layer |",
        "| :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, r in df.iterrows():
        d = f"{r.get('depth', 0):.0f}"
        t = f"{r.get('climatological_temp', 0.0):.2f} ± {r.get('temp_std', 0.0):.2f}"
        s = f"{r.get('climatological_salinity', 0.0):.2f} ± {r.get('salinity_std', 0.0):.2f}"
        rho = f"{r.get('potential_density', 0.0):.2f}"
        wm = r.get('water_mass', 'Water Column')
        lines.append(f"| {d} | {t} | {s} | {rho} | {wm} |")
    
    return "\n".join(lines)


def _format_vertical_profile_table(df: pd.DataFrame) -> str:
    """Formats full continuous CTD vertical soundings (0m to 2000m) into a markdown table."""
    if df.empty:
        return "No vertical profile sounding data available."
    
    fid = df['float_id'].iloc[0] if 'float_id' in df.columns else "ARGO CTD"
    reg = df['region'].iloc[0] if 'region' in df.columns else "Open Ocean"
    s_temp = df['temperature'].iloc[0]
    b_temp = df['temperature'].iloc[-1]
    min_sound = df.loc[df['sound_speed'].idxmin()] if 'sound_speed' in df.columns else None
    
    lines = [
        f"### High-Resolution Vertical Profile Sounding (Float {fid} - {reg})",
        f"- Depth Range: {df['depth'].min():.0f}m to {df['depth'].max():.0f}m (Pressure: {df['pressure_dbar'].max():.0f} dbar)",
        f"- Thermal Gradient: {s_temp:.2f}°C (Surface) down to {b_temp:.2f}°C (Abyssal)",
    ]
    if min_sound is not None:
        lines.append(f"- Acoustic SOFAR Axis: Minimum Sound Speed {min_sound['sound_speed']:.1f} m/s at {min_sound['depth']:.0f}m depth")
    
    lines.extend([
        "",
        "| Depth (m) | Pressure (dbar) | In-Situ Temp (°C) | Salinity (PSU) | Density σθ (kg/m³) | Sound Speed (m/s) | Stratification Zone |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])
    for _, r in df.iterrows():
        d = f"{r.get('depth', 0):.0f}"
        p = f"{r.get('pressure_dbar', 0):.0f}"
        t = f"{r.get('temperature', 0.0):.2f}"
        s = f"{r.get('salinity', 0.0):.2f}"
        rho = f"{r.get('density_sigma', 0.0):.2f}"
        c = f"{r.get('sound_speed', 0.0):.1f}"
        layer = r.get('layer', 'Water Column')
        lines.append(f"| {d} | {p} | {t} | {s} | {rho} | {c} | {layer} |")
    
    return "\n".join(lines)


def _format_current_observations_table(df: pd.DataFrame) -> str:
    """Formats current float observations into a readable summary."""
    if df.empty:
        return "No recent float observation records available."
    sample_rows = min(len(df), 12)
    lines = [
        "### Recent In-Situ Float Telemetry Records",
        "| Float ID | Latitude | Longitude | Depth (m) | Temperature (°C) | Salinity (PSU) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df.head(sample_rows).iterrows():
        lat = row.get("latitude", 0.0)
        lon = row.get("longitude", 0.0)
        temp = row.get("temperature", 0.0)
        sal = row.get("salinity", 0.0)
        depth = row.get("depth", 0.0)
        fid = row.get("float_id", "UNKNOWN")
        lines.append(f"| {fid} | {lat:.2f}° | {lon:.2f}° | {depth:.1f} | {temp:.2f} | {sal:.2f} |")
    return "\n".join(lines)


class OceanLangChainAgent:
    """
    Orchestration Agent combining LangChain patterns, ChromaDB RAG,
    NLP Intent Recognition, Specialized Oceanographic Data Retrieval,
    and OpenRouter Multi-Turn Reasoning.
    """
    def __init__(self):
        self.rag = rag_store

    def process_query(
        self,
        user_message: str,
        history: List[Dict[str, Any]],
        call_llm_func,
        fetch_locations_func=None,
        fetch_data_func=None,
        create_map_func=None,
        create_graph_func=None,
    ) -> Dict[str, Any]:
        """
        Executes the full NLP -> RAG -> Telemetry Retrieval -> LLM Reasoning -> Chart/Map pipeline.
        Supports Time Series, Monthly Averages, Gridded Climatology, and Full Vertical Profiles.
        """
        _fetch_locations = fetch_locations_func or fetch_argo_float_locations
        _fetch_data = fetch_data_func or fetch_real_time_argo_data
        _create_map = create_map_func or create_map
        _create_graph = create_graph_func or create_graph

        # 1. NLP Intent & Entity & Modality Extraction
        parsed: OceanQueryIntent = parse_ocean_query_nlp(user_message)
        print(f"[NLP] intent={parsed.intent}, data_type={parsed.data_type}, region={parsed.region}, vars={parsed.variables}")

        # 2. ChromaDB RAG Retrieval
        rag_context = self.rag.format_context_for_prompt(user_message, n_results=2)
        print(f"[RAG] Retrieved ChromaDB context ({len(rag_context)} chars)")

        # 3. Retrieve Domain Data Based on Detected Modality
        live_data_summary = ""
        live_df: Optional[pd.DataFrame] = None

        try:
            if parsed.data_type == "time_series":
                float_id = parsed.float_id or "2903334"
                live_df = get_time_series_data(region=parsed.region, float_id=float_id, days=180)
                live_data_summary = _format_time_series_table(live_df)

            elif parsed.data_type == "monthly_averages":
                live_df = get_monthly_averages_data(region=parsed.region, float_id=parsed.float_id)
                live_data_summary = _format_monthly_averages_table(live_df)

            elif parsed.data_type == "gridded_climatology":
                live_df = get_gridded_climatology_data(region=parsed.region)
                live_data_summary = _format_gridded_climatology_table(live_df)

            elif parsed.data_type == "vertical_profile":
                float_id = parsed.float_id or "2903334"
                live_df = get_full_vertical_profile_data(float_id=float_id, region=parsed.region, max_depth=2000.0)
                live_data_summary = _format_vertical_profile_table(live_df)

            elif parsed.intent == "map_visualization":
                live_df = _fetch_locations(parsed.region)
                live_data_summary = _format_current_observations_table(live_df)

            else:
                live_df = _fetch_data(parsed.region, days_back=7)
                live_data_summary = _format_current_observations_table(live_df)

        except Exception as data_err:
            print(f"[Data] Notice fetching domain data ({parsed.data_type}): {data_err}")

        # 4. Build Augmented Prompt with Domain Rules
        modality_instructions = {
            "time_series": (
                "The user requested TIME SERIES DATA. You MUST:\n"
                "1. Reference the quantitative cycle-by-cycle numbers from the telemetry table (exact dates, temperatures, anomalies, MLD, heat content).\n"
                "2. Provide deep physical oceanographic inferences explaining the temporal trends, seasonal heating/cooling rates, mixed layer deepening, and steric height implications.\n"
                "3. Conclude with an oceanographic outlook on the water column stability."
            ),
            "monthly_averages": (
                "The user requested MONTHLY AVERAGES (Climatology). You MUST:\n"
                "1. Explicitly present and analyze the 12-month annual cycle table (temperatures, salinities, standard deviations, and MLDs).\n"
                "2. Explain the physical mechanisms driving the annual cycle (monsoonal reversals, winter convective overturn, summer stratification, upwelling/downwelling dynamics).\n"
                "3. Detail the thermocline depth shoaling and deepening phases across the year."
            ),
            "gridded_climatology": (
                "The user requested GRIDDED CLIMATOLOGY DATA. You MUST:\n"
                "1. Reference the standard depth horizons (0m to 2000m) from objectively analyzed Argo products (RG-Argo / WOA).\n"
                "2. Identify the key oceanographic water masses across depth (Mixed Layer, Central Water, Antarctic Intermediate Water / Red Sea Water, Deep Water).\n"
                "3. Explain the potential density (sigma-theta) stratification and isobaric objective interpolation methodology."
            ),
            "vertical_profile": (
                "The user requested FULL VERTICAL PROFILES (CTD Soundings). You MUST:\n"
                "1. Reference the exact soundings from the surface (0m) to the bathypelagic abyss (2000m), citing exact temperatures, practical salinities, and densities.\n"
                "2. Systematically dissect each stratification zone: Mixed Layer Depth (MLD), Main Thermocline, Halocline, Pycnocline stability barrier, and SOFAR acoustic sound channel axis.\n"
                "3. Provide physical oceanographic conclusions on water column stability and buoyancy frequency."
            ),
            "current_observations": (
                "Provide accurate, scientifically rigorous inferences and insights on the recent ARGO float measurements, "
                "referencing exact float IDs, coordinates, temperatures, and salinities."
            )
        }

        specific_instruction = modality_instructions.get(parsed.data_type, modality_instructions["current_observations"])

        system_instruction = (
            "You are ARGO-AI, an expert oceanographic intelligence assistant powered by global ARGO float data.\n"
            "You provide rigorous scientific inferences, quantitative data analysis, and rich physical explanations.\n\n"
            f"=== Modality Mandate ===\n{specific_instruction}\n\n"
            "Format your response with clean markdown headings, quantitative references to the provided telemetry, "
            "and concise oceanographic takeaways."
        )

        augmented_parts = []
        if rag_context:
            augmented_parts.append(f"=== Verified Oceanographic Knowledge (ChromaDB Vector Store) ===\n{rag_context}")
        if live_data_summary:
            augmented_parts.append(f"=== Oceanographic Telemetry & Climatology Dataset ===\n{live_data_summary}")
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

        # 7. Execute Visualization Tools Based on NLP Intent & Modality
        map_html = None
        has_map = False
        graph_img = None
        has_graph = False

        if parsed.intent == "map_visualization":
            try:
                if live_df is None or live_df.empty:
                    live_df = _fetch_locations(parsed.region)
                map_html = _create_map(user_message, live_df, ai_reply)
                has_map = True
            except Exception as map_err:
                print(f"[MapTool] Error: {map_err}")

        elif parsed.intent == "graph_analytics":
            try:
                if live_df is None or live_df.empty:
                    if parsed.data_type == "time_series":
                        live_df = get_time_series_data(parsed.region, parsed.float_id or "2903334")
                    elif parsed.data_type == "monthly_averages":
                        live_df = get_monthly_averages_data(parsed.region, parsed.float_id)
                    elif parsed.data_type == "gridded_climatology":
                        live_df = get_gridded_climatology_data(parsed.region)
                    elif parsed.data_type == "vertical_profile":
                        live_df = get_full_vertical_profile_data(parsed.float_id or "2903334", parsed.region)
                    else:
                        live_df = _fetch_data(parsed.region, days_back=7)

                graph_img = _create_graph(user_message, live_df, ai_reply, data_type=parsed.data_type)
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
            "data_type": parsed.data_type,
            "data_summary": live_data_summary,
            "rag_context_used": bool(rag_context),
            "assistant_message": {
                "role": "assistant",
                "content": ai_reply,
                "reasoning_details": reasoning_details,
            },
        }


# Singleton instance consumed by main.py
ocean_agent = OceanLangChainAgent()
