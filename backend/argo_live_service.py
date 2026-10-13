"""
Compatibility shim - forwards to app.rag.erddap_retriever
"""
from app.rag.erddap_retriever import (
    REAL_ARGO_PLATFORMS,
    OCEAN_REGIONS,
    generate_realistic_profile,
    get_realtime_dashboard_data,
    get_float_profile,
    fetch_real_time_argo_data,
    fetch_argo_float_locations,
)

__all__ = [
    "REAL_ARGO_PLATFORMS",
    "OCEAN_REGIONS",
    "generate_realistic_profile",
    "get_realtime_dashboard_data",
    "get_float_profile",
    "fetch_real_time_argo_data",
    "fetch_argo_float_locations",
]
