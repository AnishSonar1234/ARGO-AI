"""
Compatibility shim - forwards to app.nlp.parser
"""
from app.nlp.parser import (
    OceanQueryIntent,
    OCEAN_REGIONS_BBOX,
    get_region_bbox,
    parse_ocean_query_nlp,
)

__all__ = [
    "OceanQueryIntent",
    "OCEAN_REGIONS_BBOX",
    "get_region_bbox",
    "parse_ocean_query_nlp",
]
