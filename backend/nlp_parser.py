import re
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class OceanQueryIntent(BaseModel):
    intent: Literal["graph_analytics", "map_visualization", "general_qa"] = Field(
        default="general_qa",
        description="The primary visualization or answer intent of the user."
    )
    region: Optional[str] = Field(
        default=None,
        description="The geographic ocean region (e.g. Pacific, North Atlantic, Indian Ocean, Arctic, Southern Ocean)."
    )
    variables: List[str] = Field(
        default_factory=list,
        description="Target oceanographic variables (temperature, salinity, depth, pressure, density)."
    )
    depth_range: Optional[List[float]] = Field(
        default=None,
        description="Depth range in meters, e.g. [0, 2000]."
    )
    float_id: Optional[str] = Field(
        default=None,
        description="Specific ARGO float WMO ID / platform number if mentioned."
    )


# Bounding boxes for ERDDAP / Geocoding
OCEAN_REGIONS_BBOX = {
    "north atlantic": {"lat_min": 5.0, "lat_max": 65.0, "lon_min": -80.0, "lon_max": 0.0},
    "south atlantic": {"lat_min": -60.0, "lat_max": 0.0, "lon_min": -70.0, "lon_max": 20.0},
    "atlantic": {"lat_min": -50.0, "lat_max": 65.0, "lon_min": -80.0, "lon_max": 20.0},
    "north pacific": {"lat_min": 10.0, "lat_max": 60.0, "lon_min": 120.0, "lon_max": -110.0},
    "south pacific": {"lat_min": -60.0, "lat_max": 10.0, "lon_min": 140.0, "lon_max": -70.0},
    "pacific": {"lat_min": -60.0, "lat_max": 60.0, "lon_min": 120.0, "lon_max": -70.0},
    "indian ocean": {"lat_min": -50.0, "lat_max": 25.0, "lon_min": 30.0, "lon_max": 115.0},
    "indian": {"lat_min": -50.0, "lat_max": 25.0, "lon_min": 30.0, "lon_max": 115.0},
    "arabian sea": {"lat_min": 8.0, "lat_max": 26.0, "lon_min": 50.0, "lon_max": 78.0},
    "bay of bengal": {"lat_min": 5.0, "lat_max": 22.0, "lon_min": 80.0, "lon_max": 95.0},
    "arctic ocean": {"lat_min": 65.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "arctic": {"lat_min": 65.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "southern ocean": {"lat_min": -80.0, "lat_max": -50.0, "lon_min": -180.0, "lon_max": 180.0},
    "southern": {"lat_min": -80.0, "lat_max": -50.0, "lon_min": -180.0, "lon_max": 180.0},
    "mediterranean": {"lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 36.0}
}


def parse_ocean_query_nlp(query: str) -> OceanQueryIntent:
    """
    NLP Intent & Entity Parser for Oceanographic Queries.
    Combines rule-based semantic analysis with named-entity extraction.
    """
    text = query.lower().strip()
    
    # 1. Intent Detection
    intent: Literal["graph_analytics", "map_visualization", "general_qa"] = "general_qa"
    
    map_triggers = [
        "where are", "map", "location", "position", "locate", "coordinates", 
        "float location", "tracking", "where is", "geographic", "geographic distribution"
    ]
    graph_triggers = [
        "graph", "plot", "chart", "profile", "depth profile", "temperature vs", 
        "salinity vs", "trend", "histogram", "time series", "bathymetry chart", 
        "correlation", "visualize temperature", "visualize salinity"
    ]
    
    has_map = any(trigger in text for trigger in map_triggers)
    has_graph = any(trigger in text for trigger in graph_triggers)
    
    if has_graph and not (has_map and ("where" in text or "location" in text)):
        intent = "graph_analytics"
    elif has_map:
        intent = "map_visualization"
    
    # 2. Region Extraction
    detected_region = None
    for region_name in OCEAN_REGIONS_BBOX.keys():
        if region_name in text:
            detected_region = region_name.title()
            break
            
    # 3. Variable Extraction
    vars_found = []
    if "temp" in text or "temperature" in text or "heat" in text or "thermal" in text:
        vars_found.append("temperature")
    if "salin" in text or "salt" in text or "halocline" in text or "psu" in text:
        vars_found.append("salinity")
    if "depth" in text or "pressure" in text or "dbar" in text or "meter" in text or "bathymetry" in text:
        vars_found.append("depth")
    if "oxygen" in text or "do" in text or "bgc" in text:
        vars_found.append("oxygen")
        
    # 4. Depth Extraction (e.g. 2000m or 0-1000m)
    depth_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:m|meter|meters|dbar)', text)
    depth_range = None
    if depth_match:
        depth_range = [float(depth_match.group(1)), float(depth_match.group(2))]
    else:
        single_depth = re.search(r'(\d+)\s*(?:m|meter|meters|dbar)', text)
        if single_depth:
            depth_range = [0.0, float(single_depth.group(1))]

    # 5. Float ID extraction (e.g. float 6902745 or WMO 5903421)
    float_match = re.search(r'(?:float|wmo|platform)\s*(?:#|no|id)?\s*(\d{5,7})', text)
    float_id = float_match.group(1) if float_match else None

    return OceanQueryIntent(
        intent=intent,
        region=detected_region,
        variables=vars_found,
        depth_range=depth_range,
        float_id=float_id
    )


def get_region_bbox(region_name: Optional[str]) -> Optional[Dict[str, float]]:
    """Returns the geographic bounding box for an ocean region name."""
    if not region_name:
        return None
    key = region_name.lower().strip()
    return OCEAN_REGIONS_BBOX.get(key)
