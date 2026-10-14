import re
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class OceanQueryIntent(BaseModel):
    intent: Literal["graph_analytics", "map_visualization", "general_qa"] = Field(
        default="general_qa",
        description="The primary visualization or answer intent of the user."
    )
    data_type: Literal[
        "time_series", 
        "monthly_averages", 
        "gridded_climatology", 
        "vertical_profile", 
        "current_observations"
    ] = Field(
        default="current_observations",
        description="The specific oceanographic data modality (time series, monthly averages, gridded climatology, vertical profile, or current observations)."
    )
    region: Optional[str] = Field(
        default=None,
        description="The geographic ocean region (e.g. Pacific, North Atlantic, Indian Ocean, Arctic, Southern Ocean, Mediterranean, Red Sea, Coral Sea, etc.)."
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


# Comprehensive bounding boxes for all major oceans, marginal seas, gulfs and polar bodies on Earth
OCEAN_REGIONS_BBOX = {
    # 1. Pacific Ocean Basins
    "north pacific": {"lat_min": 10.0, "lat_max": 65.0, "lon_min": 120.0, "lon_max": -110.0},
    "south pacific": {"lat_min": -60.0, "lat_max": 10.0, "lon_min": 140.0, "lon_max": -70.0},
    "equatorial pacific": {"lat_min": -10.0, "lat_max": 10.0, "lon_min": 130.0, "lon_max": -80.0},
    "tropical pacific": {"lat_min": -15.0, "lat_max": 15.0, "lon_min": 130.0, "lon_max": -80.0},
    "pacific ocean": {"lat_min": -60.0, "lat_max": 65.0, "lon_min": 120.0, "lon_max": -70.0},
    "pacific": {"lat_min": -60.0, "lat_max": 65.0, "lon_min": 120.0, "lon_max": -70.0},
    
    # 2. Atlantic Ocean Basins
    "north atlantic": {"lat_min": 5.0, "lat_max": 65.0, "lon_min": -80.0, "lon_max": 0.0},
    "south atlantic": {"lat_min": -60.0, "lat_max": 5.0, "lon_min": -70.0, "lon_max": 20.0},
    "equatorial atlantic": {"lat_min": -10.0, "lat_max": 10.0, "lon_min": -45.0, "lon_max": 10.0},
    "tropical atlantic": {"lat_min": -15.0, "lat_max": 15.0, "lon_min": -50.0, "lon_max": 12.0},
    "sargasso sea": {"lat_min": 20.0, "lat_max": 35.0, "lon_min": -75.0, "lon_max": -40.0},
    "atlantic ocean": {"lat_min": -60.0, "lat_max": 65.0, "lon_min": -80.0, "lon_max": 20.0},
    "atlantic": {"lat_min": -60.0, "lat_max": 65.0, "lon_min": -80.0, "lon_max": 20.0},
    
    # 3. Indian Ocean Basins
    "north indian ocean": {"lat_min": 0.0, "lat_max": 28.0, "lon_min": 45.0, "lon_max": 100.0},
    "south indian ocean": {"lat_min": -50.0, "lat_max": 0.0, "lon_min": 30.0, "lon_max": 115.0},
    "equatorial indian ocean": {"lat_min": -10.0, "lat_max": 10.0, "lon_min": 50.0, "lon_max": 100.0},
    "central indian ocean": {"lat_min": -25.0, "lat_max": 5.0, "lon_min": 55.0, "lon_max": 90.0},
    "arabian sea": {"lat_min": 8.0, "lat_max": 26.0, "lon_min": 50.0, "lon_max": 78.0},
    "bay of bengal": {"lat_min": 5.0, "lat_max": 23.0, "lon_min": 80.0, "lon_max": 96.0},
    "andaman sea": {"lat_min": 6.0, "lat_max": 16.0, "lon_min": 92.0, "lon_max": 98.0},
    "indian ocean": {"lat_min": -50.0, "lat_max": 26.0, "lon_min": 30.0, "lon_max": 115.0},
    "indian": {"lat_min": -50.0, "lat_max": 26.0, "lon_min": 30.0, "lon_max": 115.0},
    
    # 4. Polar Oceans & Seas
    "southern ocean": {"lat_min": -80.0, "lat_max": -50.0, "lon_min": -180.0, "lon_max": 180.0},
    "antarctic ocean": {"lat_min": -80.0, "lat_max": -50.0, "lon_min": -180.0, "lon_max": 180.0},
    "antarctic": {"lat_min": -80.0, "lat_max": -50.0, "lon_min": -180.0, "lon_max": 180.0},
    "southern": {"lat_min": -80.0, "lat_max": -50.0, "lon_min": -180.0, "lon_max": 180.0},
    "weddell sea": {"lat_min": -78.0, "lat_max": -60.0, "lon_min": -60.0, "lon_max": -15.0},
    "ross sea": {"lat_min": -80.0, "lat_max": -65.0, "lon_min": 160.0, "lon_max": -150.0},
    "amundsen sea": {"lat_min": -75.0, "lat_max": -68.0, "lon_min": -125.0, "lon_max": -100.0},
    "bellingshausen sea": {"lat_min": -75.0, "lat_max": -66.0, "lon_min": -100.0, "lon_max": -70.0},
    "scotia sea": {"lat_min": -62.0, "lat_max": -52.0, "lon_min": -65.0, "lon_max": -30.0},
    
    "arctic ocean": {"lat_min": 65.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "arctic": {"lat_min": 65.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "barents sea": {"lat_min": 68.0, "lat_max": 80.0, "lon_min": 18.0, "lon_max": 65.0},
    "kara sea": {"lat_min": 70.0, "lat_max": 81.0, "lon_min": 55.0, "lon_max": 105.0},
    "laptev sea": {"lat_min": 71.0, "lat_max": 81.0, "lon_min": 105.0, "lon_max": 140.0},
    "beaufort sea": {"lat_min": 69.0, "lat_max": 76.0, "lon_min": -155.0, "lon_max": -125.0},
    "chukchi sea": {"lat_min": 66.0, "lat_max": 76.0, "lon_min": -175.0, "lon_max": -156.0},
    "greenland sea": {"lat_min": 70.0, "lat_max": 82.0, "lon_min": -20.0, "lon_max": 10.0},
    "norwegian sea": {"lat_min": 62.0, "lat_max": 72.0, "lon_min": -5.0, "lon_max": 15.0},
    "labrador sea": {"lat_min": 50.0, "lat_max": 65.0, "lon_min": -65.0, "lon_max": -45.0},
    "irminger sea": {"lat_min": 58.0, "lat_max": 66.0, "lon_min": -42.0, "lon_max": -25.0},
    
    # 5. Mediterranean, Middle East & European Marginal Seas
    "mediterranean sea": {"lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 36.0},
    "mediterranean": {"lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 36.0},
    "levantine sea": {"lat_min": 31.0, "lat_max": 36.5, "lon_min": 28.0, "lon_max": 36.0},
    "aegean sea": {"lat_min": 35.0, "lat_max": 41.0, "lon_min": 23.0, "lon_max": 28.0},
    "ionian sea": {"lat_min": 35.5, "lat_max": 40.5, "lon_min": 15.0, "lon_max": 22.0},
    "adriatic sea": {"lat_min": 40.0, "lat_max": 46.0, "lon_min": 12.0, "lon_max": 20.0},
    "tyrrhenian sea": {"lat_min": 38.0, "lat_max": 44.0, "lon_min": 9.5, "lon_max": 16.0},
    "red sea": {"lat_min": 12.0, "lat_max": 28.5, "lon_min": 32.0, "lon_max": 44.0},
    "gulf of aden": {"lat_min": 10.0, "lat_max": 15.0, "lon_min": 43.0, "lon_max": 53.0},
    "persian gulf": {"lat_min": 24.0, "lat_max": 30.5, "lon_min": 48.0, "lon_max": 57.0},
    "arabian gulf": {"lat_min": 24.0, "lat_max": 30.5, "lon_min": 48.0, "lon_max": 57.0},
    "gulf of oman": {"lat_min": 22.5, "lat_max": 26.0, "lon_min": 56.0, "lon_max": 60.5},
    "black sea": {"lat_min": 40.5, "lat_max": 47.0, "lon_min": 27.0, "lon_max": 42.0},
    "baltic sea": {"lat_min": 53.0, "lat_max": 66.0, "lon_min": 10.0, "lon_max": 30.0},
    "north sea": {"lat_min": 51.0, "lat_max": 62.0, "lon_min": -4.0, "lon_max": 9.0},
    "celtic sea": {"lat_min": 49.0, "lat_max": 52.0, "lon_min": -11.0, "lon_max": -5.0},
    
    # 6. Americas & Caribbean Marginal Seas
    "caribbean sea": {"lat_min": 9.0, "lat_max": 22.0, "lon_min": -88.0, "lon_max": -60.0},
    "caribbean": {"lat_min": 9.0, "lat_max": 22.0, "lon_min": -88.0, "lon_max": -60.0},
    "gulf of mexico": {"lat_min": 18.0, "lat_max": 30.5, "lon_min": -98.0, "lon_max": -80.0},
    "gulf of alaska": {"lat_min": 54.0, "lat_max": 60.0, "lon_min": -155.0, "lon_max": -133.0},
    "baffin bay": {"lat_min": 67.0, "lat_max": 78.0, "lon_min": -75.0, "lon_max": -55.0},
    
    # 7. Indo-Pacific & Australasian Seas
    "coral sea": {"lat_min": -25.0, "lat_max": -10.0, "lon_min": 142.0, "lon_max": 160.0},
    "tasman sea": {"lat_min": -46.0, "lat_max": -30.0, "lon_min": 148.0, "lon_max": 175.0},
    "south china sea": {"lat_min": 3.0, "lat_max": 23.0, "lon_min": 100.0, "lon_max": 121.0},
    "east china sea": {"lat_min": 24.0, "lat_max": 33.0, "lon_min": 117.0, "lon_max": 131.0},
    "philippine sea": {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 122.0, "lon_max": 145.0},
    "yellow sea": {"lat_min": 32.0, "lat_max": 40.0, "lon_min": 119.0, "lon_max": 127.0},
    "sea of japan": {"lat_min": 35.0, "lat_max": 48.0, "lon_min": 128.0, "lon_max": 142.0},
    "east sea": {"lat_min": 35.0, "lat_max": 48.0, "lon_min": 128.0, "lon_max": 142.0},
    "sea of okhotsk": {"lat_min": 44.0, "lat_max": 62.0, "lon_min": 135.0, "lon_max": 162.0},
    "bering sea": {"lat_min": 52.0, "lat_max": 66.0, "lon_min": 162.0, "lon_max": -158.0},
    "java sea": {"lat_min": -7.0, "lat_max": -3.0, "lon_min": 106.0, "lon_max": 118.0},
    "banda sea": {"lat_min": -7.5, "lat_max": -4.0, "lon_min": 125.0, "lon_max": 133.0},
    "celebes sea": {"lat_min": 1.0, "lat_max": 6.0, "lon_min": 118.0, "lon_max": 126.0},
    "timor sea": {"lat_min": -14.0, "lat_max": -9.0, "lon_min": 123.0, "lon_max": 131.0},
    "arafura sea": {"lat_min": -12.0, "lat_max": -6.0, "lon_min": 131.0, "lon_max": 141.0},
    "solomon sea": {"lat_min": -11.0, "lat_max": -5.0, "lon_min": 148.0, "lon_max": 158.0},
    
    # 8. Global Ocean
    "global ocean": {"lat_min": -75.0, "lat_max": 75.0, "lon_min": -180.0, "lon_max": 180.0},
    "world ocean": {"lat_min": -75.0, "lat_max": 75.0, "lon_min": -180.0, "lon_max": 180.0},
    "global": {"lat_min": -75.0, "lat_max": 75.0, "lon_min": -180.0, "lon_max": 180.0}
}


def parse_ocean_query_nlp(query: str) -> OceanQueryIntent:
    """
    NLP Intent & Entity Parser for Oceanographic Queries.
    Extracts visualization intent, data modality (time series, monthly averages, gridded climatology, vertical profile),
    geographic regions across all oceans on Earth, depth scopes, and float platform IDs.
    """
    text = query.lower().strip()
    
    # 1. Modality / Data Type Detection
    data_type: Literal[
        "time_series", "monthly_averages", "gridded_climatology", "vertical_profile", "current_observations"
    ] = "current_observations"

    time_series_triggers = [
        "time series", "timeseries", "over time", "temporal", "trend", "past 30", "past 60", "past 90",
        "past days", "drift history", "progression", "cycle by cycle", "cycle to cycle", "time-series",
        "day by day", "over days", "over months", "drift over", "temporal evolution", "longitudinal"
    ]
    monthly_triggers = [
        "monthly average", "monthly averages", "monthly mean", "by month", "each month", "month by month",
        "month-by-month", "seasonal cycle", "annual cycle", "seasonal variation", "monthly variation",
        "monthly climatology", "seasonal climatology", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october", "november", "december"
    ]
    gridded_triggers = [
        "gridded", "climatology", "rg-argo", "woa", "world ocean atlas", "gridded climatology",
        "gridded data", "objective analysis", "basin climatology", "isas", "standard depths",
        "standard levels", "grid", "gridded field"
    ]
    vertical_triggers = [
        "vertical profile", "full profile", "full vertical profile", "vertical profiles",
        "water column", "ctd profile", "depth profile", "temperature vs depth", "salinity vs depth",
        "thermocline", "halocline", "pycnocline", "0 to 2000", "0-2000", "surface to depth",
        "water column profile", "sound speed profile", "mld", "mixed layer depth", "sounding", "ctd sounding"
    ]

    if any(t in text for t in monthly_triggers):
        data_type = "monthly_averages"
    elif any(t in text for t in time_series_triggers):
        data_type = "time_series"
    elif any(t in text for t in gridded_triggers):
        data_type = "gridded_climatology"
    elif any(t in text for t in vertical_triggers):
        data_type = "vertical_profile"

    # 2. Intent Detection
    intent: Literal["graph_analytics", "map_visualization", "general_qa"] = "general_qa"
    
    map_triggers = [
        "where are", "map", "location", "position", "locate", "coordinates", 
        "float location", "tracking", "where is", "geographic", "geographic distribution",
        "spatial distribution"
    ]
    graph_triggers = [
        "graph", "plot", "chart", "profile", "depth profile", "temperature vs", 
        "salinity vs", "trend", "histogram", "time series", "bathymetry chart", 
        "correlation", "visualize temperature", "visualize salinity", "show data",
        "give data", "data table", "curves", "evolution", "anomaly"
    ]
    
    has_map = any(trigger in text for trigger in map_triggers)
    has_graph = any(trigger in text for trigger in graph_triggers) or data_type in ("time_series", "monthly_averages", "gridded_climatology", "vertical_profile")
    
    if has_graph and not (has_map and ("where" in text or "location" in text)):
        intent = "graph_analytics"
    elif has_map:
        intent = "map_visualization"
    
    # 3. Dynamic Region Extraction Across All Ocean Bodies on Earth
    detected_region = None
    # Check known ocean bodies in order of specificity (longest name first)
    for region_name in sorted(OCEAN_REGIONS_BBOX.keys(), key=len, reverse=True):
        if region_name in text:
            detected_region = region_name.title()
            break
            
    # Dynamic extraction if not in dictionary (matches e.g. "Weddell Sea", "Coral Sea", "Gulf of Alaska")
    if not detected_region:
        match = re.search(
            r'\b(?:in|of|around|near|across|for|from|at)\s+(?:the\s+)?([a-z\s]+?\s+(?:ocean|sea|gulf|bay|basin|strait|channel|passage|trench|gyre|current|shelf))\b',
            text
        )
        if match:
            candidate = match.group(1).strip()
            for prefix in ("the ", "all ", "any ", "our ", "this "):
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix):]
            if len(candidate) > 2:
                detected_region = candidate.title()
        else:
            match_direct = re.search(r'\b([a-z\s]+?\s+(?:ocean|sea|gulf|bay|basin))\b', text)
            if match_direct:
                candidate = match_direct.group(1).strip()
                for prefix in ("the ", "all ", "any ", "our ", "this ", "show ", "give "):
                    if candidate.startswith(prefix):
                        candidate = candidate[len(prefix):]
                if len(candidate) > 2:
                    detected_region = candidate.title()
            
    # 4. Variable Extraction
    vars_found = []
    if "temp" in text or "temperature" in text or "heat" in text or "thermal" in text:
        vars_found.append("temperature")
    if "salin" in text or "salt" in text or "halocline" in text or "psu" in text:
        vars_found.append("salinity")
    if "depth" in text or "pressure" in text or "dbar" in text or "meter" in text or "bathymetry" in text:
        vars_found.append("depth")
    if "oxygen" in text or "do" in text or "bgc" in text:
        vars_found.append("oxygen")
    if "density" in text or "pycnocline" in text or "sigma" in text:
        vars_found.append("density")
        
    # 5. Depth Extraction (e.g. 2000m or 0-1000m)
    depth_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:m|meter|meters|dbar)', text)
    depth_range = None
    if depth_match:
        depth_range = [float(depth_match.group(1)), float(depth_match.group(2))]
    else:
        single_depth = re.search(r'(\d+)\s*(?:m|meter|meters|dbar)', text)
        if single_depth:
            depth_range = [0.0, float(single_depth.group(1))]

    # 6. Float ID extraction (e.g. float 6902745 or WMO 5903421)
    float_match = re.search(r'(?:float|wmo|platform)\s*(?:#|no|id)?\s*(\d{5,7})', text)
    float_id = float_match.group(1) if float_match else None

    return OceanQueryIntent(
        intent=intent,
        data_type=data_type,
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
    if key in OCEAN_REGIONS_BBOX:
        return OCEAN_REGIONS_BBOX[key]
    for k, v in OCEAN_REGIONS_BBOX.items():
        if k in key or key in k:
            return v
    return {"lat_min": -60.0, "lat_max": 60.0, "lon_min": -180.0, "lon_max": 180.0}
