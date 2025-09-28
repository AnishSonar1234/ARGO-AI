import os
import base64
import io
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)


def create_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def fetch_real_time_argo_data(region=None, days_back=7):
    """Fetch real-time ARGO float data from ERDDAP server"""
    try:
        # ERDDAP server URL for ARGO data
        base_url = "https://polarwatch.noaa.gov/erddap/tabledap/argoFloats.json"
        
        # Calculate date range (last 7 days by default)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Build query parameters
        params = {
            'time>=': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time<=': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'latitude>=': -90,
            'latitude<=': 90,
            'longitude>=': -180,
            'longitude<=': 180
        }
        
        # Add region-specific constraints if specified
        if region:
            if 'pacific' in region.lower():
                params['longitude>='] = -180
                params['longitude<='] = -100
            elif 'atlantic' in region.lower():
                params['longitude>='] = -100
                params['longitude<='] = 20
            elif 'indian' in region.lower():
                params['longitude>='] = 20
                params['longitude<='] = 180
        
        # Make request to ERDDAP
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'table' in data and 'rows' in data['table']:
                # Convert to DataFrame
                columns = [col['name'] for col in data['table']['columnNames']]
                rows = data['table']['rows']
                
                df = pd.DataFrame(rows, columns=columns)
                
                # Clean and process the data
                df = df.dropna(subset=['latitude', 'longitude'])
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                df['temperature'] = pd.to_numeric(df.get('temperature', 0), errors='coerce')
                df['salinity'] = pd.to_numeric(df.get('salinity', 0), errors='coerce')
                df['depth'] = pd.to_numeric(df.get('depth', 0), errors='coerce')
                
                # Add float ID and status
                df['float_id'] = df.get('platform_number', 'UNKNOWN')
                df['status'] = 'active'  # Assume active for real-time data
                df['deployment_date'] = pd.to_datetime(df.get('time', datetime.now()))
                
                # Ensure all required columns exist
                if 'temperature' not in df.columns:
                    df['temperature'] = np.random.normal(15, 5, len(df))
                if 'salinity' not in df.columns:
                    df['salinity'] = np.random.normal(35, 2, len(df))
                if 'depth' not in df.columns:
                    df['depth'] = np.random.uniform(0, 2000, len(df))
                
                return df.dropna()
        
        # Fallback to sample data if real-time fetch fails
        return generate_sample_data("real-time fallback")
        
    except Exception as e:
        print(f"Error fetching real-time ARGO data: {e}")
        # Return sample data as fallback
        return generate_sample_data("real-time fallback")


def fetch_argo_float_locations(region=None):
    """Fetch current ARGO float locations for mapping"""
    try:
        # Use a different endpoint for float locations
        base_url = "https://polarwatch.noaa.gov/erddap/tabledap/argoFloats.json"
        
        # Get data from last 30 days to ensure we have recent locations
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            'time>=': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time<=': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'latitude>=': -90,
            'latitude<=': 90,
            'longitude>=': -180,
            'longitude<=': 180
        }
        
        # Add region constraints
        if region:
            if 'pacific' in region.lower():
                params['longitude>='] = -180
                params['longitude<='] = -100
            elif 'atlantic' in region.lower():
                params['longitude>='] = -100
                params['longitude<='] = 20
            elif 'indian' in region.lower():
                params['longitude>='] = 20
                params['longitude<='] = 180
        
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'table' in data and 'rows' in data['table']:
                columns = [col['name'] for col in data['table']['columnNames']]
                rows = data['table']['rows']
                
                df = pd.DataFrame(rows, columns=columns)
                
                # Process location data
                df = df.dropna(subset=['latitude', 'longitude'])
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                
                # Get unique float locations (latest position for each float)
                df = df.groupby('platform_number').agg({
                    'latitude': 'last',
                    'longitude': 'last',
                    'time': 'last'
                }).reset_index()
                
                # Add float metadata
                df['float_id'] = df['platform_number']
                df['status'] = 'active'
                df['deployment_date'] = pd.to_datetime(df['time'])
                df['temperature'] = np.random.normal(15, 5, len(df))  # Placeholder
                df['salinity'] = np.random.normal(35, 2, len(df))  # Placeholder
                df['depth'] = np.random.uniform(0, 2000, len(df))  # Placeholder
                
                # Ensure all required columns exist with proper data types
                required_columns = ['latitude', 'longitude', 'float_id', 'status', 'deployment_date', 'temperature', 'salinity', 'depth']
                for col in required_columns:
                    if col not in df.columns:
                        if col == 'status':
                            df[col] = 'active'
                        elif col == 'float_id':
                            df[col] = df.get('platform_number', 'UNKNOWN')
                        elif col == 'deployment_date':
                            df[col] = pd.to_datetime(df.get('time', datetime.now()))
                        else:
                            df[col] = 0
                
                return df.dropna()
        
        # Fallback to sample data
        return generate_sample_data("location fallback")
        
    except Exception as e:
        print(f"Error fetching ARGO float locations: {e}")
        return generate_sample_data("location fallback")


def is_graph_query(message: str) -> bool:
    """Detect if the query is asking for a graph/plot/visualization"""
    graph_keywords = [
        'graph', 'plot', 'chart', 'visualize', 'show', 'display', 'trend', 
        'line graph', 'bar chart', 'scatter plot', 'histogram', 'heatmap',
        'time series', 'over time', 'comparison', 'distribution', 'correlation'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in graph_keywords)


def is_map_query(message: str) -> bool:
    """Detect if the query is asking for a map/location visualization"""
    map_keywords = [
        'map', 'location', 'where', 'position', 'coordinates', 'latitude', 'longitude',
        'argo float', 'float location', 'deployment', 'tracking', 'geographic',
        'ocean map', 'world map', 'global', 'regional', 'near', 'around'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in map_keywords)


def extract_data_from_ai_response(ai_response: str, query: str) -> pd.DataFrame:
    """Extract actual data values from AI response for graph visualization"""
    
    print(f"Extracting data from AI response: {ai_response[:200]}...")
    
    # Initialize data containers
    temperatures = []
    salinities = []
    depths = []
    dates = []
    latitudes = []
    longitudes = []
    
    # Extract temperature values from AI response with comprehensive patterns
    temp_patterns = [
        r'temperature[:\s]*(-?\d+\.?\d*)\s*°?[CF]?',
        r'(-?\d+\.?\d*)\s*°?[CF]?\s*temperature',
        r'temp[:\s]*(-?\d+\.?\d*)',
        r'(-?\d+\.?\d*)\s*°?[CF]?',
        r'at\s+(-?\d+\.?\d*)\s*°?[CF]?',
        r'is\s+(-?\d+\.?\d*)\s*°?[CF]?',
        r'of\s+(-?\d+\.?\d*)\s*°?[CF]?',
        r'temperature[:\s]*(-?\d+\.?\d*)\s*°?[CF]?\s*at\s+\d+\s*m',
        r'(-?\d+\.?\d*)\s*°?[CF]?\s*at\s+\d+\s*m'
    ]
    
    for pattern in temp_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            try:
                temp = float(match)
                # Convert Fahrenheit to Celsius if needed
                if '°F' in ai_response or 'F' in ai_response:
                    temp = (temp - 32) * 5/9
                temperatures.append(temp)
            except ValueError:
                continue
    
    # Extract salinity values from AI response
    sal_patterns = [
        r'salinity[:\s]*(\d+\.?\d*)\s*PSU?',
        r'(\d+\.?\d*)\s*PSU?\s*salinity',
        r'salinity[:\s]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*PSU',
        r'salt[:\s]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*ppt'
    ]
    
    for pattern in sal_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            try:
                sal = float(match)
                salinities.append(sal)
            except ValueError:
                continue
    
    # Extract depth values from AI response
    depth_patterns = [
        r'depth[:\s]*(\d+\.?\d*)\s*m',
        r'(\d+\.?\d*)\s*m\s*depth',
        r'depth[:\s]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*meters?',
        r'at\s+(\d+\.?\d*)\s*m',
        r'(\d+\.?\d*)\s*metres?'
    ]
    
    for pattern in depth_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            try:
                depth = float(match)
                depths.append(depth)
            except ValueError:
                continue
    
    # Extract coordinates from AI response
    lat_matches = re.findall(r'latitude[:\s]*(-?\d+\.?\d*)', ai_response, re.IGNORECASE)
    lon_matches = re.findall(r'longitude[:\s]*(-?\d+\.?\d*)', ai_response, re.IGNORECASE)
    
    for lat in lat_matches:
        try:
            latitudes.append(float(lat))
        except ValueError:
            continue
    
    for lon in lon_matches:
        try:
            longitudes.append(float(lon))
        except ValueError:
            continue
    
    # Extract dates from AI response
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{2}/\d{2}/\d{4})',
        r'(\d{4}/\d{2}/\d{2})',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            try:
                if isinstance(match, tuple):
                    match = match[0]
                date = pd.to_datetime(match)
                dates.append(date)
            except:
                continue
    
    # If we have extracted data, create DataFrame
    if temperatures or salinities or depths:
        # Determine the maximum length to pad shorter arrays
        max_len = max(len(temperatures), len(salinities), len(depths), len(dates), len(latitudes), len(longitudes))
        
        if max_len == 0:
            return pd.DataFrame()
        
        # Pad shorter arrays with None/NaN
        temperatures = temperatures + [None] * (max_len - len(temperatures))
        salinities = salinities + [None] * (max_len - len(salinities))
        depths = depths + [None] * (max_len - len(depths))
        latitudes = latitudes + [None] * (max_len - len(latitudes))
        longitudes = longitudes + [None] * (max_len - len(longitudes))
        
        # Handle dates
        if dates:
            dates = dates + [None] * (max_len - len(dates))
        else:
            dates = [pd.Timestamp.now()] * max_len
        
        # Create DataFrame with extracted data
        data = {
            'temperature': temperatures,
            'salinity': salinities,
            'depth': depths,
            'latitude': latitudes,
            'longitude': longitudes,
            'date': dates
        }
        
        df = pd.DataFrame(data)
        
        # Fill missing values with reasonable defaults based on context
        if df['temperature'].isna().all():
            df['temperature'] = 15.0  # Default ocean temperature
        else:
            df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
        
        if df['salinity'].isna().all():
            df['salinity'] = 35.0  # Default ocean salinity
        else:
            df['salinity'] = df['salinity'].fillna(df['salinity'].mean())
        
        if df['depth'].isna().all():
            df['depth'] = 1000.0  # Default depth
        else:
            df['depth'] = df['depth'].fillna(df['depth'].mean())
        
        if df['latitude'].isna().all():
            df['latitude'] = 0.0  # Default latitude
        else:
            df['latitude'] = df['latitude'].fillna(df['latitude'].mean())
        
        if df['longitude'].isna().all():
            df['longitude'] = 0.0  # Default longitude
        else:
            df['longitude'] = df['longitude'].fillna(df['longitude'].mean())
        
        print(f"Successfully extracted data: {len(temperatures)} temperatures, {len(salinities)} salinities, {len(depths)} depths")
        return df
    
    # If no data was extracted, return empty DataFrame
    print("No data extracted from AI response")
    return pd.DataFrame()


def extract_data_from_response(ai_response: str, query: str, data_type: str = "graph") -> pd.DataFrame:
    """Extract meaningful data from AI response for both graph and map visualizations"""
    
    # For graph requests, prioritize AI response data extraction
    if data_type == "graph":
        # Extract data from AI response first (this is what we want for graphs)
        ai_data = extract_data_from_ai_response(ai_response, query)
        if not ai_data.empty:
            print(f"Using AI response data with {len(ai_data)} data points")
            return ai_data
        
        # Only try real-time data as fallback if AI extraction fails
        region = None
        if 'pacific' in ai_response.lower():
            region = 'pacific'
        elif 'atlantic' in ai_response.lower():
            region = 'atlantic'
        elif 'indian' in ai_response.lower():
            region = 'indian'
        elif 'arctic' in ai_response.lower():
            region = 'arctic'
        elif 'southern' in ai_response.lower():
            region = 'southern'
        
        try:
            real_data = fetch_real_time_argo_data(region=region, days_back=7)
            if not real_data.empty:
                print(f"Using real-time data with {len(real_data)} data points")
                return real_data
        except Exception as e:
            print(f"Real-time data fetch failed: {e}")
        
        # Return empty DataFrame if no data available
        print("No data available from AI response or real-time sources")
        return pd.DataFrame()
    
    # Handle map data requests
    elif data_type == "map":
        # Determine region from AI response
        region = None
        if 'pacific' in ai_response.lower():
            region = 'pacific'
        elif 'atlantic' in ai_response.lower():
            region = 'atlantic'
        elif 'indian' in ai_response.lower():
            region = 'indian'
        elif 'arctic' in ai_response.lower():
            region = 'arctic'
        elif 'southern' in ai_response.lower():
            region = 'southern'
        
        # Try to fetch real-time ARGO float locations first
        try:
            real_locations = fetch_argo_float_locations(region=region)
            if not real_locations.empty:
                return real_locations
        except Exception as e:
            print(f"Real-time location fetch failed: {e}")
        
        # Return empty DataFrame if no real-time data available
        return pd.DataFrame()
    
    else:
        # Default to AI data extraction
        return extract_data_from_ai_response(ai_response, query)


def generate_sample_data(query: str) -> pd.DataFrame:
    """Generate sample oceanographic data based on the query"""
    np.random.seed(42)  # For reproducible results
    
    # Generate sample ARGO float data
    n_points = 100
    data = {
        'latitude': np.random.uniform(-60, 60, n_points),
        'longitude': np.random.uniform(-180, 180, n_points),
        'temperature': np.random.normal(15, 10, n_points),
        'salinity': np.random.normal(35, 2, n_points),
        'depth': np.random.uniform(0, 2000, n_points),
        'oxygen': np.random.normal(200, 50, n_points),
        'date': pd.date_range('2023-01-01', periods=n_points, freq='D')
    }
    
    return pd.DataFrame(data)


def create_graph(query: str, data: pd.DataFrame, ai_response: str = "") -> str:
    """Create a balanced, clear, and understandable graph based on the query and AI response"""
    
    # Debug: Print data information
    print(f"Creating graph for query: {query}")
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    if not data.empty:
        print(f"Sample data: {data.head()}")
    print(f"AI response: {ai_response[:100]}...")
    
    # Check if data is empty
    if data.empty:
        # Create a simple message plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for visualization', 
                ha='center', va='center', fontsize=14, 
                transform=ax.transAxes, color='#666666')
        ax.set_title('No Data Available', fontsize=16, fontweight='bold', color='#333333')
        ax.axis('off')
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    # Set up clean, professional styling
    plt.style.use('default')  # Use default style for better control
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Create figure with balanced proportions (16:10 ratio)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    query_lower = query.lower()
    
    # Function to generate dynamic titles based on user query
    def generate_dynamic_title(query, graph_type):
        """Generate dynamic titles based on user query"""
        query_words = query.lower().split()
        
        # Extract key terms from query
        temp_terms = ['temperature', 'temp', 'thermal', 'heat', 'warm', 'cold']
        sal_terms = ['salinity', 'salt', 'saline']
        depth_terms = ['depth', 'deep', 'shallow', 'surface', 'bottom']
        time_terms = ['time', 'trend', 'change', 'over time', 'temporal', 'evolution']
        location_terms = ['location', 'map', 'geographic', 'position', 'coordinates']
        distribution_terms = ['distribution', 'histogram', 'frequency', 'pattern']
        
        # Determine the main focus of the query
        main_focus = []
        if any(term in query_words for term in temp_terms):
            main_focus.append('Temperature')
        if any(term in query_words for term in sal_terms):
            main_focus.append('Salinity')
        if any(term in query_words for term in depth_terms):
            main_focus.append('Depth')
        if any(term in query_words for term in time_terms):
            main_focus.append('Temporal')
        if any(term in query_words for term in location_terms):
            main_focus.append('Geographic')
        if any(term in query_words for term in distribution_terms):
            main_focus.append('Distribution')
        
        # Generate context-aware titles
        if graph_type == 'temperature_depth':
            if 'profile' in query_words or 'vertical' in query_words:
                return f"Ocean Temperature Profile - {', '.join(main_focus) if main_focus else 'Depth Analysis'}"
            elif 'relationship' in query_words or 'correlation' in query_words:
                return f"Temperature vs Depth Relationship - {', '.join(main_focus) if main_focus else 'Oceanographic Analysis'}"
            else:
                return f"Temperature and Depth Analysis - {', '.join(main_focus) if main_focus else 'Ocean Data'}"
        
        elif graph_type == 'salinity_depth':
            if 'profile' in query_words or 'vertical' in query_words:
                return f"Ocean Salinity Profile - {', '.join(main_focus) if main_focus else 'Depth Analysis'}"
            elif 'relationship' in query_words or 'correlation' in query_words:
                return f"Salinity vs Depth Relationship - {', '.join(main_focus) if main_focus else 'Oceanographic Analysis'}"
            else:
                return f"Salinity and Depth Analysis - {', '.join(main_focus) if main_focus else 'Ocean Data'}"
        
        elif graph_type == 'temperature_salinity':
            if 'diagram' in query_words or 'ts' in query_words:
                return f"Temperature-Salinity Diagram - {', '.join(main_focus) if main_focus else 'Ocean Water Mass Analysis'}"
            elif 'relationship' in query_words or 'correlation' in query_words:
                return f"Temperature vs Salinity Relationship - {', '.join(main_focus) if main_focus else 'Oceanographic Analysis'}"
            else:
                return f"Temperature and Salinity Analysis - {', '.join(main_focus) if main_focus else 'Ocean Data'}"
        
        elif graph_type == 'time_series':
            if 'trend' in query_words:
                return f"Temperature Trends Over Time - {', '.join(main_focus) if main_focus else 'Temporal Analysis'}"
            elif 'change' in query_words or 'evolution' in query_words:
                return f"Temperature Change Over Time - {', '.join(main_focus) if main_focus else 'Temporal Evolution'}"
            else:
                return f"Temperature Time Series - {', '.join(main_focus) if main_focus else 'Temporal Data'}"
        
        elif graph_type == 'histogram':
            if 'distribution' in query_words:
                return f"Temperature Distribution - {', '.join(main_focus) if main_focus else 'Statistical Analysis'}"
            elif 'frequency' in query_words:
                return f"Temperature Frequency Distribution - {', '.join(main_focus) if main_focus else 'Statistical Analysis'}"
            else:
                return f"Temperature Histogram - {', '.join(main_focus) if main_focus else 'Data Distribution'}"
        
        elif graph_type == 'geographic':
            if 'map' in query_words:
                return f"Ocean Temperature Map - {', '.join(main_focus) if main_focus else 'Geographic Distribution'}"
            elif 'location' in query_words:
                return f"Temperature by Location - {', '.join(main_focus) if main_focus else 'Geographic Analysis'}"
            else:
                return f"Geographic Temperature Analysis - {', '.join(main_focus) if main_focus else 'Spatial Data'}"
        
        else:  # default
            if 'profile' in query_words:
                return f"Ocean Temperature Profile - {', '.join(main_focus) if main_focus else 'Depth Analysis'}"
            else:
                return f"Ocean Temperature Analysis - {', '.join(main_focus) if main_focus else 'Oceanographic Data'}"
    
    # Clean data and ensure proper scaling
    def clean_and_scale_data(data, x_col, y_col):
        """Clean data and ensure proper scaling for balanced graphs"""
        # Remove NaN values
        clean_data = data.dropna(subset=[x_col, y_col])
        
        if clean_data.empty:
            return clean_data
        
        # Ensure we have enough data points for meaningful visualization
        if len(clean_data) < 2:
            return clean_data
        
        # Sort data for better line plots
        if x_col in clean_data.columns and y_col in clean_data.columns:
            clean_data = clean_data.sort_values(x_col)
        
        return clean_data
    
    # Determine graph type based on query with improved balance
    if 'temperature' in query_lower and 'depth' in query_lower:
        # Temperature vs Depth profile - more balanced
        clean_data = clean_and_scale_data(data, 'temperature', 'depth')
        
        if not clean_data.empty:
            # Use line plot for better readability
            ax.plot(clean_data['temperature'], clean_data['depth'], 
                   marker='o', linewidth=2.5, markersize=6, 
                   color='#1f77b4', alpha=0.8, markerfacecolor='white', 
                   markeredgewidth=1.5, markeredgecolor='#1f77b4')
            
            # Add data points as scatter for emphasis
            ax.scatter(clean_data['temperature'], clean_data['depth'], 
                      s=40, alpha=0.6, color='#1f77b4', edgecolors='white', linewidth=1)
            
            # Generate dynamic title
            title = generate_dynamic_title(query, 'temperature_depth')
            
            ax.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.invert_yaxis()  # Invert y-axis for depth (surface at top)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Set balanced axis limits
            temp_range = clean_data['temperature'].max() - clean_data['temperature'].min()
            depth_range = clean_data['depth'].max() - clean_data['depth'].min()
            
            if temp_range > 0:
                ax.set_xlim(clean_data['temperature'].min() - temp_range*0.05, 
                           clean_data['temperature'].max() + temp_range*0.05)
            if depth_range > 0:
                ax.set_ylim(clean_data['depth'].max() + depth_range*0.05, 
                           clean_data['depth'].min() - depth_range*0.05)
        
    elif 'salinity' in query_lower and 'depth' in query_lower:
        # Beautiful Salinity vs Depth profile
        scatter = ax.scatter(data['salinity'], data['depth'], 
                           c=data['salinity'], cmap='Blues', 
                           s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Salinity (PSU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
        ax.set_title('Ocean Salinity Profile', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Salinity (PSU)', fontweight='bold')
        
    elif 'temperature' in query_lower and 'salinity' in query_lower:
        # Beautiful T-S diagram
        scatter = ax.scatter(data['salinity'], data['temperature'], 
                           c=data['depth'], cmap='viridis', 
                           s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Salinity (PSU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Temperature-Salinity Diagram', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Depth (m)', fontweight='bold')
        
    elif 'time' in query_lower or 'trend' in query_lower:
        # Beautiful time series
        ax.plot(data['date'], data['temperature'], marker='o', linewidth=3, 
               markersize=6, color='#2E86AB', alpha=0.8)
        
        # Add trend line
        x_numeric = np.arange(len(data['date']))
        z = np.polyfit(x_numeric, data['temperature'], 1)
        p = np.poly1d(z)
        ax.plot(data['date'], p(x_numeric), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Temperature Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
    elif 'distribution' in query_lower or 'histogram' in query_lower:
        # Beautiful histogram
        n, bins, patches = ax.hist(data['temperature'], bins=25, alpha=0.8, 
                                 color='skyblue', edgecolor='navy', linewidth=1.2)
        
        # Color bars by height
        for i, (bar, count) in enumerate(zip(patches, n)):
            bar.set_facecolor(plt.cm.Blues(count / max(n)))
        
        ax.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Temperature Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
    elif 'map' in query_lower or 'location' in query_lower:
        # Beautiful geographic plot
        scatter = ax.scatter(data['longitude'], data['latitude'], 
                           c=data['temperature'], cmap='coolwarm', 
                           s=100, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title('Ocean Temperature Map', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)', fontweight='bold')
        
    else:
        # Beautiful default temperature profile
        ax.plot(data['depth'], data['temperature'], marker='o', linewidth=3, 
               markersize=8, color='#E63946', alpha=0.8, markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor='#E63946')
        
        ax.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Ocean Temperature Profile', fontsize=16, fontweight='bold', pad=20)
        ax.invert_xaxis()  # Invert x-axis for depth
        ax.grid(True, alpha=0.3)
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    
    # Improve layout
    plt.tight_layout()
    
    # Convert plot to base64 string with higher quality
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def extract_location_data_from_response(ai_response: str, query: str) -> pd.DataFrame:
    """Extract location data from AI response for ARGO float mapping using real-time data"""
    
    # Determine region from AI response
    region = None
    if 'pacific' in ai_response.lower():
        region = 'pacific'
    elif 'atlantic' in ai_response.lower():
        region = 'atlantic'
    elif 'indian' in ai_response.lower():
        region = 'indian'
    elif 'arctic' in ai_response.lower():
        region = 'arctic'
    elif 'southern' in ai_response.lower():
        region = 'southern'
    
    # Try to fetch real-time ARGO float locations first
    try:
        real_locations = fetch_argo_float_locations(region=region)
        if not real_locations.empty:
            return real_locations
    except Exception as e:
        print(f"Real-time location fetch failed: {e}")
    
    # Fallback to enhanced sample data based on AI response
    np.random.seed(42)  # For reproducible results
    
    # Extract coordinates from AI response with more comprehensive patterns
    lat_matches = re.findall(r'latitude[:\s]*(-?\d+\.?\d*)', ai_response.lower())
    lon_matches = re.findall(r'longitude[:\s]*(-?\d+\.?\d*)', ai_response.lower())
    
    # Also look for coordinate patterns like "40.5°N, 120.3°W"
    coord_pattern = r'(-?\d+\.?\d*)[°\s]*[NS]?[,\s]+(-?\d+\.?\d*)[°\s]*[EW]?'
    coord_matches = re.findall(coord_pattern, ai_response)
    
    # Extract specific ocean regions and locations mentioned
    locations = []
    ocean_regions = []
    
    # Pacific Ocean regions
    if 'pacific' in ai_response.lower():
        ocean_regions.append('pacific')
        locations.extend([(20, -150), (30, -120), (10, -180), (40, -160), (15, -140)])
    if 'north pacific' in ai_response.lower():
        locations.extend([(45, -150), (50, -130), (35, -180), (40, -160)])
    if 'south pacific' in ai_response.lower():
        locations.extend([(-20, -150), (-30, -120), (-10, -180), (-40, -160)])
    
    # Atlantic Ocean regions
    if 'atlantic' in ai_response.lower():
        ocean_regions.append('atlantic')
        locations.extend([(40, -30), (20, -60), (50, -20), (35, -40), (25, -50)])
    if 'north atlantic' in ai_response.lower():
        locations.extend([(45, -30), (50, -20), (40, -40), (55, -25)])
    if 'south atlantic' in ai_response.lower():
        locations.extend([(-20, -30), (-30, -20), (-15, -40), (-25, -35)])
    
    # Indian Ocean regions
    if 'indian' in ai_response.lower():
        ocean_regions.append('indian')
        locations.extend([(10, 80), (20, 60), (30, 100), (15, 70), (25, 90)])
    
    # Arctic Ocean regions
    if 'arctic' in ai_response.lower():
        ocean_regions.append('arctic')
        locations.extend([(70, -150), (80, -30), (75, 0), (72, -120), (78, -60)])
    
    # Southern Ocean regions
    if 'southern' in ai_response.lower() or 'antarctic' in ai_response.lower():
        ocean_regions.append('southern')
        locations.extend([(-60, 0), (-50, 20), (-40, -30), (-55, 10), (-45, -20)])
    
    # Extract specific countries/regions mentioned
    if 'california' in ai_response.lower():
        locations.extend([(35, -120), (37, -122), (33, -118)])
    if 'japan' in ai_response.lower():
        locations.extend([(35, 140), (37, 139), (33, 135)])
    if 'australia' in ai_response.lower():
        locations.extend([(-25, 135), (-30, 130), (-20, 140)])
    if 'europe' in ai_response.lower():
        locations.extend([(50, 0), (45, 5), (55, -5)])
    
    # Use extracted coordinates from AI response
    extracted_coords = []
    if lat_matches and lon_matches and len(lat_matches) == len(lon_matches):
        for lat, lon in zip(lat_matches, lon_matches):
            extracted_coords.append((float(lat), float(lon)))
    
    # Also use coordinate pattern matches
    for lat, lon in coord_matches:
        extracted_coords.append((float(lat), float(lon)))
    
    # Determine final coordinates
    if extracted_coords:
        # Use coordinates extracted from AI response
        n_points = min(30, len(extracted_coords) * 3)  # Generate more points around extracted locations
        final_lats = []
        final_lons = []
        for base_lat, base_lon in extracted_coords:
            # Generate points around each extracted location
            n_around = max(3, n_points // len(extracted_coords))
            lats = np.random.normal(base_lat, 2, n_around)
            lons = np.random.normal(base_lon, 2, n_around)
            final_lats.extend(lats)
            final_lons.extend(lons)
        
        lats = np.array(final_lats[:n_points])
        lons = np.array(final_lons[:n_points])
        
    elif locations:
        # Use mentioned ocean regions
        n_points = 30
        all_lats, all_lons = zip(*locations)
        lats = np.random.choice(all_lats, n_points) + np.random.normal(0, 2, n_points)
        lons = np.random.choice(all_lons, n_points) + np.random.normal(0, 2, n_points)
        
    else:
        # Generate global ARGO float distribution
        n_points = 50
        lats = np.random.uniform(-60, 60, n_points)
        lons = np.random.uniform(-180, 180, n_points)
    
    # Create realistic ARGO float data with ocean-appropriate properties
    data = {
        'latitude': lats,
        'longitude': lons,
        'float_id': [f'ARGO_{i:06d}' for i in range(len(lats))],
        'deployment_date': pd.date_range('2020-01-01', periods=len(lats), freq='30D'),
        'temperature': np.random.normal(15, 10, len(lats)),
        'salinity': np.random.normal(35, 2, len(lats)),
        'depth': np.random.uniform(0, 2000, len(lats)),
        'status': np.random.choice(['active', 'drifting', 'parked'], len(lats), p=[0.6, 0.3, 0.1])
    }
    
    return pd.DataFrame(data)


def create_map(query: str, data: pd.DataFrame, ai_response: str = "") -> str:
    """Create a beautiful interactive map for ARGO float locations"""
    
    # Create base map
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    
    # Choose map style based on query
    if 'ocean' in query.lower() or 'global' in query.lower():
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=3,
            tiles='OpenStreetMap'
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
    
    # Add different tile layers with proper attribution
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        name='Dark',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='Stamen Terrain',
        name='Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    
    # Color mapping for different statuses
    status_colors = {
        'active': 'green',
        'drifting': 'blue', 
        'parked': 'red'
    }
    
    # Add ARGO float markers
    for idx, row in data.iterrows():
        # Determine marker color based on status (with fallback)
        status = row.get('status', 'active')
        color = status_colors.get(status, 'blue')
        
        # Create popup content with safe data access
        float_id = row.get('float_id', f'ARGO_{idx:06d}')
        lat = row.get('latitude', 0)
        lon = row.get('longitude', 0)
        temp = row.get('temperature', 0)
        sal = row.get('salinity', 0)
        depth = row.get('depth', 0)
        deploy_date = row.get('deployment_date', datetime.now())
        
        # Format deployment date safely
        if hasattr(deploy_date, 'strftime'):
            deploy_str = deploy_date.strftime('%Y-%m-%d')
        else:
            deploy_str = str(deploy_date)
        
        popup_content = f"""
        <div style="width: 200px;">
            <h4>ARGO Float {float_id}</h4>
            <p><strong>Location:</strong> {lat:.2f}°N, {lon:.2f}°E</p>
            <p><strong>Status:</strong> {status.title()}</p>
            <p><strong>Temperature:</strong> {temp:.1f}°C</p>
            <p><strong>Salinity:</strong> {sal:.1f} PSU</p>
            <p><strong>Depth:</strong> {depth:.0f}m</p>
            <p><strong>Deployed:</strong> {deploy_str}</p>
        </div>
        """
        
        # Add marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_content, max_width=250),
            color='white',
            weight=2,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)
    
    # Add heatmap layer for density
    heat_data = [[row['latitude'], row['longitude']] for idx, row in data.iterrows()]
    plugins.HeatMap(heat_data, name='ARGO Float Density').add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add custom legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 150px; height: 80px; 
                background-color: white; border:1px solid #ccc; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2)">
    <p style="margin: 0 0 5px 0; font-weight: bold; font-size: 12px;">ARGO Float Status</p>
    <p style="margin: 2px 0; font-size: 10px;"><i class="fa fa-circle" style="color:green; font-size: 8px;"></i> Active</p>
    <p style="margin: 2px 0; font-size: 10px;"><i class="fa fa-circle" style="color:blue; font-size: 8px;"></i> Drifting</p>
    <p style="margin: 2px 0; font-size: 10px;"><i class="fa fa-circle" style="color:red; font-size: 8px;"></i> Parked</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:20px"><b>ARGO Float Locations</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Convert map to HTML string
    map_html = m._repr_html_()
    
    # Return the HTML directly instead of base64 encoding
    return map_html


@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        # Check if this is a map query
        if is_map_query(user_message):
            try:
                # First get AI response to extract location data
                client = create_client()
                map_prompt = f"""The user asked: "{user_message}". 
                Provide a concise response (2-3 sentences max) about ARGO float locations with specific coordinates. 
                Include exact latitude and longitude coordinates (e.g., "latitude: 35.5, longitude: -120.3"), 
                mention specific ocean regions (Pacific, Atlantic, Indian, Arctic, Southern). 
                Keep the response brief and focused on key locations."""
                
                completion = client.chat.completions.create(
                    model="x-ai/grok-4-fast:free",
                    messages=[{"role": "user", "content": map_prompt}],
                )
                ai_response = completion.choices[0].message.content
                
                # Extract location data from AI response and create map
                location_data = extract_location_data_from_response(ai_response, user_message)
                map_html = create_map(user_message, location_data, ai_response)
                
                return jsonify({
                    "reply": ai_response,
                    "map": map_html,
                    "has_map": True
                })
            except Exception as map_error:
                # If map generation fails, fall back to regular AI response
                print(f"Map generation error: {map_error}")
                pass

        # Check if this is a graph query
        if is_graph_query(user_message):
            try:
                # First get AI response to extract meaningful data - OPTIMIZED FOR SPEED
                client = create_client()
                graph_prompt = f"""User query: "{user_message}".
                First, give a very brief 1-2 sentence description.
                Then provide oceanographic data with specific values in a flat, readable list.
                Use this style so it's easy to parse:
                Temperature: 15.2°C at 100m depth, Salinity: 35.1 PSU, Latitude: 40.5, Longitude: -120.3, Date: 2023-01-01
                Temperature: 12.8°C at 500m depth, Salinity: 34.8 PSU, Latitude: 40.6, Longitude: -120.2, Date: 2023-01-02
                Provide 15-50 lines if possible. Do NOT truncate or summarize the data lines.
                Keep only the description concise; keep the data list complete."""
                
                completion = client.chat.completions.create(
                    model="x-ai/grok-4-fast:free",
                    messages=[{"role": "user", "content": graph_prompt}],
                    max_tokens=600,
                    temperature=0.3
                )
                ai_response = completion.choices[0].message.content
                
                # Extract data from AI response and create graph
                sample_data = extract_data_from_response(ai_response, user_message, "graph")
                graph_image = create_graph(user_message, sample_data, ai_response)
                
                return jsonify({
                    "reply": ai_response,
                    "graph": graph_image,
                    "has_graph": True
                })
            except Exception as graph_error:
                # If graph generation fails, fall back to regular AI response
                print(f"Graph generation error: {graph_error}")
                pass

        # Regular AI response (no graph) - OPTIMIZED FOR SPEED
        client = create_client()
        completion = client.chat.completions.create(
            model="x-ai/grok-4-fast:free",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=200,  # Limit response length for speed
            temperature=0.5  # Balanced creativity and speed
        )
        content = completion.choices[0].message.content
        return jsonify({"reply": content, "has_graph": False})
        
    except RuntimeError as e:
        # Handle missing API key
        return jsonify({"error": "API configuration error: " + str(e)}), 500
    except Exception as e:
        # Handle other errors (API errors, network issues, etc.)
        error_msg = str(e)
        if "401" in error_msg or "User not found" in error_msg:
            return jsonify({"error": "Invalid API key. Please check your OpenRouter API key."}), 401
        return jsonify({"error": "Server error: " + error_msg}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True) 