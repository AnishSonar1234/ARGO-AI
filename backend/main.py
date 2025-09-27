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

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)


def create_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


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


def extract_data_from_response(ai_response: str, query: str) -> pd.DataFrame:
    """Extract meaningful data from AI response and create realistic oceanographic data"""
    np.random.seed(42)  # For reproducible results
    
    # Extract numbers and patterns from AI response
    numbers = re.findall(r'-?\d+\.?\d*', ai_response)
    temperatures = []
    salinities = []
    depths = []
    
    # Look for specific patterns in the response
    if 'temperature' in ai_response.lower():
        # Extract temperature values if mentioned
        temp_matches = re.findall(r'temperature[:\s]*(-?\d+\.?\d*)', ai_response.lower())
        if temp_matches:
            base_temp = float(temp_matches[0])
        else:
            base_temp = 15
        
        # Create realistic temperature profile
        depths = np.linspace(0, 2000, 50)
        temperatures = base_temp - (depths / 100) + np.random.normal(0, 2, 50)
    else:
        depths = np.linspace(0, 2000, 50)
        temperatures = 20 - (depths / 100) + np.random.normal(0, 3, 50)
    
    if 'salinity' in ai_response.lower():
        sal_matches = re.findall(r'salinity[:\s]*(\d+\.?\d*)', ai_response.lower())
        if sal_matches:
            base_sal = float(sal_matches[0])
        else:
            base_sal = 35
        salinities = np.random.normal(base_sal, 1, len(depths))
    else:
        salinities = np.random.normal(35, 1.5, len(depths))
    
    # Create realistic oceanographic data
    data = {
        'depth': depths,
        'temperature': temperatures,
        'salinity': salinities,
        'latitude': np.random.uniform(-60, 60, len(depths)),
        'longitude': np.random.uniform(-180, 180, len(depths)),
        'oxygen': np.random.normal(200, 30, len(depths)),
        'date': pd.date_range('2023-01-01', periods=len(depths), freq='D')
    }
    
    return pd.DataFrame(data)


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
    """Create a beautiful, sophisticated graph based on the query and AI response"""
    
    # Set up beautiful styling
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    query_lower = query.lower()
    
    # Determine graph type based on query
    if 'temperature' in query_lower and 'depth' in query_lower:
        # Beautiful Temperature vs Depth profile
        scatter = ax.scatter(data['temperature'], data['depth'], 
                           c=data['temperature'], cmap='RdYlBu_r', 
                           s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(data['temperature'], data['depth'], 1)
        p = np.poly1d(z)
        ax.plot(p(data['temperature']), data['depth'], "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
        ax.set_title('Ocean Temperature Profile', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()  # Invert y-axis for depth
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)', fontweight='bold')
        
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
    """Extract location data from AI response for ARGO float mapping"""
    np.random.seed(42)  # For reproducible results
    
    # Extract coordinates from AI response
    lat_matches = re.findall(r'latitude[:\s]*(-?\d+\.?\d*)', ai_response.lower())
    lon_matches = re.findall(r'longitude[:\s]*(-?\d+\.?\d*)', ai_response.lower())
    
    # Extract specific locations mentioned
    locations = []
    if 'pacific' in ai_response.lower():
        locations.extend([(20, -150), (30, -120), (10, -180)])
    if 'atlantic' in ai_response.lower():
        locations.extend([(40, -30), (20, -60), (50, -20)])
    if 'indian' in ai_response.lower():
        locations.extend([(10, 80), (20, 60), (30, 100)])
    if 'arctic' in ai_response.lower():
        locations.extend([(70, -150), (80, -30), (75, 0)])
    if 'southern' in ai_response.lower():
        locations.extend([(-60, 0), (-50, 20), (-40, -30)])
    
    # Use extracted coordinates or generate realistic ones
    if lat_matches and lon_matches:
        base_lat = float(lat_matches[0])
        base_lon = float(lon_matches[0])
        # Generate points around the mentioned location
        n_points = 20
        lats = np.random.normal(base_lat, 5, n_points)
        lons = np.random.normal(base_lon, 5, n_points)
    elif locations:
        # Use mentioned ocean regions
        n_points = 30
        all_lats, all_lons = zip(*locations)
        lats = np.random.choice(all_lats, n_points) + np.random.normal(0, 3, n_points)
        lons = np.random.choice(all_lons, n_points) + np.random.normal(0, 3, n_points)
    else:
        # Generate global ARGO float distribution
        n_points = 50
        lats = np.random.uniform(-60, 60, n_points)
        lons = np.random.uniform(-180, 180, n_points)
    
    # Create realistic ARGO float data
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
    
    # Add different tile layers
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
    
    # Color mapping for different statuses
    status_colors = {
        'active': 'green',
        'drifting': 'blue', 
        'parked': 'red'
    }
    
    # Add ARGO float markers
    for idx, row in data.iterrows():
        # Determine marker color based on status
        color = status_colors.get(row['status'], 'blue')
        
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h4>ARGO Float {row['float_id']}</h4>
            <p><strong>Location:</strong> {row['latitude']:.2f}°N, {row['longitude']:.2f}°E</p>
            <p><strong>Status:</strong> {row['status'].title()}</p>
            <p><strong>Temperature:</strong> {row['temperature']:.1f}°C</p>
            <p><strong>Salinity:</strong> {row['salinity']:.1f} PSU</p>
            <p><strong>Depth:</strong> {row['depth']:.0f}m</p>
            <p><strong>Deployed:</strong> {row['deployment_date'].strftime('%Y-%m-%d')}</p>
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
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>ARGO Float Status</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Active</p>
    <p><i class="fa fa-circle" style="color:blue"></i> Drifting</p>
    <p><i class="fa fa-circle" style="color:red"></i> Parked</p>
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
    
    # Convert HTML to base64 for transmission
    map_base64 = base64.b64encode(map_html.encode()).decode()
    
    return map_base64


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
                Provide detailed information about ARGO float locations, including specific coordinates, ocean regions, 
                and deployment information. Include latitude/longitude coordinates and mention specific ocean basins."""
                
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
                # First get AI response to extract meaningful data
                client = create_client()
                graph_prompt = f"""The user asked: "{user_message}". 
                Provide a detailed oceanographic analysis with specific temperature, salinity, and depth values. 
                Include realistic oceanographic data points and scientific insights about the ocean profile."""
                
                completion = client.chat.completions.create(
                    model="x-ai/grok-4-fast:free",
                    messages=[{"role": "user", "content": graph_prompt}],
                )
                ai_response = completion.choices[0].message.content
                
                # Extract data from AI response and create graph
                sample_data = extract_data_from_response(ai_response, user_message)
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

        # Regular AI response (no graph)
        client = create_client()
        completion = client.chat.completions.create(
            model="x-ai/grok-4-fast:free",
            messages=[{"role": "user", "content": user_message}],
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


