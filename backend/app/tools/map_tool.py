"""
Map Tool - Folium-based interactive map generator for ARGO float locations.
Extracted from main.py create_map() function into the app/tools package.
"""
import folium
from folium import plugins
import pandas as pd
from datetime import datetime


def create_map(query: str, data: pd.DataFrame, ai_response: str = '') -> str:
    """Create a Folium interactive HTML map for ARGO float locations. Returns raw HTML string."""
    if data.empty:
        return '<p style="text-align:center;color:#666">No location data available.</p>'

    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    zoom = 3 if ('ocean' in query.lower() or 'global' in query.lower()) else 6

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='OpenStreetMap')

    status_colors = {'active': 'green', 'drifting': 'blue', 'parked': 'red'}

    for idx, row in data.iterrows():
        status = row.get('status', 'active')
        color = status_colors.get(status, 'blue')
        float_id = row.get('float_id', f'ARGO_{idx:06d}')
        lat = row.get('latitude', 0)
        lon = row.get('longitude', 0)
        temp = row.get('temperature', 0)
        sal = row.get('salinity', 0)
        depth = row.get('depth', 0)
        deploy_date = row.get('deployment_date', datetime.now())
        deploy_str = deploy_date.strftime('%Y-%m-%d') if hasattr(deploy_date, 'strftime') else str(deploy_date)

        popup_html = f"""
        <div style="width:200px;">
            <h4>ARGO Float {float_id}</h4>
            <p><strong>Location:</strong> {lat:.2f}°N, {lon:.2f}°E</p>
            <p><strong>Status:</strong> {str(status).title()}</p>
            <p><strong>Temperature:</strong> {temp:.1f}°C</p>
            <p><strong>Salinity:</strong> {sal:.1f} PSU</p>
            <p><strong>Depth:</strong> {depth:.0f}m</p>
            <p><strong>Deployed:</strong> {deploy_str}</p>
        </div>"""

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=250),
            color='white',
            weight=2,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)

    heat_data = [[r['latitude'], r['longitude']] for _, r in data.iterrows()]
    plugins.HeatMap(heat_data, name='ARGO Float Density').add_to(m)
    folium.LayerControl().add_to(m)

    legend_html = """
    <div style="position:fixed;top:10px;right:10px;width:150px;height:80px;
                background-color:white;border:1px solid #ccc;z-index:9999;
                font-size:11px;padding:8px;border-radius:5px;box-shadow:0 2px 5px rgba(0,0,0,0.2)">
    <p style="margin:0 0 5px 0;font-weight:bold;font-size:12px;">ARGO Float Status</p>
    <p style="margin:2px 0;font-size:10px;"><i class="fa fa-circle" style="color:green;font-size:8px;"></i> Active</p>
    <p style="margin:2px 0;font-size:10px;"><i class="fa fa-circle" style="color:blue;font-size:8px;"></i> Drifting</p>
    <p style="margin:2px 0;font-size:10px;"><i class="fa fa-circle" style="color:red;font-size:8px;"></i> Parked</p>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element('<h3 align="center" style="font-size:20px"><b>ARGO Float Locations</b></h3>'))

    return m._repr_html_()
