"""
Chart Tool - Matplotlib depth profile & oceanographic chart generator.
Extracted from main.py create_graph() function into the app/tools package.
"""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_chart(query: str, data: pd.DataFrame, ai_response: str = '') -> str:
    """Create a balanced, clear chart based on the query and data. Returns base64 PNG string."""
    print(f'[ChartTool] Creating chart for: {query} | data shape: {data.shape}')

    if data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for visualization',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes, color='#666666')
        ax.set_title('No Data Available', fontsize=16, fontweight='bold', color='#333333')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        return b64

    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 14, 'axes.labelsize': 11})
    fig, ax = plt.subplots(figsize=(12, 7.5))
    q = query.lower()

    def _clean(df, xc, yc):
        df = df.dropna(subset=[xc, yc])
        if len(df) >= 2:
            df = df.sort_values(xc)
        return df

    if 'temperature' in q and 'depth' in q:
        d = _clean(data, 'temperature', 'depth')
        if not d.empty:
            ax.plot(d['temperature'], d['depth'], marker='o', linewidth=2.5, markersize=6,
                    color='#1f77b4', alpha=0.8, markerfacecolor='white',
                    markeredgewidth=1.5, markeredgecolor='#1f77b4')
            ax.scatter(d['temperature'], d['depth'], s=40, alpha=0.6, color='#1f77b4',
                       edgecolors='white', linewidth=1)
            ax.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
            ax.set_title('Ocean Temperature–Depth Profile', fontsize=14, fontweight='bold', pad=15)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
    elif 'salinity' in q and 'depth' in q:
        sc = ax.scatter(data['salinity'], data['depth'], c=data['salinity'],
                        cmap='Blues', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Salinity (PSU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
        ax.set_title('Ocean Salinity Profile', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax).set_label('Salinity (PSU)', fontweight='bold')
    elif 'temperature' in q and 'salinity' in q:
        sc = ax.scatter(data['salinity'], data['temperature'], c=data['depth'],
                        cmap='viridis', s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Salinity (PSU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Temperature–Salinity Diagram', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax).set_label('Depth (m)', fontweight='bold')
    elif 'time' in q or 'trend' in q:
        ax.plot(data['date'], data['temperature'], marker='o', linewidth=3,
                markersize=6, color='#2E86AB', alpha=0.8)
        x_num = np.arange(len(data['date']))
        z = np.polyfit(x_num, data['temperature'], 1)
        ax.plot(data['date'], np.poly1d(z)(x_num), 'r--', alpha=0.8, linewidth=2)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Temperature Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    elif 'distribution' in q or 'histogram' in q:
        n, bins, patches = ax.hist(data['temperature'], bins=25, alpha=0.8,
                                   color='skyblue', edgecolor='navy', linewidth=1.2)
        for bar, count in zip(patches, n):
            bar.set_facecolor(plt.cm.Blues(count / max(n) if max(n) else 1))
        ax.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Temperature Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    else:
        ax.plot(data['depth'], data['temperature'], marker='o', linewidth=3,
                markersize=8, color='#E63946', alpha=0.8,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#E63946')
        ax.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Ocean Temperature Profile', fontsize=16, fontweight='bold', pad=20)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return b64


# Alias for compatibility with ocean_agent callers
create_graph = create_chart
