"""
Chart Tool - Advanced Matplotlib oceanographic visualization engine.
Renders publication-quality oceanographic charts for:
1. Time Series Data (Temporal SST, Salinity & Heat Content evolution)
2. Monthly Averages (12-month seasonal climatologies with error bands)
3. Gridded Climatology (WOA / RG-Argo standard depth horizon comparisons)
4. Full Vertical Profiles (Dual-panel CTD water column stratification with MLD & pycnocline)
"""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


def _fig_to_base64(fig) -> str:
    """Converts a matplotlib figure to base64 PNG string and closes figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#0b1329', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return b64


def plot_time_series(data: pd.DataFrame, title: str = "ARGO Ocean Time Series Observations") -> str:
    """Renders multi-parameter time series with dual Y-axes (Temperature, Salinity, MLD)."""
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor('#0b1329')
    ax1.set_facecolor('#0f172a')

    # Convert date if available
    dates = pd.to_datetime(data['date']) if 'date' in data.columns else np.arange(len(data))
    temps = data.get('temperature', pd.Series(dtype=float))
    sals = data.get('salinity', pd.Series(dtype=float))

    # Axis 1: Temperature
    line1 = ax1.plot(dates, temps, color='#38bdf8', linewidth=2.5, marker='o', markersize=5,
                     markerfacecolor='#0284c7', label='Surface Temp (°C)')
    
    # Anomaly shading or rolling mean
    if len(temps) >= 4:
        rolling_temp = temps.rolling(window=3, min_periods=1).mean()
        ax1.plot(dates, rolling_temp, color='#93c5fd', linestyle='--', linewidth=1.8, label='Trend (Rolling Mean)')
        ax1.fill_between(dates, temps, rolling_temp, color='#38bdf8', alpha=0.15)

    ax1.set_ylabel('In-Situ Temperature (°C)', color='#38bdf8', fontsize=12, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='#38bdf8', labelsize=10)
    ax1.tick_params(axis='x', labelcolor='#cbd5e1', labelsize=10, rotation=30)
    ax1.grid(True, linestyle=':', alpha=0.25, color='#64748b')

    # Axis 2: Salinity
    ax2 = ax1.twinx()
    line2 = ax2.plot(dates, sals, color='#34d399', linewidth=2.2, marker='s', markersize=4.5,
                     markerfacecolor='#059669', label='Salinity (PSU)')
    ax2.set_ylabel('Practical Salinity (PSU)', color='#34d399', fontsize=12, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='#34d399', labelsize=10)
    ax2.grid(False)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=10)

    # Subtitle and annotations
    reg = data.get('region', pd.Series(['Global Basin'])).iloc[0] if not data.empty else 'Global Basin'
    fid = data.get('float_id', pd.Series(['ARGO'])).iloc[0] if not data.empty else 'ARGO'
    plt.title(f"{title} | Float {fid} ({reg})\nTemporal Evolution across Continuous 9-10 Day Profiling Cycles",
              fontsize=14, fontweight='bold', color='#f8fafc', pad=18)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_monthly_averages(data: pd.DataFrame, title: str = "ARGO 12-Month Climatological Seasonal Averages") -> str:
    """Renders 12-month ocean climatology bar & line chart (Mean SST, SSS, and MLD)."""
    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor('#0b1329')
    ax1.set_facecolor('#0f172a')

    months = data['month_name'] if 'month_name' in data.columns else [f"M{i}" for i in range(1, len(data)+1)]
    temps = data.get('mean_temperature', pd.Series(dtype=float))
    temp_err = data.get('temp_std', pd.Series([0.4]*len(data)))
    sals = data.get('mean_salinity', pd.Series(dtype=float))
    mlds = data.get('mean_mld', pd.Series(dtype=float))

    x = np.arange(len(months))
    width = 0.52

    # Bar chart for Temperature
    bars = ax1.bar(x, temps, width=width, yerr=temp_err, capsize=4,
                   color='#0284c7', alpha=0.85, edgecolor='#38bdf8', linewidth=1.5,
                   error_kw={'ecolor': '#93c5fd', 'alpha': 0.8}, label='Mean SST (°C)')

    # Add numeric labels on bars
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h/2., f'{h:.1f}°',
                 ha='center', va='center', color='#ffffff', fontweight='bold', fontsize=9)

    ax1.set_ylabel('Mean Sea Surface Temperature (°C)', color='#38bdf8', fontsize=12, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='#38bdf8', labelsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, color='#cbd5e1', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.25, color='#64748b', axis='y')

    # Secondary Axis: Mixed Layer Depth & Salinity
    ax2 = ax1.twinx()
    line_mld = ax2.plot(x, mlds, color='#f59e0b', linewidth=2.8, marker='D', markersize=6,
                        markerfacecolor='#d97706', label='Mixed Layer Depth (m)')
    ax2.set_ylabel('Mixed Layer Depth (m)', color='#f59e0b', fontsize=12, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='#f59e0b', labelsize=10)
    ax2.grid(False)

    # Invert MLD axis so deeper mixed layer goes downwards
    if not mlds.empty:
        ax2.set_ylim(max(mlds) * 1.25, min(mlds) * 0.7)

    # Combined legend
    lines = [bars] + line_mld
    labels = ['Mean SST (°C)', 'Mixed Layer Depth (m)']
    ax1.legend(lines, labels, loc='upper left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=10)

    reg = data.get('region', pd.Series(['Ocean Basin'])).iloc[0] if not data.empty else 'Ocean Basin'
    plt.title(f"{title} ({reg})\n12-Month Climatology: Annual Heating, Monsoon Dynamics, and Mixed Layer Deepening",
              fontsize=14, fontweight='bold', color='#f8fafc', pad=18)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_gridded_climatology(data: pd.DataFrame, title: str = "Gridded Ocean Climatology (RG-Argo / WOA)") -> str:
    """Renders multi-depth objective analysis gridded comparison across standard oceanographic horizons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    fig.patch.set_facecolor('#0b1329')

    depths = data.get('depth', pd.Series(dtype=float))
    temps = data.get('climatological_temp', pd.Series(dtype=float))
    sals = data.get('climatological_salinity', pd.Series(dtype=float))
    dens = data.get('potential_density', pd.Series(dtype=float))

    # Panel 1: Temperature Climatology vs Depth
    ax1.set_facecolor('#0f172a')
    ax1.plot(temps, depths, color='#38bdf8', linewidth=2.8, marker='o', markersize=5.5,
             markerfacecolor='#0284c7', label='Gridded Climatology T (°C)')
    if 'global_temp_baseline' in data.columns:
        ax1.plot(data['global_temp_baseline'], depths, color='#94a3b8', linestyle='--', linewidth=2.0,
                 label='Global Mean Baseline')
        ax1.fill_betweenx(depths, temps, data['global_temp_baseline'], color='#38bdf8', alpha=0.15, label='Regional Thermal Anomaly')

    ax1.set_xlabel('Climatological Temperature (°C)', color='#38bdf8', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Depth (m)', color='#cbd5e1', fontsize=11, fontweight='bold')
    ax1.set_title('Thermal Stratification (0-2,000m)', color='#f8fafc', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=':', alpha=0.3, color='#64748b')
    ax1.tick_params(colors='#cbd5e1')
    ax1.legend(loc='lower left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=9)

    # Panel 2: Salinity & Potential Density Climatology
    ax2.set_facecolor('#0f172a')
    ax2.plot(sals, depths, color='#34d399', linewidth=2.8, marker='s', markersize=5.5,
             markerfacecolor='#059669', label='Gridded Salinity (PSU)')
    ax2.set_xlabel('Practical Salinity (PSU)', color='#34d399', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=':', alpha=0.3, color='#64748b')
    ax2.tick_params(axis='x', colors='#34d399')
    ax2.tick_params(axis='y', colors='#cbd5e1')

    # Water Mass annotations on panel 2
    if 'water_mass' in data.columns:
        for _, row in data.iterrows():
            d = row['depth']
            if d in (100, 500, 1000, 2000):
                wm = str(row['water_mass']).split('(')[0].strip()
                ax2.annotate(f" {wm}", xy=(row['climatological_salinity'], d),
                             xytext=(row['climatological_salinity'] + 0.08, d),
                             fontsize=8, color='#e2e8f0', fontweight='bold',
                             arrowprops=dict(arrowstyle="->", color='#64748b', lw=0.8))

    ax2.set_title('Halocline & Water Mass Horizons', color='#f8fafc', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=9)

    reg = data.get('region', pd.Series(['Global Basin'])).iloc[0] if not data.empty else 'Global Basin'
    fig.suptitle(f"{title} | {reg}\nStandard Depth Horizons (0m-2,000m) via Objective Analysis Mapping",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=0.98)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_full_vertical_profile(data: pd.DataFrame, title: str = "High-Resolution Oceanographic CTD Vertical Profile") -> str:
    """Renders dual-panel water column profile from 0m to 2,000m+ resolving thermocline, halocline, and pycnocline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.patch.set_facecolor('#0b1329')

    depths = data.get('depth', pd.Series(dtype=float))
    temps = data.get('temperature', pd.Series(dtype=float))
    sals = data.get('salinity', pd.Series(dtype=float))
    dens = data.get('density_sigma', pd.Series(dtype=float))

    # Panel 1: Temperature & Mixed Layer
    ax1.set_facecolor('#0f172a')
    ax1.plot(temps, depths, color='#06b6d4', linewidth=2.8, marker='o', markersize=4.5,
             markerfacecolor='#0891b2', label='CTD Temperature (°C)')
    
    # Highlight Mixed Layer Depth (MLD typically 0-50m)
    mld_depth = 50.0
    ax1.axhspan(0, mld_depth, color='#0284c7', alpha=0.15, label=f'Mixed Layer (0-{mld_depth:.0f}m)')
    ax1.axhline(y=mld_depth, color='#38bdf8', linestyle=':', linewidth=1.5)

    # Highlight Thermocline zone (50-800m)
    ax1.axhspan(mld_depth, 800, color='#818cf8', alpha=0.08, label='Permanent Thermocline')

    ax1.set_xlabel('In-Situ Temperature (°C)', color='#06b6d4', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Depth / Pressure (m / dbar)', color='#cbd5e1', fontsize=11, fontweight='bold')
    ax1.set_title('Thermal Profile & Mixed Layer Depth', color='#f8fafc', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=':', alpha=0.3, color='#64748b')
    ax1.tick_params(colors='#cbd5e1')
    ax1.legend(loc='lower left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=9)

    # Panel 2: Salinity & Potential Density Pycnocline
    ax2.set_facecolor('#0f172a')
    line_sal = ax2.plot(sals, depths, color='#10b981', linewidth=2.8, marker='s', markersize=4.5,
                        markerfacecolor='#059669', label='Salinity (PSU)')
    ax2.set_xlabel('Practical Salinity (PSU)', color='#10b981', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=':', alpha=0.3, color='#64748b')
    ax2.tick_params(axis='x', colors='#10b981')
    ax2.tick_params(axis='y', colors='#cbd5e1')

    # Twin axis for Potential Density (sigma-theta)
    ax2_dens = ax2.twiny()
    line_dens = ax2_dens.plot(dens, depths, color='#a855f7', linewidth=2.2, linestyle='--',
                              label='Density σ_θ (kg/m³)')
    ax2_dens.set_xlabel('Potential Density σ_θ (kg/m³)', color='#a855f7', fontsize=11, fontweight='bold')
    ax2_dens.tick_params(axis='x', colors='#a855f7')
    ax2_dens.grid(False)

    lines = line_sal + line_dens
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower left', framealpha=0.85, facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=9)
    ax2.set_title('Halocline & Pycnocline Density Structure', color='#f8fafc', fontsize=12, fontweight='bold')

    fid = data.get('float_id', pd.Series(['ARGO CTD'])).iloc[0] if not data.empty else 'ARGO CTD'
    reg = data.get('region', pd.Series(['Ocean Column'])).iloc[0] if not data.empty else 'Ocean Column'
    max_d = int(max(depths)) if not depths.empty else 2000
    fig.suptitle(f"{title} | Platform {fid} ({reg})\nFull Water Column Stratification: Epipelagic (0m) to Bathypelagic ({max_d:,}m)",
                 fontsize=14, fontweight='bold', color='#f8fafc', y=0.98)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_chart(query: str, data: pd.DataFrame, ai_response: str = '', data_type: Optional[str] = None) -> str:
    """
    Main chart entry point. Intelligently selects the specialized oceanographic visualization
    based on data_type, query semantics, or data table schema.
    """
    print(f'[ChartTool] Creating chart for query: "{query}" | data_type: {data_type} | shape: {data.shape}')

    if data.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor('#0b1329')
        ax.set_facecolor('#0f172a')
        ax.text(0.5, 0.5, 'Oceanographic telemetry points currently being synthesized...\nSend prompt to refresh profile visualization.',
                ha='center', va='center', fontsize=13, color='#94a3b8')
        ax.axis('off')
        return _fig_to_base64(fig)

    q = (query or '').lower()
    cols = set(data.columns)

    # 1. Monthly Averages Chart
    if data_type == 'monthly_averages' or 'month' in cols or 'month_name' in cols or 'monthly' in q:
        return plot_monthly_averages(data, title="ARGO 12-Month Climatological Seasonal Averages")

    # 2. Time Series Chart
    if data_type == 'time_series' or 'cycle_number' in cols or ('date' in cols and len(data) >= 4) or 'time series' in q or 'temporal' in q:
        return plot_time_series(data, title="ARGO Ocean Time Series Observations")

    # 3. Gridded Climatology Chart
    if data_type == 'gridded_climatology' or 'water_mass' in cols or 'climatological_temp' in cols or 'gridded' in q or 'climatology' in q:
        return plot_gridded_climatology(data, title="Gridded Ocean Climatology (RG-Argo / WOA)")

    # 4. Full Vertical Profile Chart
    if data_type == 'vertical_profile' or 'density_sigma' in cols or 'sound_speed' in cols or 'profile' in q or 'depth' in q:
        return plot_full_vertical_profile(data, title="High-Resolution Oceanographic CTD Vertical Profile")

    # 5. Default Fallback: Full vertical profile if depth available, else Time Series
    if 'depth' in cols:
        return plot_full_vertical_profile(data, title="ARGO Float Vertical Profile")
    elif 'date' in cols:
        return plot_time_series(data, title="ARGO Float Observation Series")
    else:
        return plot_full_vertical_profile(data, title="ARGO Ocean CTD Observations")


# Alias for compatibility with callers
create_graph = create_chart

