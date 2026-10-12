import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import "echarts-gl";
import earth from "../assets/earth.jpg";
import stars from "../assets/starfield.jpg";

export interface EarthGlobeProps {
  floatData?: any[];
  selectedFloatId?: string | null;
  onSelectFloat?: (float: any) => void;
}

export function EarthGlobe({ floatData = [], selectedFloatId, onSelectFloat }: EarthGlobeProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const chartInstance = useRef<echarts.EChartsType | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    // Map floats into ECharts scatter3D points
    const scatterData = (floatData || []).map((f) => {
      const isSelected = selectedFloatId && f.id === selectedFloatId;
      const isBGC = f.type?.includes("BGC");
      const isDeep = f.type?.includes("Deep");
      
      let pointColor = "#38bdf8"; // Core ARGO: Sky / Cyan
      if (isBGC) pointColor = "#34d399"; // BGC ARGO: Emerald
      if (isDeep) pointColor = "#a78bfa"; // Deep ARGO: Purple
      if (isSelected) pointColor = "#f43f5e"; // Selected: Rose

      return {
        name: `Float ${f.id}`,
        value: [f.lon, f.lat, isSelected ? 3 : 0.5],
        symbolSize: isSelected ? 16 : 8,
        itemStyle: {
          color: pointColor,
          opacity: isSelected ? 1.0 : 0.88,
          borderWidth: isSelected ? 2 : 1,
          borderColor: "#ffffff"
        },
        floatData: f
      };
    });

    const option: any = {
      backgroundColor: "#030712",
      tooltip: {
        show: true,
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        borderColor: "#334155",
        borderWidth: 1,
        textStyle: {
          color: "#f8fafc",
          fontSize: 12
        },
        formatter: (params: any) => {
          const f = params.data?.floatData;
          if (!f) return params.name;
          return `
            <div style="font-family: sans-serif; padding: 4px;">
              <div style="font-weight: 700; color: #38bdf8; font-size: 13px; margin-bottom: 2px;">
                Float ${f.id} <span style="font-size: 11px; opacity: 0.8;">(${f.type})</span>
              </div>
              <div style="color: #cbd5e1; font-size: 11px;">Region: ${f.region}</div>
              <div style="color: #cbd5e1; font-size: 11px;">Coordinates: ${f.lat >= 0 ? f.lat + '°N' : Math.abs(f.lat) + '°S'}, ${f.lon >= 0 ? f.lon + '°E' : Math.abs(f.lon) + '°W'}</div>
              <div style="display: flex; gap: 8px; margin-top: 4px; font-weight: 600;">
                <span style="color: #fb7185;">${f.temp}°C</span>
                <span style="color: #22d3ee;">${f.salinity} PSU</span>
                <span style="color: #a78bfa;">${f.maxDepth}</span>
              </div>
              <div style="color: #94a3b8; font-size: 10px; margin-top: 2px;">Inst: ${f.institution} • ${f.lastUpdate || 'Active'}</div>
            </div>
          `;
        }
      },
      globe: {
        baseTexture: earth,
        heightTexture: earth,
        environment: stars,
        shading: "lambert",
        atmosphere: {
          show: true,
          offset: 4,
          color: "#38bdf8",
          glowPower: 4
        },
        viewControl: {
          autoRotate: true,
          autoRotateSpeed: 3,
          autoRotateAfterStill: 2,
          distance: 180,
          minDistance: 120,
          maxDistance: 320,
          alpha: 25,
          beta: 40
        },
        light: {
          ambient: {
            intensity: 0.35,
          },
          main: {
            intensity: 1.25,
            shadow: false,
          },
        },
      },
      series: [
        {
          type: "scatter3D",
          coordinateSystem: "globe",
          blendMode: "lighter",
          data: scatterData,
          emphasis: {
            itemStyle: {
              color: "#fbbf24",
              borderColor: "#ffffff",
              borderWidth: 2
            }
          }
        }
      ],
    };

    chartInstance.current.setOption(option);

    const handleClick = (params: any) => {
      if (params.data?.floatData && onSelectFloat) {
        onSelectFloat(params.data.floatData);
      }
    };

    chartInstance.current.on("click", handleClick);

    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener("resize", handleResize);

    return () => {
      chartInstance.current?.off("click", handleClick);
      window.removeEventListener("resize", handleResize);
    };
  }, [floatData, selectedFloatId, onSelectFloat]);

  return <div ref={chartRef} className="w-full h-full min-h-[450px]" />;
}

export default EarthGlobe;