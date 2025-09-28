import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import "echarts-gl";
import earth from "../assets/earth.jpg";
import stars from "../assets/starfield.jpg";

function EarthGlobe() {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const chartInstance = useRef<echarts.EChartsType | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const chartDom = chartRef.current;
    chartInstance.current = echarts.init(chartDom);

    const option: echarts.EChartsOption = {
      backgroundColor: "#000",

      globe: {
        baseTexture: earth,
        heightTexture: earth,
        environment: stars,
        shading: "lambert",
        atmosphere: {
          show: true,
        },
        light: {
          ambient: {
            intensity: 0.1,
          },
          displacementQuality: "ultra",
          main: {
            intensity: 1,
          },
        },
      },
      series: [],
    };

    chartInstance.current.setOption(option);

    return () => {
      chartInstance.current?.dispose();
    };
  }, []);

  return <div ref={chartRef} style={{ width: "50vw", height: "50vh" }} />;
}

export default EarthGlobe;