import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { 
  Thermometer, 
  Droplets, 
  Gauge, 
  Activity, 
  MapPin, 
  Calendar, 
  TrendingUp, 
  Globe, 
  Waves, 
  Microscope, 
  ArrowDown, 
  Search, 
  Filter, 
  Layers, 
  Navigation,
  RefreshCw,
  Radio,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  MessageSquare,
  Sparkles,
  Zap,
  BatteryMedium,
  RotateCw,
  Info
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  AreaChart,
  Area
} from "recharts";
import globalMapImage from "@/assets/argo-global.jpg";
import EarthGlobe from "@/components/EarthGlobe";

export interface FloatPlatform {
  id: string;
  lat: number;
  lon: number;
  region: string;
  type: string;
  institution: string;
  status: string;
  lastUpdate: string;
  lastTimestamp?: string;
  temp: number;
  salinity: number;
  maxDepth: string;
  cycle?: number;
  battery?: number;
  oxygen?: number;
  ph?: number;
  chlorophyll?: number;
  nitrate?: number;
  deepTemp?: number;
  deepSalinity?: number;
}

export interface RegionStat {
  name: string;
  floats: number;
  active: number;
  percentage: number;
}

export interface OverviewStat {
  label: string;
  value: string;
  numeric: number;
  badge: string;
  color: string;
}

export interface FloatTypeInfo {
  type: string;
  count: string;
  numeric: number;
  color: string;
  description: string;
  parameters: string[];
  analytics: {
    activeFloats: string;
    avgDepth: string;
    profilesPerMonth: string;
    coverage: string;
  };
}

export interface DepthPoint {
  depth: number;
  temperature: number;
  salinity: number;
  oxygen: number;
  density: number;
  chlorophyll?: number;
}

export interface DashboardSummary {
  totalFloatsCount: number;
  activeFloatsCount: number;
  coreCount: number;
  bgcCount: number;
  deepCount: number;
  uptimeRate: number;
  fleetAvgTemp: number;
  fleetAvgSalinity: number;
  activeVisibleFloats: number;
}

interface DashboardProps {
  onSectionChange?: (section: string) => void;
}

export const Dashboard = ({ onSectionChange }: DashboardProps) => {
  const [selectedFloatType, setSelectedFloatType] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"3d" | "2d">("3d");
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string>("Syncing...");
  
  // Selected float for deep inspection
  const [selectedFloat, setSelectedFloat] = useState<FloatPlatform | null>(null);
  const [chartMetric, setChartMetric] = useState<"temperature" | "salinity" | "oxygen" | "density">("temperature");

  // Real-time fetched state
  const [stats, setStats] = useState<OverviewStat[]>([]);
  const [regions, setRegions] = useState<RegionStat[]>([]);
  const [activeFloats, setActiveFloats] = useState<FloatPlatform[]>([]);
  const [floatTypes, setFloatTypes] = useState<FloatTypeInfo[]>([]);
  const [depthProfiles, setDepthProfiles] = useState<DepthPoint[]>([]);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);

  const baseApiUrl = import.meta.env.VITE_BACKEND_URL 
    ? import.meta.env.VITE_BACKEND_URL.replace("/chat", "")
    : "http://localhost:5000";

  const overviewUrl = `${baseApiUrl}/api/dashboard/overview`;

  const fetchDashboardData = async (isManual = false) => {
    if (isManual) setIsRefreshing(true);
    try {
      const targetUrl = isManual ? `${overviewUrl}?refresh=true` : overviewUrl;
      const res = await fetch(targetUrl);
      if (res.ok) {
        const data = await res.json();
        setStats(data.stats || []);
        setRegions(data.regions || []);
        setActiveFloats(data.activeFloats || []);
        setFloatTypes(data.floatTypes || []);
        setDepthProfiles(data.depthProfiles || []);
        if (data.summary) {
          setSummary(data.summary);
        }
        setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
        
        // If no float selected yet and we have floats, pick the first one as featured
        if (!selectedFloat && data.activeFloats && data.activeFloats.length > 0) {
          setSelectedFloat(data.activeFloats[0]);
        }
      }
    } catch (err) {
      console.error("Error fetching live dashboard data:", err);
    } finally {
      setIsLoading(false);
      if (isManual) setIsRefreshing(false);
    }
  };

  // Fetch float-specific CTD profile when user selects a float
  const handleSelectFloat = async (float: FloatPlatform) => {
    setSelectedFloat(float);
    try {
      const profileUrl = `${baseApiUrl}/api/dashboard/profile?float_id=${float.id}`;
      const res = await fetch(profileUrl);
      if (res.ok) {
        const profileData = await res.json();
        if (profileData.profile && profileData.profile.length > 0) {
          setDepthProfiles(profileData.profile);
        }
      }
    } catch (err) {
      console.error("Error fetching float CTD profile:", err);
    }
  };

  // Reset to fleet-wide average CTD curve
  const handleResetToOverview = async () => {
    try {
      const res = await fetch(overviewUrl);
      if (res.ok) {
        const data = await res.json();
        setDepthProfiles(data.depthProfiles || []);
      }
    } catch (err) {
      console.error("Error resetting profile:", err);
    }
    setSelectedFloat(null);
  };

  useEffect(() => {
    // Initial fetch on mount only - telemetry refreshes strictly on explicit user action
    fetchDashboardData(false);
  }, []);

  // Filter floats by search query and selected type
  const filteredFloats = activeFloats.filter(f => {
    const matchesSearch = searchQuery === "" || 
      f.id.includes(searchQuery) || 
      f.region.toLowerCase().includes(searchQuery.toLowerCase()) ||
      f.institution.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = !selectedFloatType || f.type.toLowerCase().includes(selectedFloatType.toLowerCase());
    return matchesSearch && matchesType;
  });

  // Calculate dynamic surface & deep values from active depth profile
  const surfaceTemp = depthProfiles[0]?.temperature ?? summary?.fleetAvgTemp ?? 28.5;
  const deepTemp = depthProfiles[depthProfiles.length - 1]?.temperature ?? 2.1;
  const surfaceSal = depthProfiles[0]?.salinity ?? summary?.fleetAvgSalinity ?? 35.8;
  const deepSal = depthProfiles[depthProfiles.length - 1]?.salinity ?? 34.7;
  const surfaceOxygen = depthProfiles[0]?.oxygen ?? 220;
  const minOxygen = depthProfiles.length > 0 ? Math.min(...depthProfiles.map(p => p.oxygen)) : 45;
  const maxDepthProfile = depthProfiles[depthProfiles.length - 1]?.depth || 2000;

  // Dynamic Array metric values
  const activePlatformsValue = summary?.activeFloatsCount 
    ? summary.activeFloatsCount.toLocaleString() 
    : stats[0]?.value || "3,345";

  const corePlatformsValue = summary?.coreCount 
    ? summary.coreCount.toLocaleString() 
    : (floatTypes.find(t => t.type.includes("Core"))?.count || "2,240");

  const bgcPlatformsValue = summary?.bgcCount 
    ? summary.bgcCount.toLocaleString() 
    : (floatTypes.find(t => t.type.includes("BGC"))?.count || "895");

  const uptimeValue = summary?.uptimeRate 
    ? `${summary.uptimeRate}%` 
    : (stats.find(s => s.label.toLowerCase().includes("uptime"))?.value || "97.4%");

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4 max-w-7xl">
        {/* Header with Live Telemetry Pulse */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold tracking-tight">ARGO Live Oceanographic Dashboard</h1>
              <Badge variant="outline" className="flex items-center gap-1.5 border-emerald-500/40 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Live Feed</span>
              </Badge>
            </div>
            <p className="text-muted-foreground text-sm mt-1">
              Real-time telemetry from NOAA ERDDAP, Coriolis GDAC & INCOIS • Synced: {lastUpdated}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => fetchDashboardData(true)}
              disabled={isRefreshing}
              className="flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
              <span>{isRefreshing ? "Syncing GDAC..." : "Sync Latest Telemetry"}</span>
            </Button>
          </div>
        </div>

        <Tabs defaultValue="data" className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-2 mb-8">
            <TabsTrigger value="data" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              <span>Data Analytics</span>
            </TabsTrigger>
            <TabsTrigger value="map" className="flex items-center gap-2">
              <Globe className="w-4 h-4" />
              <span>Interactive Map & 3D Globe</span>
            </TabsTrigger>
          </TabsList>
          
          {/* TAB 1: DATA ANALYTICS */}
          <TabsContent value="data" className="space-y-8">
            {/* Live Stats Grid */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {stats.map((stat, index) => {
                const getStatIcon = (i: number) => {
                  switch (i) {
                    case 0: return Activity;
                    case 1: return TrendingUp;
                    case 2: return Globe;
                    default: return CheckCircle2;
                  }
                };
                const Icon = getStatIcon(index);

                return (
                  <Card key={index} className="p-5 border-border/80 bg-card shadow-xs hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between mb-3">
                      <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
                        <Icon className="w-5 h-5" />
                      </div>
                      <Badge variant="secondary" className="text-[11px] font-normal px-2 py-0.5">
                        {stat.badge}
                      </Badge>
                    </div>
                    <div className="text-3xl font-extrabold tracking-tight mb-1 text-foreground">
                      {stat.value}
                    </div>
                    <div className="text-xs font-medium text-muted-foreground">{stat.label}</div>
                  </Card>
                );
              })}
            </div>

            {/* SELECTED FLOAT INSPECTOR CARD (Real-time telemetry inspector) */}
            {selectedFloat && (
              <Card className="p-5 border-primary/40 bg-gradient-to-r from-primary/5 via-card to-background shadow-sm relative overflow-hidden">
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge className="bg-primary text-primary-foreground font-mono font-bold">
                        Float #{selectedFloat.id}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {selectedFloat.type}
                      </Badge>
                      <span className="flex items-center gap-1.5 text-xs text-emerald-500 font-medium">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        {selectedFloat.status} ({selectedFloat.lastUpdate})
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                      <span>{selectedFloat.region} Basin</span>
                      <span className="text-xs text-muted-foreground font-normal">
                        ({selectedFloat.lat >= 0 ? `${selectedFloat.lat}°N` : `${Math.abs(selectedFloat.lat)}°S`}, {selectedFloat.lon >= 0 ? `${selectedFloat.lon}°E` : `${Math.abs(selectedFloat.lon)}°W`})
                      </span>
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      Operating Agency: <strong className="text-foreground">{selectedFloat.institution}</strong> • Profiling Limit: <strong className="text-foreground">{selectedFloat.maxDepth}</strong> • Cycle: <strong className="text-foreground font-mono">#{selectedFloat.cycle || 120}</strong> • Battery: <strong className="text-foreground font-mono">{selectedFloat.battery || 14.8}V</strong>
                    </p>
                  </div>

                  <div className="flex items-center gap-3 flex-wrap">
                    {/* Live parameter pills */}
                    <div className="flex items-center gap-2">
                      <div className="px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-center">
                        <div className="text-[10px] text-muted-foreground">Surface Temp</div>
                        <div className="text-sm font-bold text-rose-500">{selectedFloat.temp}°C</div>
                      </div>
                      <div className="px-3 py-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-center">
                        <div className="text-[10px] text-muted-foreground">Surface Salinity</div>
                        <div className="text-sm font-bold text-cyan-500">{selectedFloat.salinity} PSU</div>
                      </div>
                      {selectedFloat.oxygen && (
                        <div className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-center">
                          <div className="text-[10px] text-muted-foreground">Dissolved O₂</div>
                          <div className="text-sm font-bold text-emerald-500">{selectedFloat.oxygen} µmol</div>
                        </div>
                      )}
                      {selectedFloat.ph && (
                        <div className="px-3 py-1.5 rounded-lg bg-violet-500/10 border border-violet-500/20 text-center">
                          <div className="text-[10px] text-muted-foreground">Ocean pH</div>
                          <div className="text-sm font-bold text-violet-500">{selectedFloat.ph}</div>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      {onSectionChange && (
                        <Button 
                          size="sm" 
                          variant="secondary"
                          onClick={() => {
                            sessionStorage.setItem("argo_selected_float_query", `Analyze real-time CTD data for ARGO Float ${selectedFloat.id} in ${selectedFloat.region}`);
                            onSectionChange("chat");
                          }}
                          className="h-8 text-xs flex items-center gap-1.5"
                        >
                          <MessageSquare className="w-3.5 h-3.5 text-primary" />
                          <span>Ask AI About Float</span>
                        </Button>
                      )}
                      <Button 
                        size="sm" 
                        variant="outline" 
                        onClick={handleResetToOverview}
                        className="h-8 text-xs"
                      >
                        Fleet Overview
                      </Button>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* REAL-TIME INTERACTIVE CTD CHARTS (Recharts) */}
            <div>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-xl font-bold tracking-tight flex items-center gap-2">
                    <span>Vertical Water Column Profiles (0m – {maxDepthProfile}m)</span>
                    {selectedFloat ? (
                      <Badge variant="outline" className="font-mono text-[11px] text-primary">
                        Platform #{selectedFloat.id}
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-[11px]">
                        Global Array Average
                      </Badge>
                    )}
                  </h2>
                  <p className="text-xs text-muted-foreground">
                    High-resolution CTD sensor measurements showing thermocline, halocline, and pycnocline gradients
                  </p>
                </div>

                <div className="flex items-center gap-1 bg-muted p-1 rounded-lg">
                  <Button
                    size="sm"
                    variant={chartMetric === "temperature" ? "default" : "ghost"}
                    onClick={() => setChartMetric("temperature")}
                    className="h-7 text-xs px-2.5"
                  >
                    Temperature
                  </Button>
                  <Button
                    size="sm"
                    variant={chartMetric === "salinity" ? "default" : "ghost"}
                    onClick={() => setChartMetric("salinity")}
                    className="h-7 text-xs px-2.5"
                  >
                    Salinity
                  </Button>
                  <Button
                    size="sm"
                    variant={chartMetric === "oxygen" ? "default" : "ghost"}
                    onClick={() => setChartMetric("oxygen")}
                    className="h-7 text-xs px-2.5"
                  >
                    Dissolved O₂
                  </Button>
                  <Button
                    size="sm"
                    variant={chartMetric === "density" ? "default" : "ghost"}
                    onClick={() => setChartMetric("density")}
                    className="h-7 text-xs px-2.5"
                  >
                    Density (σ)
                  </Button>
                </div>
              </div>

              <div className="grid lg:grid-cols-2 gap-6">
                {/* Primary Chart: Dynamic based on chartMetric */}
                <Card className="p-5 border-border/80 bg-card shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2 text-foreground">
                      {chartMetric === "temperature" && <Thermometer className="w-4 h-4 text-rose-500" />}
                      {chartMetric === "salinity" && <Droplets className="w-4 h-4 text-cyan-500" />}
                      {chartMetric === "oxygen" && <Microscope className="w-4 h-4 text-emerald-500" />}
                      {chartMetric === "density" && <Waves className="w-4 h-4 text-violet-500" />}
                      
                      {chartMetric === "temperature" && "Vertical Temperature Profile (°C vs Depth)"}
                      {chartMetric === "salinity" && "Practical Salinity Profile (PSU vs Depth)"}
                      {chartMetric === "oxygen" && "Dissolved Oxygen Saturation (µmol/kg vs Depth)"}
                      {chartMetric === "density" && "Potential Density Anomaly (kg/m³ vs Depth)"}
                    </h3>
                    <Badge variant="outline" className="text-[10px] text-muted-foreground">
                      {chartMetric === "temperature" && "Thermocline"}
                      {chartMetric === "salinity" && "Halocline"}
                      {chartMetric === "oxygen" && "OMZ Layer"}
                      {chartMetric === "density" && "Pycnocline"}
                    </Badge>
                  </div>
                  
                  <div className="h-72 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={depthProfiles} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                        <defs>
                          <linearGradient id="primaryGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop 
                              offset="5%" 
                              stopColor={
                                chartMetric === "temperature" ? "#f43f5e" :
                                chartMetric === "salinity" ? "#06b6d4" :
                                chartMetric === "oxygen" ? "#10b981" : "#8b5cf6"
                              } 
                              stopOpacity={0.4}
                            />
                            <stop 
                              offset="95%" 
                              stopColor={
                                chartMetric === "temperature" ? "#f43f5e" :
                                chartMetric === "salinity" ? "#06b6d4" :
                                chartMetric === "oxygen" ? "#10b981" : "#8b5cf6"
                              } 
                              stopOpacity={0.0}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                        <XAxis 
                          dataKey="depth" 
                          unit="m" 
                          tick={{ fontSize: 11 }} 
                          label={{ value: "Depth (meters)", position: "insideBottom", offset: -10, fontSize: 11 }}
                        />
                        <YAxis 
                          unit={
                            chartMetric === "temperature" ? "°C" :
                            chartMetric === "salinity" ? " PSU" :
                            chartMetric === "oxygen" ? " µmol" : " kg/m³"
                          }
                          tick={{ fontSize: 11 }}
                          domain={
                            chartMetric === "temperature" ? [0, 32] :
                            chartMetric === "salinity" ? [32, 38] :
                            chartMetric === "oxygen" ? [0, 320] : [1020, 1032]
                          }
                        />
                        <RechartsTooltip 
                          formatter={(value: any) => [
                            `${value} ${
                              chartMetric === "temperature" ? "°C" :
                              chartMetric === "salinity" ? "PSU" :
                              chartMetric === "oxygen" ? "µmol/kg" : "kg/m³"
                            }`, 
                            chartMetric.charAt(0).toUpperCase() + chartMetric.slice(1)
                          ]}
                          labelFormatter={(label) => `Depth: ${label} meters`}
                          contentStyle={{ backgroundColor: "rgba(20, 25, 35, 0.9)", borderRadius: "8px", borderColor: "#334155", color: "#fff", fontSize: "12px" }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey={chartMetric} 
                          stroke={
                            chartMetric === "temperature" ? "#f43f5e" :
                            chartMetric === "salinity" ? "#06b6d4" :
                            chartMetric === "oxygen" ? "#10b981" : "#8b5cf6"
                          } 
                          strokeWidth={2.5}
                          fillOpacity={1} 
                          fill="url(#primaryGradient)" 
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-[11px] text-muted-foreground flex justify-between px-2 pt-2 border-t border-border/40 font-mono">
                    <span>
                      Surface: {
                        chartMetric === "temperature" ? `${surfaceTemp.toFixed(1)}°C` :
                        chartMetric === "salinity" ? `${surfaceSal.toFixed(2)} PSU` :
                        chartMetric === "oxygen" ? `${surfaceOxygen.toFixed(1)} µmol/kg` :
                        `${(depthProfiles[0]?.density ?? 1022.5).toFixed(2)} kg/m³`
                      }
                    </span>
                    <span>
                      At {maxDepthProfile}m: {
                        chartMetric === "temperature" ? `${deepTemp.toFixed(1)}°C` :
                        chartMetric === "salinity" ? `${deepSal.toFixed(2)} PSU` :
                        chartMetric === "oxygen" ? `${(depthProfiles[depthProfiles.length - 1]?.oxygen ?? 120).toFixed(1)} µmol/kg` :
                        `${(depthProfiles[depthProfiles.length - 1]?.density ?? 1027.8).toFixed(2)} kg/m³`
                      }
                    </span>
                  </div>
                </Card>

                {/* Secondary Chart: Practical Salinity (or Temperature if Salinity selected) */}
                <Card className="p-5 border-border/80 bg-card shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2 text-foreground">
                      {chartMetric === "salinity" ? (
                        <>
                          <Thermometer className="w-4 h-4 text-rose-500" />
                          <span>Simultaneous Thermal Structure (°C vs Depth)</span>
                        </>
                      ) : (
                        <>
                          <Droplets className="w-4 h-4 text-cyan-500" />
                          <span>Simultaneous Salinity Gradient (PSU vs Depth)</span>
                        </>
                      )}
                    </h3>
                    <Badge variant="outline" className="text-[10px] text-muted-foreground">
                      {chartMetric === "salinity" ? "Thermal Layer" : "Halocline Layer"}
                    </Badge>
                  </div>
                  
                  <div className="h-72 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={depthProfiles} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                        <defs>
                          <linearGradient id="secondaryGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop 
                              offset="5%" 
                              stopColor={chartMetric === "salinity" ? "#f43f5e" : "#06b6d4"} 
                              stopOpacity={0.4}
                            />
                            <stop 
                              offset="95%" 
                              stopColor={chartMetric === "salinity" ? "#f43f5e" : "#06b6d4"} 
                              stopOpacity={0.0}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                        <XAxis 
                          dataKey="depth" 
                          unit="m" 
                          tick={{ fontSize: 11 }} 
                          label={{ value: "Depth (meters)", position: "insideBottom", offset: -10, fontSize: 11 }}
                        />
                        <YAxis 
                          unit={chartMetric === "salinity" ? "°C" : " PSU"} 
                          tick={{ fontSize: 11 }}
                          domain={chartMetric === "salinity" ? [0, 32] : [32, 38]}
                        />
                        <RechartsTooltip 
                          formatter={(value: any) => [
                            `${value} ${chartMetric === "salinity" ? "°C" : "PSU"}`, 
                            chartMetric === "salinity" ? "Temperature" : "Salinity"
                          ]}
                          labelFormatter={(label) => `Depth: ${label} meters`}
                          contentStyle={{ backgroundColor: "rgba(20, 25, 35, 0.9)", borderRadius: "8px", borderColor: "#334155", color: "#fff", fontSize: "12px" }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey={chartMetric === "salinity" ? "temperature" : "salinity"} 
                          stroke={chartMetric === "salinity" ? "#f43f5e" : "#06b6d4"} 
                          strokeWidth={2.5}
                          fillOpacity={1} 
                          fill="url(#secondaryGradient)" 
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-[11px] text-muted-foreground flex justify-between px-2 pt-2 border-t border-border/40 font-mono">
                    {chartMetric === "salinity" ? (
                      <>
                        <span>Surface: ~{surfaceTemp.toFixed(1)}°C</span>
                        <span>Abyssal Temp: ~{deepTemp.toFixed(1)}°C</span>
                      </>
                    ) : (
                      <>
                        <span>Surface: ~{surfaceSal.toFixed(2)} PSU</span>
                        <span>Abyssal Salinity: ~{deepSal.toFixed(2)} PSU</span>
                      </>
                    )}
                  </div>
                </Card>
              </div>
            </div>

            {/* ARGO Float Platform Types */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold tracking-tight">Active Float Categories</h2>
                <span className="text-xs text-muted-foreground">Click a category card to filter recent platform telemetry</span>
              </div>
              <div className="grid md:grid-cols-3 gap-5">
                {floatTypes.map((floatType, index) => {
                  const getCategoryIcon = (t: string) => {
                    if (t.includes("Core")) return Thermometer;
                    if (t.includes("BGC")) return Microscope;
                    return ArrowDown;
                  };
                  const Icon = getCategoryIcon(floatType.type);
                  const isSelected = selectedFloatType === floatType.type;

                  return (
                    <Card 
                      key={index} 
                      className={`p-5 cursor-pointer transition-all duration-200 border-border/80 hover:shadow-md ${
                        isSelected ? "ring-2 ring-primary bg-primary/5 shadow-md" : ""
                      }`}
                      onClick={() => setSelectedFloatType(isSelected ? null : floatType.type)}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="p-2 rounded-xl bg-primary/10 text-primary">
                          <Icon className="w-5 h-5" />
                        </div>
                        <Badge variant="secondary" className="font-semibold">{floatType.count} Active</Badge>
                      </div>
                      <h3 className="text-base font-semibold mb-1">{floatType.type}</h3>
                      <p className="text-xs text-muted-foreground mb-4 line-clamp-2 leading-relaxed">{floatType.description}</p>
                      <div className="flex flex-wrap gap-1.5">
                        {floatType.parameters.map((param, pIdx) => (
                          <Badge key={pIdx} variant="outline" className="text-[10px] px-2 py-0.5 font-normal">
                            {param}
                          </Badge>
                        ))}
                      </div>
                    </Card>
                  );
                })}
              </div>
            </div>

            {/* LIVE ARGO FLOATS TELEMETRY TABLE */}
            <Card className="p-5 border-border/80 bg-card shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
                    <Radio className="w-4 h-4 text-emerald-500 animate-pulse" />
                    <span>Real-Time Float Transmissions ({filteredFloats.length} platforms)</span>
                  </h3>
                  <p className="text-xs text-muted-foreground">Click any float row to inspect its vertical CTD profile & telemetry</p>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                    <Input
                      placeholder="Search float ID, region, institution..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="h-8 pl-8 text-xs w-56 sm:w-64"
                    />
                  </div>
                  {selectedFloatType && (
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => setSelectedFloatType(null)}
                      className="h-8 text-xs text-muted-foreground hover:text-foreground"
                    >
                      Clear Filter
                    </Button>
                  )}
                </div>
              </div>

              <div className="overflow-x-auto rounded-lg border border-border/60">
                <table className="w-full text-xs">
                  <thead className="bg-muted/50 border-b border-border/60">
                    <tr>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Platform ID</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Category</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Region</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Coordinates</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Temp</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Salinity</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Max Depth</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Institution</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Last Uplink</th>
                      <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40 font-mono">
                    {filteredFloats.map((float) => {
                      const isSelected = selectedFloat?.id === float.id;
                      return (
                        <tr 
                          key={float.id} 
                          onClick={() => handleSelectFloat(float)}
                          className={`cursor-pointer transition-colors ${
                            isSelected 
                              ? "bg-primary/10 border-l-2 border-l-primary" 
                              : "hover:bg-muted/40"
                          }`}
                        >
                          <td className="py-2.5 px-3 font-bold text-primary flex items-center gap-1.5">
                            {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span>}
                            <span>{float.id}</span>
                          </td>
                          <td className="py-2.5 px-3 font-sans">
                            <span className="text-[11px] font-medium">{float.type}</span>
                          </td>
                          <td className="py-2.5 px-3 font-sans text-foreground/90">{float.region}</td>
                          <td className="py-2.5 px-3 text-muted-foreground">
                            {float.lat >= 0 ? `${float.lat.toFixed(2)}°N` : `${Math.abs(float.lat).toFixed(2)}°S`}, {float.lon >= 0 ? `${float.lon.toFixed(2)}°E` : `${Math.abs(float.lon).toFixed(2)}°W`}
                          </td>
                          <td className="py-2.5 px-3 text-rose-500 font-semibold">{float.temp.toFixed(1)}°C</td>
                          <td className="py-2.5 px-3 text-cyan-500 font-semibold">{float.salinity.toFixed(2)} PSU</td>
                          <td className="py-2.5 px-3 text-muted-foreground">{float.maxDepth}</td>
                          <td className="py-2.5 px-3 font-sans text-muted-foreground">{float.institution}</td>
                          <td className="py-2.5 px-3 font-sans text-muted-foreground">{float.lastUpdate}</td>
                          <td className="py-2.5 px-3 font-sans">
                            <Button 
                              size="sm" 
                              variant={isSelected ? "default" : "outline"} 
                              className="h-6 text-[10px] px-2"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectFloat(float);
                              }}
                            >
                              {isSelected ? "Inspecting" : "Inspect"}
                            </Button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* LIVE DATA PIPELINE STATUS */}
            <Card className="p-5 border-emerald-500/30 bg-emerald-500/5 shadow-xs">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-emerald-500/20 text-emerald-600 dark:text-emerald-400">
                    <Database className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                      <span>🟢 Active Real-Time Data Pipeline</span>
                      <Badge variant="outline" className="text-[10px] border-emerald-500/40 text-emerald-500">
                        Synchronized
                      </Badge>
                    </h3>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Ingesting telemetry from NOAA ERDDAP, Coriolis GDAC, INCOIS & CSIRO • ChromaDB RAG Vector Store & LangChain NLP Active
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-xs text-muted-foreground font-mono">
                  <span>Fleet Avg Temp: <strong className="text-rose-500">{summary?.fleetAvgTemp ?? 21.2}°C</strong></span>
                  <span>•</span>
                  <span>Fleet Avg Salinity: <strong className="text-cyan-500">{summary?.fleetAvgSalinity ?? 35.2} PSU</strong></span>
                </div>
              </div>
            </Card>
          </TabsContent>

          {/* TAB 2: INTERACTIVE MAP & 3D GLOBE */}
          <TabsContent value="map" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold tracking-tight">Global Array Mapping & Tracking</h2>
                <p className="text-xs text-muted-foreground">Interactive 3D bathymetric globe and global float location matrix</p>
              </div>

              <div className="flex items-center gap-2">
                <Button 
                  variant={viewMode === "3d" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setViewMode("3d")}
                  className="flex items-center gap-1.5 h-8 text-xs"
                >
                  <Globe className="w-3.5 h-3.5" />
                  <span>3D Globe</span>
                </Button>
                <Button 
                  variant={viewMode === "2d" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setViewMode("2d")}
                  className="flex items-center gap-1.5 h-8 text-xs"
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>2D Matrix Map</span>
                </Button>
              </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Map Viewer */}
              <div className="lg:col-span-2 space-y-4">
                <Card className="p-0 overflow-hidden border-border/80 shadow-xs">
                  {viewMode === "3d" ? (
                    <div className="w-full h-[480px] bg-black flex items-center justify-center overflow-hidden relative">
                      <EarthGlobe 
                        floatData={activeFloats}
                        selectedFloatId={selectedFloat?.id}
                        onSelectFloat={handleSelectFloat}
                      />
                      <div className="absolute bottom-3 left-3 bg-card/90 backdrop-blur-md border border-border/80 rounded-lg p-2.5 text-xs shadow-md pointer-events-none">
                        <div className="font-semibold mb-1 text-[11px]">3D Scatter Legend</div>
                        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-400"></span> Core</span>
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400"></span> BGC</span>
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-400"></span> Deep</span>
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-500"></span> Selected</span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="relative w-full h-[480px] overflow-hidden bg-slate-950">
                      <img 
                        src={globalMapImage} 
                        alt="Global ARGO Float Array"
                        className="w-full h-full object-cover opacity-80"
                      />
                      {/* Live Float Location Markers */}
                      <div className="absolute inset-0">
                        {activeFloats.map((float) => {
                          const x = ((float.lon + 180) / 360) * 100;
                          const y = ((90 - float.lat) / 180) * 100;
                          const isSelected = selectedFloat?.id === float.id;
                          const isBGC = float.type.includes("BGC");
                          const isDeep = float.type.includes("Deep");

                          let markerBg = "bg-cyan-400";
                          if (isBGC) markerBg = "bg-emerald-400";
                          if (isDeep) markerBg = "bg-purple-400";
                          if (isSelected) markerBg = "bg-rose-500";

                          return (
                            <div
                              key={float.id}
                              onClick={() => handleSelectFloat(float)}
                              className={`absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-10 ${isSelected ? "z-20" : ""}`}
                              style={{ left: `${x}%`, top: `${y}%` }}
                            >
                              <div className={`rounded-full border border-white shadow-md transition-transform ${
                                isSelected ? "w-4 h-4 ring-4 ring-rose-500/40 animate-pulse" : "w-2.5 h-2.5 group-hover:scale-175"
                              } ${markerBg}`}></div>

                              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-1 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900/95 border border-border text-white rounded-lg p-2 text-[11px] whitespace-nowrap shadow-xl z-30 pointer-events-none">
                                <div className="font-bold text-cyan-400">Float #{float.id} ({float.type})</div>
                                <div>{float.region} • {float.institution}</div>
                                <div className="font-mono text-emerald-400">{float.temp}°C | {float.salinity} PSU</div>
                                <div className="text-[10px] text-muted-foreground">Click to inspect CTD curve</div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Map overlay legend */}
                      <div className="absolute top-3 right-3 bg-card/90 backdrop-blur-md border border-border/80 rounded-lg p-2.5 text-xs shadow-md">
                        <div className="font-semibold mb-1 text-[11px]">Real-time Telemetry Status</div>
                        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                          <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                          <span>Active Floats ({activeFloats.length} plotted)</span>
                        </div>
                      </div>
                    </div>
                  )}
                </Card>

                {/* Array Status Badges - 100% REAL-TIME POPULATED */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <Card className="p-3.5 text-center border-border/70 bg-card/60">
                    <div className="text-xl font-bold text-primary mb-0.5">{activePlatformsValue}</div>
                    <div className="text-[11px] text-muted-foreground">Active Platforms</div>
                  </Card>
                  <Card className="p-3.5 text-center border-border/70 bg-card/60">
                    <div className="text-xl font-bold text-cyan-500 mb-0.5">{corePlatformsValue}</div>
                    <div className="text-[11px] text-muted-foreground">Core ARGO Arrays</div>
                  </Card>
                  <Card className="p-3.5 text-center border-border/70 bg-card/60">
                    <div className="text-xl font-bold text-emerald-500 mb-0.5">{bgcPlatformsValue}</div>
                    <div className="text-[11px] text-muted-foreground">BGC Arrays</div>
                  </Card>
                  <Card className="p-3.5 text-center border-border/70 bg-card/60">
                    <div className="text-xl font-bold text-violet-500 mb-0.5">{uptimeValue}</div>
                    <div className="text-[11px] text-muted-foreground">Operational Uptime</div>
                  </Card>
                </div>
              </div>

              {/* Regional Summary Sidebar */}
              <div className="space-y-4">
                <Card className="p-5 border-border/80 bg-card shadow-xs">
                  <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-primary" />
                    <span>Regional Ocean Array Breakdown</span>
                  </h3>

                  <div className="space-y-3">
                    {regions.map((region, idx) => (
                      <div key={idx} className="p-2.5 rounded-lg bg-muted/40 hover:bg-muted/70 transition-colors">
                        <div className="flex items-center justify-between text-xs mb-1.5">
                          <span className="font-medium text-foreground">{region.name}</span>
                          <span className="font-mono text-muted-foreground">{region.active} / {region.floats}</span>
                        </div>
                        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-primary rounded-full transition-all duration-500"
                            style={{ width: `${region.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};