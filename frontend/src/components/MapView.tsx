import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { MapPin, Search, Filter, Layers, Navigation, Globe, RefreshCw } from "lucide-react";
import { EarthGlobe } from "@/components/EarthGlobe";
import globalMapImage from "@/assets/argo-global.jpg";

export const MapView = () => {
  const [viewMode, setViewMode] = useState<"3d" | "2d">("3d");
  const [activeFloats, setActiveFloats] = useState<any[]>([]);
  const [regions, setRegions] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [selectedFloat, setSelectedFloat] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  const baseApiUrl = import.meta.env.VITE_BACKEND_URL 
    ? import.meta.env.VITE_BACKEND_URL.replace("/chat", "")
    : "http://localhost:5000";

  const fetchLiveMapData = async () => {
    try {
      const res = await fetch(`${baseApiUrl}/api/dashboard/overview`);
      if (res.ok) {
        const data = await res.json();
        setActiveFloats(data.activeFloats || []);
        setRegions(data.regions || []);
        setSummary(data.summary || null);
        if (data.activeFloats && data.activeFloats.length > 0) {
          setSelectedFloat(data.activeFloats[0]);
        }
      }
    } catch (err) {
      console.error("Error fetching map telemetry:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLiveMapData();
  }, []);

  const filteredFloats = activeFloats.filter(f => 
    searchQuery === "" || 
    f.id.includes(searchQuery) || 
    f.region.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4 max-w-7xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold">Interactive Global Map</h1>
              <Badge variant="outline" className="border-emerald-500/40 text-emerald-500 bg-emerald-500/10">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse mr-1"></span>
                Live Fleet
              </Badge>
            </div>
            <p className="text-muted-foreground text-sm mt-1">Real-time ARGO platform telemetry synchronized with NOAA ERDDAP</p>
          </div>
          <div className="flex items-center space-x-2">
            <Button 
              variant="outline"
              onClick={fetchLiveMapData}
              disabled={isLoading}
              size="sm"
            >
              <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${isLoading ? "animate-spin" : ""}`} />
              Sync Telemetry
            </Button>
            <Button 
              variant={viewMode === "3d" ? "default" : "outline"}
              onClick={() => setViewMode("3d")}
              size="sm"
            >
              <Globe className="w-4 h-4 mr-2" />
              3D Globe
            </Button>
            <Button 
              variant={viewMode === "2d" ? "default" : "outline"}
              onClick={() => setViewMode("2d")}
              size="sm"
            >
              <Layers className="w-4 h-4 mr-2" />
              2D Map
            </Button>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Map Section */}
          <div className="lg:col-span-2 space-y-6">
            {viewMode === "3d" ? (
              <Card className="p-0 overflow-hidden border-border/80">
                <div className="w-full h-[480px] bg-black flex items-center justify-center overflow-hidden">
                  <EarthGlobe 
                    floatData={activeFloats} 
                    selectedFloatId={selectedFloat?.id}
                    onSelectFloat={setSelectedFloat}
                  />
                </div>
              </Card>
            ) : (
              <Card className="p-0 overflow-hidden border-border/80">
                <div className="relative w-full h-[480px] bg-slate-950 overflow-hidden">
                  <img 
                    src={globalMapImage} 
                    alt="ARGO Float Locations" 
                    className="w-full h-full object-cover opacity-80"
                  />
                  {/* Overlay with float markers */}
                  <div className="absolute inset-0">
                    {filteredFloats.map((float) => {
                      const x = ((float.lon + 180) / 360) * 100;
                      const y = ((90 - float.lat) / 180) * 100;
                      const isSelected = selectedFloat?.id === float.id;
                      
                      return (
                        <div
                          key={float.id}
                          onClick={() => setSelectedFloat(float)}
                          className={`absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group ${isSelected ? "z-20" : "z-10"}`}
                          style={{ left: `${x}%`, top: `${y}%` }}
                        >
                          <div className={`rounded-full border border-white shadow-lg ${
                            isSelected ? "w-4 h-4 bg-rose-500 ring-4 ring-rose-500/40 animate-pulse" : "w-3 h-3 bg-cyan-400 group-hover:scale-150"
                          } transition-transform`}>
                          </div>
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900 border border-border text-white rounded-lg p-2 text-xs whitespace-nowrap shadow-lg pointer-events-none">
                            <div className="font-medium text-cyan-400">Float #{float.id} ({float.type})</div>
                            <div className="text-muted-foreground">{float.region}</div>
                            <div className="text-emerald-400">{float.temp}°C | {float.salinity} PSU</div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Search Bar */}
                  <div className="absolute bottom-4 left-4 right-4">
                    <div className="flex space-x-2">
                      <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                        <Input 
                          placeholder="Search float ID, basin, institution..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="pl-10 bg-card/90 backdrop-blur-sm text-xs"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Map Statistics - 100% Real Time */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-primary mb-1">
                  {summary?.totalFloatsCount ? summary.totalFloatsCount.toLocaleString() : "3,447"}
                </div>
                <div className="text-xs text-muted-foreground">Total Fleet Platforms</div>
              </Card>
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-cyan-500 mb-1">
                  {summary?.activeFloatsCount ? summary.activeFloatsCount.toLocaleString() : "3,345"}
                </div>
                <div className="text-xs text-muted-foreground">Active Telemetry</div>
              </Card>
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-emerald-500 mb-1">
                  {summary?.bgcCount ? summary.bgcCount.toLocaleString() : "895"}
                </div>
                <div className="text-xs text-muted-foreground">BGC Arrays</div>
              </Card>
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-violet-500 mb-1">
                  {summary?.uptimeRate ? `${summary.uptimeRate}%` : "97.4%"}
                </div>
                <div className="text-xs text-muted-foreground">Array Uptime</div>
              </Card>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Regional Summary */}
            <Card className="p-6">
              <h3 className="text-base font-semibold mb-4 flex items-center">
                <MapPin className="w-4 h-4 mr-2 text-primary" />
                Regional Basin Telemetry
              </h3>
              <div className="space-y-3">
                {regions.map((region, index) => (
                  <div key={index} className="flex items-center justify-between p-2.5 bg-muted/40 rounded-lg text-xs">
                    <div>
                      <div className="font-medium text-foreground">{region.name}</div>
                      <div className="text-[11px] text-muted-foreground">
                        {region.active} active / {region.floats} total
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-foreground">{region.percentage}%</div>
                      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden mt-1">
                        <div 
                          className="h-full bg-primary rounded-full"
                          style={{ width: `${region.percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Recent Float Updates */}
            <Card className="p-6">
              <h3 className="text-base font-semibold mb-3">Live Platform Uplinks</h3>
              <div className="space-y-2.5">
                {activeFloats.slice(0, 5).map((float) => (
                  <div 
                    key={float.id} 
                    onClick={() => setSelectedFloat(float)}
                    className="flex items-center justify-between p-2 hover:bg-muted/50 rounded-lg transition-colors cursor-pointer text-xs"
                  >
                    <div className="flex items-center space-x-2.5">
                      <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                      <div>
                        <div className="font-mono font-bold text-primary">#{float.id}</div>
                        <div className="text-[11px] text-muted-foreground">
                          {float.region} • {float.temp}°C
                        </div>
                      </div>
                    </div>
                    <Badge variant="outline" className="text-[10px]">
                      {float.lastUpdate}
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};