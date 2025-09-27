import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { MapPin, Search, Filter, Layers, Navigation, Globe } from "lucide-react";
import { EarthGlobe } from "@/components/EarthGlobe";
import globalMapImage from "@/assets/argo-global.jpg";

export const MapView = () => {
  const [viewMode, setViewMode] = useState("3d");
  
  const activeFloats = [
    { id: "2902746", lat: 19.8, lon: 64.7, status: "Active", lastUpdate: "2h ago", temp: 28.4, salinity: 35.2, pressure: 1013 },
    { id: "5906423", lat: -15.2, lon: 73.1, status: "Active", lastUpdate: "4h ago", temp: 26.8, salinity: 34.8, pressure: 1012 },
    { id: "2903351", lat: 8.5, lon: 76.3, status: "Active", lastUpdate: "1h ago", temp: 29.1, salinity: 35.5, pressure: 1015 },
    { id: "5906198", lat: 22.1, lon: 68.9, status: "Maintenance", lastUpdate: "12h ago", temp: 27.6, salinity: 35.0, pressure: 1011 },
    { id: "3901234", lat: 5.3, lon: 82.4, status: "Active", lastUpdate: "3h ago", temp: 28.9, salinity: 35.3, pressure: 1014 }
  ];

  const regions = [
    { name: "Arabian Sea", floats: 67, active: 64 },
    { name: "Bay of Bengal", floats: 89, active: 86 },
    { name: "Central Indian Ocean", floats: 156, active: 148 },
    { name: "Southern Ocean", floats: 203, active: 195 }
  ];

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Interactive Map</h1>
            <p className="text-muted-foreground">Global ARGO float locations and real-time tracking</p>
          </div>
          <div className="flex space-x-2">
            <Button 
              variant={viewMode === "3d" ? "default" : "outline"}
              onClick={() => setViewMode("3d")}
            >
              <Globe className="w-4 h-4 mr-2" />
              3D Globe
            </Button>
            <Button 
              variant={viewMode === "2d" ? "default" : "outline"}
              onClick={() => setViewMode("2d")}
            >
              <Layers className="w-4 h-4 mr-2" />
              2D Map
            </Button>
            <Button variant="outline">
              <Filter className="w-4 h-4 mr-2" />
              Filters
            </Button>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Map Section */}
          <div className="lg:col-span-2">
            {viewMode === "3d" ? (
              <Card className="p-0 overflow-hidden">
                <EarthGlobe floatData={activeFloats} />
              </Card>
            ) : (
              <Card className="p-0 overflow-hidden">
                <div className="relative">
                  <img 
                    src={globalMapImage} 
                    alt="ARGO Float Locations" 
                    className="w-full h-96 object-cover"
                  />
                  {/* Overlay with float markers */}
                  <div className="absolute inset-0">
                    {activeFloats.map((float, index) => {
                      // Convert lat/lon to approximate pixel positions (simplified)
                      const x = ((float.lon + 180) / 360) * 100;
                      const y = ((90 - float.lat) / 180) * 100;
                      
                      return (
                        <div
                          key={float.id}
                          className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
                          style={{ left: `${x}%`, top: `${y}%` }}
                        >
                          <div className={`w-3 h-3 rounded-full border-2 border-white shadow-lg ${
                            float.status === "Active" ? "bg-accent" : "bg-secondary"
                          } group-hover:scale-150 transition-transform`}>
                          </div>
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity bg-card border border-border rounded-lg p-2 text-xs whitespace-nowrap shadow-lg z-10">
                            <div className="font-medium">Float {float.id}</div>
                            <div className="text-muted-foreground">{float.lat}°, {float.lon}°</div>
                            <div className="text-muted-foreground">Temp: {float.temp}°C</div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Map Controls */}
                  <div className="absolute top-4 right-4 flex flex-col space-y-2">
                    <Button size="sm" variant="secondary" className="w-10 h-10 p-0">
                      <Navigation className="w-4 h-4" />
                    </Button>
                    <div className="bg-card border border-border rounded-lg p-2 text-xs">
                      <div className="font-medium mb-1">Legend</div>
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="w-2 h-2 bg-accent rounded-full"></div>
                        <span>Active</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-secondary rounded-full"></div>
                        <span>Maintenance</span>
                      </div>
                    </div>
                  </div>

                  {/* Search Bar */}
                  <div className="absolute bottom-4 left-4 right-4">
                    <div className="flex space-x-2">
                      <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                        <Input 
                          placeholder="Search by coordinates, float ID, or region..."
                          className="pl-10 bg-card/90 backdrop-blur-sm"
                        />
                      </div>
                      <Button className="bg-primary/90 backdrop-blur-sm">Search</Button>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Map Statistics */}
            <div className="grid md:grid-cols-4 gap-4 mt-6">
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-accent mb-1">4,247</div>
                <div className="text-sm text-muted-foreground">Total Floats</div>
              </Card>
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-data-temperature mb-1">4,089</div>
                <div className="text-sm text-muted-foreground">Active</div>
              </Card>
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-data-salinity mb-1">158</div>
                <div className="text-sm text-muted-foreground">Maintenance</div>
              </Card>
              <Card className="p-4 text-center">
                <div className="text-2xl font-bold text-data-depth mb-1">97%</div>
                <div className="text-sm text-muted-foreground">Operational</div>
              </Card>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Regional Summary */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <MapPin className="w-5 h-5 mr-2" />
                Regional Summary
              </h3>
              <div className="space-y-3">
                {regions.map((region, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                    <div>
                      <div className="font-medium text-sm">{region.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {region.active} active of {region.floats} total
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">{Math.round((region.active / region.floats) * 100)}%</div>
                      <div className="w-12 h-1 bg-secondary rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-accent rounded-full"
                          style={{ width: `${(region.active / region.floats) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Recent Float Updates */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Updates</h3>
              <div className="space-y-3">
                {activeFloats.slice(0, 4).map((float, index) => (
                  <div key={index} className="flex items-center justify-between p-2 hover:bg-secondary/50 rounded-lg transition-colors">
                    <div className="flex items-center space-x-3">
                      <div className={`w-2 h-2 rounded-full ${
                        float.status === "Active" ? "bg-accent" : "bg-secondary"
                      }`}></div>
                      <div>
                        <div className="font-mono text-sm">{float.id}</div>
                        <div className="text-xs text-muted-foreground">
                          {float.lat}°, {float.lon}°
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant="outline" className="text-xs">
                        {float.lastUpdate}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Quick Actions */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
              <div className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <MapPin className="w-4 h-4 mr-2" />
                  Find Nearest Float
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter by Data Type
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Search className="w-4 h-4 mr-2" />
                  Search by Region
                </Button>
              </div>
            </Card>
          </div>
        </div>

        {/* Backend Notice */}
        <Card className="mt-8 p-6 bg-accent/10 border-accent/20">
          <h3 className="text-lg font-semibold mb-2 text-accent">Interactive Earth Visualization</h3>
          <p className="text-muted-foreground">
            Experience oceanographic data through our interactive 3D globe with atmospheric overlays inspired by earth.nullschool.net. 
            Switch between temperature, wind patterns, and sea level pressure visualizations while tracking ARGO float positions in real-time.
          </p>
        </Card>
      </div>
    </div>
  );
};