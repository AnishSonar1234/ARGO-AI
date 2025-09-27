import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Thermometer, Droplets, Gauge, Activity, MapPin, Calendar, TrendingUp, Globe, Waves, Microscope, ArrowDown, Search, Filter, Layers, Navigation } from "lucide-react";
import globalMapImage from "@/assets/argo-global.jpg";
import { useState } from "react";
import { EarthGlobe } from "@/components/EarthGlobe";

export const Dashboard = () => {
  const [selectedFloatType, setSelectedFloatType] = useState<string | null>(null);
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
  const stats = [
    { label: "Active Floats", value: "4,247", icon: Activity, color: "text-data-temperature" },
    { label: "Recent Profiles", value: "18,423", icon: TrendingUp, color: "text-data-salinity" },
    { label: "Countries", value: "30+", icon: Globe, color: "text-data-depth" },
    { label: "Data Points", value: "2.3M", icon: Gauge, color: "text-data-bgc" }
  ];

  const recentFloatsByType = {
    "Core ARGO": [
      { id: "2902746", lat: "19.8°N", lon: "64.7°E", temp: "28.4°C", salinity: "35.2 PSU", depth: "1,847m", status: "Active" },
      { id: "5906423", lat: "15.2°S", lon: "73.1°E", temp: "26.8°C", salinity: "34.8 PSU", depth: "2,002m", status: "Active" },
      { id: "2903351", lat: "8.5°N", lon: "76.3°E", temp: "29.1°C", salinity: "35.0 PSU", depth: "1,923m", status: "Active" }
    ],
    "BGC ARGO": [
      { id: "5906789", lat: "12.3°N", lon: "68.2°E", oxygen: "210 μmol/kg", ph: "8.1", chlorophyll: "0.8 mg/m³", depth: "1,950m", status: "Active" },
      { id: "2903456", lat: "18.7°S", lon: "71.5°E", oxygen: "195 μmol/kg", ph: "8.0", chlorophyll: "1.2 mg/m³", depth: "2,100m", status: "Active" },
      { id: "5906890", lat: "25.1°N", lon: "65.8°E", oxygen: "225 μmol/kg", ph: "8.2", chlorophyll: "0.6 mg/m³", depth: "1,875m", status: "Maintenance" }
    ],
    "Deep ARGO": [
      { id: "7901234", lat: "20.5°N", lon: "67.3°E", temp: "2.1°C", salinity: "34.9 PSU", depth: "4,850m", status: "Active" },
      { id: "7901567", lat: "16.8°S", lon: "72.1°E", temp: "1.8°C", salinity: "34.7 PSU", depth: "5,200m", status: "Active" },
      { id: "7901890", lat: "22.9°N", lon: "69.4°E", temp: "2.3°C", salinity: "35.1 PSU", depth: "4,650m", status: "Active" }
    ]
  };

  const floatTypes = [
    {
      type: "Core ARGO",
      count: "3,247",
      icon: Thermometer,
      color: "bg-data-temperature",
      description: "Standard temperature, salinity, and pressure profiles",
      parameters: ["Temperature", "Salinity", "Pressure"],
      analytics: {
        activeFloats: "3,247",
        avgDepth: "2,000m",
        dataPoints: "1.8M",
        coverage: "Global Ocean"
      }
    },
    {
      type: "BGC ARGO",
      count: "856",
      icon: Microscope,
      color: "bg-data-bgc",
      description: "Biogeochemical sensors for ocean health monitoring",
      parameters: ["Oxygen", "pH", "Nitrate", "Chlorophyll"],
      analytics: {
        activeFloats: "856",
        avgDepth: "2,000m",
        dataPoints: "423K",
        coverage: "Major Ocean Basins"
      }
    },
    {
      type: "Deep ARGO",
      count: "144",
      icon: ArrowDown,
      color: "bg-data-depth",
      description: "Full ocean depth profiling to 6,000m",
      parameters: ["Deep Temperature", "Deep Salinity", "Full Ocean Column"],
      analytics: {
        activeFloats: "144",
        avgDepth: "4,500m",
        dataPoints: "87K",
        coverage: "Deep Ocean Regions"
      }
    }
  ];

  const dataTypes = [
    { name: "Temperature", count: "847K", color: "bg-data-temperature", description: "CTD temperature profiles" },
    { name: "Salinity", count: "847K", color: "bg-data-salinity", description: "Conductivity measurements" },
    { name: "Pressure/Depth", count: "847K", color: "bg-data-depth", description: "Depth pressure readings" },
    { name: "BGC Parameters", count: "234K", color: "bg-data-bgc", description: "Bio-geochemical data" }
  ];

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">ARGO Dashboard</h1>
            <p className="text-muted-foreground">Comprehensive oceanographic monitoring and analysis</p>
          </div>
          <Button className="bg-gradient-ocean">
            <Calendar className="w-4 h-4 mr-2" />
            View Reports
          </Button>
        </div>

        <Tabs defaultValue="data" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="data">Data Analytics</TabsTrigger>
            <TabsTrigger value="map">Interactive Map</TabsTrigger>
          </TabsList>
          
          <TabsContent value="data" className="space-y-8">
            {/* Stats Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {stats.map((stat, index) => {
                const Icon = stat.icon;
                return (
                  <Card key={index} className="p-6">
                    <div className="flex items-center justify-between mb-2">
                      <Icon className={`w-5 h-5 ${stat.color}`} />
                      <Badge variant="secondary">Live</Badge>
                    </div>
                    <div className="text-2xl font-bold mb-1">{stat.value}</div>
                    <div className="text-sm text-muted-foreground">{stat.label}</div>
                  </Card>
                );
              })}
            </div>

            {/* Float Types */}
            <div>
              <h2 className="text-2xl font-bold mb-6">ARGO Float Types</h2>
              <div className="grid md:grid-cols-3 gap-6 mb-6">
                {floatTypes.map((floatType, index) => {
                  const Icon = floatType.icon;
                  const isSelected = selectedFloatType === floatType.type;
                  return (
                    <Card 
                      key={index} 
                      className={`p-6 cursor-pointer transition-all duration-200 hover:shadow-lg ${
                        isSelected ? 'ring-2 ring-primary bg-primary/5' : ''
                      }`}
                      onClick={() => setSelectedFloatType(isSelected ? null : floatType.type)}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <Icon className={`w-6 h-6 text-white`} />
                        <Badge variant="secondary">{floatType.count}</Badge>
                      </div>
                      <h3 className="text-lg font-semibold mb-2">{floatType.type}</h3>
                      <p className="text-sm text-muted-foreground mb-4">{floatType.description}</p>
                      <div className="space-y-2">
                        {floatType.parameters.map((param, paramIndex) => (
                          <Badge key={paramIndex} variant="outline" className="mr-2 text-xs">
                            {param}
                          </Badge>
                        ))}
                      </div>
                    </Card>
                  );
                })}
              </div>

              {/* Analytics and Measurements for Selected Float Type */}
              {selectedFloatType && (
                <div className="space-y-6">
                  <Card className="p-6 bg-accent/5 border-accent/20">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <Activity className="w-5 h-5 mr-2" />
                      {selectedFloatType} Analytics
                    </h3>
                    {(() => {
                      const analytics = floatTypes.find(f => f.type === selectedFloatType)?.analytics;
                      return analytics ? (
                        <div className="grid md:grid-cols-4 gap-4">
                          <div className="bg-background/50 p-4 rounded-lg">
                            <div className="text-sm text-muted-foreground">Active Floats</div>
                            <div className="text-2xl font-bold">{analytics.activeFloats}</div>
                          </div>
                          <div className="bg-background/50 p-4 rounded-lg">
                            <div className="text-sm text-muted-foreground">Average Depth</div>
                            <div className="text-2xl font-bold">{analytics.avgDepth}</div>
                          </div>
                          <div className="bg-background/50 p-4 rounded-lg">
                            <div className="text-sm text-muted-foreground">Data Points</div>
                            <div className="text-2xl font-bold">{analytics.dataPoints}</div>
                          </div>
                          <div className="bg-background/50 p-4 rounded-lg">
                            <div className="text-sm text-muted-foreground">Coverage</div>
                            <div className="text-2xl font-bold text-sm">{analytics.coverage}</div>
                          </div>
                        </div>
                      ) : null;
                    })()}
                  </Card>

                  {/* Recent Measurements for Selected Float Type */}
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      {selectedFloatType === "Core ARGO" && <Thermometer className="w-5 h-5 mr-2 text-data-temperature" />}
                      {selectedFloatType === "BGC ARGO" && <Microscope className="w-5 h-5 mr-2 text-data-bgc" />}
                      {selectedFloatType === "Deep ARGO" && <ArrowDown className="w-5 h-5 mr-2 text-data-depth" />}
                      {selectedFloatType} - Recent Measurements
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left py-3 px-4">Float ID</th>
                            <th className="text-left py-3 px-4">Location</th>
                            {selectedFloatType === "Core ARGO" && (
                              <>
                                <th className="text-left py-3 px-4">Temperature</th>
                                <th className="text-left py-3 px-4">Salinity</th>
                              </>
                            )}
                            {selectedFloatType === "BGC ARGO" && (
                              <>
                                <th className="text-left py-3 px-4">Oxygen</th>
                                <th className="text-left py-3 px-4">pH</th>
                                <th className="text-left py-3 px-4">Chlorophyll</th>
                              </>
                            )}
                            {selectedFloatType === "Deep ARGO" && (
                              <>
                                <th className="text-left py-3 px-4">Deep Temperature</th>
                                <th className="text-left py-3 px-4">Deep Salinity</th>
                              </>
                            )}
                            <th className="text-left py-3 px-4">Max Depth</th>
                            <th className="text-left py-3 px-4">Status</th>
                            <th className="text-left py-3 px-4">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {recentFloatsByType[selectedFloatType]?.map((float, index) => (
                            <tr key={index} className="border-b border-border/50">
                              <td className="py-3 px-4 font-mono text-sm">{float.id}</td>
                              <td className="py-3 px-4">{float.lat}, {float.lon}</td>
                              {selectedFloatType === "Core ARGO" && (
                                <>
                                  <td className="py-3 px-4 flex items-center">
                                    <Thermometer className="w-4 h-4 mr-1 text-data-temperature" />
                                    {float.temp}
                                  </td>
                                  <td className="py-3 px-4 flex items-center">
                                    <Droplets className="w-4 h-4 mr-1 text-data-salinity" />
                                    {float.salinity}
                                  </td>
                                </>
                              )}
                              {selectedFloatType === "BGC ARGO" && (
                                <>
                                  <td className="py-3 px-4 flex items-center">
                                    <Waves className="w-4 h-4 mr-1 text-data-bgc" />
                                    {float.oxygen}
                                  </td>
                                  <td className="py-3 px-4">{float.ph}</td>
                                  <td className="py-3 px-4">{float.chlorophyll}</td>
                                </>
                              )}
                              {selectedFloatType === "Deep ARGO" && (
                                <>
                                  <td className="py-3 px-4 flex items-center">
                                    <Thermometer className="w-4 h-4 mr-1 text-data-temperature" />
                                    {float.temp}
                                  </td>
                                  <td className="py-3 px-4 flex items-center">
                                    <Droplets className="w-4 h-4 mr-1 text-data-salinity" />
                                    {float.salinity}
                                  </td>
                                </>
                              )}
                              <td className="py-3 px-4 flex items-center">
                                <Gauge className="w-4 h-4 mr-1 text-data-depth" />
                                {float.depth}
                              </td>
                              <td className="py-3 px-4">
                                <Badge variant={float.status === "Active" ? "default" : "secondary"}>
                                  {float.status}
                                </Badge>
                              </td>
                              <td className="py-3 px-4">
                                <Button variant="ghost" size="sm">View Profile</Button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </div>
              )}
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Global Map */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <MapPin className="w-5 h-5 mr-2" />
                  Global Float Distribution
                </h3>
                <div className="relative">
                  <img 
                    src={globalMapImage} 
                    alt="Global ARGO Float Distribution" 
                    className="w-full h-64 object-cover rounded-lg"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-primary-deep/20 to-transparent rounded-lg"></div>
                  <div className="absolute bottom-4 left-4 text-primary-foreground">
                    <div className="text-sm font-medium">Indian Ocean Region</div>
                    <div className="text-xs opacity-80">423 active floats</div>
                  </div>
                </div>
              </Card>

              {/* Data Types */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Available Data Types</h3>
                <div className="space-y-4">
                  {dataTypes.map((type, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${type.color}`}></div>
                        <div>
                          <div className="font-medium">{type.name}</div>
                          <div className="text-xs text-muted-foreground">{type.description}</div>
                        </div>
                      </div>
                      <Badge variant="outline">{type.count}</Badge>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Sample Visualizations */}
            <div className="grid lg:grid-cols-2 gap-8">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Thermometer className="w-5 h-5 mr-2 text-data-temperature" />
                  Temperature Profile
                </h3>
                <div className="h-64 bg-gradient-to-b from-data-temperature/20 to-data-temperature/5 rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <div className="text-sm mb-2">Sample Temperature vs Depth</div>
                    <div className="text-xs">Interactive chart would appear here</div>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Droplets className="w-5 h-5 mr-2 text-data-salinity" />
                  Salinity Analysis
                </h3>
                <div className="h-64 bg-gradient-to-b from-data-salinity/20 to-data-salinity/5 rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <div className="text-sm mb-2">Regional Salinity Comparison</div>
                    <div className="text-xs">Interactive chart would appear here</div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Backend Integration Notice */}
            <Card className="p-6 bg-accent/10 border-accent/20">
              <h3 className="text-lg font-semibold mb-2 text-accent">Backend Integration Required</h3>
              <p className="text-muted-foreground mb-4">
                To display real ARGO data, process NetCDF files, and enable the full RAG pipeline with vector databases, you'll need to integrate with a backend system.
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">PostgreSQL</Badge>
                <Badge variant="outline">Vector Database</Badge>
                <Badge variant="outline">NetCDF Processing</Badge>
                <Badge variant="outline">LLM Integration</Badge>
                <Badge variant="outline">Real-time Data</Badge>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="map" className="space-y-8">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold mb-2">Interactive Map</h2>
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

            {/* Interactive Earth Visualization Notice */}
            <Card className="p-6 bg-accent/10 border-accent/20">
              <h3 className="text-lg font-semibold mb-2 text-accent">Interactive Earth Visualization</h3>
              <p className="text-muted-foreground">
                Experience oceanographic data through our interactive 3D globe with atmospheric overlays inspired by earth.nullschool.net. 
                Switch between temperature, wind patterns, and sea level pressure visualizations while tracking ARGO float positions in real-time.
              </p>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};