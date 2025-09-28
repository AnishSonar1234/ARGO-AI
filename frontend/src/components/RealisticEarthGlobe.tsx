import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Text, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MapPin, Thermometer, Droplets, Wind } from 'lucide-react';

interface RealisticEarthGlobeProps {
  floatData?: Array<{
    id: string;
    lat: number;
    lon: number;
    status: string;
    temp: number;
    salinity?: number;
    pressure?: number;
  }>;
}

// Convert lat/lon to 3D coordinates
const latLonToVector3 = (lat: number, lon: number, radius: number = 1) => {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
};

// Create realistic Earth texture
const createRealisticEarthTexture = () => {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d')!;
  
  // Base ocean color - realistic blue
  const oceanGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  oceanGradient.addColorStop(0, '#0f172a'); // Deep blue at poles
  oceanGradient.addColorStop(0.2, '#1e3a8a'); // Deep blue
  oceanGradient.addColorStop(0.4, '#3b82f6'); // Ocean blue
  oceanGradient.addColorStop(0.6, '#3b82f6'); // Ocean blue
  oceanGradient.addColorStop(0.8, '#1e3a8a'); // Deep blue
  oceanGradient.addColorStop(1, '#0f172a'); // Deep blue at poles
  ctx.fillStyle = oceanGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Add realistic continent shapes
  ctx.fillStyle = '#16a34a'; // Forest green for land
  
  // North America
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.18, canvas.height * 0.25, 120, 140, 0, 0, Math.PI * 2);
  ctx.fill();
  // Alaska
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.12, canvas.height * 0.15, 60, 80, 0, 0, Math.PI * 2);
  ctx.fill();
  
  // South America
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.22, canvas.height * 0.6, 70, 100, 0, 0, Math.PI * 2);
  ctx.fill();
  
  // Africa
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.48, canvas.height * 0.35, 80, 140, 0, 0, Math.PI * 2);
  ctx.fill();
  
  // Europe
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.5, canvas.height * 0.2, 50, 60, 0, 0, Math.PI * 2);
  ctx.fill();
  
  // Asia
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.7, canvas.height * 0.25, 140, 100, 0, 0, Math.PI * 2);
  ctx.fill();
  
  // Australia
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.8, canvas.height * 0.7, 70, 50, 0, 0, Math.PI * 2);
  ctx.fill();
  
  // Antarctica
  ctx.fillStyle = '#f8fafc';
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.5, canvas.height * 0.05, 200, 30, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.5, canvas.height * 0.95, 200, 30, 0, 0, Math.PI * 2);
  ctx.fill();
  
  return new THREE.CanvasTexture(canvas);
};

// Realistic Earth component
const RealisticEarth: React.FC = () => {
  const earthRef = useRef<THREE.Mesh>(null);
  const earthTexture = useMemo(() => createRealisticEarthTexture(), []);
  
  useFrame((state) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.001; // Slow rotation
    }
  });
  
  return (
    <Sphere ref={earthRef} args={[1, 256, 128]}>
      <meshPhongMaterial 
        map={earthTexture}
        shininess={100}
        specular={0x222222}
        transparent={false}
      />
    </Sphere>
  );
};

// Float markers component
const FloatMarkers: React.FC<{ floats: RealisticEarthGlobeProps['floatData'] }> = ({ floats = [] }) => {
  return (
    <>
      {floats.map((float) => {
        const position = latLonToVector3(float.lat, float.lon, 1.02);
        const color = float.status === 'Active' ? '#10b981' : '#f59e0b';
        
        return (
          <group key={float.id} position={position}>
            <Sphere args={[0.008, 12, 12]}>
              <meshBasicMaterial color={color} transparent opacity={0.9} />
            </Sphere>
            {/* Glow effect */}
            <Sphere args={[0.012, 8, 8]}>
              <meshBasicMaterial color={color} transparent opacity={0.3} />
            </Sphere>
          </group>
        );
      })}
    </>
  );
};

export const RealisticEarthGlobe: React.FC<RealisticEarthGlobeProps> = ({ floatData = [] }) => {
  const [showData, setShowData] = useState(false);
  
  // Sample ARGO float data
  const sampleFloats = floatData.length > 0 ? floatData : [
    { id: "2902746", lat: 19.8, lon: 64.7, status: "Active", temp: 28.4, salinity: 34.2, pressure: 1013.2 },
    { id: "5906423", lat: -15.2, lon: 73.1, status: "Active", temp: 26.8, salinity: 35.1, pressure: 1015.8 },
    { id: "2903351", lat: 8.5, lon: 76.3, status: "Active", temp: 29.1, salinity: 33.8, pressure: 1011.5 },
    { id: "5906198", lat: 22.1, lon: 68.9, status: "Maintenance", temp: 27.6, salinity: 34.7, pressure: 1014.2 },
    { id: "3901234", lat: 5.3, lon: 82.4, status: "Active", temp: 28.9, salinity: 34.0, pressure: 1012.1 },
    { id: "4901234", lat: 35.0, lon: -120.0, status: "Active", temp: 15.2, salinity: 33.5, pressure: 1018.1 },
    { id: "3901235", lat: -30.0, lon: 150.0, status: "Active", temp: 18.5, salinity: 35.8, pressure: 1016.3 },
    { id: "2901236", lat: 45.0, lon: -30.0, status: "Active", temp: 12.3, salinity: 35.2, pressure: 1020.5 }
  ];

  return (
    <div className="relative w-full h-[700px] bg-gradient-to-b from-slate-900 to-slate-800 rounded-lg overflow-hidden">
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 2.2], fov: 50 }}
        className="absolute inset-0"
      >
        <ambientLight intensity={0.3} />
        <directionalLight position={[10, 10, 5]} intensity={1.2} color="#ffffff" />
        <directionalLight position={[-10, -10, -5]} intensity={0.4} color="#87CEEB" />
        <pointLight position={[0, 0, 10]} intensity={0.3} color="#ffffff" />
        
        <Stars radius={300} depth={50} count={3000} factor={4} saturation={0} fade />
        
        <RealisticEarth />
        <FloatMarkers floats={sampleFloats} />
        
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={1.3}
          maxDistance={4}
          autoRotate={true}
          autoRotateSpeed={0.3}
          rotateSpeed={1.0}
          zoomSpeed={1.2}
        />
      </Canvas>

      {/* Controls */}
      <div className="absolute top-4 left-4 flex flex-col space-y-2">
        <Button
          onClick={() => setShowData(!showData)}
          className="bg-primary hover:bg-primary/90"
        >
          {showData ? 'Hide Data' : 'Show Data'}
        </Button>
      </div>

      {/* Data Panel */}
      {showData && (
        <div className="absolute top-16 left-4 right-4 bottom-4">
          <Card className="p-6 bg-slate-900/95 backdrop-blur-sm border-slate-700 h-full overflow-auto">
            <h3 className="text-lg font-semibold mb-4 text-slate-100">ARGO Float Data</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sampleFloats.map((float) => (
                <Card key={float.id} className="p-4 bg-slate-800/50 border-slate-600">
                  <div className="flex justify-between items-start mb-3">
                    <span className="text-sm font-mono text-slate-200">{float.id}</span>
                    <Badge 
                      variant={float.status === 'Active' ? 'default' : 'secondary'}
                      className="text-xs"
                    >
                      {float.status}
                    </Badge>
                  </div>
                  <div className="space-y-2 text-sm text-slate-300">
                    <div className="flex justify-between">
                      <span>Location:</span>
                      <span>{float.lat}°, {float.lon}°</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="flex items-center">
                        <Thermometer className="w-4 h-4 mr-1" />
                        Temperature:
                      </span>
                      <span>{float.temp}°C</span>
                    </div>
                    {float.salinity && (
                      <div className="flex justify-between items-center">
                        <span className="flex items-center">
                          <Droplets className="w-4 h-4 mr-1" />
                          Salinity:
                        </span>
                        <span>{float.salinity} PSU</span>
                      </div>
                    )}
                    {float.pressure && (
                      <div className="flex justify-between items-center">
                        <span className="flex items-center">
                          <Wind className="w-4 h-4 mr-1" />
                          Pressure:
                        </span>
                        <span>{float.pressure} hPa</span>
                      </div>
                    )}
                  </div>
                </Card>
              ))}
            </div>
            
            <div className="mt-6 p-4 bg-slate-800/30 rounded-lg">
              <h4 className="text-sm font-semibold text-slate-200 mb-2">Summary Statistics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-slate-400">Total Floats:</span>
                  <div className="text-slate-200 font-semibold">{sampleFloats.length}</div>
                </div>
                <div>
                  <span className="text-slate-400">Active:</span>
                  <div className="text-green-400 font-semibold">
                    {sampleFloats.filter(f => f.status === 'Active').length}
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Avg Temperature:</span>
                  <div className="text-slate-200 font-semibold">
                    {(sampleFloats.reduce((sum, f) => sum + f.temp, 0) / sampleFloats.length).toFixed(1)}°C
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Avg Salinity:</span>
                  <div className="text-slate-200 font-semibold">
                    {(sampleFloats.reduce((sum, f) => sum + (f.salinity || 0), 0) / sampleFloats.length).toFixed(1)} PSU
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Globe Info */}
      {!showData && (
        <div className="absolute bottom-4 left-4 right-4">
          <Card className="p-4 bg-slate-900/90 backdrop-blur-sm border-slate-600">
            <div className="flex justify-between items-center text-sm">
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <MapPin className="w-4 h-4 text-green-400" />
                  <span className="text-slate-200">Active Floats: {sampleFloats.filter(f => f.status === 'Active').length}</span>
                </div>
                <div className="text-slate-300">
                  Global ARGO Network
                </div>
              </div>
              <div className="text-slate-400 text-xs">
                Drag to rotate • Scroll to zoom • Click markers for details
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};
