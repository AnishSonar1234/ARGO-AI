import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart3, 
  Database, 
  ArrowRight, 
  Globe2, 
  Compass, 
  Radio, 
  Layers, 
  Waves, 
  Sparkles,
  ShieldCheck,
  TrendingDown,
  Navigation,
  Activity,
  CheckCircle2,
  GitCommit,
  GitBranch,
  Code2,
  Cpu,
  History,
  FolderTree,
  Zap,
  Tag
} from "lucide-react";
import { WhaleIcon } from "@/components/WhaleIcon";
import argoCycleGif from "@/assets/argo-float-cycle.gif";
import argoGlobeGif from "@/assets/argo-global-globe.gif";
import whaleDiveGif from "@/assets/whale-dive.gif";

interface HeroProps {
  onSectionChange: (section: string) => void;
}

export const Hero = ({ onSectionChange }: HeroProps) => {
  const [activeMediaTab, setActiveMediaTab] = useState<"cycle" | "global">("cycle");
  const [activeUpdateCategory, setActiveUpdateCategory] = useState<"all" | "architecture" | "ai" | "ui" | "telemetry">("all");

  const features = [
    {
      icon: WhaleIcon,
      title: "Natural Language Ocean Queries",
      description: "Ask complex oceanographic questions in plain English. Powered by LangChain RAG & Nemotron-3 reasoning.",
      badge: "AI Powered"
    },
    {
      icon: BarChart3,
      title: "Interactive CTD Depth Profiles",
      description: "Visualize continuous vertical water column measurements: thermocline, halocline, oxygen, and pycnocline gradients.",
      badge: "0m – 6,000m"
    },
    {
      icon: Globe2,
      title: "Global Array 3D Tracking",
      description: "Track over 3,800 active autonomous floats across all major ocean basins on an interactive bathymetric globe.",
      badge: "Global Fleet"
    },
    {
      icon: Database,
      title: "Live GDAC & ERDDAP Ingestion",
      description: "Direct real-time synchronization with NOAA PolarWatch, Coriolis, INCOIS, CSIRO, and international GDAC centers.",
      badge: "Live Telemetry"
    }
  ];

  const missionStages = [
    {
      step: "01",
      title: "Deployment & Descent",
      desc: "Float deflates internal hydraulic bladder to sink to 1,000m parking depth."
    },
    {
      step: "02",
      title: "Subsurface Drift",
      desc: "Drifts neutrally with deep geostrophic currents for 9 days, mapping mid-depth circulation."
    },
    {
      step: "03",
      title: "Deep CTD Profiling",
      desc: "Descends to 2,000m, then ascends while recording continuous temperature, salinity & pressure curves."
    },
    {
      step: "04",
      title: "Satellite Uplink",
      desc: "Reaches surface to beam CTD profiles to NOAA & Coriolis via Iridium RUDICS satellites."
    }
  ];

  const fleetStats = [
    { name: "Pacific Basin Fleet", count: "1,420 Floats", pct: "41%", color: "text-cyan-500" },
    { name: "Atlantic Basin Fleet", count: "890 Floats", pct: "26%", color: "text-blue-500" },
    { name: "Indian Ocean Fleet", count: "680 Floats", pct: "20%", color: "text-emerald-500" },
    { name: "Southern & Arctic Polar Fleet", count: "450 Floats", pct: "13%", color: "text-violet-500" },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-primary/20">
      {/* 1. HERO HEADER SECTION */}
      <section className="relative overflow-hidden pt-12 pb-16 md:pt-20 md:pb-24 border-b border-border/60">
        {/* Ambient ocean aura background */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[900px] h-[550px] bg-gradient-to-b from-primary/15 via-cyan-500/10 to-transparent blur-3xl opacity-70"></div>
          <div className="absolute top-1/2 -left-40 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 -right-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl"></div>
        </div>

        <div className="container mx-auto px-4 max-w-6xl relative z-10 text-center">
          {/* Header pill badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/30 bg-primary/10 text-primary text-xs font-medium mb-6 shadow-xs animate-in fade-in duration-300">
            <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span>
            <WhaleIcon className="w-3.5 h-3.5 text-primary" />
            <span>Autonomous Ocean Intelligence • NOAA ERDDAP & Coriolis Ingestion</span>
          </div>

          {/* Main Title */}
          <h1 className="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight mb-6 leading-tight max-w-4xl mx-auto">
            Explore Ocean Data with{" "}
            <span className="bg-gradient-to-r from-cyan-400 via-primary to-blue-600 bg-clip-text text-transparent">
              AI Whale Intelligence
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed font-normal">
            Query real-time ARGO oceanographic observations through natural language. Explore 3D bathymetric profiles, thermocline gradients, and global float arrays.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 max-w-md mx-auto mb-10">
            <Button 
              size="lg" 
              className="w-full sm:w-auto h-12 px-7 text-sm font-semibold rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 shadow-md shadow-primary/20 flex items-center justify-center gap-2 group transition-all"
              onClick={() => onSectionChange("chat")}
            >
              <WhaleIcon className="w-4 h-4 text-primary-foreground group-hover:scale-110 transition-transform" />
              <span>Chat with Whale</span>
              <ArrowRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
            </Button>

            <Button 
              size="lg" 
              variant="outline" 
              className="w-full sm:w-auto h-12 px-7 text-sm font-medium rounded-xl border-border bg-card/60 hover:bg-muted/80 backdrop-blur-xs flex items-center justify-center gap-2 transition-all"
              onClick={() => onSectionChange("dashboard")}
            >
              <BarChart3 className="w-4 h-4 text-cyan-500" />
              <span>Live Dashboard</span>
            </Button>
          </div>

          {/* Trust telemetry pills */}
          <div className="flex items-center justify-center flex-wrap gap-3.5 text-xs text-muted-foreground/90 font-mono">
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-card/80 border border-border/60 shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>3,345 Active Platforms</span>
            </div>
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-card/80 border border-border/60 shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              <span>0m – 6,000m CTD Depths</span>
            </div>
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-card/80 border border-border/60 shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-violet-500"></span>
              <span>Iridium Satellite Uplinks</span>
            </div>
          </div>
        </div>
      </section>

      {/* 2. INTERACTIVE ANIMATED SHOWCASE SECTION (GIFS INSTEAD OF STATIC PHOTOS) */}
      <section className="py-14 md:py-20 bg-muted/20 border-b border-border/60">
        <div className="container mx-auto px-4 max-w-6xl">
          {/* Section Header with Animation Switcher Tabs */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-8">
            <div>
              <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-primary mb-1">
                <Activity className="w-3.5 h-3.5" />
                <span>Autonomous Profiling Technology</span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
                Live Ocean Profiling & Global Tracking
              </h2>
            </div>

            {/* Animation Toggle Tabs */}
            <div className="flex items-center p-1 rounded-xl bg-card border border-border shadow-2xs">
              <button
                type="button"
                onClick={() => setActiveMediaTab("cycle")}
                className={`px-4 py-2 text-xs font-medium rounded-lg transition-all flex items-center gap-2 ${
                  activeMediaTab === "cycle"
                    ? "bg-primary text-primary-foreground shadow-xs font-semibold"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Waves className="w-3.5 h-3.5" />
                <span>10-Day Profiling Cycle</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveMediaTab("global")}
                className={`px-4 py-2 text-xs font-medium rounded-lg transition-all flex items-center gap-2 ${
                  activeMediaTab === "global"
                    ? "bg-primary text-primary-foreground shadow-xs font-semibold"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Globe2 className="w-3.5 h-3.5" />
                <span>Global Fleet Orbit</span>
              </button>
            </div>
          </div>

          {/* Featured Showcase Card with Animated GIF */}
          <Card className="overflow-hidden border-border/80 bg-card shadow-sm rounded-2xl">
            {activeMediaTab === "cycle" ? (
              <div className="grid lg:grid-cols-12 gap-0 items-center">
                {/* Left Column: 10-Day Profiling Cycle Animated GIF */}
                <div className="lg:col-span-7 bg-slate-950 p-4 sm:p-6 flex flex-col items-center justify-center relative overflow-hidden min-h-[380px] border-b lg:border-b-0 lg:border-r border-border/60">
                  {/* High-tech viewport frame */}
                  <div className="w-full max-w-xl rounded-xl overflow-hidden shadow-2xl border border-white/10 group bg-slate-900/60">
                    {/* Viewport header bar */}
                    <div className="px-3.5 py-2 bg-slate-900/90 border-b border-white/10 flex items-center justify-between text-xs font-mono">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span className="text-slate-200 text-[11px] font-semibold">ARGO Float Mission Telemetry</span>
                      </div>
                      <Badge className="bg-primary/20 text-primary-foreground text-[10px] border border-primary/30">
                        Autonomous Loop
                      </Badge>
                    </div>

                    {/* Animated GIF */}
                    <div className="relative">
                      <img 
                        src={argoCycleGif} 
                        alt="Animated 10-day ARGO profiling float cycle simulation" 
                        className="w-full h-auto object-cover transform group-hover:scale-[1.01] transition-transform duration-500"
                      />
                    </div>
                  </div>

                  <div className="mt-3.5 text-center">
                    <p className="text-xs text-slate-400 font-mono">
                      Cycle Protocol: Descent (1,000m) → Neutral Drift (9 Days) → Deep CTD Ascent (2,000m) → Satellite Uplink
                    </p>
                  </div>
                </div>

                {/* Right Column: Step-by-Step Breakdown */}
                <div className="lg:col-span-5 p-6 sm:p-8 space-y-5">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary">
                    <Navigation className="w-4 h-4" />
                    <span>The 10-Day Observation Cycle</span>
                  </div>

                  <div className="space-y-4">
                    {missionStages.map((stage, idx) => (
                      <div key={idx} className="flex items-start gap-3.5 group">
                        <div className="w-7 h-7 rounded-lg bg-primary/10 text-primary font-mono text-xs font-bold flex items-center justify-center flex-shrink-0 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                          {stage.step}
                        </div>
                        <div className="space-y-0.5">
                          <h4 className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors">
                            {stage.title}
                          </h4>
                          <p className="text-xs text-muted-foreground leading-relaxed">
                            {stage.desc}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="pt-2 border-t border-border/40">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => onSectionChange("dashboard")}
                      className="w-full text-xs h-9 justify-center gap-2"
                    >
                      <span>View Live Floats on Dashboard</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid lg:grid-cols-12 gap-0 items-center">
                {/* Left Column: Rotating Global Fleet Array GIF */}
                <div className="lg:col-span-7 bg-slate-950 p-4 sm:p-8 flex flex-col items-center justify-center relative overflow-hidden min-h-[380px] border-b lg:border-b-0 lg:border-r border-border/60">
                  <div className="w-full max-w-sm rounded-xl overflow-hidden shadow-2xl border border-white/10 group bg-slate-900/60 p-4 flex flex-col items-center">
                    <div className="w-full flex items-center justify-between pb-3 border-b border-white/10 mb-4 text-xs font-mono">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                        <span className="text-slate-200 text-[11px] font-semibold">Global Array 3D Orbit</span>
                      </div>
                      <Badge className="bg-cyan-500/20 text-cyan-300 text-[10px] border border-cyan-500/30">
                        120-Frame Loop
                      </Badge>
                    </div>

                    <img 
                      src={argoGlobeGif} 
                      alt="Animated 3D rotating globe of global ARGO profiling float array" 
                      className="w-56 h-56 sm:w-64 sm:h-64 object-contain rounded-full shadow-lg"
                    />

                    <div className="mt-4 text-center">
                      <span className="text-xs text-cyan-400 font-mono">
                        Official UCSD Argo Global Array Distribution
                      </span>
                    </div>
                  </div>
                </div>

                {/* Right Column: Fleet Distribution Stats */}
                <div className="lg:col-span-5 p-6 sm:p-8 space-y-5">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-cyan-500">
                    <Globe2 className="w-4 h-4" />
                    <span>Global Basin Distribution</span>
                  </div>

                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Over 3,800 autonomous profiling buoys are currently distributed across all international ocean sectors, transmitting data every 10 days.
                  </p>

                  <div className="space-y-3">
                    {fleetStats.map((stat, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2.5 rounded-lg bg-muted/40 border border-border/50 text-xs">
                        <div className="font-medium text-foreground flex items-center gap-2">
                          <CheckCircle2 className={`w-3.5 h-3.5 ${stat.color}`} />
                          <span>{stat.name}</span>
                        </div>
                        <div className="flex items-center gap-2 font-mono">
                          <span className="font-semibold text-foreground">{stat.count}</span>
                          <span className="text-muted-foreground text-[11px]">({stat.pct})</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="pt-2 border-t border-border/40">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => onSectionChange("dashboard")}
                      className="w-full text-xs h-9 justify-center gap-2"
                    >
                      <span>Explore Live Ocean Maps</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>
      </section>

      {/* 3. WHALE INTELLIGENCE & SECONDARY GIF SECTION */}
      <section className="py-14 md:py-20 border-b border-border/60">
        <div className="container mx-auto px-4 max-w-6xl">
          <div className="grid lg:grid-cols-12 gap-8 items-center">
            {/* Left Column: Text narrative */}
            <div className="lg:col-span-6 space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-600 dark:text-cyan-400 text-xs font-medium">
                <WhaleIcon className="w-3.5 h-3.5 text-cyan-500" />
                <span>Marine Science & Climate Insight</span>
              </div>

              <h2 className="text-2xl sm:text-4xl font-bold tracking-tight leading-tight">
                Why Whale? Inspired by the Deepest Ocean Navigators
              </h2>

              <p className="text-sm sm:text-base text-muted-foreground leading-relaxed font-normal">
                Just like ocean whales that migrate across global basins and dive thousands of meters beneath the waves, ARGO AI navigates the vast water column of global oceanographic telemetry.
              </p>

              <div className="space-y-3 pt-1">
                <div className="flex items-start gap-3">
                  <div className="p-1 rounded-md bg-emerald-500/10 text-emerald-500 mt-0.5">
                    <ShieldCheck className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-foreground">Biogeochemical (BGC) Monitoring</div>
                    <div className="text-xs text-muted-foreground">Monitoring ocean acidification, nitrate nutrients, chlorophyll, and dissolved oxygen minimum zones.</div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="p-1 rounded-md bg-rose-500/10 text-rose-500 mt-0.5">
                    <TrendingDown className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-foreground">Deep Ocean Warming & Heat Content</div>
                    <div className="text-xs text-muted-foreground">Measuring abyssal heat uptake to calculate planetary energy imbalances and sea-level rise.</div>
                  </div>
                </div>
              </div>

              <div className="pt-2">
                <Button 
                  onClick={() => onSectionChange("chat")}
                  className="rounded-xl text-xs h-10 px-5 gap-2 bg-primary"
                >
                  <WhaleIcon className="w-4 h-4 text-white" />
                  <span>Ask Whale Anything About Oceans</span>
                </Button>
              </div>
            </div>

            {/* Right Column: Animated Whale Dive GIF Card */}
            <div className="lg:col-span-6">
              <Card className="overflow-hidden border-border/80 bg-card shadow-sm rounded-2xl relative group">
                <div className="p-3 bg-muted/40 border-b border-border/60 flex items-center justify-between text-xs">
                  <span className="font-semibold flex items-center gap-1.5 text-foreground">
                    <WhaleIcon className="w-3.5 h-3.5 text-primary" />
                    Deep Water Column Exploration
                  </span>
                  <Badge variant="outline" className="text-[10px] border-primary/30 text-primary">
                    Marine Animation
                  </Badge>
                </div>

                <div className="relative bg-slate-950 flex flex-col items-center justify-center p-4 sm:p-6 overflow-hidden">
                  <div className="rounded-xl overflow-hidden border border-white/10 shadow-xl bg-slate-900/50 p-2">
                    <img 
                      src={whaleDiveGif} 
                      alt="Animated Humpback Whale Diving Beneath Ocean Waves"
                      className="w-full max-w-sm rounded-lg object-contain transform group-hover:scale-102 transition-transform duration-500"
                    />
                  </div>
                  
                  {/* Floating telemetry tag */}
                  <div className="w-full max-w-sm mt-3 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
                      Whale Bathymetric Dive
                    </span>
                    <span className="text-cyan-400">Epipelagic & Mesopelagic</span>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* 4. RECENT CODEBASE UPDATES & CHANGELOG TAB */}
      <section className="py-14 md:py-20 bg-muted/20 border-b border-border/60">
        <div className="container mx-auto px-4 max-w-6xl">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
            <div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-xs font-semibold text-primary mb-3">
                <GitCommit className="w-3.5 h-3.5" />
                Live Engineering Changelog
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
                Recent Codebase Updates
              </h2>
              <p className="text-sm text-muted-foreground mt-1.5 max-w-xl">
                Verified releases, architectural refactoring, vector memory enhancements, and live telemetry integrations.
              </p>
            </div>

            {/* Category Filter Tabs */}
            <div className="flex flex-wrap items-center gap-1.5 p-1 bg-background/80 border border-border/80 rounded-xl shadow-xs">
              {[
                { id: "all", label: "All Updates", count: "5" },
                { id: "architecture", label: "Architecture", count: "v2.4" },
                { id: "ai", label: "LangChain & RAG", count: "v2.5" },
                { id: "ui", label: "Claude UI", count: "v2.2" },
                { id: "telemetry", label: "ERDDAP", count: "v2.1" },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveUpdateCategory(tab.id as any)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
                    activeUpdateCategory === tab.id
                      ? "bg-primary text-primary-foreground shadow-xs font-semibold"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
                  }`}
                >
                  <span>{tab.label}</span>
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full ${
                    activeUpdateCategory === tab.id
                      ? "bg-primary-foreground/20 text-primary-foreground"
                      : "bg-muted text-muted-foreground"
                  }`}>
                    {tab.count}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Updates Grid */}
          <div className="grid md:grid-cols-2 gap-5">
            {[
              {
                id: "upd-cloud",
                version: "v2.5.0",
                category: "ai",
                date: "Latest Release",
                icon: Database,
                title: "Chroma Cloud Online Vector Store & Cloud PaaS Deployment",
                badge: "Live Cloud",
                badgeColor: "bg-cyan-500/10 text-cyan-500 border-cyan-500/20",
                description: "Connected the platform to official Chroma Cloud (api.trychroma.com) with remote vector storage, automated knowledge seeding, and dynamic port binding for Render/Vercel.",
                highlights: [
                  "Integrated chromadb.CloudClient with secure tenant & database authentication",
                  "Pre-seeded 6 comprehensive oceanographic domain knowledge corpora into Chroma Cloud",
                  "Added standalone seed_chroma.py diagnostic utility and dynamic $PORT handling in Dockerfile",
                  "Seamless fallback to local persistent ChromaDB whenever cloud credentials are not supplied"
                ],
                tags: ["#ChromaCloud", "#CloudDeployment", "#VectorDatabase", "#RenderPaaS", "#Vercel"]
              },
              {
                id: "upd-1",
                version: "v2.4.0",
                category: "architecture",
                date: "Current Release",
                icon: FolderTree,
                title: "Modular App Architecture & Codebase Pruning",
                badge: "Active",
                badgeColor: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
                description: "Restructured the backend into an industry-standard package hierarchy (app/nlp, app/rag, app/tools, app/agent.py) and removed 1,223 lines of dead code.",
                highlights: [
                  "Extracted dedicated packages: app/nlp/parser.py, app/rag/vectorstore.py, and app/tools/",
                  "Streamlined main.py from 1,037 lines down to 164 lines of modular routing",
                  "Deleted unused heuristic scrapers, duplicate generators, and legacy root shims",
                  "Standardized requests payload handling for streaming reasoning traces"
                ],
                tags: ["#ModularBackend", "#CleanCode", "#ProductionReady", "#FastBoot"]
              },
              {
                id: "upd-2",
                version: "v2.3.0",
                category: "ai",
                date: "Recent Update",
                icon: Cpu,
                title: "ChromaDB Persistent RAG & LangChain Reasoning",
                badge: "Verified",
                badgeColor: "bg-cyan-500/10 text-cyan-500 border-cyan-500/20",
                description: "Implemented persistent vector retrieval using ChromaDB and Pydantic NLP intent recognition for context-augmented oceanographic intelligence.",
                highlights: [
                  "ChromaDB persistent collection indexed with 6 verified oceanographic knowledge docs",
                  "Pydantic NLP query intent parser extracts spatial regions, variables, and float IDs",
                  "Preserved multi-turn reasoning_details across iterative conversational follow-ups",
                  "Dynamic tool execution: Folium map generators and Matplotlib CTD depth profiles"
                ],
                tags: ["#ChromaDB", "#LangChain", "#RAG", "#OpenRouter", "#Pydantic"]
              },
              {
                id: "upd-3",
                version: "v2.2.0",
                category: "ui",
                date: "Recent Update",
                icon: WhaleIcon,
                title: "Claude AI-Inspired Chat & Whale Mascot Avatar",
                badge: "Enhanced",
                badgeColor: "bg-blue-500/10 text-blue-500 border-blue-500/20",
                description: "Overhauled the chat interface to match the balanced Claude AI design system, featuring collapsible sidebars, suggestion pills, and a friendly whale mascot.",
                highlights: [
                  "Fixed viewport layout with zero double-scrolling and auto-expanding input box",
                  "Real-time whale mascot animation with live thinking & responding state badges",
                  "Rich markdown rendering with mathematical symbols, tables, and expandable reasoning",
                  "Interactive oceanographic query suggestions for rapid one-click research"
                ],
                tags: ["#ClaudeStyle", "#WhaleMascot", "#ResponsiveUI", "#MicroAnimations"]
              },
              {
                id: "upd-4",
                version: "v2.1.0",
                category: "telemetry",
                date: "Recent Update",
                icon: Radio,
                title: "Live ERDDAP Sync & Explicit Telemetry Controls",
                badge: "Live Telemetry",
                badgeColor: "bg-indigo-500/10 text-indigo-500 border-indigo-500/20",
                description: "Direct real-time ingestion from NOAA ERDDAP, Coriolis, and INCOIS GDAC feeds with on-demand user refresh controls to eliminate API timeouts.",
                highlights: [
                  "Over 3,800 active floats monitored across 8 major global ocean basins",
                  "Explicit on-demand refresh controls replace aggressive background polling",
                  "Interactive 3D Earth Globe with coordinate pinpointing and status filters",
                  "Full vertical CTD depth profiling (0m–6,000m) with live gradient calculations"
                ],
                tags: ["#NOAA", "#ERDDAP", "#LiveTelemetry", "#3DGlobe", "#CTDProfiles"]
              }
            ]
              .filter(u => activeUpdateCategory === "all" || u.category === activeUpdateCategory)
              .map((update) => {
                const IconComponent = update.icon;
                return (
                  <Card 
                    key={update.id} 
                    className="p-5 sm:p-6 border-border/80 bg-card hover:border-primary/40 hover:shadow-md transition-all rounded-2xl flex flex-col justify-between"
                  >
                    <div>
                      {/* Top Meta Header */}
                      <div className="flex items-center justify-between mb-3.5">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                            <IconComponent className="w-4 h-4" />
                          </div>
                          <div>
                            <span className="text-xs font-bold text-foreground font-mono">{update.version}</span>
                            <span className="text-[11px] text-muted-foreground ml-2">• {update.date}</span>
                          </div>
                        </div>
                        <Badge variant="outline" className={`text-[10px] px-2 py-0.5 border ${update.badgeColor}`}>
                          {update.badge}
                        </Badge>
                      </div>

                      {/* Title & Description */}
                      <h3 className="text-base font-semibold mb-2 text-foreground">{update.title}</h3>
                      <p className="text-xs text-muted-foreground leading-relaxed mb-4">{update.description}</p>

                      {/* Detailed Highlights */}
                      <div className="space-y-2 mb-4 bg-muted/30 p-3 rounded-xl border border-border/40">
                        <div className="text-[11px] font-semibold text-foreground/80 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                          <Zap className="w-3 h-3 text-primary" /> Key Improvements
                        </div>
                        {update.highlights.map((item, idx) => (
                          <div key={idx} className="flex items-start gap-2 text-xs text-muted-foreground">
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 mt-0.5 shrink-0" />
                            <span className="leading-snug">{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Tags Footer */}
                    <div className="flex flex-wrap gap-1.5 pt-2 border-t border-border/50">
                      {update.tags.map((tag, tIdx) => (
                        <span 
                          key={tIdx} 
                          className="text-[10px] font-mono px-2 py-0.5 bg-muted/60 text-muted-foreground rounded-md border border-border/40"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </Card>
                );
              })}
          </div>

          {/* Quick Architecture Summary Banner */}
          <div className="mt-8 p-4 rounded-xl border border-border/80 bg-card/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="font-semibold text-foreground">Active Architecture:</span>
              <span className="text-muted-foreground font-mono">Flask 3.x + LangChain + ChromaDB (Local Persisted) + OpenRouter + Vite React</span>
            </div>
            <div className="flex items-center gap-3">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => onSectionChange("chat")}
                className="text-xs h-8 rounded-lg gap-1.5"
              >
                <WhaleIcon className="w-3.5 h-3.5 text-primary" />
                <span>Test in Whale Chat</span>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* 5. FOOTER SUMMARY */}
      <footer className="py-8 bg-background text-xs text-muted-foreground text-center">
        <div className="container mx-auto px-4 max-w-6xl flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-primary/10 flex items-center justify-center">
              <WhaleIcon className="w-3.5 h-3.5 text-primary" />
            </div>
            <span className="font-semibold text-foreground">ARGO AI Platform</span>
            <span>• Open Ocean Science</span>
          </div>
          <div>
            Data sourced from NOAA ERDDAP, Coriolis GDAC, INCOIS & CSIRO
          </div>
        </div>
      </footer>
    </div>
  );
};