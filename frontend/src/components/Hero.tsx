import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { MessageCircle, BarChart3, Database, Zap } from "lucide-react";
import heroImage from "@/assets/argo-underwater.jpg";

interface HeroProps {
  onSectionChange: (section: string) => void;
}

export const Hero = ({ onSectionChange }: HeroProps) => {
  const features = [
    {
      icon: MessageCircle,
      title: "Natural Language Queries",
      description: "Ask questions about ocean data in plain English"
    },
    {
      icon: BarChart3,
      title: "Interactive Visualizations",
      description: "Explore depth profiles, trajectories, and BGC parameters"
    },
    {
      icon: Database,
      title: "Real-time Data Access",
      description: "Query live ARGO float data from global ocean networks"
    },
    {
      icon: Zap,
      title: "AI-Powered Insights",
      description: "Get intelligent analysis and recommendations"
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-primary-deep/40 via-primary-deep/20 to-primary-deep/50"></div>
        </div>
        
        <div className="relative z-10 text-center text-primary-foreground max-w-4xl mx-auto px-4">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight drop-shadow-lg">
            Explore Ocean Data with
            <span className="block bg-gradient-to-r from-accent to-primary-surface bg-clip-text text-transparent">
              AI Intelligence
            </span>
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-primary-foreground/95 max-w-2xl mx-auto drop-shadow-md">
            Revolutionary AI-powered platform for querying and visualizing ARGO oceanographic data through natural language conversations.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              size="lg" 
              className="bg-accent hover:bg-accent/90 text-accent-foreground px-8 py-3 text-lg"
              onClick={() => onSectionChange("chat")}
            >
              <MessageCircle className="w-5 h-5 mr-2" />
              Start Exploring
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              className="border-white/30 text-white hover:bg-white/20 hover:text-white bg-white/10 backdrop-blur-sm px-8 py-3 text-lg font-semibold shadow-lg"
              onClick={() => onSectionChange("dashboard")}
            >
              <BarChart3 className="w-5 h-5 mr-2" />
              View Dashboard
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-background">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Democratizing Ocean Science</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Bridge the gap between complex oceanographic data and meaningful insights through AI-powered natural language processing.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow">
                  <div className="w-12 h-12 bg-gradient-ocean rounded-lg flex items-center justify-center mx-auto mb-4">
                    <Icon className="w-6 h-6 text-primary-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* ARGO Info Section */}
      <section className="py-20 bg-gradient-wave">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-bold text-center mb-8">About ARGO Floats</h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div>
                <p className="text-lg text-muted-foreground mb-6">
                  The ARGO program deploys autonomous profiling floats across the world's oceans, creating the largest ocean observation network in history. These floats collect essential data on temperature, salinity, and biogeochemical parameters.
                </p>
                <ul className="space-y-3 text-muted-foreground">
                  <li className="flex items-center">
                    <div className="w-2 h-2 bg-accent rounded-full mr-3"></div>
                    Over 4,000 active floats worldwide
                  </li>
                  <li className="flex items-center">
                    <div className="w-2 h-2 bg-accent rounded-full mr-3"></div>
                    Profiles from surface to 2000m depth
                  </li>
                  <li className="flex items-center">
                    <div className="w-2 h-2 bg-accent rounded-full mr-3"></div>
                    Real-time data transmission via satellite
                  </li>
                  <li className="flex items-center">
                    <div className="w-2 h-2 bg-accent rounded-full mr-3"></div>
                    Essential for climate and weather prediction
                  </li>
                </ul>
              </div>
              <div className="relative">
                <img 
                  src={heroImage} 
                  alt="ARGO Float" 
                  className="rounded-lg shadow-lg"
                />
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};