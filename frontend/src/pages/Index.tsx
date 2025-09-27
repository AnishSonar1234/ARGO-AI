import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { Hero } from "@/components/Hero";
import { ChatInterface } from "@/components/ChatInterface";
import { Dashboard } from "@/components/Dashboard";
import { MapView } from "@/components/MapView";

const Index = () => {
  const [activeSection, setActiveSection] = useState("home");

  const renderSection = () => {
    switch (activeSection) {
      case "chat":
        return <ChatInterface />;
      case "dashboard":
        return <Dashboard />;
      default:
        return <Hero onSectionChange={setActiveSection} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar activeSection={activeSection} onSectionChange={setActiveSection} />
      {renderSection()}
    </div>
  );
};

export default Index;
