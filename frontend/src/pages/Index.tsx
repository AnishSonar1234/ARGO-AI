import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { Hero } from "@/components/Hero";
import { ChatInterface } from "@/components/ChatInterface";
import { Dashboard } from "@/components/Dashboard";

const Index = () => {
  const [activeSection, setActiveSection] = useState("home");

  const renderSection = () => {
    switch (activeSection) {
      case "chat":
        return <ChatInterface />;
      case "dashboard":
        return <Dashboard onSectionChange={setActiveSection} />;
      default:
        return <Hero onSectionChange={setActiveSection} />;
    }
  };

  const isChat = activeSection === "chat";

  return (
    <div className={`bg-background ${isChat ? "h-screen flex flex-col overflow-hidden" : "min-h-screen"}`}>
      <Navbar activeSection={activeSection} onSectionChange={setActiveSection} />
      <div className={isChat ? "flex-1 min-h-0 overflow-hidden" : ""}>
        {renderSection()}
      </div>
    </div>
  );
};

export default Index;
