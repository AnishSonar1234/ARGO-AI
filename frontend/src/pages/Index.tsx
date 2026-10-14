import { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { Hero } from "@/components/Hero";
import { ChatInterface } from "@/components/ChatInterface";
import { Dashboard } from "@/components/Dashboard";

const Index = () => {
  const [activeSection, setActiveSection] = useState("home");
  const isChat = activeSection === "chat";

  // Dispatch window resize event on section switch so charts & WebGL globes resize properly
  useEffect(() => {
    const timer = setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 50);
    return () => clearTimeout(timer);
  }, [activeSection]);

  return (
    <div className={`bg-background ${isChat ? "h-screen flex flex-col overflow-hidden" : "min-h-screen"}`}>
      <Navbar activeSection={activeSection} onSectionChange={setActiveSection} />
      
      {/* Whale Chat: Kept persistently mounted so ongoing AI queries, timers, and active chat states are never interrupted or lost on tab switch */}
      <div className={isChat ? "flex-1 min-h-0 overflow-hidden flex flex-col h-full" : "hidden"}>
        <ChatInterface isActive={isChat} />
      </div>

      {/* Dashboard Section */}
      <div className={activeSection === "dashboard" ? "block" : "hidden"}>
        <Dashboard onSectionChange={setActiveSection} />
      </div>

      {/* Home Section */}
      <div className={activeSection !== "chat" && activeSection !== "dashboard" ? "block" : "hidden"}>
        <Hero onSectionChange={setActiveSection} />
      </div>
    </div>
  );
};

export default Index;

