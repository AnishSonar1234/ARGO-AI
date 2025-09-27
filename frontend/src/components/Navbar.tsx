import { Button } from "@/components/ui/button";
import { MessageCircle, BarChart3, Map, Menu } from "lucide-react";
import { useState } from "react";
import { ThemeToggle } from "@/components/ThemeToggle";

interface NavbarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export const Navbar = ({ activeSection, onSectionChange }: NavbarProps) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navItems = [
    { id: "home", label: "Home", icon: null },
    { id: "chat", label: "AI Chat", icon: MessageCircle },
    { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  ];

  return (
    <nav className="bg-background/80 backdrop-blur-sm border-b border-border sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-ocean rounded-lg flex items-center justify-center">
              <div className="w-4 h-4 bg-primary-foreground rounded-full"></div>
            </div>
            <span className="text-xl font-bold bg-gradient-ocean bg-clip-text text-transparent">
              ARGO AI
            </span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Button
                  key={item.id}
                  variant={activeSection === item.id ? "default" : "ghost"}
                  onClick={() => onSectionChange(item.id)}
                  className="flex items-center space-x-2"
                >
                  {Icon && <Icon className="w-4 h-4" />}
                  <span>{item.label}</span>
                </Button>
              );
            })}
            <div className="ml-4">
              <ThemeToggle />
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex items-center space-x-2 md:hidden">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <Menu className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden py-4 border-t border-border">
            <div className="flex flex-col space-y-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Button
                    key={item.id}
                    variant={activeSection === item.id ? "default" : "ghost"}
                    onClick={() => {
                      onSectionChange(item.id);
                      setIsMenuOpen(false);
                    }}
                    className="justify-start flex items-center space-x-2"
                  >
                    {Icon && <Icon className="w-4 h-4" />}
                    <span>{item.label}</span>
                  </Button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};