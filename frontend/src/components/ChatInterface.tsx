import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User, Maximize2, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
  hasGraph?: boolean;
  graphImage?: string;
  hasMap?: boolean;
  mapHtml?: string;
}

export const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm your AI oceanographic assistant. I can help you explore ARGO float data using natural language queries and create visualizations. Try asking me to 'Show me a graph of temperature vs depth', 'Where are the ARGO floats located?', or 'Plot salinity distribution over time' to see interactive charts and maps!",
      sender: "ai",
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [fullscreenMap, setFullscreenMap] = useState<{ isOpen: boolean; mapHtml?: string }>({
    isOpen: false,
    mapHtml: undefined
  });

  // Handle ESC key to close fullscreen
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && fullscreenMap.isOpen) {
        setFullscreenMap({ isOpen: false, mapHtml: undefined });
      }
    };

    if (fullscreenMap.isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [fullscreenMap.isOpen]);


  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const res = await fetch(import.meta.env.VITE_BACKEND_URL ?? "http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content })
      });
      const data = await res.json();
      const reply = data?.reply ?? data?.error ?? "Sorry, something went wrong.";
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: reply,
        sender: "ai",
        timestamp: new Date(),
        hasGraph: data?.has_graph || false,
        graphImage: data?.graph || undefined,
        hasMap: data?.has_map || false,
        mapHtml: data?.map || undefined
      };
      setMessages(prev => [...prev, aiResponse]);
    } catch (err) {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "Network error. Please try again.",
        sender: "ai",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiResponse]);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gradient-wave py-8">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="w-12 h-12 bg-gradient-ocean rounded-lg flex items-center justify-center mr-3">
              <Bot className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-3xl font-bold">AI Ocean Explorer</h1>
          </div>
          <p className="text-muted-foreground">
            Ask questions about ARGO oceanographic data in natural language
          </p>
        </div>


        {/* Chat Messages */}
        <Card className="h-[700px] mb-4">
          <ScrollArea className="h-full p-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.sender === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-secondary-foreground"
                    }`}
                  >
                    <div className="flex items-start space-x-2">
                      {message.sender === "ai" && (
                        <Bot className="w-4 h-4 mt-1 flex-shrink-0" />
                      )}
                      {message.sender === "user" && (
                        <User className="w-4 h-4 mt-1 flex-shrink-0" />
                      )}
                      <div className="flex-1">
                        {message.sender === "ai" ? (
                          <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent hover:scrollbar-thumb-gray-400">
                            <div className="prose prose-sm dark:prose-invert max-w-none">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {message.content}
                              </ReactMarkdown>
                              {message.hasGraph && message.graphImage && (
                                <div className="mt-4">
                                  <img 
                                    src={`data:image/png;base64,${message.graphImage}`}
                                    alt="Generated graph"
                                    className="max-w-full h-auto rounded-lg border border-border shadow-sm"
                                  />
                                </div>
                              )}
                              {message.hasMap && message.mapHtml && (
                                <div className="mt-4 relative">
                                  <div 
                                    className="w-full h-96 rounded-lg border border-border shadow-sm overflow-hidden"
                                    dangerouslySetInnerHTML={{ 
                                      __html: message.mapHtml 
                                    }}
                                  />
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    className="absolute top-2 right-2 bg-white/95 hover:bg-white shadow-lg border-2 hover:border-blue-300 transition-all duration-200"
                                    onClick={() => setFullscreenMap({ isOpen: true, mapHtml: message.mapHtml })}
                                    title="Open map in fullscreen"
                                  >
                                    <Maximize2 className="w-4 h-4 mr-1" />
                                    Fullscreen
                                  </Button>
                                </div>
                              )}
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm">{message.content}</p>
                        )}
                        <p className="text-xs opacity-70 mt-1">
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-secondary text-secondary-foreground rounded-lg p-3 max-w-[80%]">
                    <div className="flex items-center space-x-2">
                      <Bot className="w-4 h-4" />
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                        <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </Card>

        {/* Input Area */}
        <div className="flex space-x-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about ocean data... (e.g., 'Show me temperature profiles in the Pacific')"
            className="flex-1"
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            disabled={isLoading}
          />
          <Button 
            onClick={handleSendMessage} 
            disabled={isLoading || !inputValue.trim()}
            className="px-6"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        {/* Backend Notice */}
        <Card className="mt-4 p-3 bg-accent/10 border-accent/20">
          <p className="text-xs text-center text-muted-foreground">
            <strong>Note:</strong> This is a demonstration interface. To enable full AI-powered querying of real ARGO data, you'll need to connect a backend system with database integration.
          </p>
        </Card>
      </div>

      {/* Fullscreen Map Modal */}
      {fullscreenMap.isOpen && fullscreenMap.mapHtml && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-2xl w-full h-full max-w-7xl max-h-[95vh] flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-xl font-semibold">ARGO Float Locations - Fullscreen View</h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFullscreenMap({ isOpen: false, mapHtml: undefined })}
                className="hover:bg-gray-100"
              >
                <X className="w-5 h-5" />
              </Button>
            </div>
            
            {/* Map Container */}
            <div className="flex-1 p-4">
              <div 
                className="w-full h-full rounded-lg border border-border shadow-sm overflow-hidden"
                dangerouslySetInnerHTML={{ 
                  __html: fullscreenMap.mapHtml 
                }}
              />
            </div>
            
            {/* Modal Footer */}
            <div className="p-4 border-t bg-gray-50">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-600">
                  Interactive ARGO float map - Click markers for details, use controls to switch layers
                </p>
                <Button
                  variant="outline"
                  onClick={() => setFullscreenMap({ isOpen: false, mapHtml: undefined })}
                >
                  Close Fullscreen
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};