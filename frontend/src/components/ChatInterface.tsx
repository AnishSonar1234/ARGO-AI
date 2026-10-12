import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  User, 
  Send, 
  ArrowUp,
  Maximize2, 
  X, 
  Plus, 
  Trash2, 
  MessageSquare, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Download, 
  RotateCcw, 
  Copy, 
  Check, 
  Sparkles, 
  Brain, 
  ChevronDown, 
  ChevronUp,
  MapPin,
  BarChart2,
  ThumbsUp,
  ThumbsDown,
  Edit2,
  CheckCircle2,
  Flame,
  Globe2,
  Compass,
  ArrowDown,
  Waves,
  Radio,
  Activity,
  Wifi,
  WifiOff,
  AlertCircle
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { WhaleIcon } from "@/components/WhaleIcon";

export interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: string;
  hasGraph?: boolean;
  graphImage?: string;
  hasMap?: boolean;
  mapHtml?: string;
  reasoningDetails?: any;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

const STORAGE_KEY_SESSIONS = "argo_ai_chat_sessions_v3";
const STORAGE_KEY_ACTIVE_ID = "argo_ai_active_session_id_v3";

const INITIAL_SUGGESTIONS = [
  {
    icon: BarChart2,
    title: "Temperature vs Depth",
    subtitle: "Pacific Ocean Bathymetry",
    prompt: "Show me a graph of temperature vs depth in the Pacific Ocean"
  },
  {
    icon: MapPin,
    title: "Float Coordinates",
    subtitle: "North Atlantic Ingestion",
    prompt: "Where are active ARGO floats located right now in the North Atlantic?"
  },
  {
    icon: Compass,
    title: "Salinity Distribution",
    subtitle: "Water Column Halocline",
    prompt: "Plot salinity distribution across depths and explain the halocline"
  },
  {
    icon: Globe2,
    title: "Southern Ocean Drift",
    subtitle: "Antarctic Circumpolar Flow",
    prompt: "What are the latest ARGO float observations in the Southern Ocean?"
  }
];

export const ChatInterface = () => {
  // Load or initialize sessions
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_SESSIONS);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.error("Error loading chat sessions:", e);
    }
    const defaultSession: ChatSession = {
      id: `session_${Date.now()}`,
      title: "New Whale Conversation",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: []
    };
    return [defaultSession];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    try {
      const savedId = localStorage.getItem(STORAGE_KEY_ACTIVE_ID);
      if (savedId) return savedId;
    } catch (e) {}
    return sessions[0]?.id || `session_${Date.now()}`;
  });

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [serverStatus, setServerStatus] = useState<"connected" | "offline" | "checking">("checking");
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitleInput, setEditTitleInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [feedbackMap, setFeedbackMap] = useState<Record<string, "up" | "down">>({});
  const [showScrollBottom, setShowScrollBottom] = useState(false);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});

  // Fullscreen modals
  const [fullscreenMap, setFullscreenMap] = useState<{ isOpen: boolean; mapHtml?: string }>({
    isOpen: false,
    mapHtml: undefined
  });
  const [fullscreenGraph, setFullscreenGraph] = useState<{ isOpen: boolean; graphImage?: string }>({
    isOpen: false,
    graphImage: undefined
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatScrollContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Active session helper
  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
  const messages = activeSession?.messages || [];

  // Check backend server connection on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const baseUrl = import.meta.env.VITE_BACKEND_URL 
          ? import.meta.env.VITE_BACKEND_URL.replace("/chat", "") 
          : "http://localhost:5000";
        const res = await fetch(`${baseUrl}/api/rag/status`);
        if (res.ok) {
          setServerStatus("connected");
        } else {
          setServerStatus("offline");
        }
      } catch {
        setServerStatus("offline");
      }
    };
    checkServer();
  }, []);

  // Timer while Whale is responding
  useEffect(() => {
    let timer: any = null;
    if (isLoading) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      setElapsedSeconds(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLoading]);

  // Check if a prefilled prompt was set by Dashboard (e.g. "Ask AI About Float")
  useEffect(() => {
    const prefill = sessionStorage.getItem("argo_selected_float_query");
    if (prefill) {
      setInputValue(prefill);
      sessionStorage.removeItem("argo_selected_float_query");
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, []);

  // Save sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
    } catch (e) {
      console.error("Error saving sessions:", e);
    }
  }, [sessions]);

  // Save active session id
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_ACTIVE_ID, activeSessionId);
    } catch (e) {}
  }, [activeSessionId]);

  // Auto-scroll to bottom on messages change
  useEffect(() => {
    scrollToBottom("smooth");
  }, [messages.length, isLoading]);

  // Adjust textarea height dynamically
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(Math.max(scrollHeight, 44), 180)}px`;
    }
  }, [inputValue]);

  // Handle ESC key for modals
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (fullscreenMap.isOpen) setFullscreenMap({ isOpen: false, mapHtml: undefined });
        if (fullscreenGraph.isOpen) setFullscreenGraph({ isOpen: false, graphImage: undefined });
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [fullscreenMap.isOpen, fullscreenGraph.isOpen]);

  // Scroll detection for scroll-to-bottom button
  const handleScroll = () => {
    if (!chatScrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = chatScrollContainerRef.current;
    const isUp = scrollHeight - scrollTop - clientHeight > 180;
    setShowScrollBottom(isUp);
  };

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  // Create new session (Claude style)
  const handleCreateNewSession = () => {
    const newSession: ChatSession = {
      id: `session_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      title: "New Whale Conversation",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: []
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setInputValue("");
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  // Delete session
  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessions(prev => {
      const filtered = prev.filter(s => s.id !== id);
      if (filtered.length === 0) {
        const fresh: ChatSession = {
          id: `session_${Date.now()}`,
          title: "New Whale Conversation",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: []
        };
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (activeSessionId === id) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  // Start editing title
  const handleStartRename = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSessionId(session.id);
    setEditTitleInput(session.title);
  };

  // Save session title
  const handleSaveRename = (id: string) => {
    if (editTitleInput.trim()) {
      setSessions(prev => prev.map(s => s.id === id ? { ...s, title: editTitleInput.trim() } : s));
    }
    setEditingSessionId(null);
  };

  // Clear current chat
  const handleClearCurrentChat = () => {
    if (window.confirm("Clear all messages in this conversation?")) {
      setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, messages: [], updatedAt: Date.now() } : s));
    }
  };

  // Export conversation as Markdown
  const handleExportChat = () => {
    if (messages.length === 0) return;
    let md = `# ${activeSession.title}\n\n*Exported from ARGO Whale Oceanographer on ${new Date().toLocaleString()}*\n\n---\n\n`;
    messages.forEach(m => {
      md += `### ${m.sender === "user" ? "👤 You" : "🐋 ARGO Whale"}\n*${m.timestamp}*\n\n${m.content}\n\n---\n\n`;
    });
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${activeSession.title.replace(/[^a-z0-9]/gi, "_").toLowerCase()}_whale_chat.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Copy message text
  const handleCopyText = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Dynamic status text for responding phases
  const getRespondingStageText = () => {
    if (elapsedSeconds <= 3) {
      return "Whale diving into ChromaDB ocean knowledge...";
    } else if (elapsedSeconds <= 7) {
      return "Whale retrieving real-time ARGO CTD telemetry...";
    } else {
      return "Whale synthesizing multi-depth Nemotron reasoning...";
    }
  };

  // Send message
  const handleSendMessage = async (textToSend?: string) => {
    const prompt = (textToSend ?? inputValue).trim();
    if (!prompt || isLoading) return;

    const userMessage: Message = {
      id: `msg_u_${Date.now()}`,
      content: prompt,
      sender: "user",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };

    // Auto-update session title on first message
    const updatedMessages = [...messages, userMessage];
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        const isFirst = s.messages.length === 0;
        const newTitle = isFirst ? (prompt.length > 36 ? prompt.substring(0, 36) + "..." : prompt) : s.title;
        return {
          ...s,
          title: newTitle,
          updatedAt: Date.now(),
          messages: updatedMessages
        };
      }
      return s;
    }));

    setInputValue("");
    if (textareaRef.current) textareaRef.current.style.height = "44px";
    setIsLoading(true);

    try {
      // Build conversation history including preserved reasoning_details
      const history = messages.map(m => {
        if (m.sender === "user") {
          return { role: "user", content: m.content };
        }
        const asst: any = { role: "assistant", content: m.content };
        if (m.reasoningDetails) {
          asst.reasoning_details = m.reasoningDetails;
        }
        return asst;
      });

      const res = await fetch(import.meta.env.VITE_BACKEND_URL ?? "http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt,
          history: history
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      const reply = data?.reply ?? data?.error ?? "Whale reached maximum depth but returned empty telemetry.";

      const whaleResponse: Message = {
        id: `msg_a_${Date.now()}`,
        content: reply,
        sender: "ai",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        hasGraph: data?.has_graph || false,
        graphImage: data?.graph || undefined,
        hasMap: data?.has_map || false,
        mapHtml: data?.map || undefined,
        reasoningDetails: data?.reasoning_details
      };

      setServerStatus("connected");
      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            updatedAt: Date.now(),
            messages: [...s.messages, whaleResponse]
          };
        }
        return s;
      }));
    } catch (err: any) {
      setServerStatus("offline");
      const errorMsg: Message = {
        id: `msg_err_${Date.now()}`,
        content: `⚠️ **Whale Connection Alert**: Could not receive response from ARGO backend server.\n\n*Details: ${err?.message || "Connection refused"}*\n\nPlease make sure the Python Flask backend is running on \`http://localhost:5000\`.`,
        sender: "ai",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      };
      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            updatedAt: Date.now(),
            messages: [...s.messages, errorMsg]
          };
        }
        return s;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  // Keyboard shortcut in textarea: Enter to send, Shift+Enter for newline
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Group sessions by date
  const groupSessionsByDate = () => {
    const today: ChatSession[] = [];
    const yesterday: ChatSession[] = [];
    const older: ChatSession[] = [];
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startOfYesterday = startOfToday - 86400000;

    sessions.forEach(s => {
      if (s.updatedAt >= startOfToday) {
        today.push(s);
      } else if (s.updatedAt >= startOfYesterday) {
        yesterday.push(s);
      } else {
        older.push(s);
      }
    });

    return { today, yesterday, older };
  };

  const { today, yesterday, older } = groupSessionsByDate();

  return (
    <div className="h-full w-full flex overflow-hidden bg-background select-none md:select-auto">
      {/* 1. LEFT SIDEBAR: Claude-Style Chat History */}
      <aside 
        className={`${
          isSidebarOpen ? "w-72 border-r" : "w-0 border-none"
        } transition-all duration-300 ease-in-out flex flex-col flex-shrink-0 bg-card/60 dark:bg-card/40 backdrop-blur-md border-border h-full overflow-hidden z-20`}
      >
        <div className="p-3 border-b border-border/70 flex items-center justify-between">
          <Button 
            onClick={handleCreateNewSession}
            variant="outline" 
            className="w-full flex items-center justify-start gap-2 bg-background hover:bg-muted/80 border-border shadow-xs text-sm font-medium rounded-xl h-10 px-3"
          >
            <WhaleIcon className="w-4 h-4 text-primary" />
            <span>New Whale Chat</span>
            <kbd className="ml-auto text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded border border-border">Ctrl+K</kbd>
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsSidebarOpen(false)}
            className="md:hidden ml-1 h-9 w-9 text-muted-foreground"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Sessions list */}
        <div className="flex-1 overflow-y-auto px-2 py-3 space-y-4 scrollbar-thin">
          {today.length > 0 && (
            <div>
              <div className="px-2 pb-1 text-[11px] font-semibold tracking-wider uppercase text-muted-foreground">Today</div>
              <div className="space-y-1">
                {today.map(renderSessionItem)}
              </div>
            </div>
          )}

          {yesterday.length > 0 && (
            <div>
              <div className="px-2 pb-1 text-[11px] font-semibold tracking-wider uppercase text-muted-foreground">Yesterday</div>
              <div className="space-y-1">
                {yesterday.map(renderSessionItem)}
              </div>
            </div>
          )}

          {older.length > 0 && (
            <div>
              <div className="px-2 pb-1 text-[11px] font-semibold tracking-wider uppercase text-muted-foreground">Previous Explorations</div>
              <div className="space-y-1">
                {older.map(renderSessionItem)}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Footer with Whale Branding */}
        <div className="p-3 border-t border-border/70 bg-card/40 text-xs text-muted-foreground flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-primary/10 flex items-center justify-center">
              <WhaleIcon className="w-3.5 h-3.5 text-primary" />
            </div>
            <span className="font-semibold text-foreground text-xs">ARGO Whale</span>
          </div>
          <Badge variant="outline" className="text-[10px] font-mono border-primary/30 text-primary">
            v3.2
          </Badge>
        </div>
      </aside>

      {/* 2. MAIN CHAT AREA */}
      <main className="flex-1 flex flex-col h-full min-w-0 bg-background relative overflow-hidden">
        {/* Top Header Bar */}
        <header className="h-14 border-b border-border/70 px-4 flex items-center justify-between flex-shrink-0 bg-background/80 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3 min-w-0">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="h-9 w-9 text-muted-foreground hover:text-foreground"
              title={isSidebarOpen ? "Collapse sidebar" : "Open sidebar"}
            >
              {isSidebarOpen ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
            </Button>
            
            <div className="flex items-center gap-2.5 truncate">
              <span className="font-semibold text-sm truncate max-w-[200px] sm:max-w-xs">
                {activeSession?.title}
              </span>

              {/* INDICATION: CLEARLY SHOWS WHETHER CHAT IS RESPONDING OR NOT */}
              {isLoading ? (
                <Badge variant="outline" className="flex items-center gap-1.5 text-[11px] font-semibold border-cyan-500/50 bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 shadow-xs animate-pulse">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                  </span>
                  <Waves className="w-3.5 h-3.5 animate-bounce text-cyan-500" />
                  <span>Whale Responding ({elapsedSeconds}s)</span>
                </Badge>
              ) : serverStatus === "offline" ? (
                <Badge variant="outline" className="flex items-center gap-1.5 text-[11px] font-medium border-rose-500/40 bg-rose-500/10 text-rose-500">
                  <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                  <WifiOff className="w-3 h-3 text-rose-500" />
                  <span>Whale Offline</span>
                </Badge>
              ) : (
                <Badge variant="outline" className="flex items-center gap-1.5 text-[11px] font-normal border-emerald-500/40 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                  <WhaleIcon className="w-3 h-3 text-emerald-500" />
                  <span>Whale Ready</span>
                </Badge>
              )}

              <Badge variant="outline" className="hidden lg:inline-flex items-center gap-1 text-[11px] text-muted-foreground border-border/80 font-normal">
                <Brain className="w-3 h-3 text-primary" />
                <span>Nemotron-3 Reasoning</span>
              </Badge>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleExportChat}
              disabled={messages.length === 0}
              className="h-8 text-xs text-muted-foreground hover:text-foreground hidden sm:flex items-center gap-1"
              title="Export conversation as Markdown"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearCurrentChat}
              disabled={messages.length === 0}
              className="h-8 text-xs text-muted-foreground hover:text-destructive flex items-center gap-1"
              title="Clear current messages"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear</span>
            </Button>
          </div>
        </header>

        {/* Scrollable Conversation Feed */}
        <div 
          ref={chatScrollContainerRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth scrollbar-thin"
        >
          {messages.length === 0 ? (
            /* Claude Signature Empty / Greeting State with Whale Branding */
            <div className="max-w-2xl mx-auto h-full flex flex-col justify-center items-center text-center py-8">
              <div className="w-16 h-16 rounded-3xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/25 mb-5 ring-4 ring-cyan-500/20">
                <WhaleIcon className="w-9 h-9 text-white" />
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight mb-2 bg-gradient-to-r from-foreground via-foreground/90 to-foreground/70 bg-clip-text">
                Ask ARGO Whale
              </h2>
              <p className="text-sm text-muted-foreground max-w-lg mb-8 leading-relaxed">
                Your autonomous oceanographic navigator. Ask questions about global ARGO float measurements, 3D underwater CTD curves, and multi-depth ocean intelligence.
              </p>

              {/* 2x2 Suggestion Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
                {INITIAL_SUGGESTIONS.map((s, idx) => {
                  const Icon = s.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => handleSendMessage(s.prompt)}
                      className="flex items-start gap-3 p-3.5 rounded-xl border border-border/70 bg-card hover:bg-muted/70 hover:border-primary/40 transition-all text-left shadow-2xs group cursor-pointer"
                    >
                      <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:scale-105 transition-transform">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-semibold text-foreground truncate">{s.title}</div>
                        <div className="text-[11px] text-muted-foreground truncate">{s.subtitle}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            /* Conversation Messages */
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <div key={message.id} className="group transition-all">
                  {message.sender === "user" ? (
                    /* User Message: Clean right-aligned pill */
                    <div className="flex justify-end items-start gap-2 pl-12">
                      <div className="relative max-w-[85%] rounded-2xl px-4 py-3 bg-primary text-primary-foreground shadow-xs">
                        <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
                        <div className="mt-1 flex items-center justify-end gap-2 text-[10px] text-primary-foreground/70">
                          <span>{message.timestamp}</span>
                          <button
                            onClick={() => handleCopyText(message.id, message.content)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity hover:text-white"
                            title="Copy text"
                          >
                            {copiedId === message.id ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* Whale Message: Ocean theme with Whale avatar */
                    <div className="flex items-start gap-3.5 pr-2">
                      <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 flex items-center justify-center flex-shrink-0 shadow-xs mt-1 border border-cyan-400/30">
                        <WhaleIcon className="w-4.5 h-4.5 text-white" />
                      </div>
                      <div className="flex-1 min-w-0 space-y-3">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-xs text-foreground flex items-center gap-1.5">
                            <span>ARGO Whale</span>
                            <Badge variant="outline" className="text-[9px] py-0 px-1 border-primary/30 text-primary">
                              AI Oceanographer
                            </Badge>
                          </span>
                          <span className="text-[10px] text-muted-foreground">{message.timestamp}</span>
                        </div>

                        {/* Claude 3.7 Thinking Process Accordion */}
                        {message.reasoningDetails && (
                          <div className="rounded-xl border border-border/70 bg-muted/40 dark:bg-muted/20 overflow-hidden">
                            <button
                              onClick={() => {
                                setExpandedThinking(prev => ({
                                  ...prev,
                                  [message.id]: !prev[message.id]
                                }));
                              }}
                              className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <div className="flex items-center gap-1.5">
                                <Brain className="w-3.5 h-3.5 text-primary" />
                                <span>Whale Reasoning Process</span>
                              </div>
                              {expandedThinking[message.id] ? (
                                <ChevronUp className="w-3.5 h-3.5" />
                              ) : (
                                <ChevronDown className="w-3.5 h-3.5" />
                              )}
                            </button>
                            {expandedThinking[message.id] && (
                              <div className="p-3 border-t border-border/50 bg-background/50 text-[11px] font-mono text-muted-foreground whitespace-pre-wrap max-h-60 overflow-y-auto scrollbar-thin">
                                {typeof message.reasoningDetails === "string"
                                  ? message.reasoningDetails
                                  : JSON.stringify(message.reasoningDetails, null, 2)}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Content Markdown */}
                        <div className="prose prose-sm dark:prose-invert max-w-none text-foreground leading-relaxed">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>

                        {/* Embedded Interactive Ocean Map */}
                        {message.hasMap && message.mapHtml && (
                          <div className="mt-3 rounded-xl border border-border overflow-hidden bg-card shadow-xs">
                            <div className="bg-muted/60 px-3 py-2 border-b border-border flex items-center justify-between text-xs">
                              <span className="font-medium flex items-center gap-1.5 text-foreground">
                                <MapPin className="w-3.5 h-3.5 text-primary" />
                                Interactive ARGO Float Positions Map
                              </span>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => setFullscreenMap({ isOpen: true, mapHtml: message.mapHtml })}
                                className="h-6 text-[11px] px-2"
                              >
                                <Maximize2 className="w-3 h-3 mr-1" />
                                Expand
                              </Button>
                            </div>
                            <div 
                              className="w-full h-80 overflow-hidden"
                              dangerouslySetInnerHTML={{ __html: message.mapHtml }}
                            />
                          </div>
                        )}

                        {/* Embedded Bathymetry / Depth Graph */}
                        {message.hasGraph && message.graphImage && (
                          <div className="mt-3 rounded-xl border border-border overflow-hidden bg-card shadow-xs">
                            <div className="bg-muted/60 px-3 py-2 border-b border-border flex items-center justify-between text-xs">
                              <span className="font-medium flex items-center gap-1.5 text-foreground">
                                <BarChart2 className="w-3.5 h-3.5 text-primary" />
                                CTD Depth Profile Chart
                              </span>
                              <div className="flex items-center gap-1">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => {
                                    const link = document.createElement("a");
                                    link.href = `data:image/png;base64,${message.graphImage}`;
                                    link.download = "argo_ctd_profile.png";
                                    link.click();
                                  }}
                                  className="h-6 text-[11px] px-2"
                                >
                                  <Download className="w-3 h-3 mr-1" />
                                  Save
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => setFullscreenGraph({ isOpen: true, graphImage: message.graphImage })}
                                  className="h-6 text-[11px] px-2"
                                >
                                  <Maximize2 className="w-3 h-3 mr-1" />
                                  Expand
                                </Button>
                              </div>
                            </div>
                            <div className="p-3 bg-slate-950 flex justify-center">
                              <img
                                src={`data:image/png;base64,${message.graphImage}`}
                                alt="Oceanographic Depth Profile"
                                className="max-h-96 rounded-lg object-contain cursor-pointer hover:opacity-95 transition-opacity"
                                onClick={() => setFullscreenGraph({ isOpen: true, graphImage: message.graphImage })}
                              />
                            </div>
                          </div>
                        )}

                        {/* Message Action Bar */}
                        <div className="flex items-center gap-2 pt-1 text-xs text-muted-foreground">
                          <button
                            onClick={() => handleCopyText(message.id, message.content)}
                            className="p-1.5 rounded-md hover:bg-muted hover:text-foreground transition-colors"
                            title="Copy response"
                          >
                            {copiedId === message.id ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                          </button>
                          <button
                            onClick={() => setFeedbackMap(prev => ({ ...prev, [message.id]: "up" }))}
                            className={`p-1.5 rounded-md hover:bg-muted transition-colors ${
                              feedbackMap[message.id] === "up" ? "text-emerald-500 bg-muted" : "hover:text-foreground"
                            }`}
                            title="Helpful"
                          >
                            <ThumbsUp className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => setFeedbackMap(prev => ({ ...prev, [message.id]: "down" }))}
                            className={`p-1.5 rounded-md hover:bg-muted transition-colors ${
                              feedbackMap[message.id] === "down" ? "text-rose-500 bg-muted" : "hover:text-foreground"
                            }`}
                            title="Not helpful"
                          >
                            <ThumbsDown className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* WHALE RESPONDING INDICATION: CLEAR VISUAL FEEDBACK WITH SONAR BARS & STAGE TEXT */}
              {isLoading && (
                <div className="flex items-start gap-3.5 pr-2 animate-in fade-in duration-300">
                  <div className="relative w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 flex items-center justify-center flex-shrink-0 shadow-md border border-cyan-400/40">
                    <WhaleIcon className="w-4.5 h-4.5 text-white animate-pulse" />
                    <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
                    </span>
                  </div>

                  <div className="flex-1 min-w-0 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-xs text-foreground flex items-center gap-1.5">
                        <span>ARGO Whale</span>
                        <Badge variant="outline" className="text-[10px] py-0 px-1.5 border-cyan-500/40 text-cyan-600 dark:text-cyan-400 bg-cyan-500/10 font-mono">
                          Responding ({elapsedSeconds}s)
                        </Badge>
                      </span>
                    </div>

                    <div className="p-3.5 rounded-2xl bg-card border border-primary/20 shadow-xs space-y-2.5 max-w-md">
                      <div className="flex items-center gap-2 text-xs text-foreground font-medium">
                        <Waves className="w-4 h-4 text-cyan-500 animate-pulse flex-shrink-0" />
                        <span className="truncate">{getRespondingStageText()}</span>
                      </div>

                      {/* Animated Ocean Sonar Equalizer Waves */}
                      <div className="flex items-center gap-1.5 pt-1">
                        <span className="w-1.5 h-4 bg-cyan-500 rounded-full animate-pulse"></span>
                        <span className="w-1.5 h-6 bg-blue-500 rounded-full animate-[pulse_1.2s_ease-in-out_infinite_200ms]"></span>
                        <span className="w-1.5 h-3 bg-indigo-500 rounded-full animate-[pulse_0.8s_ease-in-out_infinite_400ms]"></span>
                        <span className="w-1.5 h-5 bg-cyan-400 rounded-full animate-[pulse_1.1s_ease-in-out_infinite_600ms]"></span>
                        <span className="text-[11px] text-muted-foreground ml-2 font-mono">
                          CTD acoustic ping active...
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Floating Scroll to Bottom Button */}
        {showScrollBottom && (
          <button
            onClick={() => scrollToBottom("smooth")}
            className="absolute bottom-24 right-8 p-2 rounded-full bg-card border border-border shadow-md hover:bg-muted text-muted-foreground hover:text-foreground transition-all z-20"
            title="Scroll to bottom"
          >
            <ArrowDown className="w-4 h-4" />
          </button>
        )}

        {/* 3. CLAUDE SIGNATURE FLOATING INPUT BAR (FIXED AT BOTTOM) */}
        <div className="flex-shrink-0 w-full px-4 pb-4 pt-2 bg-gradient-to-t from-background via-background/95 to-transparent z-10">
          <div className="max-w-3xl mx-auto">
            {/* Input card container */}
            <div className={`rounded-2xl border bg-card shadow-sm transition-all p-2.5 ${
              isLoading 
                ? "border-cyan-500/40 ring-2 ring-cyan-500/10 shadow-md" 
                : "border-input hover:shadow-md focus-within:shadow-md focus-within:border-primary/60"
            }`}>
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  isLoading 
                    ? "Whale is diving & responding, please wait..." 
                    : "Ask Whale about ocean data... (e.g., 'Show me temperature vs depth in Pacific')"
                }
                rows={1}
                disabled={isLoading}
                className="w-full bg-transparent resize-none border-none outline-none text-sm text-foreground placeholder:text-muted-foreground px-2 py-1 max-h-[180px] overflow-y-auto scrollbar-thin disabled:opacity-60"
              />

              {/* Bottom toolbar */}
              <div className="flex items-center justify-between pt-1.5 px-1 border-t border-border/40 mt-1">
                <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-none py-0.5">
                  <button
                    type="button"
                    disabled={isLoading}
                    onClick={() => setInputValue("Show me a map of ARGO float locations in ")}
                    className="text-[11px] px-2 py-0.5 rounded-md bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors whitespace-nowrap disabled:opacity-50"
                  >
                    🗺️ Map
                  </button>
                  <button
                    type="button"
                    disabled={isLoading}
                    onClick={() => setInputValue("Show me a graph of temperature vs depth in ")}
                    className="text-[11px] px-2 py-0.5 rounded-md bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors whitespace-nowrap disabled:opacity-50"
                  >
                    📊 Graph
                  </button>
                  <button
                    type="button"
                    disabled={isLoading}
                    onClick={() => setInputValue("Analyze salinity anomalies in ")}
                    className="text-[11px] px-2 py-0.5 rounded-md bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors whitespace-nowrap disabled:opacity-50"
                  >
                    🧂 Salinity
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    size="icon"
                    onClick={() => handleSendMessage()}
                    disabled={!inputValue.trim() || isLoading}
                    className="h-8 w-8 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-all shadow-xs flex-shrink-0"
                    title={isLoading ? "Whale is responding..." : "Send prompt to Whale"}
                  >
                    {isLoading ? (
                      <Waves className="w-4 h-4 animate-spin text-white" />
                    ) : (
                      <ArrowUp className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              </div>
            </div>

            {/* Footer Disclaimer */}
            <div className="text-center mt-2 text-[11px] text-muted-foreground/70 flex items-center justify-center gap-1.5">
              <WhaleIcon className="w-3 h-3 text-cyan-500 inline" />
              <span>ARGO Whale Intelligence • NOAA ERDDAP & Coriolis Ingestion • Nemotron 3 Reasoning</span>
            </div>
          </div>
        </div>
      </main>

      {/* Fullscreen Map Modal */}
      {fullscreenMap.isOpen && fullscreenMap.mapHtml && (
        <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-md flex flex-col p-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div className="flex items-center gap-2">
              <MapPin className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-base">ARGO Float Global Coordinates</h3>
            </div>
            <Button
              variant="outline"
              size="icon"
              onClick={() => setFullscreenMap({ isOpen: false, mapHtml: undefined })}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          <div 
            className="flex-1 w-full rounded-xl overflow-hidden mt-3 border border-border"
            dangerouslySetInnerHTML={{ __html: fullscreenMap.mapHtml }}
          />
        </div>
      )}

      {/* Fullscreen Graph Modal */}
      {fullscreenGraph.isOpen && fullscreenGraph.graphImage && (
        <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-md flex flex-col p-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div className="flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-base">Oceanographic Depth & Bathymetry Chart</h3>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const link = document.createElement("a");
                  link.href = `data:image/png;base64,${fullscreenGraph.graphImage}`;
                  link.download = "argo_ocean_graph.png";
                  link.click();
                }}
              >
                <Download className="w-4 h-4 mr-1" />
                Download
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setFullscreenGraph({ isOpen: false, graphImage: undefined })}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center p-4 overflow-auto">
            <img
              src={`data:image/png;base64,${fullscreenGraph.graphImage}`}
              alt="High-resolution oceanographic chart"
              className="max-h-full max-w-full rounded-xl shadow-lg border border-border"
            />
          </div>
        </div>
      )}
    </div>
  );

  // Helper renderer for sidebar items
  function renderSessionItem(session: ChatSession) {
    const isActive = session.id === activeSessionId;
    const isEditing = editingSessionId === session.id;

    return (
      <div
        key={session.id}
        onClick={() => {
          setActiveSessionId(session.id);
          if (window.innerWidth < 768) setIsSidebarOpen(false);
        }}
        className={`group relative flex items-center gap-2 px-2.5 py-2 rounded-xl text-xs cursor-pointer transition-colors ${
          isActive 
            ? "bg-primary/10 text-primary font-medium" 
            : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
        }`}
      >
        <MessageSquare className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? "text-primary" : "text-muted-foreground"}`} />
        
        {isEditing ? (
          <input
            type="text"
            value={editTitleInput}
            onChange={(e) => setEditTitleInput(e.target.value)}
            onBlur={() => handleSaveRename(session.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSaveRename(session.id);
              if (e.key === "Escape") setEditingSessionId(null);
            }}
            autoFocus
            className="flex-1 bg-background border border-primary rounded px-1.5 py-0.5 text-xs text-foreground outline-none"
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className="flex-1 truncate">{session.title}</span>
        )}

        <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
          <button
            onClick={(e) => handleStartRename(session, e)}
            className="p-1 hover:text-foreground rounded"
            title="Rename conversation"
          >
            <Edit2 className="w-3 h-3" />
          </button>
          <button
            onClick={(e) => handleDeleteSession(session.id, e)}
            className="p-1 hover:text-destructive rounded"
            title="Delete conversation"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      </div>
    );
  }
};