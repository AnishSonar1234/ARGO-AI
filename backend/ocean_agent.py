"""
Compatibility shim - forwards to app.agent
"""
from app.agent import OceanLangChainAgent, ocean_agent

__all__ = ["OceanLangChainAgent", "ocean_agent"]
