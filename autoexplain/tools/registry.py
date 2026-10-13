"""Tool registry for discovering and instantiating XAI tools."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

from autoexplain.tools.base import BaseXAITool


class ToolRegistry:
    """Registry maintaining available XAI diagnostic tools."""

    _tools: Dict[str, Type[BaseXAITool]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseXAITool]], Type[BaseXAITool]]:
        """Decorator to register an XAI tool class."""
        def decorator(tool_cls: Type[BaseXAITool]) -> Type[BaseXAITool]:
            cls._tools[name.lower()] = tool_cls
            return tool_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseXAITool]:
        """Retrieve a tool class by name."""
        name_lower = name.lower()
        if name_lower not in cls._tools:
            available = list(cls._tools.keys())
            raise KeyError(f"Tool '{name}' not found in registry. Available tools: {available}")
        return cls._tools[name_lower]

    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tool names."""
        return sorted(list(cls._tools.keys()))

    @classmethod
    def get_tool_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """Return metadata for all registered tools."""
        return {
            name: {
                "name": tool_cls.name,
                "description": tool_cls.description,
                "supported_model_types": tool_cls.supported_model_types,
            }
            for name, tool_cls in cls._tools.items()
        }


def register_tool(name: str):
    """Convenience alias for ToolRegistry.register."""
    return ToolRegistry.register(name)
