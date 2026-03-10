"""LangChain-based architecture for nanobot.

This module provides LangChain-compatible implementations of nanobot's core components,
allowing nanobot to leverage LangChain's ecosystem while maintaining its lightweight design.

Components:
- NanobotChatModel: LangChain-compatible ChatModel wrapper
- NanobotToolAdapter: Adapter for nanobot tools to LangChain format
- NanobotFileMemory: LangChain-compatible memory with file persistence
- LangChainAgent: LangChain-based agent implementation

Example:
    ```python
    from nanobot.agent.langchain import (
        NanobotChatModel,
        LangChainAgent,
        NanobotFileMemory
    )

    # Create a model
    model = NanobotChatModel(provider=provider, model="claude-opus-4-5")

    # Create an agent
    agent = LangChainAgent(
        provider=provider,
        tools=tool_registry,
        workspace=workspace,
        session_manager=session_manager
    )

    # Process a message
    response = await agent.process_message(
        content="Hello!",
        session_key="telegram:user123"
    )
    ```
"""

from nanobot.agent.langchain.agent import (
    LangChainAgent,
    ProgressCallback,
    create_langchain_agent,
)
from nanobot.agent.langchain.chat_model import (
    NanobotChatModel,
    create_nanobot_model,
)
from nanobot.agent.langchain.memory import (
    ConsolidatedMemory,
    NanobotChatHistory,
    NanobotFileMemory,
    create_memory,
)
from nanobot.agent.langchain.tools import (
    MCPToolAdapter,
    NanobotToolAdapter,
    convert_mcp_tools,
    convert_nanobot_tools,

__all__ = [
    # Agent
    "LangChainAgent",
    "ProgressCallback",
    "create_langchain_agent",
    # Chat Model
    "NanobotChatModel",
    "create_nanobot_model",
    # Memory
    "ConsolidatedMemory",
    "NanobotChatHistory",
    "NanobotFileMemory",
    "create_memory",
    # Tools
    "MCPToolAdapter",
    "NanobotToolAdapter",
    "convert_mcp_tools",
    "convert_nanobot_tools",
)
