#!/usr/bin/env python3
"""Test script for LangChain-based agent architecture."""

import asyncio
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.session.manager import SessionManager
from nanobot.agent.langchain import LangChainAgent, NanobotChatModel


async def test_langchain_agent():
    """Test the LangChain-based agent."""
    print("🔧 Testing LangChain Agent Architecture\n")

    # Load config
    config = load_config()
    workspace = config.workspace_path

    # Create provider
    provider_config = config.get_provider()
    if not provider_config or not provider_config.api_key:
        print("❌ No API key configured. Please set up a provider in config.json")
        return

    provider = LiteLLMProvider(
        api_key=provider_config.api_key,
        api_base=config.get_api_base(),
    )

    # Create tool registry
    tools = ToolRegistry()
    for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
        tools.register(cls(workspace=workspace, allowed_dir=None))
    tools.register(ExecTool(
        working_dir=str(workspace),
        timeout=60,
        restrict_to_workspace=False,
        path_append="",
    ))

    # Create session manager
    session_manager = SessionManager(workspace)

    # Create LangChain agent
    print("📦 Creating LangChain Agent...")
    agent = LangChainAgent(
        provider=provider,
        tools=tools,
        workspace=workspace,
        session_manager=session_manager,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        memory_window=config.agents.defaults.memory_window,
    )

    # Test a simple message
    session_key = "test:langchain_user"
    test_message = "Hello! Can you tell me what files are in the current directory?"

    print(f"\n🤖 Processing message: '{test_message}'")
    print("=" * 60)

    response = await agent.process_message(
        content=test_message,
        session_key=session_key,
        channel="test",
        chat_id="langchain_user",
    )

    print("=" * 60)
    print(f"\n✅ Response:\n{response}")

    # Test the model directly
    print("\n\n🔧 Testing NanobotChatModel directly...")
    model = NanobotChatModel(
        provider=provider,
        model=config.agents.defaults.model,
        temperature=0.1,
    )

    from langchain_core.messages import HumanMessage
    msg = HumanMessage(content="Say 'LangChain integration works!' in a single sentence.")
    result = await model.ainvoke([msg])
    print(f"📝 Model response: {result.content}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_langchain_agent())
