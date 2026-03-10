"""LangChain-based agent implementation for nanobot."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import (
    BaseOutputParser,
    OutputParserException,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from loguru import logger

from nanobot.agent.langchain.chat_model import NanobotChatModel
from nanobot.agent.langchain.memory import NanobotFileMemory
from nanobot.agent.langchain.tools import convert_nanobot_tools

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


class ProgressCallback(BaseCallbackHandler):
    """Callback handler for streaming agent progress."""

    def __init__(self, on_progress: Optional[Callable[[str], Any]] = None):
        """Initialize the callback handler."""
        super().__init__()
        self.on_progress = on_progress

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when agent action is taken."""
        if self.on_progress:
            # Format: tool_name("argument_preview")
            args_preview = ""
            if action.tool_input:
                first_value = next(iter(action.tool_input.values()), "")
                if isinstance(first_value, str):
                    args_preview = (
                        f'("{first_value[:40]}…")'
                        if len(first_value) > 40
                        else f'("{first_value}")'
                    )
            self.on_progress(f'{action.tool}{args_preview}')

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM emits a new token."""
        # Note: This is called for every token, which may be too frequent
        # Use with caution and consider debouncing
        pass


class LangChainAgent:
    """
    LangChain-based agent implementation for nanobot.

    This agent uses LangChain's Agent framework while maintaining
    nanobot's tool system, memory, and provider abstraction.

    Example:
        ```python
        from nanobot.agent.langchain.agent import LangChainAgent

        agent = LangChainAgent(
            provider=provider,
            tools=tool_registry,
            workspace=Path("~/.nanobot/workspace"),
            session_manager=session_manager,
        )

        response = await agent.process_message(
            content="Hello!",
            session_key="telegram:user123",
            channel="telegram",
            chat_id="user123"
        )
        ```
    """

    provider: "LLMProvider" = Field(description="The LLM provider")
    tools: "ToolRegistry" = Field(description="Tool registry")
    workspace: Path = Field(description="Path to workspace")
    session_manager: "SessionManager" = Field(description="Session manager")
    bus: Optional["MessageBus"] = Field(default=None, description="Message bus for routing")

    model: str = Field(default="")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4096)
    reasoning_effort: Optional[str] = Field(default=None)
    memory_window: int = Field(default=100)

    _model: Optional[NanobotChatModel] = None
    _executor: Optional[Any] = None
    _prompt_template: Optional[ChatPromptTemplate] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        provider: "LLMProvider",
        tools: "ToolRegistry",
        workspace: Path,
        session_manager: "SessionManager",
        bus: Optional["MessageBus"] = None,
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        memory_window: int = 100,
    ):
        """Initialize the LangChain agent."""
        # Use Pydantic's initialization
        super().__init__(
            provider=provider,
            tools=tools,
            workspace=workspace,
            session_manager=session_manager,
            bus=bus,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            memory_window=memory_window,
        )

        # Initialize model
        self._model = NanobotChatModel(
            provider=provider,
            model=model or provider.get_default_model(),
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        # Initialize prompt template
        self._prompt_template = self._create_prompt_template()

        # Convert tools
        langchain_tools = convert_nanobot_tools(tools._tools)

        # Create agent executor
        # Note: Using tool-calling agent which is the modern approach
        from langchain.agents import create_tool_calling_agent, AgentExecutor

        agent = create_tool_calling_agent(
            self._model,
            langchain_tools,
            self._prompt_template,
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=langchain_tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=40,
        )

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for the agent."""
        # Import ContextBuilder to get the system prompt
        from nanobot.agent.context import ContextBuilder

        context_builder = ContextBuilder(self.workspace)

        # Build system prompt
        system_prompt = context_builder.build_system_prompt()

        # Create prompt template with system prompt, history, and user input
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return template

    async def process_message(
        self,
        content: str,
        session_key: str,
        channel: str = "cli",
        chat_id: str = "default",
        media: Optional[List[str]] = None,
        on_progress: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """
        Process a message and return the response.

        Args:
            content: The message content
            session_key: Unique session identifier
            channel: Channel name (e.g., "telegram", "discord")
            chat_id: Chat ID within the channel
            media: Optional list of media file paths
            on_progress: Optional callback for progress updates

        Returns:
            The agent's response
        """
        # Create memory for this session
        memory = NanobotFileMemory(
            workspace=self.workspace,
            session_key=session_key,
            memory_window=self.memory_window,
            session_manager=self.session_manager,
        )

        # Prepare input
        agent_input = {"input": content}

        # Add history if available
        history_vars = memory.load_memory_variables({})
        if history_vars.get("history"):
            agent_input["history"] = history_vars["history"]

        # Create callback handler
        callbacks = []
        if on_progress:
            callbacks.append(ProgressCallback(on_progress=on_progress))

        # Run the agent
        try:
            config = RunnableConfig(callbacks=callbacks) if callbacks else {}
            result = await self._executor.ainvoke(agent_input, config=config)

            # Save the conversation to memory
            memory.save_context(
                inputs={"input": content},
                outputs={"output": result.get("output", "")},
            )

            return result.get("output", "")

        except Exception as e:
            logger.exception("Error in LangChain agent: {}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def process_message_stream(
        self,
        content: str,
        session_key: str,
        channel: str = "cli",
        chat_id: str = "default",
    ) -> AsyncIterator[str]:
        """
        Process a message and stream the response.

        Args:
            content: The message content
            session_key: Unique session identifier
            channel: Channel name
            chat_id: Chat ID within the channel

        Yields:
            Chunks of the response as they are generated
        """
        # Create memory for this session
        memory = NanobotFileMemory(
            workspace=self.workspace,
            session_key=session_key,
            memory_window=self.memory_window,
            session_manager=self.session_manager,
        )

        # Prepare input
        agent_input = {"input": content}

        # Add history if available
        history_vars = memory.load_memory_variables({})
        if history_vars.get("history"):
            agent_input["history"] = history_vars["history"]

        # Stream the agent output
        async for chunk in self._executor.astream(agent_input):
            if "output" in chunk:
                yield chunk["output"]

        # Save final context to memory
        # Note: This won't work perfectly with streaming - need to accumulate
        pass

    def reset_session(self, session_key: str) -> None:
        """Reset a session's memory."""
        session = self.session_manager.get_or_create(session_key)
        session.clear()
        self.session_manager.save(session)
        self.session_manager.invalidate(session_key)


def create_langchain_agent(
    provider: "LLMProvider",
    tools: "ToolRegistry",
    workspace: Path,
    session_manager: "SessionManager",
    bus: Optional["MessageBus"] = None,
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    reasoning_effort: Optional[str] = None,
    memory_window: int = 100,
) -> LangChainAgent:
    """
    Create a LangChain agent with the given parameters.

    Helper function for easy agent creation.

    Args:
        provider: The LLM provider
        tools: Tool registry
        workspace: Path to workspace
        session_manager: Session manager
        bus: Optional message bus
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        reasoning_effort: Reasoning effort for thinking models
        memory_window: Number of messages to keep in context

    Returns:
        A configured LangChainAgent instance
    """
    return LangChainAgent(
        provider=provider,
        tools=tools,
        workspace=workspace,
        session_manager=session_manager,
        bus=bus,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        memory_window=memory_window,
    )
