"""LangChain-compatible memory implementations for nanobot."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import Field

if TYPE_CHECKING:
    from nanobot.session.manager import Session, SessionManager


class NanobotFileMemory:
    """
    LangChain-compatible memory that uses nanobot's file-based storage.

    This memory implementation:
    - Persists messages to disk
    - Supports memory consolidation
    - Works with nanobot's SessionManager
    - Integrates with LangChain's Agent framework

    Example:
        ```python
        from nanobot.agent.langchain.memory import NanobotFileMemory

        memory = NanobotFileMemory(
            workspace=Path("~/.nanobot/workspace"),
            session_key="telegram:user123",
            memory_window=100
        )

        # Use with LangChain Agent
        from langchain.agents import AgentExecutor
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory
        )
        ```
    """

    workspace: Path = Field(description="Path to the workspace directory")
    session_key: str = Field(description="Unique key for this session")
    memory_window: int = Field(default=100, description="Number of messages to keep in context")
    session_manager: Optional[Any] = Field(default=None, description="SessionManager instance")

    _session: Optional[Any] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize the memory."""
        super().__init__(**kwargs)
        if self.session_manager:
            self._session = self.session_manager.get_or_create(self.session_key)
        else:
            from nanobot.session.manager import SessionManager
            manager = SessionManager(self.workspace)
            self._session = manager.get_or_create(self.session_key)

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return ["history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables."""
        if not self._session:
            return {"history": []}

        history = self._session.get_history(max_messages=self.memory_window)
        return {"history": self._convert_to_langchain_messages(history)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory."""
        if not self._session:
            return

        # Get the input message
        input_text = inputs.get("input", "")
        if input_text:
            self._session.messages.append({
                "role": "user",
                "content": input_text,
                "timestamp": datetime.now().isoformat(),
            })

        # Get the output message
        output_text = outputs.get("output", "")
        if output_text:
            self._session.messages.append({
                "role": "assistant",
                "content": output_text,
                "timestamp": datetime.now().isoformat(),
            })

        # Save the session
        if self.session_manager:
            self.session_manager.save(self._session)
        else:
            from nanobot.session.manager import SessionManager
            manager = SessionManager(self.workspace)
            manager.save(self._session)

    def clear(self) -> None:
        """Clear memory."""
        if self._session:
            self._session.clear()

    @staticmethod
    def _convert_to_langchain_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """Convert nanobot messages to LangChain format."""
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))

        return result


class NanobotChatHistory(BaseChatMessageHistory):
    """
    LangChain chat history that uses nanobot's file-based storage.

    This implements the BaseChatMessageHistory interface for use with
    LangChain's conversation chain and other components.

    Example:
        ```python
        from nanobot.agent.langchain.memory import NanobotChatHistory

        history = NanobotChatHistory(
            workspace=Path("~/.nanobot/workspace"),
            session_key="telegram:user123"
        )

        # Use with ConversationBufferMemory
        from langchain.memory import ConversationBufferMemory
        memory = ConversationBufferMemory(
            chat_memory=history,
            return_messages=True
        )
        ```
    """

    workspace: Path = Field(description="Path to the workspace directory")
    session_key: str = Field(description="Unique key for this session")
    session_manager: Optional[Any] = Field(default=None)

    _session: Optional[Any] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize the chat history."""
        super().__init__(**kwargs)
        if self.session_manager:
            self._session = self.session_manager.get_or_create(self.session_key)
        else:
            from nanobot.session.manager import SessionManager
            manager = SessionManager(self.workspace)
            self._session = manager.get_or_create(self.session_key)

    @property
    def messages(self) -> List[BaseMessage]:
        """Get the messages from the session."""
        if not self._session:
            return []

        return self._convert_to_langchain_messages(self._session.messages)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history."""
        if not self._session:
            return

        # Convert LangChain message to nanobot format
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"

        self._session.messages.append({
            "role": role,
            "content": message.content,
            "timestamp": datetime.now().isoformat(),
        })

        # Save the session
        if self.session_manager:
            self.session_manager.save(self._session)
        else:
            from nanobot.session.manager import SessionManager
            manager = SessionManager(self.workspace)
            manager.save(self._session)

    def clear(self) -> None:
        """Clear the chat history."""
        if self._session:
            self._session.clear()

    @staticmethod
    def _convert_to_langchain_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """Convert nanobot messages to LangChain format."""
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))

        return result


class ConsolidatedMemory:
    """
    Memory with automatic consolidation for long-running conversations.

    This memory implementation periodically consolidates old messages
    into a summary to maintain context while reducing token usage.
    """

    base_memory: NanobotFileMemory = Field(description="Base memory instance")
    consolidation_threshold: int = Field(
        default=100,
        description="Messages count threshold for consolidation"
    )
    provider: Optional[Any] = Field(default=None, description="LLM provider for consolidation")
    consolidation_model: Optional[str] = Field(default=None, description="Model to use for consolidation")

    _consolidating: bool = False

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables."""
        return self.base_memory.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables, checking if consolidation is needed."""
        # Check if consolidation is needed
        if self._should_consolidate() and not self._consolidating:
            asyncio.run(self._consolidate())

        return self.base_memory.load_memory_variables(inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory."""
        self.base_memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear memory."""
        self.base_memory.clear()

    def _should_consolidate(self) -> bool:
        """Check if consolidation is needed."""
        if not self.base_memory._session:
            return False

        unconsolidated = (
            len(self.base_memory._session.messages) -
            self.base_memory._session.last_consolidated
        )
        return unconsolidated >= self.consolidation_threshold

    async def _consolidate(self) -> None:
        """Perform memory consolidation."""
        if not self.provider or not self.base_memory._session:
            return

        self._consolidating = True
        try:
            from nanobot.agent.memory import MemoryStore
            memory_store = MemoryStore(self.base_memory.workspace)
            await memory_store.consolidate(
                self.base_memory._session,
                self.provider,
                self.consolidation_model or self.provider.get_default_model(),
                memory_window=self.base_memory.memory_window,
            )
        finally:
            self._consolidating = False


def create_memory(
    workspace: Path,
    session_key: str,
    memory_window: int = 100,
    enable_consolidation: bool = False,
    consolidation_threshold: int = 100,
    provider: Optional[Any] = None,
    session_manager: Optional[Any] = None,
):
    """
    Create a memory instance with the specified configuration.

    Helper function for easy memory creation.

    Args:
        workspace: Path to workspace directory
        session_key: Unique session identifier
        memory_window: Number of messages to keep in context
        enable_consolidation: Whether to enable automatic consolidation
        consolidation_threshold: Threshold for consolidation
        provider: LLM provider for consolidation
        session_manager: SessionManager instance

    Returns:
        A configured memory instance
    """
    base_memory = NanobotFileMemory(
        workspace=workspace,
        session_key=session_key,
        memory_window=memory_window,
        session_manager=session_manager,
    )

    if enable_consolidation and provider:
        return ConsolidatedMemory(
            base_memory=base_memory,
            consolidation_threshold=consolidation_threshold,
            provider=provider,
        )

    return base_memory
