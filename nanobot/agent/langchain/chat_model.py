"""LangChain-compatible ChatModel wrapper for nanobot providers."""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, agenerate_from_stream
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class NanobotChatModel(BaseChatModel):
    """
    LangChain-compatible ChatModel wrapper for nanobot's LLM providers.

    This allows nanobot to use LangChain's Agent, Chain, and Tool ecosystem
    while maintaining its own provider abstraction layer.

    Example:
        ```python
        from nanobot.providers.litellm_provider import LiteLLMProvider
        from nanobot.agent.langchain.chat_model import NanobotChatModel

        provider = LiteLLMProvider(api_key="...", model="claude-opus-4-5")
        model = NanobotChatModel(provider=provider)

        # Use with LangChain Agent
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        agent = create_tool_calling_agent(model, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        ```
    """

    provider: "LLMProvider" = Field(description="The underlying nanobot LLM provider")
    model: str = Field(description="Model name to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum tokens to generate")
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort for thinking models")

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "nanobot"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to nanobot format."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    # Handle multimodal content
                    formatted_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            formatted_content.append(item)
                    result.append({"role": "user", "content": formatted_content if formatted_content else str(content)})
                else:
                    result.append({"role": "user", "content": str(content)})
            elif isinstance(msg, AIMessage):
                msg_dict: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                # Handle tool calls
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"], ensure_ascii=False)
                            }
                        })
                    msg_dict["tool_calls"] = tool_calls
                result.append(msg_dict)
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "content": msg.content
                })
            elif isinstance(msg, ChatMessage):
                role_mapping = {"human": "user", "ai": "assistant", "system": "system"}
                role = role_mapping.get(msg.role, msg.role)
                result.append({"role": role, "content": msg.content})
            else:
                # Fallback for other message types
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _convert_tools(self, tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
        """Convert LangChain tools to nanobot format."""
        tool_defs = []
        for tool in tools:
            schema = tool.args_schema
            if schema:
                # Get the JSON schema from the args_schema
                if hasattr(schema, "schema"):
                    properties = schema.schema().get("properties", {})
                else:
                    properties = {}

                tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": getattr(schema, "required_fields", [])
                        }
                    }
                })
        return tool_defs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        import asyncio
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat completion."""
        # Convert messages
        nanobot_messages = self._convert_messages(messages)

        # Convert tools if present in kwargs
        tools = kwargs.get("tools")
        tool_definitions = []
        if tools:
            tool_definitions = self._convert_tools(tools)

        # Call the provider
        response = await self.provider.chat(
            messages=nanobot_messages,
            tools=tool_definitions,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )

        # Convert response back to LangChain format
        message = AIMessage(content=response.content or "")

        # Add tool calls if present
        if response.has_tool_calls:
            tool_calls = []
            for tc in response.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.arguments,
                })
            message.tool_calls = tool_calls

        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def create_nanobot_model(
    provider: "LLMProvider",
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    reasoning_effort: Optional[str] = None,
) -> NanobotChatModel:
    """
    Create a NanobotChatModel instance with the given parameters.

    Helper function for easy model creation.

    Args:
        provider: The nanobot LLM provider
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        reasoning_effort: Reasoning effort for thinking models

    Returns:
        A configured NanobotChatModel instance
    """
    return NanobotChatModel(
        provider=provider,
        model=model or provider.get_default_model(),
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
