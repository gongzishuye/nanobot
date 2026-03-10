"""LangChain tool adapters for nanobot."""

from __future__ import annotations

import asyncio
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

if TYPE_CHECKING:
    from nanobot.agent.tools.base import Tool


T = TypeVar("T", bound=BaseModel)


class NanobotToolAdapter(BaseTool):
    """
    Adapter to convert nanobot tools to LangChain format.

    This allows nanobot's custom tools to work seamlessly with LangChain's
    Agent ecosystem while maintaining nanobot's tool execution semantics.

    Example:
        ```python
        from nanobot.agent.tools.shell import ExecTool
        from nanobot.agent.langchain.tools import NanobotToolAdapter

        nanobot_tool = ExecTool(working_dir="/workspace", timeout=30)
        langchain_tool = NanobotToolAdapter(
            name="exec",
            tool=nanobot_tool
        )

        # Use with LangChain Agent
        from langchain.agents import AgentExecutor
        executor = AgentExecutor(agent=agent, tools=[langchain_tool])
        ```
    """

    name: str = Field(description="The name of the tool")
    description: str = Field(default="", description="Description of what the tool does")
    tool: "Tool" = Field(description="The underlying nanobot tool")

    args_schema: Optional[Type[BaseModel]] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, name: str, tool: "Tool", description: str = ""):
        """Initialize the adapter."""
        # Get description from tool if not provided
        if not description:
            description = getattr(tool, "description", "")
            if not description:
                # Try to get from docstring
                description = tool.__class__.__doc__ or ""

        # Create args schema from tool's parameters
        args_schema = self._create_args_schema(tool)

        super().__init__(
            name=name,
            description=description,
            tool=tool,
            args_schema=args_schema,
        )

    @staticmethod
    def _create_args_schema(tool: "Tool") -> Optional[Type[BaseModel]]:
        """Create a Pydantic schema from a nanobot tool's parameters."""
        # Try to get schema from tool
        if hasattr(tool, "get_schema"):
            schema = tool.get_schema()
            if schema:
                return schema

        # Try to infer from __call__ signature
        if hasattr(tool, "__call__"):
            sig = inspect.signature(tool.__call__)
            fields = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Determine the type
                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    param_type = str

                # Determine if required
                default = ...
                if param.default != inspect.Parameter.empty:
                    default = param.default

                # Create field
                field_info = Field(
                    default=default,
                    description=f"Parameter {param_name}"
                )
                fields[param_name] = (param_type, field_info)

            if fields:
                return create_model(
                    f"{tool.__class__.__name__}Schema",
                    **fields
                )

        return None

    def _run(self, **kwargs: Any) -> Any:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> Any:
        """Run the tool asynchronously."""
        # Call the nanobot tool
        if asyncio.iscoroutinefunction(self.tool.__call__):
            return await self.tool(**kwargs)
        else:
            return self.tool(**kwargs)


def convert_nanobot_tools(
    tools: Dict[str, "Tool"],
) -> List[BaseTool]:
    """
    Convert a dictionary of nanobot tools to LangChain format.

    Args:
        tools: Dictionary mapping tool names to nanobot Tool instances

    Returns:
        List of LangChain BaseTool instances

    Example:
        ```python
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.langchain.tools import convert_nanobot_tools

        registry = ToolRegistry()
        # ... register tools ...

        langchain_tools = convert_nanobot_tools(registry._tools)
        ```
    """
    result = []
    for name, tool in tools.items():
        # Get description
        description = getattr(tool, "description", "")
        if not description and tool.__class__.__doc__:
            description = tool.__class__.__doc__.strip()

        # Create adapter
        adapter = NanobotToolAdapter(
            name=name,
            tool=tool,
            description=description,
        )
        result.append(adapter)

    return result


class MCPToolAdapter(BaseTool):
    """
    Adapter for MCP (Model Context Protocol) tools to work with LangChain.

    This allows MCP tools to be used seamlessly in LangChain agents.
    """

    name: str = Field(description="The name of the tool")
    description: str = Field(description="Description of what the tool does")
    mcp_tool: Any = Field(description="The underlying MCP tool")
    execute_fn: Callable = Field(description="The function to execute the tool")

    args_schema: Optional[Type[BaseModel]] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> Any:
        """Run the tool asynchronously."""
        if asyncio.iscoroutinefunction(self.execute_fn):
            return await self.execute_fn(self.name, kwargs)
        else:
            return self.execute_fn(self.name, kwargs)


def convert_mcp_tools(mcp_tools: List[Dict[str, Any]]) -> List[BaseTool]:
    """
    Convert MCP tools to LangChain format.

    Args:
        mcp_tools: List of MCP tool definitions

    Returns:
        List of LangChain BaseTool instances
    """
    result = []
    for tool_def in mcp_tools:
        name = tool_def.get("name", "")
        description = tool_def.get("description", "")

        # Create schema from inputSchema
        input_schema = tool_def.get("inputSchema", {})
        args_schema = None
        if input_schema:
            # Build Pydantic model from JSON schema
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            fields = {}
            for prop_name, prop_def in properties.items():
                prop_type = str  # Default to string
                if "type" in prop_def:
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": float,
                        "boolean": bool,
                        "array": List,
                        "object": Dict,
                    }
                    prop_type = type_map.get(prop_def["type"], str)

                is_required = prop_name in required
                default = ... if is_required else None

                field_info = Field(
                    default=default,
                    description=prop_def.get("description", "")
                )
                fields[prop_name] = (prop_type, field_info)

            if fields:
                args_schema = create_model(
                    f"{name}Schema",
                    **fields
                )

        # Note: The actual execute function should be provided by the MCP server
        # This is a placeholder - the caller needs to inject the actual execution logic
        result.append(
            MCPToolAdapter(
                name=name,
                description=description,
                mcp_tool=tool_def,
                execute_fn=lambda n, a: {"error": "MCP execution not configured"},
                args_schema=args_schema,
            )
        )

    return result
