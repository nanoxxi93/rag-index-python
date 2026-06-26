import json
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core import Settings
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.workflow import Context
from typing import List, Optional

from llama_index.llms.nvidia import NVIDIA
from workflows.events import StopEvent
from utils.debug import debug_print, debug_section, debug_info
from config.settings import settings


class AgentManager:
    """
    Manages the creation and execution of agents for query processing.

    This class handles the AgentWorkflow
    for more flexible and conversational query handling.
    """

    def __init__(self, tools: List, llm=None):
        """
        Initialize the AgentManager.

        Args:
            tools: List of tools (QueryEngineTool, FunctionTool, etc.)
            llm: LLM instance (uses Settings.llm if None)
        """
        self.tools = tools
        self.llm = llm or Settings.llm
        self.agent: Optional[AgentWorkflow] = None
        self.ctx: Optional[Context] = None
        self.chat_history = []

    async def create_agent(self) -> AgentWorkflow:
        """
        Create the agent with tools or functions.

        Returns:
            AgentWorkflow: Configured agent instance
        """
        debug_section("Creating Agent")

        debug_info("LLM", self.llm.model if hasattr(
            self.llm, 'model') else str(self.llm))

        # --- START OF REVISED PATCH NVIDIA/LLAMAINDEX ---
        def _safe_get_tool_calls_from_response(instance, response, error_on_no_tool_call=True, **kwargs):
            if not response or not hasattr(response, 'message') or not response.message:
                return []

            message = response.message

            # 1. Attempt to retrieve the tool calls already processed/accumulated by the message
            tool_calls = getattr(message, 'tool_calls', [])

            # 2. If they are not there, look for them in the additional arguments (fallback)
            if not tool_calls:
                tool_calls = message.additional_kwargs.get("tool_calls", [])

            if not tool_calls:
                return []

            parsed_tool_calls = []
            for tool_call in tool_calls:
                # Validate if it is a native LlamaIndex object that has already been processed
                if isinstance(tool_call, ToolSelection):
                    parsed_tool_calls.append(tool_call)
                    continue

                if not hasattr(tool_call, 'function') or not tool_call.function:
                    continue

                name = tool_call.function.name
                raw_args = tool_call.function.arguments

                tool_kwargs = {}
                if raw_args:
                    try:
                        if isinstance(raw_args, dict):
                            tool_kwargs = raw_args
                        else:
                            # If the string contains escape characters or duplicate segments, clean it up.
                            clean_args = str(raw_args).strip()
                            tool_kwargs = json.loads(clean_args)
                    except Exception:
                        tool_kwargs = {}

                parsed_tool_calls.append(
                    ToolSelection(
                        tool_id=getattr(tool_call, 'id', f"call_{name}"),
                        tool_name=name,
                        tool_kwargs=tool_kwargs
                    )
                )

            return parsed_tool_calls

        # Replace the defective method
        NVIDIA.get_tool_calls_from_response = _safe_get_tool_calls_from_response
        # --- END OF PATCH ---

        if settings.is_multi_mode():
            tool_mapping = SimpleToolNodeMapping.from_objects(list(self.tools))

            obj_index = ObjectIndex.from_objects(
                list(self.tools),
                tool_mapping,
                index_cls=VectorStoreIndex
            )

            tool_retriever = obj_index.as_retriever(similarity_top_k=3)

            async def call_relevant_tool(query: str) -> str:
                top_3_tools = tool_retriever.retrieve(query)

                if not top_3_tools:
                    return "No relevant sources of information were found for that query."

                results = []
                for tool in top_3_tools:
                    try:
                        engine_response = await tool.query_engine.aquery(query)
                        results.append(
                            f"--- Source: {tool.metadata.name} ---\n{str(engine_response)}")
                    except Exception as e:
                        results.append(
                            f"Error querying {tool.metadata.name}: {str(e)}")

                return "\n\n".join(results)

            self.agent = AgentWorkflow.from_tools_or_functions(
                tools_or_functions=[call_relevant_tool],
                llm=self.llm,
                verbose=settings.AGENT_VERBOSE,
                system_prompt=(
                    "You are an expert agent. Choose from the available tools to answer. "
                    "Always use the exact tool names as provided. Do not invent tool names."
                )
            )
            debug_print("Agent created successfully with call_relevent_tool")
        else:
            self.agent = AgentWorkflow.from_tools_or_functions(
                self.tools,
                llm=self.llm,
                verbose=settings.AGENT_VERBOSE,
            )
            debug_print(
                f"Agent created successfully with {len(self.tools)} tools")

        self.ctx = Context(self.agent)

        debug_print(f"Max steps: {settings.AGENT_MAX_STEPS}")
        debug_print(f"Verbose: {settings.AGENT_VERBOSE}")

        return self.agent

    async def _run_agent(self, message: str) -> str:
        if self.agent is None:
            await self.create_agent()

        handler = self.agent.run(
            user_msg=message,
            ctx=self.ctx,
            max_iterations=settings.AGENT_MAX_STEPS,
        )
        result = None

        async for event in handler.stream_events():
            if isinstance(event, StopEvent):
                debug_print(f"Workflow completed with result: {event.result}")
            else:
                debug_print(f"Received event: {event}")

        result = await handler

        text = str(result) if result else "No response from agent"
        self.add_to_history("user", message)
        self.add_to_history("assistant", text)
        return text

    async def query(self, question: str, reset_history: bool = True) -> str:
        """
        Execute a query using the agent.

        Args:
            question: The question to ask
            reset_history: Whether to reset chat history before query

        Returns:
            str: Agent response
        """
        if reset_history:
            await self.reset_chat_history()
        elif self.agent is None:
            await self.create_agent()

        debug_section("Agent Query")
        debug_print(f"Question: {question}")

        response = await self._run_agent(question)

        return response

    async def chat(self, message: str) -> str:
        """
        Send a chat message to the agent (maintains conversation history).

        Args:
            message: The chat message

        Returns:
            str: Agent response
        """
        if self.agent is None:
            await self.create_agent()

        debug_section("Agent Chat")
        debug_print(f"Message: {message}")

        response = await self._run_agent(message)

        return response

    async def reset_chat_history(self):
        """Reset the chat history and workflow context."""
        self.chat_history = []
        if self.agent is None:
            await self.create_agent()
        else:
            self.ctx = Context(self.agent)
        debug_print("Chat history reset")

    def get_chat_history(self):
        """Get the current chat history."""
        return self.chat_history

    def add_to_history(self, role: str, content: str):
        """
        Manually add a message to chat history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.chat_history.append({"role": role, "content": content})
        debug_print(f"Added to history: {role} - {content[:50]}...")
