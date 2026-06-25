from config.settings import settings
from core.data_loader import DataLoader
from core.index_manager import IndexManager
from core.query_engine import QueryEngineManager
from core.agent_manager import AgentManager
from utils.debug import debug_print, debug_section, debug_info


class RAGOrchestrator:
    """
    Orchestrates the entire RAG pipeline.

    This class manages the creation of indices, tools, and the execution
    mode (router or agent) based on configuration.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the orchestrator.

        Args:
            data_dir: Directory containing the documents
        """
        self.data_dir = data_dir
        self.nodes = None
        self.summary_index = None
        self.vector_index = None
        self.tools = None
        self.query_manager = None
        self.agent_manager = None

    async def initialize(self):
        """
        Initialize all components in the pipeline.

        Returns:
            self: For method chaining
        """
        debug_section("Initializing RAG Pipeline")

        # 1. Load data
        debug_print("Step 1: Loading data...")
        data_loader = DataLoader(self.data_dir)
        self.nodes = await data_loader.get_nodes()

        # 2. Create indices
        debug_print("Step 2: Creating indices...")
        index_manager = IndexManager(self.nodes)
        self.summary_index, self.vector_index = await index_manager.get_indices()

        # 3. Create tools
        debug_print("Step 3: Creating query tools...")
        self.query_manager = QueryEngineManager(
            self.summary_index, self.vector_index)
        self.tools = await self.query_manager.create_tools()

        debug_print("Initialization complete!")

        return self

    async def setup_router(self):
        """
        Set up the router-based execution mode.

        Returns:
            QueryEngineManager: Configured query manager
        """
        debug_section("Setting up Router Mode")
        await self.query_manager.create_router(self.tools)
        return self.query_manager

    async def setup_agent(self):
        """
        Set up the agent-based execution mode.

        Returns:
            AgentManager: Configured agent manager
        """
        debug_section("Setting up Agent Mode")

        # Get tools as list
        summary_tool, vector_tool = self.tools
        tools = [summary_tool, vector_tool]

        # Create agent manager
        self.agent_manager = AgentManager(tools)
        await self.agent_manager.create_agent()

        return self.agent_manager

    async def query(self, question: str, reset_history: bool = True) -> str:
        """
        Execute a query based on the configured execution mode.

        Args:
            question: The question to ask
            reset_history: Whether to reset agent chat history (agent mode only)

        Returns:
            str: Response
        """
        debug_section(f"Executing Query in {settings.EXECUTION_MODE} mode")

        if settings.is_agent_mode():
            if self.agent_manager is None:
                await self.setup_agent()
            return await self.agent_manager.query(question, reset_history=reset_history)
        else:
            if self.query_manager.router_query_engine is None:
                await self.setup_router()
            return await self.query_manager.query(question)

    async def chat(self, message: str) -> str:
        """
        Send a chat message (agent mode only).

        Args:
            message: The chat message

        Returns:
            str: Response
        """
        if not settings.is_agent_mode():
            raise ValueError(
                "Chat mode is only available in agent execution mode")

        if self.agent_manager is None:
            await self.setup_agent()

        return await self.agent_manager.chat(message)

    async def reset_chat(self):
        """Reset chat history (agent mode only)."""
        if self.agent_manager:
            await self.agent_manager.reset_chat_history()
            debug_print("Chat history reset")
