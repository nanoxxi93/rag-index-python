from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from config.settings import settings
from utils.debug import debug_print, debug_info, debug_section


class QueryEngineManager:
    """
    Manages the creation and execution of query engines.

    This class handles the RouterQueryEngine for structured routing
    between different tools (summary, vector, etc.).
    """

    def __init__(self, summary_index, vector_index):
        """
        Initialize the QueryEngineManager.

        Args:
            summary_index: SummaryIndex instance
            vector_index: VectorStoreIndex instance
        """
        self.summary_index = summary_index
        self.vector_index = vector_index
        self.summary_query_engine = None
        self.vector_query_engine = None
        self.router_query_engine = None

    async def create_query_engines(self):
        """
        Create individual query engines.

        Returns:
            tuple: (summary_query_engine, vector_query_engine)
        """
        debug_section("Creating Query Engines")

        # Summary query engine
        self.summary_query_engine = self.summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True
        )

        # Vector query engine (basic)
        self.vector_query_engine = self.vector_index.as_query_engine(
            similarity_top_k=settings.SIMILARITY_TOP_K
        )

        debug_print("Query engines created successfully")
        debug_info("Summary engine", "tree_summarize mode")
        debug_info("Vector engine", f"top_k={settings.SIMILARITY_TOP_K}")

        return self.summary_query_engine, self.vector_query_engine

    async def create_tools(self, vector_tool=None) -> tuple:
        """
        Create tools from the query engines.

        Returns:
            tuple: (summary_tool, vector_tool)
        """
        debug_section("Creating Tools")

        if self.summary_query_engine is None or self.vector_query_engine is None:
            await self.create_query_engines()

        # Summary tool
        summary_tool = QueryEngineTool.from_defaults(
            name="summary_tool",
            query_engine=self.summary_query_engine,
            description=(
                "Useful for summarization questions about the entire document. "
                "Use for 'summarize' or 'overview' questions."
            )
        )

        # Vector tool
        vector_tool = QueryEngineTool.from_defaults(
            name="vector_tool",
            query_engine=self.vector_query_engine,
            description=(
                "Useful for retrieving specific details, experiences, or projects. "
                "Use for questions about 'experience', 'skills', or 'projects'."
            )
        )

        debug_print("Tools created successfully")

        return summary_tool, vector_tool

    async def create_router(self, tools: tuple = None) -> RouterQueryEngine:
        """
        Create the RouterQueryEngine with the provided tools.

        Args:
            tools: Tuple of (summary_tool, vector_tool). If None, creates them.

        Returns:
            RouterQueryEngine: Configured router
        """
        debug_section("Creating Router Query Engine")

        if tools is None:
            tools = await self.create_tools()
            
        # Unpack tools if it's a tuple
        if isinstance(tools, tuple) and len(tools) >= 2:
            summary_tool, vector_tool = tools[0], tools[1]
        else:
            # If tools is already a list or individual tools
            summary_tool, vector_tool = await self.create_tools()

        self.router_query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool],
            verbose=settings.DEBUG
        )

        debug_print("Router query engine created successfully")
        debug_info("Router selector", "LLMSingleSelector")

        return self.router_query_engine

    async def query(self, question):
        """
        Execute a query using the router.

        Args:
            question: The question to ask

        Returns:
            str: Response from the router
        """
        if self.router_query_engine is None:
            raise ValueError(
                "Router not initialized. Call create_router first.")

        debug_section("Router Query")
        debug_print(f"Question: {question}")

        response = await self.router_query_engine.aquery(question)

        return str(response)
