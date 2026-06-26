from llama_index.core.indices import VectorStoreIndex, SummaryIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine, ToolRetrieverRouterQueryEngine
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
        self._summary_index = summary_index
        self._vector_index = vector_index

        self.summary_query_engine = None
        self.vector_query_engine = None
        self.router_query_engine = None

    def _create_single_engines(self, s_index: SummaryIndex, v_index: VectorStoreIndex):
        """
        Create individual query engines.

        Returns:
            tuple: (summary_query_engine, vector_query_engine)
        """
        debug_section("Creating Query Engines")

        # Summary query engine
        summary_engine = s_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True
        )

        # Vector query engine (basic)
        vector_engine = v_index.as_query_engine(
            similarity_top_k=settings.SIMILARITY_TOP_K,
            response_mode="compact",
            use_async=True
        )

        debug_print("Query engines created successfully")
        debug_info("Summary engine", "tree_summarize mode")
        debug_info("Vector engine", f"top_k={settings.SIMILARITY_TOP_K}")

        return summary_engine, vector_engine

    async def create_tools(self, vector_tool=None, filename=None) -> tuple:
        """
        Create tools from the query engines.

        Returns:
            tuple: (summary_tool, vector_tool)
        """
        debug_section("Creating Tools")
        all_tools = []

        if isinstance(self._vector_index, dict):
            debug_print("Processing tools in MULTI MODE...")

            for filename in self._vector_index.keys():
                s_index = self._summary_index[filename]
                v_index = self._vector_index[filename]

                summary_engine, vector_engine = self._create_single_engines(
                    s_index, v_index)

                summary_tool = QueryEngineTool.from_defaults(
                    name=f"summary_tool_{filename}",
                    query_engine=summary_engine,
                    description=f"Useful for summarization questions about the entire document {filename}."
                )
                vector_tool = QueryEngineTool.from_defaults(
                    name=f"vector_tool_{filename}",
                    query_engine=vector_engine,
                    description=f"Useful for retrieving specific details of the document {filename}."
                )

                all_tools.extend([summary_tool, vector_tool])
                debug_print(
                    f"Tools successfully created for the archive: {filename}")
        else:
            debug_print("Processing tools in SINGLE MODE...")
            self.summary_query_engine, self.vector_query_engine = self._create_single_engines(
                self._summary_index, self._vector_index
            )

            summary_tool = QueryEngineTool.from_defaults(
                name="summary_tool_global",
                query_engine=self.summary_query_engine,
                description="Useful for summarization questions about all documents combined."
            )

            vector_tool = QueryEngineTool.from_defaults(
                name="vector_tool_global",
                query_engine=self.vector_query_engine,
                description="Useful for retrieving specific details across all stored documents."
            )
            all_tools.extend([summary_tool, vector_tool])

        debug_print("Tools created successfully")
        return all_tools

    async def create_router(self, tools: tuple = None):
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

        if settings.is_multi_mode():
            tool_mapping = SimpleToolNodeMapping.from_objects(list(tools))

            obj_index = ObjectIndex.from_objects(
                list(tools),
                tool_mapping,
                index_cls=VectorStoreIndex
            )

            tool_retriever = obj_index.as_retriever(similarity_top_k=3)

            self.router_query_engine = ToolRetrieverRouterQueryEngine(
                retriever=tool_retriever,
            )
            debug_info("Router selector", "ToolRetriever")
        else:
            self.router_query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(),
                query_engine_tools=list(tools),
                verbose=settings.DEBUG
            )
            debug_info("Router selector", "LLMSingleSelector")

        print(
            f"Router query engine created successfully with {len(tools)} tools")
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
