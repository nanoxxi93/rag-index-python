from typing import List
from llama_index.core.vector_stores import FilterCondition, MetadataFilters
from llama_index.core.tools import FunctionTool
from config.settings import settings


class VectorTools:
    """Tools for vector search with filters"""

    def __init__(self, vector_index):
        self.vector_index = vector_index

    def create_vector_query_tool(self):
        """Create a vector query tool with page filters"""

        def vector_query(
            query: str,
            page_numbers: List[str]
        ) -> str:
            """Perform a vector search over an index.

            query (str): the string query to be embedded.
            page_numbers (List[str]): Filter by set of pages. 
                Leave BLANK for search over all pages.
            """
            metadata_dicts = [
                {"key": "page_label", "value": p} for p in page_numbers
            ]

            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=settings.SIMILARITY_TOP_K,
                filters=MetadataFilters.from_dicts(
                    metadata_dicts,
                    condition=FilterCondition.OR
                )
            )

            response = query_engine.query(query)
            return response

        return FunctionTool.from_defaults(
            name="vector_tool",
            fn=vector_query
        )
