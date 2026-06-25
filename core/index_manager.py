from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from config.settings import settings
from utils.debug import debug_print, debug_section


class IndexManager:
    """Manage index creation and configuration"""

    def __init__(self, nodes):
        self.nodes = nodes
        self.summary_index = None
        self.vector_index = None
        self._setup_models()

    def _setup_models(self):
        """Configure LLM and embedding models"""
        debug_section("Setting up Models")

        Settings.llm = NVIDIA(
            model=settings.LLM_MODEL,
            api_key=settings.NVIDIA_API_KEY
        )

        Settings.embed_model = NVIDIAEmbedding(
            model=settings.EMBED_MODEL,
            api_key=settings.NVIDIA_API_KEY
        )

        debug_print("Models configured successfully")

    async def create_indices(self):
        """Create summary and vector indices"""
        debug_section("Creating Indices")

        self.summary_index = SummaryIndex(self.nodes)
        self.vector_index = VectorStoreIndex(self.nodes)

        debug_print(f"Summary Index created with {len(self.nodes)} nodes")
        debug_print(f"Vector Index created with {len(self.nodes)} nodes")

        return self.summary_index, self.vector_index

    async def get_indices(self):
        """Get indices (creates them if they don't exist)"""
        if self.summary_index is None or self.vector_index is None:
            await self.create_indices()
        return self.summary_index, self.vector_index
