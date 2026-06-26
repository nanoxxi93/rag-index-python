import os
from llama_index.core import SummaryIndex, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from config.settings import settings
from utils.debug import debug_print, debug_section


class IndexManager:
    """Manage index creation and configuration"""

    def __init__(self, nodes):
        self._nodes = nodes
        self.summary_index = None
        self.vector_index = None
        self._base_storage_dir = "./storage"
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

    async def _create_indices(self):
        """Create summary and vector indices"""
        debug_section("Creating Indices")

        if settings.is_single_mode():
            persist_dir = os.path.join(self._base_storage_dir, "single_mode")

            if os.path.exists(persist_dir):
                debug_print("Loading SINGLE indexes from local storage...")
                storage_context = StorageContext.from_defaults(
                    persist_dir=persist_dir)

                self.vector_index = load_index_from_storage(
                    storage_context,
                    index_id="vector",
                    llm=Settings.llm,
                    embed_model=Settings.embed_model
                )
                self.summary_index = load_index_from_storage(
                    storage_context,
                    index_id="summary",
                    llm=Settings.llm,
                    embed_model=Settings.embed_model
                )
            else:
                debug_print(
                    "No local storage found. Creating new SINGLE index (Calling API)...")
                storage_context = StorageContext.from_defaults()

                self.summary_index = SummaryIndex(
                    self._nodes, storage_context=storage_context)
                self.vector_index = VectorStoreIndex(
                    self._nodes, storage_context=storage_context)
                debug_print(
                    f"Summary Index created with {len(self._nodes)} nodes")
                debug_print(
                    f"Vector Index created with {len(self._nodes)} nodes")

                self.summary_index.set_index_id("summary")
                self.vector_index.set_index_id("vector")

                storage_context.persist(persist_dir=persist_dir)
                debug_print(
                    f"SINGLE indexes successfully saved in: {persist_dir}")
        elif settings.is_multi_mode():
            debug_print(
                "Detected MULTI MODE. Grouping nodes by source file...")

            vector_indices_dict = {}
            summary_indices_dict = {}

            nodes_by_file = {}
            for node in self._nodes:
                file_name = node.metadata.get(
                    "file_name", "unknown_document").replace(".", "_")
                if file_name not in nodes_by_file:
                    nodes_by_file[file_name] = []
                nodes_by_file[file_name].append(node)

            for file_name, file_nodes in nodes_by_file.items():
                file_persist_dir = os.path.join(
                    self._base_storage_dir, f"multi_mode_{file_name}")

                if os.path.exists(file_persist_dir):
                    debug_print(
                        f"Loading indexes for [{file_name}] from disk...")
                    storage_context = StorageContext.from_defaults(
                        persist_dir=file_persist_dir)

                    v_idx = load_index_from_storage(
                        storage_context,
                        index_id=f"vector_{file_name}",
                        llm=Settings.llm,
                        embed_model=Settings.embed_model
                    )
                    s_idx = load_index_from_storage(
                        storage_context,
                        index_id=f"summary_{file_name}",
                        llm=Settings.llm,
                        embed_model=Settings.embed_model
                    )
                else:
                    debug_print(
                        f"Indexing new file in multi mode: [{file_name}] (Calling API)...")
                    file_storage_context = StorageContext.from_defaults()

                    s_idx = SummaryIndex(
                        file_nodes, storage_context=file_storage_context)
                    v_idx = VectorStoreIndex(
                        file_nodes, storage_context=file_storage_context)

                    s_idx.set_index_id(f"summary_{file_name}")
                    v_idx.set_index_id(f"vector_{file_name}")

                    file_storage_context.persist(persist_dir=file_persist_dir)

                vector_indices_dict[file_name] = v_idx
                summary_indices_dict[file_name] = s_idx

            self.vector_index = vector_indices_dict
            self.summary_index = summary_indices_dict

        debug_print("Indexing process completed successfully.")

        return self.summary_index, self.vector_index

    async def get_indices(self):
        """Get indices (creates them if they don't exist)"""
        self._setup_models()

        if self.summary_index is None or self.vector_index is None:
            await self._create_indices()
        return self.summary_index, self.vector_index
