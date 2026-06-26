from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from config.settings import settings
from utils.debug import debug_print, debug_info, debug_section


class DataLoader:
    """Manage data loading and preparation"""

    def __init__(self, data_dir="data", file_path=""):
        self._data_dir = data_dir
        self._file_path = file_path
        self._documents = None
        self.nodes = None

    async def _load_document(self):
        """Load a single document from the file path"""
        debug_section("Loading Document")

        self._documents = SimpleDirectoryReader(
            input_files=[self._file_path]).load_data()

        debug_info("Document loaded", self._documents)
        if settings.DEBUG:
            for i, doc in enumerate(self._documents):
                print(f"Document {i}: {len(doc.text)} characters")
                print(f"Preview: {doc.text[:200]}...")
                print("-" * 50)

        return self._documents

    async def _load_documents(self):
        """Load documents from the data directory"""
        debug_section("Loading Documents")

        self._documents = SimpleDirectoryReader(
            input_dir=self._data_dir).load_data()

        debug_info("Documents loaded", self._documents)
        if settings.DEBUG:
            for i, doc in enumerate(self._documents):
                print(f"Document {i}: {len(doc.text)} characters")
                print(f"Preview: {doc.text[:200]}...")
                print("-" * 50)

        return self._documents

    async def _split_documents(self):
        """Divide documents into nodes"""
        debug_section("Splitting Documents")

        if not self._documents:
            await self._load_documents()

        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.nodes = splitter.get_nodes_from_documents(self._documents)

        debug_info("Nodes created", self.nodes)
        if settings.DEBUG:
            for i, node in enumerate(self.nodes[:3]):
                print(f"Node {i}: {len(node.text)} characters")
                print(f"Preview: {node.text[:200]}...")
                print("-" * 50)

        return self.nodes

    async def get_nodes(self):
        """Get nodes (loads and splits if necessary)"""
        if self.nodes is None:
            if (self._file_path):
                await self._load_document()
            else:
                await self._load_documents()
            await self._split_documents()
        return self.nodes
