from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from config.settings import settings
from utils.debug import debug_print, debug_info, debug_section


class DataLoader:
    """Manage data loading and preparation"""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.documents = None
        self.nodes = None

    async def load_documents(self):
        """Load documents from the data directory"""
        debug_section("Loading Documents")

        self.documents = SimpleDirectoryReader(self.data_dir).load_data()

        debug_info("Documents loaded", self.documents)
        if settings.DEBUG:
            for i, doc in enumerate(self.documents):
                print(f"Document {i}: {len(doc.text)} characters")
                print(f"Preview: {doc.text[:200]}...")
                print("-" * 50)

        return self.documents

    async def split_documents(self):
        """Divide documents into nodes"""
        debug_section("Splitting Documents")

        if not self.documents:
            await self.load_documents()

        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.nodes = splitter.get_nodes_from_documents(self.documents)

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
            await self.load_documents()
            await self.split_documents()
        return self.nodes
