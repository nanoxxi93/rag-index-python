import os
from dotenv import load_dotenv

load_dotenv()


class SettingsConfig:
    """Central application settings"""

    # API Keys
    NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', "")

    # Debug
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Mode: 'single' or 'multi'
    MODE = os.environ.get('MODE', 'single')

    # Execution Mode: 'router' or 'agent'
    EXECUTION_MODE = os.environ.get(
        'EXECUTION_MODE', 'router')  # router | agent

    # Models
    LLM_MODEL = os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct")
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embed-v1")

    # Chunking configuration
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1024'))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '20'))

    # Query configuration
    SIMILARITY_TOP_K = int(os.environ.get('SIMILARITY_TOP_K', '5'))

    # Agent configuration
    AGENT_VERBOSE = os.environ.get('AGENT_VERBOSE', 'True').lower() == 'true'
    AGENT_MAX_STEPS = int(os.environ.get('AGENT_MAX_STEPS', '10'))

    # Validation
    @classmethod
    def validate(cls):
        if not cls.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY not found in environment variables")

        if cls.EXECUTION_MODE not in ['router', 'agent']:
            raise ValueError(
                f"EXECUTION_MODE must be 'router' or 'agent', got '{cls.EXECUTION_MODE}'")

        return True

    @classmethod
    def is_single_mode(cls):
        return cls.MODE == 'single'
    
    @classmethod
    def is_multi_mode(cls):
        return cls.MODE == 'multi'
    
    @classmethod
    def is_agent_mode(cls):
        return cls.EXECUTION_MODE == 'agent'

    @classmethod
    def is_router_mode(cls):
        return cls.EXECUTION_MODE == 'router'


# Global instance
settings = SettingsConfig()
