"""
Debug utilities for the RAG pipeline.
"""

from config.settings import settings


def debug_print(*args, **kwargs):
    """Print only if DEBUG is enabled"""
    if settings.DEBUG:
        print(*args, **kwargs)


def debug_section(title):
    """Print a formatted debug section"""
    if settings.DEBUG:
        print("\n" + "="*50)
        print(f"DEBUG: {title}")
        print("="*50)


def debug_info(label, data, preview_length=200):
    """Show formatted debug information"""
    if settings.DEBUG:
        if isinstance(data, list):
            print(f"{label}: {len(data)} items")
            if data and len(data) > 0:
                preview = str(data[0])[:preview_length]
                print(f"First item preview: {preview}...")
        elif hasattr(data, '__len__') and not isinstance(data, str):
            print(f"{label}: {len(data)}")
            if len(str(data)) > preview_length:
                print(f"Preview: {str(data)[:preview_length]}...")
        else:
            print(f"{label}: {data}")


def debug_separator():
    """Print a debug separator line."""
    if settings.DEBUG:
        print("-" * 40)
