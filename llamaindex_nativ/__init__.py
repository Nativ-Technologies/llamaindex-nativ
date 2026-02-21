"""LlamaIndex tools for Nativ -- AI-powered localization.

Quickstart::

    from llamaindex_nativ import NativToolSpec

    tools = NativToolSpec().to_tool_list()       # reads NATIV_API_KEY from env
    tools = NativToolSpec(api_key="nativ_...").to_tool_list()
"""

from llamaindex_nativ.tools import NativToolSpec

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NativToolSpec",
]
