# llamaindex-nativ

[![PyPI](https://img.shields.io/pypi/v/llamaindex-nativ)](https://pypi.org/project/llamaindex-nativ/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[LlamaIndex](https://www.llamaindex.ai/) integration for **[Nativ](https://usenativ.com)** — AI-powered localization.

Give any LlamaIndex agent the ability to translate text, search translation memory, manage terminology, and more — all backed by your team's brand voice and style guides.

## Installation

```bash
pip install llamaindex-nativ
```

## Quick start

```python
from llamaindex_nativ import NativToolSpec

spec = NativToolSpec()  # reads NATIV_API_KEY from env
tools = spec.to_tool_list()

# Use a single tool directly
result = spec.translate("Hello world", target_language="French")
print(result)
# Bonjour le monde
# Rationale: Standard greeting translated with neutral register.
```

## Use with a LlamaIndex agent

```python
from llamaindex_nativ import NativToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

tools = NativToolSpec().to_tool_list()
llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

response = agent.chat(
    "Translate 'Welcome back!' to French, German, and Japanese"
)
```

## Available tools

| Tool | Description |
|------|-------------|
| `translate` | Translate text with cultural adaptation |
| `translate_batch` | Translate multiple texts to one language |
| `search_translation_memory` | Fuzzy-search existing translations |
| `add_translation_memory_entry` | Store an approved translation for reuse |
| `get_languages` | List configured target languages |
| `get_style_guides` | Get style guide content |
| `get_brand_voice` | Get the brand voice prompt |
| `get_translation_memory_stats` | Get TM statistics |

## Configuration

Pass `api_key` and `base_url` directly:

```python
spec = NativToolSpec(api_key="nativ_...", base_url="https://api.usenativ.com")
```

Or set the environment variable:

```bash
export NATIV_API_KEY=nativ_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Use individual tools

```python
from llamaindex_nativ import NativToolSpec

spec = NativToolSpec()

# Search translation memory
print(spec.search_translation_memory("Welcome"))
# Found 3 match(es):
# - [98% exact] "Welcome" -> "Bienvenue"
# - [85% fuzzy] "Welcome back" -> "Content de vous revoir"
# ...

# Get configured languages
print(spec.get_languages())
# Configured languages:
# - French (fr) -- formality: formal
# - German (de) -- formality: formal
# ...
```

## License

MIT
