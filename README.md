# Inference Server CLI

A powerful CLI for interacting with large language models (LLMs) with advanced memory management capabilities.

## Overview

The inference server CLI provides an interactive chat interface with support for multiple LLM models, persona-based memory management, and intelligent context retrieval using both keyword and vector-based semantic search.

## Quick Start

```bash
# Start interactive chat with default settings
cargo run --bin cli

# List available models
cargo run --bin cli -- --list-models

# Chat with specific model and persona
cargo run --bin cli -- --model phi-2 --persona assistant
```

## CLI Arguments

### Model Selection

- `--model <MODEL_NAME>`

  - Select which LLM to use for generation
  - Default: First available model
  - Example: `--model phi-2`, `--model tinyllama`

- `--list-models`
  - Display all available LLM models
  - No other arguments processed when this flag is present

### System and Persona Configuration

- `--system <PROMPT>`

  - Override the default system prompt
  - Example: `--system "You are a helpful coding assistant"`

- `--persona <NAME>`
  - Select which persona memory to use
  - Personas maintain separate memory contexts
  - Default: "default"
  - Example: `--persona coder`, `--persona writer`

### Response Control

- `--stream`
  - Enable streaming responses (token-by-token output)
  - Default: False (wait for complete response)

### Memory Management

#### Memory Modes

- `--memory-mode <MODE>`
  - Control how memory is used
  - Options:
    - `off`: No memory operations
    - `read`: Read-only memory (no new memories stored)
    - `write`: Write-only memory (no memory retrieval)
    - `readwrite`: Full read/write memory access
  - Default: `readwrite`

#### Memory Policies

- `--memory-policy <POLICY>`
  - Control how new memories are stored
  - Options:
    - `append`: Always append new content to existing memories
    - `replace`: Replace existing memories with new content
    - `auto`: Intelligent policy (append for conversations, replace for personas)
  - Default: `auto`

#### Memory Types

- `--memory-types <TYPES>`
  - Specify which memory types to use for vector search
  - Comma-separated list
  - Options: `persona`, `conversation`, `fact`
  - Default: `persona,conversation`
  - Example: `--memory-types persona,fact`

#### Memory Tuning

- `--memory-k <NUMBER>`

  - Maximum number of similar memories to retrieve
  - Default: 3
  - Example: `--memory-k 5`

- `--memory-threshold <FLOAT>`
  - Similarity threshold for vector memory retrieval (0.0-1.0)
  - Lower values = more memories retrieved
  - Default: 0.78
  - Example: `--memory-threshold 0.5`

#### Memory Debugging

- `--memory-debug`
  - Enable detailed memory operation logging
  - Shows memory retrieval, storage, and embedding operations

#### Memory Management

- `--memory-wipe [PERSONA|"all"]`
  - Wipe memory for specific persona or all personas
  - With argument: Wipe specific persona
  - Without argument: Wipe all personas
  - Examples:
    - `--memory-wipe coder` (wipe "coder" persona)
    - `--memory-wipe all` (wipe all personas)
    - `--memory-wipe` (wipe all personas)

### Legacy Arguments

- `--memory-update <CONTENT>` (deprecated)
  - Legacy memory update functionality
  - Kept for backward compatibility

## Interactive Commands

Once in interactive mode, use these commands:

- `/exit` - Exit the chat session
- `/clear` - Clear conversation history and screen
- `/system <PROMPT>` - Change system prompt for current session

## Memory System

The inference server uses a sophisticated multi-layered memory system:

### Memory Types

1. **Persona Memory**: Identity, preferences, and stable facts about the persona
2. **Conversation Memory**: Dialogue context and recent interactions
3. **Fact Memory**: Explicit facts and information

### Memory Storage

- **Keyword-based**: Simple text matching and retrieval
- **Vector-based**: Semantic similarity using embeddings (requires embedding model)

### Intelligent Memory Policies

- **Auto Policy**: Automatically determines storage strategy based on content type
- **Append Policy**: Adds new information without replacing existing content
- **Replace Policy**: Updates existing memories with new information

## Examples

### Basic Chat

```bash
cargo run --bin cli
```

### Advanced Configuration

```bash
cargo run --bin cli \
  --model phi-2 \
  --persona assistant \
  --memory-mode readwrite \
  --memory-policy auto \
  --memory-types persona,conversation \
  --memory-debug
```

### Memory Management

```bash
# Wipe all memory
cargo run --bin cli -- --memory-wipe

# Wipe specific persona
cargo run --bin cli -- --memory-wipe coder

# Debug memory operations
cargo run --bin cli -- --memory-debug --memory-threshold 0.5
```

### Model Selection

```bash
# List available models
cargo run --bin cli -- --list-models

# Use specific model
cargo run --bin cli -- --model tinyllama --stream
```

## Model Setup

⚠️ **Important**: Model weights are **not included** in this repository due to their large size.

### Quick Setup (Recommended)

```bash
# Download all required models automatically
python3 download_models.py
```

### Manual Download Options

**Option 1: Using huggingface-cli**

```bash
# Install huggingface-cli
pip install huggingface_hub[cli]

# Download models
huggingface-cli download microsoft/phi-2 --local-dir models/llm/phi-2
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/llm/tinyllama
```

**Option 2: Using Python**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="microsoft/phi-2", local_dir="models/llm/phi-2")
```

**Option 3: Manual Download**
Visit [Hugging Face](https://huggingface.co) and download the following models:

- `microsoft/phi-2` → `models/llm/phi-2/`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` → `models/llm/tinyllama/`
- `sentence-transformers/all-MiniLM-L6-v2` → `models/minilm/` (for embeddings)

### Supported Models

The CLI automatically discovers models from the `models/llm/` directory. Each model directory should contain:

- `config.json` - Model configuration ✅ (committed)
- `tokenizer.json` - Tokenizer configuration ✅ (committed)
- `*.safetensors` - Model weights ❌ (download separately)

Currently supported architectures:

- Llama models (including Llama-3.2)
- Mistral models
- Phi models

## Requirements

- Rust toolchain
- Models downloaded to `models/llm/` directory
- Embedding model (optional, for vector memory features)

## Troubleshooting

### Model Not Found

- Ensure model files are in `models/llm/<model_name>/`
- Run `--list-models` to verify model discovery

### Memory Issues

- Use `--memory-debug` to see memory operations
- Check embedding model availability for vector features
- Use `--memory-wipe` to reset memory if corrupted

### Performance Issues

- Adjust `--memory-k` and `--memory-threshold` for memory retrieval tuning
- Use `--memory-mode off` to disable memory for faster responses
