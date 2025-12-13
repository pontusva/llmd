# LLM Inference Server

A client-server system for interacting with large language models. Features advanced memory management, persona-based conversations, and vector similarity search. Models are loaded via the companion `model-loader` tool.

## Architecture

The system consists of two components:

1. **`llmd` (Server)**: Long-running inference server that loads models and handles all memory/embedding logic
2. **`cli` (Client)**: Thin terminal client that connects to the server via HTTP

## Overview

The inference server provides an OpenAI-compatible API with support for multiple LLM models, persona-based memory management, and intelligent context retrieval using both keyword and vector-based semantic search.

## Quick Start

### 1. Start the Server

```bash
# Load a model using the model-loader tool (from separate repository)
# This creates a LoadPlan that gets piped to the server
model-loader --model phi-2 | cargo run --bin llmd

# Server will start on http://localhost:3000
```

### 2. Use the Client

```bash
# Start interactive chat
cargo run --bin cli

# List available models (from running server)
cargo run --bin cli -- --list-models

# Chat with specific persona
cargo run --bin cli -- --persona assistant
```

## CLI Arguments

### Model Selection

- `--list-models`

  - Query the server and display all available LLM models
  - No other arguments processed when this flag is present

- `--model <MODEL_NAME>` (optional)
  - Specify which model to use (if multiple are loaded)
  - Default: Auto-discover first available model from server

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

### Memory Control

The CLI includes memory control options that are sent to the server:

- `--persona <NAME>`
  - Select which persona memory context to use
  - Personas maintain separate memory contexts on the server
  - Default: "minimal" (fast, minimal prompt)
  - Available: "default", "minimal", "rogue", "developer", "therapist", "pirate", "yoda"

## Interactive Commands

Once in chat mode, use these commands:

- `/exit` - Exit the chat session

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

### Basic Usage

```bash
# 1. Start server with loaded model
model-loader --model phi-2 | cargo run --bin llmd

# 2. In another terminal, start chat
cargo run --bin cli
```

### Server Options

```bash
# Load different model
model-loader --model tinyllama | cargo run --bin llmd

# Server runs on http://localhost:3000 by default
```

### Client Options

```bash
# Use specific persona
cargo run --bin cli -- --persona developer

# List available models from server
cargo run --bin cli -- --list-models
```

## Model Loading

⚠️ **Important**: Models are loaded by the separate `model-loader` repository, not this one.

### Using the Model Loader

Models are loaded using the companion `model-loader` tool:

```bash
# Install model-loader (from separate repository)
git clone https://github.com/your-org/model-loader.git
cd model-loader
cargo build --release

# Load a model and start the server
./target/release/model-loader --model phi-2 | cargo run --bin llmd
```

The model-loader handles:

- Downloading model weights from Hugging Face
- Creating optimized LoadPlans
- Streaming models to the inference server

### Supported Models

The server supports models loaded via LoadPlan:

- **Llama models**: Llama 2, Llama 3, Llama 3.2
- **Mistral models**: Mistral 7B, Mixtral
- **Phi models**: Phi-2, Phi-3
- **Other architectures**: As supported by `candle-transformers`

Models are automatically detected and made available through the `/v1/models` API endpoint.

## Requirements

- Rust toolchain
- Running `model-loader` instance (separate repository)
- Server started with model LoadPlan piped from stdin

## API Endpoints

The server provides OpenAI-compatible endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions with memory support
- `GET /v1/persona/:persona/memory` - Get persona memory
- `POST /v1/persona/:persona/memory/reset` - Reset persona memory

## Troubleshooting

### Server Won't Start

- Ensure model-loader is properly configured
- Check that LoadPlan is being piped to server stdin
- Verify model files exist in model-loader's cache

### Model Not Found

- Use `cargo run --bin cli -- --list-models` to check available models
- Ensure model was properly loaded by model-loader

### Connection Issues

- Verify server is running on http://localhost:3000
- Check that model-loader successfully created LoadPlan

### Performance Issues

- Use `--persona minimal` for fastest responses
- Memory operations happen server-side automatically
