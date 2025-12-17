## llmd — Intent Compilation & Execution Runtime (LLM-Driven)

llmd is a secure, deterministic runtime for using Large Language Models (LLMs) as intent compilers, not free-form chatbots.

It translates natural language into strict, validated intents, executes them through guarded tools, and optionally returns conversational responses (Natural-language responses are generated after execution and have no authority over execution) — all while enforcing grammar, scope, and security constraints.

This architecture is designed for production systems, not demos.

⸻

## What llmd Is (and Is Not)

llmd is
• An LLM-driven intent compiler
• A grammar-enforced execution runtime
• A secure gateway to structured data
• A tool orchestration system with guardrails
• A persona-aware conversational system

llmd is not
• A “best-effort” chatbot
• A prompt-only system
• A model that queries databases directly
• A system that guesses or hallucinates missing data

If llmd cannot produce a valid, safe intent, it will reject the request.

⸻

## High-Level Architecture

llmd separates understanding, decision, and execution into explicit stages:

User Input
↓
Intent Compilation (LLM, grammar-constrained)
↓
Intent Normalization & Validation (executor)
↓
Capability & Scope Checks
↓
Tool Execution (deterministic)
↓
Optional Natural-Language Response

Each stage has hard boundaries. The LLM does not bypass them.

⸻

## Core Components

### 1. Intent Compiler (LLM)

The LLM’s primary role is to compile natural language into structured intent.

Example:

"How many measures are in building Räven?"

⬇️

```json
{
  "action": "count",
  "target": "measure",
  "scope": {
    "type": "building",
    "building_name": "Räven"
  }
}
```

The LLM:
• Must follow a strict grammar
• Must emit valid JSON
• Must not invent filters, scopes, or entities

If it violates the grammar, the request is rejected.

⸻

### 2. Intent Grammar (First-Class Concept)

All executable requests conform to a formal grammar:

| Field     | Description                                            |
| --------- | ------------------------------------------------------ |
| action    | What to do (count, list, get, aggregate, …)            |
| target    | Structural entity (building, measure, component, …)    |
| scope     | Context boundary (current_team, building, real_estate) |
| attribute | Dependent physical object (windows, doors, …)          |
| filters   | Optional constraints (status, completed, year_min, …)  |

Structural vs Physical Objects
• Structural entities → valid target
• building, measure, component, plan, project, realEstate
• Physical/dependent objects → attribute only
• windows, doors, rooms, pipes, sensors

Example:

"How many windows are in building Björk?"

```json
{
  "action": "count",
  "target": "building",
  "attribute": "windows",
  "scope": {
    "type": "building",
    "building_name": "Björk"
  }
}
```

⸻

### 3. Executor (Authority Boundary)

The executor is the source of truth.

It is responsible for:
• Validating intent shape (JSON Schema)
• Normalizing scopes and filters
• Resolving human-readable names
• Enforcing capability rules
• Executing tools deterministically

The executor can:
• Reject invalid intents
• Reject unresolved scopes
• Reject unsupported filters
• Reject unsafe execution

Only the executor can cause data access or side effects.
The LLM cannot override this.

⸻

### 4. Tool System (Deterministic Execution)

Tools are:
• Explicitly registered
• Schema-validated
• Capability-checked
• Executed with strict inputs

Example tool:
• query_intent — compile intent → GraphQL query

Tools are execution authority, not the LLM.

⸻

### 5. Name Resolution & Scoping

Human-friendly names (e.g. “Räven”) are resolved via a NameResolutionRegistry, scoped per team.
• Resolution happens before execution
• Unknown names cause rejection
• No guessing or fallback logic

This prevents:
• Cross-tenant data leaks
• Ambiguous execution
• Silent mis-scoping

⸻

### 6. Relation Mapping

Targets and scopes are connected through an explicit relation map.

Example:
• measure scoped by building →

measure → component → building

This guarantees:
• Correct query generation
• No accidental joins
• No implicit assumptions

⸻

### 7. Memory System (Supporting Role)

llmd includes a multi-layer memory system:
• Persona memory
• Conversation memory
• Fact memory
• Vector and keyword retrieval

Memory:
• Improves conversational continuity
• Provides context for intent compilation

Memory never:
• Changes execution results
• Overrides intent grammar
• Bypasses validation

⸻

## Modes of Operation

llmd operates in multiple modes, depending on user intent:

### Conversational Mode

• Free-form chat
• No tools
• Natural language only

### Compiler Mode

• User requests data or actions
• LLM emits structured intent
• Tools may be executed

### Execution Mode

• Strict schema validation
• Deterministic results
• Optional natural-language summary

These modes are enforced automatically.

⸻

## Security & Safety Model

llmd is designed for production safety:
• No prompt injection into execution
• No arbitrary code or query execution
• No implicit data access
• No silent failures

If something cannot be done safely, it is not done.

This makes llmd suitable for:
• Internal tools
• Regulated environments
• Multi-tenant systems
• Audit-heavy organizations

### Why Not "Just Use ChatGPT / Copilot"?

| Chat Systems         | llmd                    |
| -------------------- | ----------------------- |
| Free-form text       | Grammar-constrained     |
| Best-effort answers  | Deterministic execution |
| Hallucination-prone  | Rejects invalid output  |
| No scope enforcement | Explicit scope          |
| No audit trail       | Structured intents      |
| Hard to secure       | Built for security      |

llmd treats LLMs as compilers, not oracles.

⸻

## Running llmd

### Start the Server

```bash
LLMD_LLM_BACKEND=ollama \
LLMD_OLLAMA_MODEL=llama3.1:8b \
cargo run --bin llmd
```

http://localhost:3000

```bash
cargo run --bin cli
```

Optional flags:
• --persona <name>
• --stream
• --list-models

⸻

## Model Loading

Models are loaded via the companion model-loader tool (separate repository).

```bash
model-loader --model llama3.1 | cargo run --bin llmd
```

The loader handles:
• Downloads
• Caching
• Load plans
• Streaming models into llmd

⸻

## API Endpoints

• GET /v1/models
• POST /v1/chat/completions
• GET /v1/persona/:persona/memory
• POST /v1/persona/:persona/memory/reset

OpenAI-compatible where applicable.

⸻

## Design Philosophy (TL;DR)

LLMs should suggest intent.
Systems should decide execution.

llmd exists to make that separation explicit, enforceable, and safe.
