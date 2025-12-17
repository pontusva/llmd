# llmd Execution Model

This document describes how intents are executed, where authority lives, and why llmd is safe by construction.

---

## 1. Design Principle

**LLMs suggest. The compiler decides. The executor enforces.**

At no point does an LLM:

- execute queries
- resolve identifiers
- bypass schema rules
- invent joins or filters

All execution authority lives in deterministic Rust code.

---

## 2. High-Level Execution Flow

```
User Input (Natural Language)
    ↓
LLM Intent Compilation
    ↓
JSON Schema Validation
    ↓
Name Resolution (IDs only)
    ↓
Intent Lowering (Ownership rules)
    ↓
Invariant Validation
    ↓
GraphQL AST Compilation
    ↓
Pretty-Printed Query
    ↓
(Backend Execution – optional)
```

Each step is fail-fast.  
Failure at any stage aborts execution.

⸻

## 3. Intent Compilation (LLM Boundary)

The LLM's only responsibility is to output a candidate intent JSON.

### Constraints

- Output must match schema
- Tools must be explicitly invoked
- Any invalid JSON is rejected
- Hallucinated fields are dropped

If the LLM is unsure, it must respond in plain text.

---

## 4. Schema Validation

All intents are validated against a compile-time embedded JSON Schema.

### Guarantees

- No unknown fields
- No invalid enum values
- No missing required properties
- No structural ambiguity

**Schema violations = hard failure.**

⸻

## 5. Name Resolution (Human → Machine)

Human-readable identifiers (e.g. "Räven") are never executable.

Resolution happens via the NameResolutionRegistry:

```rust
resolve_building("Räven") → building-123
```

### Rules

- Names → IDs
- IDs only after resolution
- Resolution failure aborts execution
- Mock or real backends behave identically

This ensures:

- no guessing
- no leakage
- no execution on ambiguous entities

⸻

## 6. Intent Lowering (Compiler Pass)

Lowering is a deterministic transformation from logical intent to physical execution intent.

### Example

```
windows → component_type = "window"
building → component
```

Lowering is driven by:

- ownership rules
- schema introspection
- invariant checks

This step contains zero LLM logic.

⸻

## 7. Invariant Validation

Before execution, llmd enforces invariants:

### Examples

- Component queries must specify component_type
- Buildings cannot have physical attributes
- Attributes and filters cannot overlap
- Scope must match target ownership

**Invariant violations = execution blocked.**

⸻

## 8. GraphQL AST Compilation

Intents are compiled into a GraphQL AST, not raw strings.

### Why AST?

- Prevents malformed queries
- Enforces structural correctness
- Enables deterministic formatting
- Avoids injection vulnerabilities

### Example AST

```rust
GqlField::new("components")
    .arg("where", { buildingId, type })
    .select(
        GqlField::new("_count")
            .select("id")
    )
```

String output is a rendered artifact, not a source of truth.

⸻

## 9. Pretty Printing

GraphQL queries are pretty-printed for:

- readability
- debugging
- trust
- explainability

### Example output

```graphql
query {
  components(where: { buildingId: "building-123", type: "window" }) {
    _count {
      id
    }
  }
}
```

Formatting is deterministic and AST-driven.

⸻

## 10. Execution Boundary

By default, llmd does not execute GraphQL.

Instead, it returns:

- compiled query
- canonical intent
- execution context

This allows:

- dry runs
- previews
- audit logging
- backend swapping

Actual execution is an opt-in integration.

⸻

## 11. Safety Guarantees

llmd guarantees:

- ❌ No hallucinated execution
- ❌ No implicit joins
- ❌ No cross-scope access
- ❌ No schema drift
- ❌ No string-based query building
- ❌ No "LLM decides reality"

⸻

## 12. Mental Model

Think of llmd as:

| Component   | Maps To             |
| ----------- | ------------------- |
| Parser      | LLM                 |
| Compiler    | Rust                |
| Typechecker | Schema + invariants |
| Codegen     | GraphQL AST         |
| Runtime     | Optional backend    |

⸻

## 13. Why This Matters

This architecture enables:

- safe AI interfaces
- multi-tenant querying
- auditable decision making
- predictable behavior
- production-grade guarantees

**LLMs are powerful.**  
**They just aren't trusted.**
