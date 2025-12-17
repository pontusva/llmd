# llmd Intent Syntax & Grammar

llmd exposes a strict, declarative intent language compiled from natural language into a validated, executable form.

This document describes the formal grammar, normalization rules, and execution constraints of that language.

---

## 1. Overview

llmd does not execute natural language directly.

Instead, user input is compiled into a canonical Intent:

```json
{
  "action": "count",
  "target": "component",
  "scope": {
    "type": "building",
    "buildingId": "building-123"
  },
  "filters": {
    "component_type": "window"
  }
}
```

Only valid intents may be executed.
If compilation or validation fails, execution is rejected.

---

## 2. Intent Shape (Canonical Form)

Every executable intent conforms to this structure:

| Field   | Required | Description                                |
| ------- | -------- | ------------------------------------------ |
| action  | ✅       | What operation to perform                  |
| target  | ✅       | The primary entity being queried           |
| scope   | ✅       | The boundary in which execution is allowed |
| filters | ❌       | Optional constraints                       |
| limit   | ❌       | Result limit                               |
| metric  | ❌       | Aggregation metric (only for aggregate)    |

## 3. Actions

Actions describe what is being done.

| Action    | Meaning                           |
| --------- | --------------------------------- |
| count     | Count entities                    |
| list      | List entities                     |
| get       | Fetch detailed entity data        |
| aggregate | Aggregate values (sum, avg, etc.) |

### Rules

- count MUST NOT use metric
- aggregate MUST specify a valid metric
- Unsupported combinations are rejected at compile time

⸻

## 4. Targets (Structural Entities)

Targets represent structural domain entities.

Valid targets are derived from the schema:

| Target     |
| ---------- |
| building   |
| component  |
| measure    |
| realEstate |
| plan       |
| project    |

### Structural vs Physical

- Structural entities → valid targets
- Physical things → never targets (see §6)

⸻

## 5. Scope

Scope defines where the intent is allowed to execute.

```json
"scope": {
  "type": "building",
  "buildingId": "building-123"
}
```

### Scope Types

| Type         | Meaning             |
| ------------ | ------------------- |
| current_team | Entire team context |
| building     | A specific building |
| real_estate  | A real estate group |

### Rules

- Human-readable names ("Räven") are never executed
- Names are resolved → IDs before execution
- Unresolved names cause hard failure
- Scope narrowing is mandatory when explicitly mentioned

⸻

## 6. Physical Attributes (Windows, Doors, etc.)

Physical objects are not entities.
They are attributes owned by structural entities.

### Examples

- windows
- doors
- rooms
- pipes
- sensors

These must never appear as targets.

⸻

## 7. Ownership & Lowering Rules

Physical attributes are lowered deterministically using a relation table.

### Example

**User input:**

"How many windows are in building Räven?"

**Logical intent (pre-lowering):**

```json
{
  "action": "count",
  "target": "building",
  "attribute": "windows",
  "scope": {
    "type": "building",
    "buildingName": "Räven"
  }
}
```

**Lowered canonical intent:**

```json
{
  "action": "count",
  "target": "component",
  "scope": {
    "type": "building",
    "buildingId": "building-123"
  },
  "filters": {
    "component_type": "window"
  }
}
```

### Why?

Because:

```
window → component → building
```

This transformation is:

- deterministic
- schema-driven
- non-LLM
- non-heuristic

⸻

## 8. Filters

Filters restrict execution without changing meaning.

### Common filters

| Filter         | Applies To |
| -------------- | ---------- |
| component_type | components |
| building_type  | buildings  |
| status         | measures   |
| completed      | measures   |

### Rules

- Filters are validated against schema
- Unknown filters are rejected
- Filters may be auto-inserted by lowering

⸻

## 9. Grammar Invariants (Hard Rules)

The following are non-negotiable:

- ❌ Physical attributes cannot be targets
- ❌ Targets cannot retain physical attributes post-lowering
- ❌ Component targets require component_type
- ❌ Execution without resolved IDs is forbidden
- ❌ JSON schema violations abort execution
- ❌ LLM output never bypasses validation

⸻

## 10. Execution Boundary

Only lowered + validated intents reach execution.

### Pipeline

```
Natural Language
    ↓
Intent Compilation (LLM)
    ↓
Schema Validation
    ↓
Name Resolution
    ↓
Ownership Lowering
    ↓
Invariant Validation
    ↓
GraphQL AST Compilation
    ↓
Execution
```

At no point can the LLM:

- invent joins
- invent filters
- guess IDs
- execute queries directly

⸻

## 11. Why This Matters

This grammar enables:

- deterministic behavior
- auditability
- zero hallucination execution
- safe multi-tenant usage
- backend-agnostic execution

**LLMs suggest intent.**  
**llmd decides reality.**
