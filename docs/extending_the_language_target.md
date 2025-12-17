# Extending the Language

## Adding New Targets

This document explains how to add a new **target entity** to llmd’s query language in a **safe, compiler-driven** way.

Targets are **top-level queryable entities**.  
Unlike physical attributes, they can be queried directly and compiled into GraphQL root fields.

---

## 1. What Is a Target?

A **target** is an entity that:

- exists independently
- has its own lifecycle
- has a stable ID
- can be queried directly
- appears as a GraphQL root field
- has authorization semantics

### Examples

- building
- component
- measure
- project
- plan
- device

### Non-examples

- window
- door
- pipe
- room

These are **physical attributes**, not targets.  
They require **ownership lowering** before execution.

---

## 2. When Should You Add a New Target?

Add a new target **only if all of the following are true**:

- ✅ It can exist without a physical attribute
- ✅ It has its own ID
- ✅ It can be listed or counted globally
- ✅ It maps to a GraphQL root field
- ✅ It has distinct authorization semantics

If **any** of these are false → it is **not** a target.

---

## 3. Example: Adding a `device` Target

Your backend introduces a new entity:

```
Device
- id
- name
- type
- buildingId
```

You want to support:

> How many devices are in building Räven?

---

## 4. Step-by-Step: Adding a Target

### Step 1 — Add Target to the Intent Schema

Open:

```
schemas/intent.schema.json
```

Add the new value to the `Target` enum:

```json
"Target": {
  "type": "string",
  "enum": [
    "building",
    "component",
    "measure",
    "project",
    "plan",
    "device"
  ]
}
```

This immediately:

- validates incoming intents
- updates schema introspection
- constrains LLM output
- prevents hallucinated targets

---

### Step 2 — Add Target Enum Variant

Open the Rust file where `pub enum Target` is defined.

Add the enum variant:

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Target {
    Building,
    Component,
    RealEstate,
    Measure,
    Project,
    Plan,
    Device, // NEW
}
```

This step only enables the grammar and schema recognition of the new target; it does not yet enable execution or compilation logic.

---

### Step 2.5 — Declare Capabilities (Eligibility Matrix)

Targets must be explicitly allowed via the capability registry to be executable. Add a capability entry like this:

```rust
registry.add_capability(Capability {
    action: Action::Count,
    target: Target::Device,
    attribute: AttributePattern::None,
    metric: MetricPattern::None,
});
```

Without declaring capabilities, the executor will refuse to run the tool for that target and action combination.

---

### Step 3 — Add GraphQL Compilation Rule

Open:

```
src/tools/graphql.rs
```

Add the execution mapping:

```rust
(Action::Count, Target::Device, None) => {
    GqlField::new("devices")
        .arg("where", where_clause)
        .select(
            GqlField::new("_count")
                .select(GqlField::new("id"))
        )
}
```

This defines:

- how the target executes
- which GraphQL root field it maps to
- what operations are supported

Missing this step will result in an “unsupported action/target combination” compiler error.

---

### Step 4 — Define Ownership / Required Filters (Compiler-Derived)

**Do not hardcode required filters.**

Targets that require contextual ownership (e.g. `device → building`, `measure → building`)
must declare this via **entity ownership rules**.

Entity ownership rules participate in the same compiler-driven lowering phase as
physical attribute ownership. This allows queries like:

“How many measures are in building Räven?”

to be normalized into:

target = measure  
filters.buildingId = …

before GraphQL compilation.

Open:

```
src/tools/graphql/relations.rs
```

Add an entity ownership rule:

```rust
EntityOwnershipRule {
    entity: "device",
    parent: "building",
    required_filter: "buildingId",
}
```

### Entity Attributes vs Physical Attributes

Some user queries refer to entities using plural or attribute-like phrasing
(e.g. “measures”, “devices”).

These are treated as **entity-attributes**, not physical attributes.

The compiler will:

- normalize the attribute (e.g. “measures” → “measure”)
- lower it into a target
- apply entity ownership rules
- clear the attribute before GraphQL compilation

This prevents invalid shapes such as:

buildings { measures { … } }

and ensures all entity queries compile to root-level GraphQL fields.

---

## 5. What You Do Not Do

- ❌ Do not add target logic to prompts
- ❌ Do not teach the LLM new entities
- ❌ Do not hardcode rewrites
- ❌ Do not bypass schema validation
- ❌ Do not manually inject required filters

Targets are **compiler features**, not prompt features.

---

## 6. Target vs Physical Attribute (Quick Rule)

| Question                         | If Yes →  |
| -------------------------------- | --------- |
| Can it exist alone?              | Target    |
| Does it belong to something?     | Attribute |
| Can it be listed globally?       | Target    |
| Does it need ownership lowering? | Attribute |

---

## 7. Example Comparison

### ❌ Incorrect (attribute as target)

> How many windows are in Räven?

```
target = window ❌
```

### ✅ Correct (ownership lowering)

```
target = component
filters.component_type = "window"
scope.buildingId = …
```

### ✅ Correct (entity ownership lowering)

> How many measures are in building Räven?

```
target = measure
filters.buildingId = …
```

---

## 8. Security & Stability Benefits

Adding a target:

- expands the language explicitly
- updates schema validation automatically
- updates introspection automatically
- enforces ownership constraints
- fails safely if incomplete
- prevents partially-wired targets from executing

---

## 9. Testing a New Target

Minimum tests to add:

```rust
#[test]
fn device_target_is_valid() {
    let i = IntentIntrospection::from_schema();
    assert!(i.valid_targets.contains("device"));
}
```

```rust
#[test]
fn count_devices_requires_building_scope() {
    // build intent → lower → validate → expect error without buildingId
}
```

```rust
#[test]
fn count_devices_compiles() {
    // build intent → compile → assert GraphQL
}
```

---

## 10. Summary

To add a new target:

1. Add it to the schema
2. Add enum variant
3. Declare capabilities
4. Add GraphQL compilation rule
5. Define ownership via relations (physical + entity)

No prompt edits.  
No LLM retraining.  
No ambiguity.  
No unsafe execution.
