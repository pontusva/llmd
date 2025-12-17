# Extending the Language

## Adding New Physical Attributes

This document explains how to safely extend llmd's query language by adding new physical attributes (e.g. balcony, elevator, radiator), without touching the LLM or risking unsafe execution.

---

## 1. What Is a Physical Attribute?

A physical attribute is a real-world, countable thing that:

- exists inside another entity
- cannot be queried on its own
- must be owned by a parent entity

### Examples

- windows → component
- doors → component
- pipes → component
- rooms → component

### Non-examples

- buildings
- measures
- projects
- plans

Physical attributes never map directly to GraphQL roots.

⸻

## 2. Single Source of Truth: Ownership Rules

All physical attributes are defined in one place:

`OWNERSHIP_RULES`

This table is the canonical definition of:

- what the attribute is called
- who owns it
- which filter represents it

---

## 3. Adding a New Physical Attribute (Step-by-Step)

### Example: Adding balcony

#### Step 1 — Add Ownership Rule

Open:

`src/tools/graphql/relations.rs`

Add:

```rust
OwnershipRule {
    physical: "balcony",
    owner: "component",
    required_filter: "component_type",
}
```

That's it.

No other logic changes required.

⸻

#### Step 2 — Add Plural Normalization (Optional)

If users may say "balconies":

```rust
pub fn normalize_physical_attribute(attr: &str) -> String {
    match attr.to_lowercase().as_str() {
        "windows" => "window",
        "doors" => "door",
        "balconies" => "balcony",
        other => other,
    }.to_string()
}
```

This ensures natural language maps to canonical form.

#### Step 3 — Done ✅

The following now works automatically:

**"How many balconies are in building Räven?"**

Compiles to:

```graphql
query {
  components(where: { buildingId: "building-123", type: "balcony" }) {
    _count {
      id
    }
  }
}
```

## 4. What You Do Not Touch

❌ No prompt changes  
❌ No executor changes  
❌ No compiler logic  
❌ No schema edits  
❌ No LLM fine-tuning

The language grows declaratively.

---

## 5. Why This Is Safe

Ownership rules enforce:

- No attribute without an owner
- No execution on unknown entities
- No implicit joins
- No schema drift

**If a rule is missing → execution fails.**

⸻

## 6. Validation Guarantees

Every new attribute is validated through:

1. Schema introspection
2. Ownership lookup
3. Lowering pass
4. Invariant validation

**If any step fails → hard error.**

---

## 7. Adding New Ownership Types (Advanced)

You may later want:

| Attribute | Owner       | Filter         |
| --------- | ----------- | -------------- |
| meter     | building    | building_type  |
| sensor    | real_estate | estate_feature |

Just add a rule:

```rust
OwnershipRule {
    physical: "sensor",
    owner: "real_estate",
    required_filter: "estate_feature",
}
```

Lowering adapts automatically.

⸻

## 8. When to Add a New Target Instead

Add a new target only if the entity:

- has independent lifecycle
- can be listed without parent
- has its own permissions

### Examples

- measure
- project
- plan

If it must live inside something else → it's a physical attribute.

---

## 9. Mental Checklist

Before adding an attribute, ask:

- Does it exist independently? → ❌
- Does it belong to something? → ✅
- Is it countable? → ✅
- Should it be filtered, not targeted? → ✅

**If yes → add ownership rule.**

---

## 10. Summary

To extend llmd's language:

1. Add one rule
2. (Optionally) normalize plurals
3. Done

This keeps the system:

- predictable
- auditable
- schema-safe
- LLM-agnostic
