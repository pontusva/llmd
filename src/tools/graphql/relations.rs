//! Relation-driven ownership model for intent lowering
//!
//! This module defines the declarative rules for how physical attributes
//! map to their owning entities and required filters.

use std::collections::{HashMap, HashSet};

/// A rule defining ownership relationships for physical attributes
#[derive(Debug, Clone)]
pub struct OwnershipRule {
    /// The physical attribute name (e.g., "window", "door")
    pub physical: &'static str,
    /// The owning entity type (e.g., "component")
    pub owner: &'static str,
    /// The filter field required for this ownership (e.g., "component_type")
    pub required_filter: &'static str,
}

/// Static table of ownership rules - the single source of truth
/// for physical attribute relationships
pub static OWNERSHIP_RULES: &[OwnershipRule] = &[
    OwnershipRule {
        physical: "window",
        owner: "component",
        required_filter: "component_type",
    },
    OwnershipRule {
        physical: "door",
        owner: "component",
        required_filter: "component_type",
    },
    OwnershipRule {
        physical: "pipe",
        owner: "component",
        required_filter: "component_type",
    },
    OwnershipRule {
        physical: "room",
        owner: "component",
        required_filter: "component_type",
    },
    OwnershipRule {
        physical: "sensor",
        owner: "component",
        required_filter: "component_type",
    },
];

/// Introspection data derived from the JSON schema
#[derive(Debug)]
pub struct IntentIntrospection {
    pub valid_targets: HashSet<String>,
    pub valid_filters: HashSet<String>,
    pub required_filters_by_target: HashMap<String, Vec<String>>,
}

impl IntentIntrospection {
    /// Create introspection by reading the embedded JSON schema
    pub fn from_schema() -> Self {
        // Read the embedded schema to derive valid targets and filters
        let schema_json = include_str!("../../../schemas/intent.schema.json");
        let schema: serde_json::Value = serde_json::from_str(schema_json)
            .expect("Embedded intent schema must be valid JSON");

        let mut valid_targets = HashSet::new();
        let mut valid_filters = HashSet::new();
        let mut required_filters_by_target: HashMap<String, Vec<String>> = HashMap::new();

        // Extract targets from schema
        if let Some(definitions) = schema.get("definitions") {
            if let Some(target_def) = definitions.get("Target") {
                if let Some(enum_values) = target_def.get("enum") {
                    if let Some(values) = enum_values.as_array() {
                        for value in values {
                            if let Some(s) = value.as_str() {
                                valid_targets.insert(s.to_string());
                            }
                        }
                    }
                }
            }

            // Extract filters from schema
            if let Some(filters_def) = definitions.get("Filters") {
                if let Some(properties) = filters_def.get("properties") {
                    if let Some(obj) = properties.as_object() {
                        for key in obj.keys() {
                            valid_filters.insert(key.clone());
                        }
                    }
                }
            }

            // Derive required filters by target from ownership rules (single source of truth).
            // This stays schema-aligned by later validating that derived filters exist in `valid_filters`.
            {
                let mut tmp: HashMap<String, HashSet<String>> = HashMap::new();
                for rule in OWNERSHIP_RULES {
                    tmp.entry(rule.owner.to_string())
                        .or_default()
                        .insert(rule.required_filter.to_string());
                }

                for (owner, filters) in tmp {
                    let mut v: Vec<String> = filters
                        .into_iter()
                        .filter(|f| valid_filters.contains(f))
                        .collect();
                    v.sort();
                    required_filters_by_target.insert(owner, v);
                }
            }
        }

        Self {
            valid_targets,
            valid_filters,
            required_filters_by_target,
        }
    }

    /// Get the required filter keys for a given target (if any)
    pub fn required_filters_for_target(&self, target: &str) -> Option<&[String]> {
        self.required_filters_by_target.get(target).map(|v| v.as_slice())
    }
}

/// Look up ownership rule for a physical attribute
pub fn find_ownership_rule(physical: &str) -> Option<&'static OwnershipRule> {
    OWNERSHIP_RULES.iter().find(|rule| rule.physical == physical)
}

/// Check if an attribute is a known physical attribute
pub fn is_physical_attribute(attr: &str) -> bool {
    let normalized = normalize_physical_attribute(attr);
    find_ownership_rule(&normalized).is_some()
}

/// Normalize physical attributes to singular form
pub fn normalize_physical_attribute(attr: &str) -> String {
    match attr.to_lowercase().as_str() {
        "windows" => "window",
        "doors" => "door",
        "rooms" => "room",
        "pipes" => "pipe",
        "sensors" => "sensor",
        other => other,
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ownership_rules_contain_expected_physical_attributes() {
        assert!(find_ownership_rule("window").is_some());
        assert!(find_ownership_rule("door").is_some());
        assert!(find_ownership_rule("pipe").is_some());
        assert!(find_ownership_rule("room").is_some());
        assert!(find_ownership_rule("sensor").is_some());
    }

    #[test]
    fn unknown_physical_attribute_returns_none() {
        assert!(find_ownership_rule("unknown").is_none());
        assert!(find_ownership_rule("building").is_none());
    }

    #[test]
    fn schema_introspection_extracts_valid_targets() {
        let introspection = IntentIntrospection::from_schema();
        // Should contain at least "building", "component", "measure", etc.
        assert!(introspection.valid_targets.contains("building"));
        assert!(introspection.valid_targets.contains("component"));
    }

    #[test]
    fn schema_introspection_extracts_valid_filters() {
        let introspection = IntentIntrospection::from_schema();
        // Should contain "component_type", "building_type", etc.
        assert!(introspection.valid_filters.contains("component_type"));
        assert!(introspection.valid_filters.contains("building_type"));
    }

    #[test]
    fn schema_introspection_derives_required_filters_from_ownership_rules() {
        let introspection = IntentIntrospection::from_schema();
        let required = introspection
            .required_filters_for_target("component")
            .unwrap_or(&[]);
        assert!(required.iter().any(|f| f == "component_type"));
    }
}
