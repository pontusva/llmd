//! Relation-driven ownership model for intent lowering
//!
//! This module defines the declarative rules for how physical attributes
//! and entity-attributes map to their owning entities and required filters.

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

/// A rule defining ownership relationships for entity targets
#[derive(Debug, Clone)]
pub struct EntityOwnershipRule {
    /// The entity target (e.g., "device")
    pub entity: &'static str,
    /// The parent entity it must be scoped to (e.g., "building")
    pub parent: &'static str,
    /// The filter required to express that ownership (e.g., "buildingId")
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

/// Static table of entity ownership rules
/// These define required scoping filters for entity targets
pub static ENTITY_OWNERSHIP_RULES: &[EntityOwnershipRule] = &[
    EntityOwnershipRule {
        entity: "component",
        parent: "building",
        required_filter: "buildingId",
    },
    EntityOwnershipRule {
        entity: "device",
        parent: "building",
        required_filter: "buildingId",
    },
    EntityOwnershipRule {
        entity: "measure",
        parent: "building",
        required_filter: "buildingId",
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

            // Derive required filters by target from both physical ownership rules
            // and entity ownership rules (single source of truth).
            {
                let mut tmp: HashMap<String, HashSet<String>> = HashMap::new();

                // Physical attribute ownership (e.g., window -> component_type)
                for rule in OWNERSHIP_RULES {
                    tmp.entry(rule.owner.to_string())
                        .or_default()
                        .insert(rule.required_filter.to_string());
                }

                // Entity ownership (e.g., device -> buildingId)
                for rule in ENTITY_OWNERSHIP_RULES {
                    tmp.entry(rule.entity.to_string())
                        .or_default()
                        .insert(rule.required_filter.to_string());
                }

                for (target, filters) in tmp {
                    let mut v: Vec<String> = filters
                        .into_iter()
                        .filter(|f| valid_filters.contains(f))
                        .collect();
                    v.sort();
                    required_filters_by_target.insert(target, v);
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

/// Normalize entity-attributes (plural / common variants) to canonical target names.
///
/// Examples:
/// - "measures" -> "measure"
/// - "components" -> "component"
/// - "devices" -> "device"
pub fn normalize_entity_attribute(attr: &str) -> String {
    match attr.trim().to_lowercase().as_str() {
        "measures" => "measure",
        "measure" => "measure",

        "components" => "component",
        "component" => "component",

        "devices" => "device",
        "device" => "device",

        other => other,
    }
    .to_string()
}

/// True if an attribute string actually refers to a known *entity* target
/// (e.g. "measures" in "How many measures are in building Räven?").
pub fn is_entity_attribute(attr: &str) -> bool {
    let normalized = normalize_entity_attribute(attr);
    ENTITY_OWNERSHIP_RULES
        .iter()
        .any(|r| r.entity.eq_ignore_ascii_case(&normalized))
}

/// Look up an entity-ownership rule by an attribute string (plural OK).
pub fn find_entity_ownership_rule(attr: &str) -> Option<&'static EntityOwnershipRule> {
    let normalized = normalize_entity_attribute(attr);
    ENTITY_OWNERSHIP_RULES
        .iter()
        .find(|r| r.entity.eq_ignore_ascii_case(&normalized))
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
        assert!(introspection.valid_targets.contains("building"));
        assert!(introspection.valid_targets.contains("component"));
    }

    #[test]
    fn schema_introspection_extracts_valid_filters() {
        let introspection = IntentIntrospection::from_schema();
        assert!(introspection.valid_filters.contains("component_type"));
        assert!(introspection.valid_filters.contains("building_type"));
    }

    #[test]
    fn normalize_entity_attribute_singularizes_plural() {
        assert_eq!(normalize_entity_attribute("measures"), "measure".to_string());
        assert_eq!(normalize_entity_attribute("components"), "component".to_string());
        assert_eq!(normalize_entity_attribute("devices"), "device".to_string());
    }

    #[test]
    fn is_entity_attribute_recognizes_measures() {
        assert!(is_entity_attribute("measures"));
        assert!(is_entity_attribute("measure"));
        assert!(!is_entity_attribute("windows"));
    }
}
