//! Intent lowering pass - compiler-style transformation from logical to physical representation
//!
//! This module implements deterministic lowering rules that transform parsed intents
//! into their canonical executable form, driven by the ownership relation table.

use crate::tools::graphql::{Intent, Target, Filters, ScopeType};
use crate::tools::graphql::NameResolutionRegistry;
use super::relations::{
    find_ownership_rule,
    IntentIntrospection,
    is_physical_attribute,
    normalize_physical_attribute,
    is_entity_attribute,
    normalize_entity_attribute,
    find_entity_ownership_rule,
};
use std::sync::OnceLock;

/// Global introspection data - computed once at startup
static INTROSPECTION: OnceLock<IntentIntrospection> = OnceLock::new();

/// Initialize the lowering system with schema introspection
pub fn initialize_lowering() {
    let introspection = IntentIntrospection::from_schema();
    let _ = INTROSPECTION.set(introspection); // Ignore if already initialized
}

/// Intent lowering error types
#[derive(Debug, thiserror::Error)]
pub enum LoweringError {
    #[error("Unknown physical attribute: {0}")]
    UnknownPhysicalAttribute(String),
    #[error("Invalid target after lowering: {0}")]
    InvalidLoweredTarget(String),
    #[error("Invalid filter after lowering: {0}")]
    InvalidLoweredFilter(String),
    #[error("Lowering system not initialized")]
    NotInitialized,
    #[error("Missing building name for resolution")]
    MissingBuildingName,
    #[error("Unresolved building: {0}")]
    UnresolvedBuilding(String),

    #[error("Missing real estate name for resolution")]
    MissingRealEstateName,
    #[error("Unresolved real estate: {0}")]
    UnresolvedRealEstate(String),
}

/// Normalize and lower an intent from logical to physical representation
///
/// This is the single entry point for all intent transformation:
/// 1. Resolve names (buildingName → building_id)
/// 2. Lower physical attributes to ownership relations
/// 3. Validate lowered intent invariants
pub fn normalize_intent(intent: &mut Intent, registry: &dyn NameResolutionRegistry) -> Result<(), LoweringError> {
    initialize_lowering();

    resolve_names(intent, registry)?;

    // If the model already encoded a physical concept into filters.component_type,
    // canonicalize it before/after lowering.
    normalize_component_type_filter(intent);

    lower_intent(intent)?;

    // Lowering may also have produced/modified filters; canonicalize once more.
    normalize_component_type_filter(intent);

    validate_lowered_intent(intent)?;

    Ok(())
}

/// Resolve names in the intent (buildingName → building_id)
fn resolve_names(intent: &mut Intent, registry: &dyn NameResolutionRegistry) -> Result<(), LoweringError> {
    match intent.scope.r#type {
        ScopeType::Building => {
            if intent.scope.building_id.is_none() {
                let name = intent.scope.building_name
                    .as_ref()
                    .ok_or(LoweringError::MissingBuildingName)?;

                let resolved = registry
                    .resolve_building(name)
                    .ok_or_else(|| LoweringError::UnresolvedBuilding(name.clone()))?;

                intent.scope.building_id = Some(resolved.id);
            }
        }
        ScopeType::RealEstate => {
            if intent.scope.real_estate_id.is_none() {
                let name = intent
                    .scope
                    .real_estate_name
                    .as_ref()
                    .ok_or(LoweringError::MissingRealEstateName)?;

                let resolved = registry
                    .resolve_real_estate(name)
                    .ok_or_else(|| LoweringError::UnresolvedRealEstate(name.clone()))?;

                intent.scope.real_estate_id = Some(resolved.id);
            }
        }
        _ => {} // Other scope types don't need name resolution
    }
    Ok(())
}

/// Normalize physical/plural component_type values into canonical singular form.
///
/// This is intentionally conservative: it only changes values that are recognized
/// physical attributes (windows/doors/etc.) so we don't accidentally rewrite
/// domain-specific component types.
fn normalize_component_type_filter(intent: &mut Intent) {
    let Some(filters) = intent.filters.as_mut() else {
        return;
    };

    let Some(component_type) = filters.component_type.as_ref() else {
        return;
    };

    // If the component_type looks like a physical attribute (including plural forms),
    // normalize it to canonical singular form.
    if is_physical_attribute(component_type) {
        let normalized = normalize_physical_attribute(component_type);
        filters.component_type = Some(normalized);
    }
}

/// Lower an intent from logical to physical representation
///
/// This is a pure, deterministic transformation that must be called
/// after name resolution and before query compilation.
pub fn lower_intent(intent: &mut Intent) -> Result<(), LoweringError> {
    let introspection = INTROSPECTION.get().ok_or(LoweringError::NotInitialized)?;

    if let Some(attribute) = intent.attribute.clone() {
        // Normalize once up front
        let normalized = normalize_entity_attribute(&attribute);

        // Case 1: physical attribute (windows, doors, etc.)
        if is_physical_attribute(&normalized) {
            return lower_physical_attribute(intent, introspection);
        }

        // Case 2: entity attribute (measures, components, devices, etc.)
        if let Some(rule) = find_entity_ownership_rule(&normalized) {
            intent.target = match rule.entity {
                "building" => Target::Building,
                "component" => Target::Component,
                "measure" => Target::Measure,
                "real_estate" => Target::RealEstate,
                "project" => Target::Project,
                "plan" => Target::Plan,
                other => return Err(LoweringError::InvalidLoweredTarget(other.to_string())),
            };

            // Entity attributes never survive lowering
            intent.attribute = None;
            return Ok(());
        }
    }

    Ok(())
}

/// Lower a physical attribute to its canonical ownership representation
fn lower_physical_attribute(intent: &mut Intent, introspection: &IntentIntrospection) -> Result<(), LoweringError> {
    let attribute = intent.attribute.as_ref()
        .expect("Attribute must exist for physical lowering");

    // Normalize the attribute to canonical form before looking up rules
    let normalized_attribute = normalize_physical_attribute(attribute);

    let rule = find_ownership_rule(&normalized_attribute)
        .ok_or_else(|| LoweringError::UnknownPhysicalAttribute(attribute.clone()))?;

    // Validate that the target exists in the schema
    if !introspection.valid_targets.contains(rule.owner) {
        return Err(LoweringError::InvalidLoweredTarget(rule.owner.to_string()));
    }

    // Validate that the filter exists in the schema
    if !introspection.valid_filters.contains(rule.required_filter) {
        return Err(LoweringError::InvalidLoweredFilter(rule.required_filter.to_string()));
    }

    // Rewrite target to the owning entity
    intent.target = match rule.owner {
        "component" => Target::Component,
        "building" => Target::Building,
        "measure" => Target::Measure,
        "real_estate" => Target::RealEstate,
        _ => return Err(LoweringError::InvalidLoweredTarget(rule.owner.to_string())),
    };

    // Move attribute meaning into the required filter
    if intent.filters.is_none() {
        intent.filters = Some(Filters {
            component_type: None,
            building_type: None,
            year_min: None,
            year_max: None,
            completed: None,
            verified: None,
            status: None,
        });
    }

    // If the model already provided a component_type alongside attribute, make the
    // lowering deterministic by overwriting it with the canonical normalized value.
    // This prevents shape drift like component_type="windows" + attribute="windows".
    if let Some(filters) = &mut intent.filters {
        if filters.component_type.is_some() {
            filters.component_type = Some(normalized_attribute.clone());
        }
    }

    if let Some(filters) = &mut intent.filters {
        match rule.required_filter {
            "component_type" => {
                filters.component_type = Some(normalized_attribute);
            }
            "building_type" => {
                filters.building_type = Some(normalized_attribute);
            }
            _ => return Err(LoweringError::InvalidLoweredFilter(rule.required_filter.to_string())),
        }
    }

    // Clear the attribute - it's now represented in filters
    intent.attribute = None;

    Ok(())
}

/// Validate that a lowered intent satisfies all invariants
pub fn validate_lowered_intent(intent: &Intent) -> Result<(), LoweringError> {
    let introspection = INTROSPECTION.get().ok_or(LoweringError::NotInitialized)?;

    // Component targets must have component_type filter
    if intent.target == Target::Component {
        if let Some(filters) = &intent.filters {
            if filters.component_type.is_none() {
                return Err(LoweringError::InvalidLoweredTarget(
                    "Component target must have component_type filter".to_string()
                ));
            }
        } else {
            return Err(LoweringError::InvalidLoweredTarget(
                "Component target must have filters".to_string()
            ));
        }
    }

    // No intent may have both attribute and component_type
    if let (Some(_), Some(filters)) = (&intent.attribute, &intent.filters) {
        if filters.component_type.is_some() {
            return Err(LoweringError::InvalidLoweredTarget(
                "Cannot have both attribute and component_type".to_string()
            ));
        }
    }

    // Building targets may not have physical attributes
    if intent.target == Target::Building {
        if let Some(attribute) = &intent.attribute {
            if is_physical_attribute(attribute) {
                return Err(LoweringError::InvalidLoweredTarget(
                    format!("Building target cannot have physical attribute '{}'", attribute)
                ));
            }
        }
    }

    // Physical attributes must NEVER exist post-lowering
    if let Some(attribute) = &intent.attribute {
        if is_physical_attribute(attribute) {
            return Err(LoweringError::InvalidLoweredTarget(
                format!("Physical attribute '{}' must be lowered to filters, not remain as attribute", attribute)
            ));
        }
    }

    // Entity attributes must NEVER exist post-lowering
    if let Some(attribute) = &intent.attribute {
        if is_entity_attribute(attribute) {
            return Err(LoweringError::InvalidLoweredTarget(
                format!(
                    "Entity attribute '{}' must be lowered to target before compilation",
                    attribute
                )
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::graphql::{Intent, Action, Scope, ScopeType, Target};

    fn setup_test_intent(target: Target, attribute: Option<&str>) -> Intent {
        Intent {
            action: Action::Count,
            target,
            attribute: attribute.map(|s| s.to_string()),
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                real_estate_id: None,
                building_name: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        }
    }

    #[test]
    fn initialize_lowering_sets_introspection() {
        initialize_lowering();
        assert!(INTROSPECTION.get().is_some());
    }

    #[test]
    fn window_lowering_works() {
        initialize_lowering();

        let mut intent = setup_test_intent(Target::Building, Some("windows"));
        lower_intent(&mut intent).unwrap();

        assert_eq!(intent.target, Target::Component);
        assert_eq!(intent.attribute, None);
        assert_eq!(intent.filters.as_ref().unwrap().component_type, Some("window".to_string()));
    }

    #[test]
    fn door_lowering_works() {
        initialize_lowering();

        let mut intent = setup_test_intent(Target::Building, Some("doors"));
        lower_intent(&mut intent).unwrap();

        assert_eq!(intent.target, Target::Component);
        assert_eq!(intent.attribute, None);
        assert_eq!(intent.filters.as_ref().unwrap().component_type, Some("door".to_string()));
    }

    #[test]
    fn non_physical_attribute_unchanged() {
        initialize_lowering();

        let mut intent = setup_test_intent(Target::Building, Some("floors"));
        let original = intent.clone();
        lower_intent(&mut intent).unwrap();

        assert_eq!(intent.target, original.target);
        assert_eq!(intent.attribute, original.attribute);
    }

    #[test]
    fn lowered_component_requires_component_type() {
        initialize_lowering();

        let mut intent = setup_test_intent(Target::Component, None);
        let result = validate_lowered_intent(&intent);
        assert!(matches!(result, Err(LoweringError::InvalidLoweredTarget(_))));
    }

    #[test]
    fn building_with_physical_attribute_invalid() {
        initialize_lowering();

        let intent = setup_test_intent(Target::Building, Some("windows"));
        let result = validate_lowered_intent(&intent);
        assert!(matches!(result, Err(LoweringError::InvalidLoweredTarget(_))));
    }

    #[test]
    fn component_with_both_attribute_and_component_type_invalid() {
        initialize_lowering();

        let intent = Intent {
            action: Action::Count,
            target: Target::Component,
            attribute: Some("windows".to_string()),
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                real_estate_id: None,
                building_name: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: Some("windows".to_string()),
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: None,
            }),
            limit: None,
            group_by: None,
        };

        let result = validate_lowered_intent(&intent);
        assert!(matches!(result, Err(LoweringError::InvalidLoweredTarget(_))));
    }

    #[test]
    fn windows_in_building_lowers_correctly() {
        use crate::tools::graphql::{Action, Scope, ScopeType};
        use crate::tools::graphql::MockNameRegistry;

        initialize_lowering();

        let mut intent = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: Some("windows".to_string()),
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: None,
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let registry = MockNameRegistry;
        normalize_intent(&mut intent, &registry).unwrap();

        assert_eq!(intent.target, Target::Component);
        assert_eq!(intent.attribute, None);
        assert_eq!(
            intent.filters.as_ref().unwrap().component_type,
            Some("window".to_string())
        );
        assert!(intent.scope.building_id.is_some());
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
    }

    #[test]
    fn plural_component_type_filter_is_canonicalized() {
        initialize_lowering();

        let mut intent = setup_test_intent(Target::Component, None);
        intent.filters = Some(Filters {
            component_type: Some("windows".to_string()),
            building_type: None,
            year_min: None,
            year_max: None,
            completed: None,
            verified: None,
            status: None,
        });

        // No lowering should be required here, only canonicalization.
        normalize_component_type_filter(&mut intent);

        assert_eq!(intent.filters.as_ref().unwrap().component_type, Some("window".to_string()));
    }

    #[test]
    fn measures_in_building_lowers_to_measure_target() {
        use crate::tools::graphql::{Action, Scope, ScopeType};
        use crate::tools::graphql::MockNameRegistry;

        initialize_lowering();

        let mut intent = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: Some("measures".to_string()),
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: None,
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let registry = MockNameRegistry;
        normalize_intent(&mut intent, &registry).unwrap();

        assert_eq!(intent.target, Target::Measure);
        assert!(intent.attribute.is_none());
        assert!(intent.scope.building_id.is_some());
    }
}
