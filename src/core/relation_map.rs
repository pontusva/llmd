use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Represents how to traverse from a target entity to a scope entity
#[derive(Debug, Clone)]
pub enum RelationPath {
    /// Direct foreign key relationship: target.fk_field = scope_id
    Direct { fk_field: &'static str },
    /// Nested relationship path: target.path[0].path[1]...path[n].field = scope_id
    Nested { path: &'static [&'static str], field: &'static str },
}

/// Static relation map defining how targets relate to scopes
/// Key: (Target, ScopeType) -> Value: RelationPath
pub static RELATION_MAP: Lazy<HashMap<(crate::tools::graphql::Target, crate::tools::graphql::ScopeType), RelationPath>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Measure relationships
    map.insert(
        (crate::tools::graphql::Target::Measure, crate::tools::graphql::ScopeType::Building),
        RelationPath::Direct { fk_field: "buildingId" }
    );
    map.insert(
        (crate::tools::graphql::Target::Measure, crate::tools::graphql::ScopeType::RealEstate),
        RelationPath::Nested { path: &["building"], field: "realEstateId" }
    );

    // Component relationships
    map.insert(
        (crate::tools::graphql::Target::Component, crate::tools::graphql::ScopeType::Building),
        RelationPath::Direct { fk_field: "buildingId" }
    );

    // Plan relationships
    map.insert(
        (crate::tools::graphql::Target::Plan, crate::tools::graphql::ScopeType::Building),
        RelationPath::Nested { path: &["component"], field: "buildingId" }
    );

    map
});

/// Compiles a scope constraint into a GraphQL where clause fragment
pub fn compile_scope_constraint(
    relation: &RelationPath,
    resolved_id: &str
) -> serde_json::Value {
    match relation {
        RelationPath::Direct { fk_field } => {
            // Direct: { buildingId: "<id>" }
            serde_json::json!({ *fk_field: resolved_id })
        },
        RelationPath::Nested { path, field } => {
            // Nested: { building: { realEstateId: "<id>" } }
            let mut current = serde_json::json!({ *field: resolved_id });
            for &segment in path.iter().rev() {
                let mut obj = serde_json::Map::new();
                obj.insert(segment.to_string(), current);
                current = serde_json::Value::Object(obj);
            }
            current
        }
    }
}

/// Looks up the relation path for a target/scope combination
pub fn get_relation_path(
    target: &crate::tools::graphql::Target,
    scope_type: &crate::tools::graphql::ScopeType
) -> Option<&'static RelationPath> {
    RELATION_MAP.get(&(*target, *scope_type))
}
