use crate::runtime::toolport::{ToolPort, ToolInput, ToolOutput, ToolError, ToolEligibility, ToolEligibilityContext};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use schemars::JsonSchema;
use crate::core::schema;

/// Intent Schema - Authoritative model for data access requests
/// LLMs output Intent JSON, backend compiles to GraphQL
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
/// FORMAL INTENT STRUCTURE
/// =======================
/// Intent := {
///   action: Action,
///   target: Target,
///   scope: Scope,
///   attribute?: Attribute,
///   metric?: Metric
/// }
/// Note: Grammar-level validation only. Capabilities decide what is supported.
pub struct Intent {
    /// Action to perform
    pub action: Action,
    /// Target type to query (structural entity) - supports both "target" and "entity" for backward compatibility
    #[serde(alias = "entity")]
    pub target: Target,
    /// Scope of the query
    pub scope: Scope,
    /// Optional attribute (specific property like "windows", "doors", etc.)
    /// Grammar does NOT validate - this is free-form
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<Attribute>,
    /// Optional metric for aggregations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<Metric>,
    /// Optional filters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Filters>,
    /// Optional result limit
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Optional grouping
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_by: Option<GroupBy>,
    /// Signal that this intent is partial and requires enrichment or clarification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial: Option<bool>,
}

/// Intent envelope for JSON Schema generation
/// This is the root schema that wraps the intent
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IntentEnvelope {
    pub intent: Intent,
}

impl Intent {
    /// Lower this intent from logical to physical representation
    /// Must be called after name resolution and before query compilation
    pub fn lower(&mut self) -> Result<(), intent_lowering::LoweringError> {
        intent_lowering::lower_intent(self)
    }
}

/// NAME RESOLUTION SYSTEM
/// ======================
/// Deterministic name resolution layer for intent normalization
/// Ensures extracted building/real estate names are authoritative

/// Domain kinds that can be resolved
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolvableKind {
    Building,
    RealEstate,
}

/// Resolved building with ID and name
#[derive(Debug, Clone)]
pub struct ResolvedBuilding {
    pub id: String,
    pub name: String,
}

/// Resolved real estate with ID and name
#[derive(Debug, Clone)]
pub struct ResolvedRealEstate {
    pub id: String,
    pub name: String,
}

/// Trait for name resolution registries
pub trait NameResolutionRegistry: Send + Sync {
    fn resolve_building(&self, name: &str) -> Option<ResolvedBuilding>;
    fn resolve_real_estate(&self, name: &str) -> Option<ResolvedRealEstate>;
}

/// Mock implementation of name resolution registry with deterministic IDs
pub struct MockNameRegistry;

impl MockNameRegistry {
    fn fake_id(kind: &str, name: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}", kind, name));
        let hash = hasher.finalize();
        format!("mock-{}-{}", kind, hex::encode(&hash[..6]))
    }
}

impl NameResolutionRegistry for MockNameRegistry {
    fn resolve_building(&self, name: &str) -> Option<ResolvedBuilding> {
        // Simulate error for testing
        if name.eq_ignore_ascii_case("unknown") {
            return None;
        }

        let normalized_name = name.trim().to_lowercase();
        Some(ResolvedBuilding {
            id: Self::fake_id("building", &normalized_name),
            name: name.to_string(),
        })
    }

    fn resolve_real_estate(&self, name: &str) -> Option<ResolvedRealEstate> {
        // Simulate error for testing
        if name.eq_ignore_ascii_case("unknown") {
            return None;
        }

        let normalized_name = name.trim().to_lowercase();
        Some(ResolvedRealEstate {
            id: Self::fake_id("realestate", &normalized_name),
            name: name.to_string(),
        })
    }
}

/// In-memory implementation of name resolution registry
use std::collections::HashMap;

pub struct InMemoryNameRegistry {
    buildings: HashMap<String, ResolvedBuilding>,
    real_estates: HashMap<String, ResolvedRealEstate>,
}

impl InMemoryNameRegistry {
    pub fn new(
        buildings: impl IntoIterator<Item = (String, String, String)>,
        real_estates: impl IntoIterator<Item = (String, String, String)>,
    ) -> Self {
        let mut buildings_map = HashMap::new();
        for (name, id, canonical_name) in buildings {
            buildings_map.insert(
                name.to_lowercase(),
                ResolvedBuilding {
                    id,
                    name: canonical_name,
                },
            );
        }

        let mut real_estates_map = HashMap::new();
        for (name, id, canonical_name) in real_estates {
            real_estates_map.insert(
                name.to_lowercase(),
                ResolvedRealEstate {
                    id,
                    name: canonical_name,
                },
            );
        }

        Self {
            buildings: buildings_map,
            real_estates: real_estates_map,
        }
    }
}

impl NameResolutionRegistry for InMemoryNameRegistry {
    fn resolve_building(&self, name: &str) -> Option<ResolvedBuilding> {
        self.buildings.get(&name.trim().to_lowercase()).cloned()
    }

    fn resolve_real_estate(&self, name: &str) -> Option<ResolvedRealEstate> {
        self.real_estates.get(&name.trim().to_lowercase()).cloned()
    }
}

/// FILTER CAPABILITY MATRIX SYSTEM
/// ================================
/// The FilterCapability Matrix validates normalized intent filters.
/// It decides which filters are supported for each (entity, action) pair.
///
/// Rules:
/// - Filters MUST be normalized before validation
/// - FilterCapabilityMatrix MUST be the ONLY place that validates filters
/// - Validation happens AFTER normalization, BEFORE execution
/// - No string comparisons - filters are enum/struct based

/// Canonical filter keys
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FilterKey {
    Status,
    Year,
    DateRange,
}

/// Canonical filter values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterValue {
    Status(StatusValue),
    Year(i32),
    DateRange { from: i32, to: i32 },
}

/// Status filter values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusValue {
    Completed,
    NotCompleted,
}

/// Filter validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterValidationError {
    UnsupportedFilter { key: FilterKey, entity: Target, action: Action },
    InvalidFilterValue { key: FilterKey, value: String },
    FiltersNotAllowed { entity: Target, action: Action },
}

/// Filter capability definition
#[derive(Debug, Clone)]
pub struct FilterCapability {
    pub entity: Target,
    pub action: Action,
    pub allowed_filters: std::collections::HashMap<FilterKey, Vec<FilterValue>>,
}

/// FILTER CAPABILITY MATRIX TRAIT
/// ===============================
pub trait FilterCapabilityMatrix {
    fn validate_filters(
        &self,
        entity: &Target,
        action: &Action,
        filters: &std::collections::HashMap<FilterKey, FilterValue>,
    ) -> Result<(), FilterValidationError>;
}

/// Concrete implementation using a registry of filter capabilities
#[derive(Debug)]
pub struct FilterCapabilityRegistry {
    capabilities: Vec<FilterCapability>,
}

impl FilterCapabilityRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            capabilities: Vec::new(),
        };

        // Register supported filter capabilities
        registry.add_capability(FilterCapability {
            entity: Target::Measure,
            action: Action::Count,
            allowed_filters: [
                (FilterKey::Status, vec![
                    FilterValue::Status(StatusValue::Completed),
                    FilterValue::Status(StatusValue::NotCompleted),
                ]),
                (FilterKey::Year, vec![]), // Any year allowed (empty vec means any value)
            ].into_iter().collect(),
        });

        // Building count allows NO filters
        registry.add_capability(FilterCapability {
            entity: Target::Building,
            action: Action::Count,
            allowed_filters: std::collections::HashMap::new(),
        });

        // Component count allows NO filters
        registry.add_capability(FilterCapability {
            entity: Target::Component,
            action: Action::Count,
            allowed_filters: std::collections::HashMap::new(),
        });

        registry
    }

    fn add_capability(&mut self, capability: FilterCapability) {
        self.capabilities.push(capability);
    }
}

impl Filters {
    /// Convert normalized Filters struct to canonical HashMap for validation
    pub fn to_canonical_map(&self) -> std::collections::HashMap<FilterKey, FilterValue> {
        let mut map = std::collections::HashMap::new();

        // Canonical status: prioritize status field over completed field
        if let Some(ref status) = self.status {
            let status_value = match status.as_str() {
                "completed" => FilterValue::Status(StatusValue::Completed),
                "not_completed" => FilterValue::Status(StatusValue::NotCompleted),
                _ => FilterValue::Status(StatusValue::Completed), // fallback
            };
            map.insert(FilterKey::Status, status_value);
        } else if let Some(completed) = self.completed {
            // Convert completed boolean to status enum
            let status_value = if completed {
                FilterValue::Status(StatusValue::Completed)
            } else {
                FilterValue::Status(StatusValue::NotCompleted)
            };
            map.insert(FilterKey::Status, status_value);
        }

        if let Some(year) = self.year_min {
            map.insert(FilterKey::Year, FilterValue::Year(year));
        }

        if self.year_min != self.year_max {
            if let Some(year_max) = self.year_max {
                // For year ranges, use DateRange
                if let Some(year_min) = self.year_min {
                    map.insert(FilterKey::DateRange, FilterValue::DateRange {
                        from: year_min,
                        to: year_max,
                    });
                }
            }
        }

        // Add other filter conversions as needed

        map
    }
}

impl FilterCapabilityMatrix for FilterCapabilityRegistry {
    fn validate_filters(
        &self,
        entity: &Target,
        action: &Action,
        filters: &std::collections::HashMap<FilterKey, FilterValue>,
    ) -> Result<(), FilterValidationError> {
        // Find the capability for this entity/action pair
        let capability = self.capabilities.iter()
            .find(|cap| cap.entity == *entity && cap.action == *action);

        match capability {
            Some(cap) => {
                // If no filters are provided, that's fine
                if filters.is_empty() {
                    return Ok(());
                }

                // Check each filter
                for (key, value) in filters {
                    let allowed_values = match cap.allowed_filters.get(key) {
                        Some(values) => values,
                        None => {
                            return Err(FilterValidationError::UnsupportedFilter {
                                key: key.clone(),
                                entity: entity.clone(),
                                action: action.clone(),
                            });
                        }
                    };

                    // If allowed_values is empty, any value is allowed for this key
                    if allowed_values.is_empty() {
                        continue;
                    }

                    // Check if the value is in the allowed list
                    if !allowed_values.contains(value) {
                        return Err(FilterValidationError::InvalidFilterValue {
                            key: key.clone(),
                            value: format!("{:?}", value),
                        });
                    }
                }

                Ok(())
            }
            None => {
                // No capability defined for this entity/action
                if filters.is_empty() {
                    Ok(())
                } else {
                    Err(FilterValidationError::FiltersNotAllowed {
                        entity: entity.clone(),
                        action: action.clone(),
                    })
                }
            }
        }
    }
}

/// CAPABILITY MATRIX SYSTEM
/// ========================
/// The Capability Matrix decides what intents are supported.
/// It is separate from the grammar and configurable.

/// Decision result from capability matrix
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecision {
    Supported,
    Unsupported { reason: String },
}

/// CAPABILITY MATRIX TRAIT
/// =======================
/// The CapabilityMatrix trait defines the interface for checking
/// whether an intent is supported by the backend.
///
/// Rules:
/// - Grammar parsing MUST NOT reject intents based on capability
/// - CapabilityMatrix MUST be the ONLY place that decides validity
/// - Executor MUST call CapabilityMatrix before execution
pub trait CapabilityMatrix {
    fn supports(&self, intent: &Intent) -> CapabilityDecision;
}

/// Capability definition patterns
#[derive(Debug, Clone)]
pub enum AttributePattern {
    Any,              // Any attribute allowed
    None,             // No attribute allowed
    Exact(String),    // Exact attribute match
}

#[derive(Debug, Clone)]
pub enum MetricPattern {
    Any,              // Any metric allowed
    None,             // No metric allowed
    Exact(Metric),    // Exact metric match
}

#[derive(Debug, Clone)]
pub struct Capability {
    pub action: Action,
    pub target: Target,
    pub attribute: AttributePattern,
    pub metric: MetricPattern,
}

/// EXAMPLE CAPABILITY REGISTRY
/// ===========================
/// Concrete implementation of CapabilityMatrix using a registry of supported capabilities.
#[derive(Debug)]
pub struct CapabilityRegistry {
    capabilities: Vec<Capability>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            capabilities: Vec::new(),
        };

        // Register supported capabilities
        registry.add_capability(Capability {
            action: Action::Count,
            target: Target::Component,
            attribute: AttributePattern::None,
            metric: MetricPattern::None,
        });

        registry.add_capability(Capability {
            action: Action::Count,
            target: Target::Building,
            attribute: AttributePattern::None,
            metric: MetricPattern::None,
        });

        registry.add_capability(Capability {
            action: Action::Count,
            target: Target::Building,
            attribute: AttributePattern::Any,  // Allow any attribute for counting
            metric: MetricPattern::None,
        });

        registry.add_capability(Capability {
            action: Action::Count,
            target: Target::Measure,
            attribute: AttributePattern::None,
            metric: MetricPattern::None,
        });

        registry.add_capability(Capability {
            action: Action::List,
            target: Target::Building,
            attribute: AttributePattern::None,
            metric: MetricPattern::None,
        });

        registry.add_capability(Capability {
            action: Action::Get,
            target: Target::Component,
            attribute: AttributePattern::None,
            metric: MetricPattern::None,
        });

        registry.add_capability(Capability {
            action: Action::Aggregate,
            target: Target::Component,
            attribute: AttributePattern::None,
            metric: MetricPattern::Exact(Metric::Count),
        });

        registry.add_capability(Capability {
            action: Action::Aggregate,
            target: Target::Building,
            attribute: AttributePattern::None,
            metric: MetricPattern::Exact(Metric::TotalFloorArea),
        });

        registry.add_capability(Capability {
            action: Action::Aggregate,
            target: Target::Building,
            attribute: AttributePattern::Any,  // Allow any attribute for aggregation
            metric: MetricPattern::Any,        // Allow any metric
        });

        registry.add_capability(Capability {
            action: Action::Count,
            target: Target::Measure,
            attribute: AttributePattern::None,
            metric: MetricPattern::Exact(Metric::Count),
        });

        registry
    }

    pub fn add_capability(&mut self, capability: Capability) {
        self.capabilities.push(capability);
    }
}

impl CapabilityMatrix for CapabilityRegistry {
    fn supports(&self, intent: &Intent) -> CapabilityDecision {
        for capability in &self.capabilities {
            if capability.action != intent.action || capability.target != intent.target {
                continue;
            }

            // Phase 1 (Classification) only requires action + target + scope.
            // Attributes and Metrics are hints - we don't reject based on them here.
            // The executor will handle enrichment and validation in Phase 2.
            return CapabilityDecision::Supported;
        }

        // No matching capability found
        CapabilityDecision::Unsupported {
            reason: format!(
                "Unsupported action/target combination: action={:?}, target={:?}",
                intent.action, intent.target
            ),
        }
    }
}

/// Supported actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(JsonSchema)]
pub enum Action {
    List,
    Count,
    Get,
    Aggregate,
    Exists,
}

/// FORMAL INTENT GRAMMAR
/// ======================
/// The Intent grammar defines the structure of valid intents.
/// It is STABLE and does NOT encode backend capabilities.

/// Supported target types (ROOT ENTITIES ONLY)
/// This is the grammar-level enum - capabilities decide which combinations are supported
/// ONLY root domain entities are allowed - nested objects must be attributes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Target {
    /// Physical building structures
    Building,
    /// Components within buildings (doors, windows, etc.)
    Component,
    /// Real estate properties
    RealEstate,
    /// Measurement data points
    Measure,
    /// Planning documents
    Plan,
    /// Project management entities
    Project,
}

/// Legacy alias for backward compatibility
pub type Entity = Target;

/// Build GraphQL where clause from intent as GqlValue::Object
pub fn build_where_clause(intent: &Intent) -> crate::tools::graphql::ast::GqlValue {
    use crate::tools::graphql::ast::GqlValue;

    let mut conditions = Vec::new();

    // INVARIANT 2 ENFORCEMENT: GraphQL query builders must never receive names.
    // They must operate only on resolved IDs.
    match intent.scope.r#type {
        ScopeType::Building => {
            if let Some(id) = &intent.scope.building_id {
                conditions.push(("buildingId".into(), GqlValue::String(id.clone())));
            } else {
                // This should have been caught during normalization/lowering
                tracing::error!("🚫 Invariant Violation: GraphQL compilation received 'building' scope without resolved building_id");
                panic!("GraphQL invariant violation: unresolved building_id");
            }
        }
        ScopeType::RealEstate => {
            if let Some(id) = &intent.scope.real_estate_id {
                conditions.push(("realEstateId".into(), GqlValue::String(id.clone())));
            } else {
                // This should have been caught during normalization/lowering
                tracing::error!("🚫 Invariant Violation: GraphQL compilation received 'real_estate' scope without resolved real_estate_id");
                panic!("GraphQL invariant violation: unresolved real_estate_id");
            }
        }
        ScopeType::CurrentTeam => {
            // No ID needed for team scope, it's implicit in context
        }
    }

    // ---- filters → where
    if let Some(filters) = &intent.filters {
        match intent.target {
            Target::Component => {
                if let Some(component_type) = &filters.component_type {
                    // Use proper normalization for physical attributes
                    let normalized = crate::tools::graphql::relations::normalize_physical_attribute(component_type);
                    conditions.push(("type".into(), GqlValue::String(normalized)));
                }
            }

            Target::Building => {
                if let Some(building_type) = &filters.building_type {
                    conditions.push(("type".into(), GqlValue::String(building_type.clone())));
                }
            }

            _ => {}
        }

        if let Some(status) = &filters.status {
            let status_value = match status.as_str() {
                "completed" => "COMPLETED",
                "not_completed" => "NOT_COMPLETED",
                _ => status.as_str(), // Pass through unknown status values
            };
            conditions.push(("status".into(), GqlValue::String(status_value.into())));
        }

        // Handle year range filters (simplified for now)
        if let Some(year_min) = filters.year_min {
            conditions.push(("yearBuilt".into(), GqlValue::Object(vec![
                ("gte".into(), GqlValue::Number(year_min as i64)),
            ])));
        }
        if let Some(year_max) = filters.year_max {
            conditions.push(("yearBuilt".into(), GqlValue::Object(vec![
                ("lte".into(), GqlValue::Number(year_max as i64)),
            ])));
        }
    }

    GqlValue::Object(conditions)
}

/// TIME AND STATUS FILTER NORMALIZATION
/// ====================================
/// Converts natural language time/status expressions into Intent.filters
/// This runs after parsing but before capability checks
impl IntentQueryTool {
    /// Normalize time and status filters from natural language user message
    /// Mutates intent.filters only - never overwrites existing explicit filters
    pub fn normalize_time_and_status_filters(intent: &mut Intent, user_message: &str) {
        // IMPORTANT:
        // This pass must NEVER drop existing filters (e.g., component_type from lowering).
        // We temporarily take the filters to mutate them, but must always restore them
        // if anything is set.
        let mut filters = intent.filters.take().unwrap_or_default();

        // TIME NORMALIZATION
        Self::normalize_time_filters(&mut filters, user_message);

        // STATUS NORMALIZATION
        Self::normalize_status_filters(&mut filters, user_message);

        // Restore filters if *any* filter field is set.
        // This ensures lowering-inserted filters like component_type are preserved.
        let any_filter_set = filters.component_type.is_some()
            || filters.building_type.is_some()
            || filters.year_min.is_some()
            || filters.year_max.is_some()
            || filters.completed.is_some()
            || filters.verified.is_some()
            || filters.status.is_some();

        if any_filter_set {
            intent.filters = Some(filters);
        }
    }

    /// Extract time-related filters from user message
    fn normalize_time_filters(filters: &mut Filters, user_message: &str) {
        let msg = user_message.to_lowercase();

        // Skip if year filters are already set (don't overwrite explicit filters)
        if filters.year_min.is_some() || filters.year_max.is_some() {
            return;
        }

        // "in year 2025" or "year 2025"
        if let Some(year) = Self::extract_single_year(&msg, &["in year ", "year ", "for "]) {
            filters.year_min = Some(year);
            filters.year_max = Some(year);
            return;
        }

        // "between 2020 and 2023"
        if let Some((min, max)) = Self::extract_year_range(&msg) {
            filters.year_min = Some(min);
            filters.year_max = Some(max);
            return;
        }

        // "this year" - for now, assume 2024 (can be made dynamic later)
        if msg.contains("this year") {
            filters.year_min = Some(2024);
            filters.year_max = Some(2024);
            return;
        }

        // "last year" - for now, assume 2023 (can be made dynamic later)
        if msg.contains("last year") {
            filters.year_min = Some(2023);
            filters.year_max = Some(2023);
            return;
        }
    }

    /// Extract status-related filters from user message
    fn normalize_status_filters(filters: &mut Filters, user_message: &str) {
        let msg = user_message.to_lowercase();

        // Skip if completed filter is already set
        if filters.completed.is_some() {
            return;
        }

        // Completed status indicators
        if msg.contains("completed") || msg.contains("done") {
            if msg.contains("not completed") || msg.contains("incomplete") {
                filters.completed = Some(false);
            } else {
                filters.completed = Some(true);
            }
            return;
        }

        // "open" typically means not completed
        if msg.contains("open") {
            filters.completed = Some(false);
            return;
        }
    }

    /// Extract single year from patterns like "in year 2025", "year 2024", "for 2023"
    fn extract_single_year(msg: &str, patterns: &[&str]) -> Option<i32> {
        for pattern in patterns {
            if let Some(start_idx) = msg.find(pattern) {
                let year_start = start_idx + pattern.len();
                let year_str = &msg[year_start..];

                // Extract digits until non-digit
                let year_digits: String = year_str.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();

                if let Ok(year) = year_digits.parse::<i32>() {
                    if year >= 2000 && year <= 2100 { // reasonable year bounds
                        return Some(year);
                    }
                }
            }
        }
        None
    }

    /// Extract year range from "between X and Y"
    fn extract_year_range(msg: &str) -> Option<(i32, i32)> {
        if let Some(between_idx) = msg.find("between ") {
            let after_between = &msg[between_idx + 8..];

            // Look for "and" to separate the years
            if let Some(and_idx) = after_between.find(" and ") {
                let year1_str = &after_between[..and_idx];
                let year2_str = &after_between[and_idx + 5..];

                // Extract first year
                let year1_digits: String = year1_str.chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect();

                // Extract second year
                let year2_digits: String = year2_str.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();

                if let (Ok(year1), Ok(year2)) = (year1_digits.parse::<i32>(), year2_digits.parse::<i32>()) {
                    if year1 >= 2000 && year1 <= 2100 && year2 >= 2000 && year2 <= 2100 && year1 <= year2 {
                        return Some((year1, year2));
                    }
                }
            }
        }
        None
    }
}


/// FORMAL INTENT STRUCTURE
/// =======================
/// Intent := {
///   action: Action,
///   target: Target,
///   scope: Scope,
///   attribute?: Attribute,
///   metric?: Metric
/// }

/// Scope definitions (grammar level - no validation)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[derive(JsonSchema)]
pub struct Scope {
    /// Scope type discriminator
    pub r#type: ScopeType,
    /// Building ID (only when type is "building") - resolved during normalization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub building_id: Option<String>,
    /// Real estate ID (only when type is "realEstate") - resolved during normalization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_estate_id: Option<String>,
    /// LEGACY: Building name - kept for backward compatibility during transition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub building_name: Option<String>,
    /// LEGACY: Real estate name - kept for backward compatibility during transition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_estate_name: Option<String>,
}

/// Scope type discriminator (grammar level)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScopeType {
    CurrentTeam,
    Building,
    RealEstate,
}

/// Attribute type (free-form string, not validated at grammar level)
pub type Attribute = String;

/// Metrics supported in grammar
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Metric {
    Count,
    Sum,
    Avg,
    TotalFloorArea,  // Legacy support
    SumCost,
    AvgCondition,
    EnergyCosts,
}

/// Metrics for aggregations
/// Query filters
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[derive(Default)]
pub struct Filters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub building_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub year_min: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub year_max: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Grouping options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(JsonSchema)]
pub enum GroupBy {
    Building,
    RealEstate,
    ComponentType,
}

/// Request context for compilation
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub team_id: String,
    pub user_id: String,
    pub current_building_id: Option<String>,
    pub current_real_estate_id: Option<String>,
}

/// Intent query tool for structured data retrieval
///
/// This tool executes GraphQL queries against configured endpoints.
/// It is strictly read-only and enforces safety constraints:
/// Intent-based query tool for structured data retrieval
/// Accepts Intent JSON, compiles to GraphQL server-side
pub struct IntentQueryTool {
    _registry: Box<dyn NameResolutionRegistry>,
}

impl IntentQueryTool {
    pub fn new(registry: Box<dyn NameResolutionRegistry>) -> Self {
        Self { _registry: registry }
    }
}

impl ToolEligibility for IntentQueryTool {
    fn is_eligible(&self, ctx: &ToolEligibilityContext) -> bool {
        // Honor explicit tool requests - always eligible if user explicitly asked for this tool
        if ctx.explicitly_requested {
            return true;
        }

        // Must have a successfully parsed Intent - no intent = not eligible
        let intent = match ctx.intent {
            Some(intent) => intent,
            None => return false,
        };

        // Check capabilities - this replaces the old hardcoded validation
        matches!(self.check_capabilities(intent), CapabilityDecision::Supported)
    }
}

// Special eligibility logic for query_intent tool (compiler output channel)
// This is handled at the executor level to ensure query_intent is always eligible
// when the intent passes validation + normalization

impl IntentQueryTool {
    /// Enrich intent with defaults based on action and target
    /// This is Phase 2 (Execution) enrichment
    pub fn enrich_intent(intent: &mut Intent) {
        // If metric is missing for Count/Aggregate, it's a candidate for enrichment
        if intent.metric.is_none() {
            match intent.action {
                Action::Count => {
                    intent.metric = Some(Metric::Count);
                }
                Action::Aggregate => {
                    // Default to Count for generic "how many" Aggregate requests
                    intent.metric = Some(Metric::Count);
                    intent.partial = Some(true);
                }
                _ => {}
            }
        }

        // If target is building but metric is count, ensure attribute is handled correctly
        // (This happens in compile step, but we signal partiality if no attribute/filter provided)
        if intent.action == Action::Count && intent.target == Target::Building && intent.attribute.is_none() && intent.filters.is_none() {
            // "How many buildings?" is a valid but sparse intent
        }
    }

    /// Parse Intent from tool arguments
    pub fn parse_intent(args: &Value) -> Result<Intent, ToolError> {
        let obj = args.as_object()
            .ok_or_else(|| ToolError::InvalidParameters("Arguments must be an object".to_string()))?;

        // Must have intent field
        let mut intent_value = obj.get("intent")
            .ok_or_else(|| ToolError::InvalidParameters("Missing required 'intent' field".to_string()))?
            .clone();

        // PRE-GRAMMAR SCRUB: Enforce time is NEVER scope
        Self::scrub_invalid_scopes(&mut intent_value);

        // First attempt: Parse Intent JSON directly (grammar level only)
        let parse_result = serde_json::from_value::<Intent>(intent_value.clone());

        let mut intent = match parse_result {
            Ok(intent) => intent,
            Err(e) => {
                return Err(ToolError::InvalidParameters(format!("Invalid intent format: {}", e)));
            }
        };

        // ORGANIZATION NORMALIZATION (Part 4)
        // Convert organization/company references to current_team
        Self::normalize_scope(&mut intent);

        // FILTER NORMALIZATION
        // Convert natural-language filter values into canonical enum-style values
        Self::normalize_filters(&mut intent);

        // STRIP STATUS ATTRIBUTES
        // Status words must only appear in filters, never attributes
        Self::strip_status_attributes(&mut intent);

        // Validate grammar structure (NOT capabilities)
        Self::validate_grammar(&intent)?;

        Ok(intent)
    }

    /// Normalize organization references to current_team scope
    /// Part 4: Organization names MUST NOT appear in Intent
    fn normalize_scope(intent: &mut Intent) {
        // If scope is organization/company related, convert to current_team
        // This logic runs BEFORE intent validation and is separate from grammar

        match intent.scope.r#type {
            ScopeType::CurrentTeam => {
                // Already normalized - keep current_team scope
            }
            ScopeType::Building | ScopeType::RealEstate => {
                // Valid structural scopes - no change needed
                // Organization name detection would go here in a full implementation
            }
        }
    }

    /// Normalize filter values into canonical, enum-safe formats
    /// This runs after parsing but before eligibility/capability checks
    /// Pre-grammar JSON scrub: Enforce time is NEVER scope
    /// Replace invalid scope types with current_team to prevent parsing failures
    fn scrub_invalid_scopes(intent_json: &mut serde_json::Value) {
        if let Some(scope_obj) = intent_json.get_mut("scope").and_then(|s| s.as_object_mut()) {
            // Check if scope.type is invalid (temporal or unknown)
            let has_invalid_scope = if let Some(scope_type) = scope_obj.get("type") {
                if let Some(type_str) = scope_type.as_str() {
                    // Invalid scope types that should be rejected/replaced
                    matches!(type_str,
                        "year" | "time" | "date" | "period" |
                        "start_date" | "end_date" | "range")
                } else {
                    false
                }
            } else {
                false
            };

            if has_invalid_scope {
                // Replace invalid scope with current_team
                scope_obj.clear();
                scope_obj.insert("type".to_string(), serde_json::Value::String("current_team".to_string()));
            }
        }
    }

    /// Strip status words from attributes BEFORE capability checks
    /// Status words must only appear in filters, never attributes
    fn strip_status_attributes(intent: &mut Intent) {
        if let Some(ref attr) = intent.attribute {
            let normalized = attr.trim().to_lowercase();
            if matches!(normalized.as_str(),
                "open" | "closed" | "completed" | "incomplete" |
                "unfinished" | "done" | "not completed" | "not_completed") {
                intent.attribute = None;
            }
        }
    }

    /// Normalize building scope based on explicit user message references
    /// Uses authoritative name resolution to confirm extracted building names
    /// FAILS if building name is referenced but cannot be resolved
    pub fn normalize_building_scope_with_early_failure(
        intent: &mut Intent,
        user_message: &str,
        registry: &dyn NameResolutionRegistry,
    ) -> Result<(), ToolError> {
        // Only override current_team scope
        if !matches!(intent.scope.r#type, ScopeType::CurrentTeam) {
            return Ok(());
        }

        let Some(candidate) = Self::extract_building_name_from_message(user_message) else {
            // No building reference found - keep current_team scope
            return Ok(());
        };

        let resolved = registry.resolve_building(&candidate)
            .ok_or_else(|| ToolError::InvalidParameters(
                format!("Building '{}' not found", candidate)
            ))?;

        // Convert to building scope with resolved ID
        intent.scope = Scope {
            r#type: ScopeType::Building,
            building_id: Some(resolved.id.clone()),
            building_name: Some(resolved.name.clone()), // Keep name for display/debugging
            real_estate_id: None,
            real_estate_name: None,
        };

        Ok(())
    }

    /// LEGACY: Normalize building scope based on explicit user message references
    /// Uses authoritative name resolution to confirm extracted building names
    /// WARNING: This method silently ignores unknown buildings - DO NOT USE in new code
    

    /// Extract building name from user message using conservative patterns
    fn extract_building_name_from_message(message: &str) -> Option<String> {
        let lower = message.to_lowercase();
    
        // ─────────────────────────────────────────────
        // Pattern 1: "building X"
        // ─────────────────────────────────────────────
        if let Some(idx) = lower.find("building ") {
            let after_building = &message[idx + "building ".len()..];
    
            if let Some(name) = Self::extract_building_name_from_words(after_building) {
                return Some(name);
            }
        }
    
        // ─────────────────────────────────────────────
        // Pattern 2: "in X"
        // ─────────────────────────────────────────────
        if let Some(idx) = lower.find(" in ") {
            let after_in = &message[idx + 4..]; // keep original casing
            let mut words = after_in.split_whitespace();
    
            if let Some(first_word) = words.next() {
                let first_lower = first_word.to_lowercase();

                if !Self::is_rejected_building_candidate(&first_lower) {
                    let remaining: Vec<&str> = words.take(2).collect();
                    let candidate = std::iter::once(first_word)
                        .chain(remaining.iter().copied())
                        .collect::<Vec<_>>()
                        .join(" ");

                    if let Some(name) = Self::extract_building_name_from_words(&candidate) {
                        return Some(name);
                    }
                }
            }
        }
    
        None
    }

    /// Extract building name from word sequence with stop-word termination
    fn extract_building_name_from_words(text: &str) -> Option<String> {
        let mut words = text.split_whitespace();
        let mut result = Vec::new();

        while result.len() < 3 {
            if let Some(word) = words.next() {
                let clean_word = word.trim_end_matches(&['?', '!', '.', ','][..]);
                if clean_word.is_empty() {
                    break;
                }

                let lower = clean_word.to_lowercase();
                // Stop at termination words
                if matches!(lower.as_str(), "that" | "which" | "with" | "where" | "who" | "has" | "have") {
                    break;
                }

                result.push(clean_word.to_string());
            } else {
                break;
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result.join(" "))
        }
    }

    /// Check if a word should be rejected as a building name candidate
    fn is_rejected_building_candidate(word: &str) -> bool {
        // Reject if starts with digit
        if word.chars().next().unwrap_or(' ').is_ascii_digit() {
            return true;
        }

        // Reject common stop words
        matches!(word,
            "the" | "a" | "an" | "my" | "this" | "that" |
            "today" | "yesterday" | "tomorrow" | "year"
        )

        // Reject temporal words (would be extended for more temporal terms)
    }

    fn normalize_filters(intent: &mut Intent) {
        if let Some(ref mut filters) = intent.filters {
            // Canonical status normalization: convert both completed and status to canonical status
            if let Some(ref mut status) = filters.status {
                let canonical_status = match status.as_str() {
                    "not completed" | "incomplete" | "unfinished" | "open" => "not_completed",
                    "completed" | "done" | "finished" | "closed" => "completed",
                    other => other, // Keep unknown values as-is for now
                };
                *status = canonical_status.to_string();
                // Clear completed field since we're using canonical status
                filters.completed = None;
            } else if let Some(completed) = filters.completed {
                // Convert completed boolean to canonical status string
                filters.status = Some(if completed { "completed" } else { "not_completed" }.to_string());
                // Clear completed field since we're using canonical status
                filters.completed = None;
            }

            // Future: Add normalization for other filter types
            // - Time filters (year, last year, 2024)
            // - Boolean filters (verified, active)
            // - Numeric ranges
        }
    }

    /// Validate intent JSON shape BEFORE parsing
    /// Rejects structured attributes and unknown top-level fields
    pub fn validate_intent_shape(intent_json: &serde_json::Value) -> Result<(), ToolError> {
        let obj = intent_json.as_object()
            .ok_or_else(|| ToolError::InvalidParameters(
                "Intent must be a JSON object".to_string()
            ))?;

        // Check for forbidden "attributes" (plural) field
        if obj.contains_key("attributes") {
            return Err(ToolError::InvalidParameters(
                "Invalid intent: 'attributes' (plural) field is forbidden. Use 'attribute' (singular) as a string.".to_string()
            ));
        }

        // Validate "attribute" field if present
        if let Some(attr_value) = obj.get("attribute") {
            // Must be either null (for Option<String>) or a string
            if !attr_value.is_null() && !attr_value.is_string() {
                return Err(ToolError::InvalidParameters(
                    "Invalid intent: 'attribute' must be a string or null, not an object or array.".to_string()
                ));
            }
        }

        // Check for unknown top-level fields (allow only known intent fields)
        let allowed_fields = [
            "action", "target", "entity", "scope", "attribute", "metric",
            "filters", "limit", "group_by"
        ];

        for key in obj.keys() {
            if !allowed_fields.contains(&key.as_str()) {
                return Err(ToolError::InvalidParameters(
                    format!("Invalid intent: unknown field '{}'. Valid fields are: {}", key, allowed_fields.join(", "))
                ));
            }
        }

        Ok(())
    }  
    

    /// Validate intent grammar structure (NOT capabilities)
    /// This only validates the grammar - capabilities are checked separately
    fn validate_grammar(intent: &Intent) -> Result<(), ToolError> {
        // Validate scope structure (grammar-level only)
        match intent.scope.r#type {
            ScopeType::Building => {
                // Building scope requires either name (for parsing) or ID (after normalization)
                if intent.scope.building_name.is_none() && intent.scope.building_id.is_none() {
                    return Err(ToolError::InvalidParameters(
                        "Scope type 'building' requires either 'building_name' or 'building_id' field".to_string()
                    ));
                }
                if intent.scope.real_estate_name.is_some() || intent.scope.real_estate_id.is_some() {
                    return Err(ToolError::InvalidParameters(
                        "Scope type 'building' cannot have real estate fields".to_string()
                    ));
                }
            },
            ScopeType::RealEstate => {
                // Real estate scope requires either name (for parsing) or ID (after normalization)
                if intent.scope.real_estate_name.is_none() && intent.scope.real_estate_id.is_none() {
                    return Err(ToolError::InvalidParameters(
                        "Scope type 'realEstate' requires either 'real_estate_name' or 'real_estate_id' field".to_string()
                    ));
                }
                if intent.scope.building_name.is_some() || intent.scope.building_id.is_some() {
                    return Err(ToolError::InvalidParameters(
                        "Scope type 'realEstate' cannot have building fields".to_string()
                    ));
                }
            },
            ScopeType::CurrentTeam => {
                // CurrentTeam cannot have any name or ID fields
                if intent.scope.building_name.is_some() || intent.scope.building_id.is_some() ||
                   intent.scope.real_estate_name.is_some() || intent.scope.real_estate_id.is_some() {
                    return Err(ToolError::InvalidParameters(
                        "Scope type 'current_team' cannot have any name or ID fields".to_string()
                    ));
                }
            },
        }

        // Grammar validation complete - capabilities checked separately
        Ok(())
    }

    /// Check if intent is supported by capabilities
    /// This replaces the old semantic validation logic
    pub fn check_capabilities(&self, intent: &Intent) -> CapabilityDecision {
        // Use the default capability registry
        let registry = CapabilityRegistry::new();
        registry.supports(intent)
    }

    /// Validate filters using the filter capability matrix
    pub fn validate_filters(intent: &Intent) -> Result<(), FilterValidationError> {
        if let Some(ref filters) = intent.filters {
            let canonical_filters = filters.to_canonical_map();
            let registry = FilterCapabilityRegistry::new();
            registry.validate_filters(&intent.target, &intent.action, &canonical_filters)
        } else {
            // No filters - always valid
            Ok(())
        }
    }

    /// Compile Intent to GraphQL query using AST
    fn compile_intent_to_graphql(
        intent: &Intent
    ) -> Result<String, ToolError> {
        use crate::tools::graphql::ast::{GqlQuery, GqlField, GqlValue};

        // Build where clause from intent (scope + filters)
        let where_clause = build_where_clause(intent);

        // Debug view
        if let GqlValue::Object(ref conditions) = where_clause {
            tracing::info!(
                "🧱 Compiling GraphQL where clause: {:?}",
                conditions.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>()
            );
        }

        // Build GraphQL query based on action and target using AST
        let root_field = match (&intent.action, &intent.target, intent.attribute.as_deref()) {
            (Action::Count, Target::Component, _) | (Action::Aggregate, Target::Component, _) if intent.metric == Some(Metric::Count) => {
                GqlField::new("components")
                    .arg("where", where_clause)
                    .select(
                        GqlField::new("_count")
                            .select(GqlField::new("id"))
                    )
            },

            (Action::Count, Target::Building, _) | (Action::Aggregate, Target::Building, _) if intent.metric == Some(Metric::Count) => {
                GqlField::new("buildings")
                    .arg("where", where_clause)
                    .select(
                        GqlField::new("_count")
                            .select(GqlField::new("id"))
                    )
            },

            (Action::Count, Target::Measure, _) | (Action::Aggregate, Target::Measure, _) if intent.metric == Some(Metric::Count) => {
                GqlField::new("measures")
                    .arg("where", where_clause)
                    .select(
                        GqlField::new("_count")
                            .select(GqlField::new("id"))
                    )
            },

            (Action::List, Target::Building, _) => {
                GqlField::new("buildings")
                    .arg("where", where_clause)
                    .select(GqlField::new("id"))
                    .select(GqlField::new("name"))
            },

            (Action::Get, Target::Building, _) => {
                GqlField::new("buildings")
                    .arg("where", where_clause)
                    .select(GqlField::new("id"))
                    .select(GqlField::new("name"))
                    .select(GqlField::new("address"))
                    .select(GqlField::new("yearBuilt"))
            },

            (Action::Aggregate, Target::Building, _) if intent.metric == Some(Metric::TotalFloorArea) => {
                GqlField::new("buildings")
                    .arg("where", where_clause)
                    .select(
                        GqlField::new("_sum")
                            .select(GqlField::new("totalFloorArea"))
                    )
            },

            (Action::Exists, target, _) => {
                let field_name = match target {
                    Target::Building => "buildings",
                    Target::Component => "components",
                    Target::Measure => "measures",
                    Target::RealEstate => "realEstates",
                    Target::Plan => "plans",
                    Target::Project => "projects",
                };
                GqlField::new(field_name)
                    .arg("where", where_clause)
                    .arg("take", GqlValue::Number(1))
                    .select(GqlField::new("id"))
            },

            _ => return Err(ToolError::InvalidParameters(
                format!(
                    "Unsupported execution shape: action={:?}, target={:?}, metric={:?}. Attributes must be lowered to filters before compilation.",
                    intent.action,
                    intent.target,
                    intent.metric
                )
            )),
        };

        // Apply limit if specified
        let root_field = if let Some(limit) = intent.limit {
            root_field.arg("take", GqlValue::Number(limit.into()))
        } else {
            root_field
        };

        let query = GqlQuery { root: root_field };
        Ok(query.pretty_print(0))
    }
}


impl ToolPort for IntentQueryTool {
    fn name(&self) -> &str {
        "query_intent"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn execute(&self, input: ToolInput, _ctx: &crate::runtime::executor::ExecutorContext) -> Result<ToolOutput, ToolError> {
        // ─────────────────────────────────────────────
        // 0. JSON SCHEMA VALIDATION (HARD CONTRACT)
        // ─────────────────────────────────────────────
        schema::intent_schema().validate(&input.payload)
            .map_err(|e| ToolError::InvalidParameters(format!(
                "Intent schema validation failed: {}",
                e
            )))?;

        // ─────────────────────────────────────────────
        // 1. Grammar parsing (structure only)
        // ─────────────────────────────────────────────
        let mut intent = if let Some(parsed_intent) = input.parsed_intent {
            // Use the pre-parsed and normalized intent from executor
            parsed_intent
        } else {
            // Fallback: parse from payload (should not happen for normalized flow)
            // EXECUTOR GUARDRAIL: Extract and validate intent shape BEFORE parsing
            let payload_obj = input.payload.as_object()
                .ok_or_else(|| ToolError::InvalidParameters("Tool payload must be a JSON object".to_string()))?;

            let intent_value = payload_obj.get("intent")
                .ok_or_else(|| ToolError::InvalidParameters("Missing required 'intent' field in tool payload".to_string()))?;

            Self::validate_intent_shape(intent_value)?;
            Self::parse_intent(&input.payload)?
        };

        // ─────────────────────────────────────────────
        // 1.5 Execution Enrichment (Phase 2)
        // Set defaults and handle partial intents
        // ─────────────────────────────────────────────
        Self::enrich_intent(&mut intent);


        // ─────────────────────────────────────────────
        // Intent lowering (logical → physical)
        // Must run AFTER scope resolution, BEFORE compilation
        // ─────────────────────────────────────────────
        intent.lower().map_err(|e| ToolError::InvalidParameters(
            format!("Intent lowering failed: {}", e)
        ))?;

        // INVARIANT 2 ENFORCEMENT: Ensure scope resolution succeeded before compilation
        match intent.scope.r#type {
            ScopeType::Building => {
                if intent.scope.building_id.is_none() {
                    return Err(ToolError::InvalidParameters(
                        format!("Scope resolution failed: Building '{}' could not be resolved to an ID", 
                            intent.scope.building_name.as_deref().unwrap_or("unknown"))
                    ));
                }
            }
            ScopeType::RealEstate => {
                if intent.scope.real_estate_id.is_none() {
                    return Err(ToolError::InvalidParameters(
                        format!("Scope resolution failed: Real Estate '{}' could not be resolved to an ID", 
                            intent.scope.real_estate_name.as_deref().unwrap_or("unknown"))
                    ));
                }
            }
            ScopeType::CurrentTeam => {}
        }

        // SCOPE NORMALIZATION GUARANTEE: If building mentioned but scope still current_team, reject
        if input.user_message.to_lowercase().contains("building")
            && matches!(intent.scope.r#type, crate::tools::graphql::ScopeType::CurrentTeam)
        {
            return Err(ToolError::InvalidParameters(
                "Building mentioned but scope not resolved".to_string()
            ));
        }

        // ─────────────────────────────────────────────
        // 2. Natural-language normalization
        //    (time + status extraction)
        // ─────────────────────────────────────────────
        Self::normalize_time_and_status_filters(&mut intent, &input.user_message);

        // NOTE: Building scope normalization now happens BEFORE tool eligibility check in executor
        // This ensures name resolution fails early on unknown buildings

        // ─────────────────────────────────────────────
        // 3. Filter validation (FilterCapabilityMatrix)
        // ─────────────────────────────────────────────
        Self::validate_filters(&intent)
            .map_err(|e| ToolError::InvalidParameters(format!(
                "Filter validation failed: {:?}",
                e
            )))?;

        // ─────────────────────────────────────────────
        // 4. Request context (auth/session placeholder)
        // ─────────────────────────────────────────────
        let request_ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        // ─────────────────────────────────────────────
        // 5. Compile intent → GraphQL
        // ─────────────────────────────────────────────
        let graphql_query =
            Self::compile_intent_to_graphql(&intent)?;

        tracing::info!("🕸️ Compiled GraphQL query:\n{}", graphql_query);

        // ─────────────────────────────────────────────
        // 6. Execute (placeholder)
        // ─────────────────────────────────────────────
        let response = serde_json::json!({
            "data": {
                "message": "Intent compiled to GraphQL and would be executed",
                "intent": intent,
                "compiled_query": graphql_query,
                "request_context": {
                    "team_id": request_ctx.team_id,
                    "user_id": request_ctx.user_id
                },
                "note": "This is a placeholder response. Actual GraphQL execution requires backend integration."
            },
            "summary": format!(
                "Executed intent query for {:?} {:?} (attribute: {:?})",
                intent.action,
                intent.target,
                intent.attribute
            )
        });

        Ok(ToolOutput { payload: response })
    }
}


pub mod relations;
pub mod intent_lowering;

/// Schema generation utilities
impl IntentQueryTool {
    /// Generate JSON Schema from Rust types and write to file
    /// This ensures the schema stays in sync with the Rust intent types
    pub fn generate_intent_schema() -> Result<(), Box<dyn std::error::Error>> {
        use schemars::schema_for;
        use std::fs;
        use std::path::Path;

        let schema = schema_for!(IntentEnvelope);
        let schema_json = serde_json::to_string_pretty(&schema)?;

        let schema_path = Path::new("schemas/intent.schema.json");
        fs::create_dir_all(schema_path.parent().unwrap_or(Path::new(".")))?;
        fs::write(schema_path, schema_json)?;

    println!("Generated intent schema at: {}", schema_path.display());
    Ok(())
    }
}

pub mod ast;


