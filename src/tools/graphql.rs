use crate::runtime::toolport::{ToolPort, ToolInput, ToolOutput, ToolError, ToolEligibility, ToolEligibilityContext};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use schemars::JsonSchema;
use crate::core::schema;
use crate::core::relation_map;



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
}

/// Intent envelope for JSON Schema generation
/// This is the root schema that wraps the intent
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IntentEnvelope {
    pub intent: Intent,
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

            // Check attribute pattern
            match &capability.attribute {
                AttributePattern::Any => {
                    // Any attribute allowed - continue checking
                }
                AttributePattern::None => {
                    if intent.attribute.is_some() {
                        continue; // Attribute provided but not allowed
                    }
                }
                AttributePattern::Exact(expected) => {
                    if intent.attribute.as_ref() != Some(expected) {
                        continue; // Attribute doesn't match exactly
                    }
                }
            }

            // Check metric pattern
            match &capability.metric {
                MetricPattern::Any => {
                    // Any metric allowed - continue checking
                }
                MetricPattern::None => {
                    if intent.metric.is_some() {
                        continue; // Metric provided but not allowed
                    }
                }
                MetricPattern::Exact(expected) => {
                    if intent.metric.as_ref() != Some(expected) {
                        continue; // Metric doesn't match exactly
                    }
                }
            }

            // All patterns matched - this capability supports the intent
            return CapabilityDecision::Supported;
        }

        // No matching capability found
        CapabilityDecision::Unsupported {
            reason: format!(
                "Unsupported combination: action={:?}, target={:?}, attribute={:?}, metric={:?}",
                intent.action, intent.target, intent.attribute, intent.metric
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

/// TIME AND STATUS FILTER NORMALIZATION
/// ====================================
/// Converts natural language time/status expressions into Intent.filters
/// This runs after parsing but before capability checks
impl IntentQueryTool {
    /// Normalize time and status filters from natural language user message
    /// Mutates intent.filters only - never overwrites existing explicit filters
    pub fn normalize_time_and_status_filters(intent: &mut Intent, user_message: &str) {
        let mut filters = intent.filters.take().unwrap_or_default();

        // TIME NORMALIZATION
        Self::normalize_time_filters(&mut filters, user_message);

        // STATUS NORMALIZATION
        Self::normalize_status_filters(&mut filters, user_message);

        // Only set filters if we actually found something to normalize
        if filters.year_min.is_some() || filters.year_max.is_some() || filters.completed.is_some() {
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

/// ATTRIBUTE NORMALIZATION
/// =======================
/// Maps common nested domain objects to correct target + attribute format
/// This handles cases where the LLM might incorrectly emit nested objects as targets
impl IntentQueryTool {
    /// Attempt to normalize raw intent JSON by fixing invalid targets
    /// Returns the normalized JSON value if changes were made, None otherwise
    pub fn normalize_intent_json(intent_json: &serde_json::Value) -> Option<serde_json::Value> {
        let mut normalized = intent_json.clone();

        // Map of invalid targets that should be attributes
        let nested_to_attribute: std::collections::HashMap<&str, (&str, &str)> = [
            // (invalid_target, (correct_target, attribute_name))
            ("window", ("building", "windows")),
            ("door", ("building", "doors")),
            ("floor", ("building", "floors")),
            ("room", ("building", "rooms")),
            ("pipe", ("building", "pipes")),
            ("sensor", ("component", "sensors")),
            ("condition", ("component", "condition")),
            ("floor_area", ("building", "floorArea")),
            ("floorarea", ("building", "floorArea")),
            ("energy_cost", ("building", "energyCosts")),
            ("energycost", ("building", "energyCosts")),
        ].iter().cloned().collect();

        if let Some(intent_obj) = normalized.as_object_mut() {
            if let Some(target_value) = intent_obj.get("target").or_else(|| intent_obj.get("entity")) {
                if let Some(target_str) = target_value.as_str() {
                    if let Some(&(correct_target, attribute)) = nested_to_attribute.get(target_str.to_lowercase().as_str()) {
                        // Replace invalid target with correct target
                        intent_obj.insert("target".to_string(), serde_json::json!(correct_target));
                        intent_obj.insert("attribute".to_string(), serde_json::json!(attribute));

                        // Remove entity field if it exists (legacy compatibility)
                        intent_obj.remove("entity");

                        return Some(normalized);
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
    registry: Box<dyn NameResolutionRegistry>,
}

impl IntentQueryTool {
    pub fn new(registry: Box<dyn NameResolutionRegistry>) -> Self {
        Self { registry }
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

impl IntentQueryTool {
    /// Check if message appears to be conversational/small talk
    fn is_conversational(message: &str) -> bool {
        let conversational_phrases = [
            "hello", "hi", "hey", "how are you", "what's up", "good morning",
            "good afternoon", "good evening", "nice to meet", "thank you",
            "thanks", "please", "sorry", "excuse me", "can you help",
            "opinion", "think", "feel", "recommend", "suggest",
            "what do you", "can you", "would you", "should I",
            "tool", "function", "command", "execute", "run",
            "about", "believe", "suppose", "guess", "wonder"
        ];

        conversational_phrases
            .iter()
            .any(|phrase| message.contains(phrase))
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
            Err(_) => {
                // Second attempt: Try to normalize the JSON first
                if let Some(normalized_json) = Self::normalize_intent_json(&intent_value) {
                    serde_json::from_value(normalized_json)
                        .map_err(|e| ToolError::InvalidParameters(format!("Invalid intent format after normalization: {}", e)))?
                } else {
                    // No normalization possible, return original error
                    return Err(ToolError::InvalidParameters(format!("Invalid intent format: {}", parse_result.unwrap_err())));
                }
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
    pub fn normalize_explicit_building_scope(
        intent: &mut Intent,
        user_message: &str,
        registry: &dyn NameResolutionRegistry,
    ) -> Result<(), ToolError> {
        // Only override current_team scope
        if !matches!(intent.scope.r#type, ScopeType::CurrentTeam) {
            return Ok(());
        }

        let Some(candidate) = Self::extract_building_name_from_message(user_message) else {
            return Ok(());
        };

        let resolved = registry.resolve_building(&candidate)
            .ok_or_else(|| ToolError::InvalidParameters(
                format!("Building '{}' not found", candidate)
            ))?;

        // Replace scope deterministically
        intent.scope = Scope {
            r#type: ScopeType::Building,
            building_id: Some(resolved.id.clone()),
            building_name: Some(resolved.name.clone()),
            real_estate_id: None,
            real_estate_name: None,
        };

        Ok(())
    }

    /// Extract building name from user message using conservative patterns
    fn extract_building_name_from_message(message: &str) -> Option<String> {
        let lower = message.to_lowercase();
    
        // ─────────────────────────────────────────────
        // Pattern 1: "building X"
        // ─────────────────────────────────────────────
        if let Some(idx) = lower.find("building ") {
            let after_building = &message[idx + "building ".len()..];
    
            if let Some(name) = Self::extract_building_name_from_words(after_building) {
                if name.chars().next().unwrap_or(' ').is_uppercase() {
                    return Some(name);
                }
            }
        }
    
        // ─────────────────────────────────────────────
        // Pattern 2: "in X"
        // ─────────────────────────────────────────────
        if let Some(idx) = lower.find(" in ") {
            let after_in = &message[idx + 4..]; // keep original casing
            let mut words = after_in.split_whitespace();
    
            if let Some(first_word) = words.next() {
                if first_word.chars().next().unwrap_or(' ').is_uppercase() {
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

    /// Compile Intent to GraphQL query
    fn compile_intent_to_graphql(
        intent: &Intent,
        ctx: &RequestContext,
        registry: &dyn NameResolutionRegistry
    ) -> Result<String, ToolError> {
        // Resolve scope using RelationMap
        let scope_constraint = Self::compile_scope_constraint(&intent.target, &intent.scope, ctx, registry)?;

        // Compile filters if present
        let filters_constraint = intent.filters.as_ref()
            .and_then(|filters| Self::compile_filters_to_json(filters));

        // Combine scope and filters with AND semantics (merge JSON objects)
        let where_json = match (&scope_constraint, &filters_constraint) {
            (Some(scope), Some(filters)) => {
                if let (serde_json::Value::Object(mut scope_obj), serde_json::Value::Object(filters_obj)) = (scope.clone(), filters.clone()) {
                    for (k, v) in filters_obj {
                        scope_obj.insert(k, v);
                    }
                    serde_json::Value::Object(scope_obj)
                } else {
                    scope.clone()
                }
            },
            (Some(scope), None) => scope.clone(),
            (None, Some(filters)) => filters.clone(),
            (None, None) => serde_json::json!({}),
        };

        // Convert to GraphQL where clause string
        let where_clause = serde_json::to_string(&where_json)
            .map_err(|e| ToolError::InvalidParameters(format!("Failed to serialize where clause: {}", e)))?
            .trim_matches('"')
            .to_string();

        // Build GraphQL query based on action and target
        let query_body = match (&intent.action, &intent.target, intent.attribute.as_deref()) {
            (Action::Count, Target::Component, None) => {
                format!("components(where: {{{}}}) {{ _count {{ id }} }}", where_clause)
            },

            (Action::Count, Target::Building, None) => {
                format!("buildings(where: {{{}}}) {{ _count {{ id }} }}", where_clause)
            },

            (Action::Count, Target::Building, Some(attr)) => {
                format!(
                    "buildings(where: {{{}}}) {{ {} {{ _count {{ id }} }} }}",
                    where_clause,
                    attr
                )
            },

            (Action::Count, Target::Measure, None) => {
                format!(
                    "measures(where: {{{}}}) {{ _count {{ id }} }}",
                    where_clause
                )
            },

            (Action::List, Target::Building, None) => {
                format!("buildings(where: {{{}}}) {{ id name }}", where_clause)
            },
            (Action::Get, Target::Building, None) => {
                format!("buildings(where: {{{}}}) {{ id name address yearBuilt }}", where_clause)
            },
            (Action::Aggregate, Target::Component, None) => {
                if let Some(Metric::Count) = intent.metric {
                    format!("components(where: {{{}}}) {{ _count {{ id }} }}", where_clause)
                } else {
                    return Err(ToolError::InvalidParameters("Unsupported aggregation".to_string()));
                }
            },
            (Action::Aggregate, Target::Building, None) => {
                if let Some(Metric::TotalFloorArea) = intent.metric {
                    format!("buildings(where: {{{}}}) {{ _sum {{ totalFloorArea }} }}", where_clause)
                } else {
                    return Err(ToolError::InvalidParameters("Unsupported building aggregation".to_string()));
                }
            },
            (Action::Aggregate, Target::Building, Some(attr)) => {
                // Aggregate attribute of buildings (e.g., count windows in buildings)
                match intent.metric {
                    Some(Metric::Count) => format!("buildings(where: {{{}}}) {{ {} {{ _count {{ id }} }} }}", where_clause, attr),
                    Some(Metric::Sum) => format!("buildings(where: {{{}}}) {{ {} {{ _sum {{ value }} }} }}", where_clause, attr),
                    Some(Metric::Avg) => format!("buildings(where: {{{}}}) {{ {} {{ _avg {{ value }} }} }}", where_clause, attr),
                    _ => return Err(ToolError::InvalidParameters(format!("Unsupported metric for attribute '{}': {:?}", attr, intent.metric))),
                }
            },
            _ => return Err(ToolError::InvalidParameters(format!("Unsupported action/target combination: {:?}/{:?} with attribute {:?}", intent.action, intent.target, intent.attribute))),
        };

        // Apply limit if specified
        let limited_query = if let Some(limit) = intent.limit {
            format!("{}(take: {})", query_body, limit)
        } else {
            query_body
        };

        Ok(format!("query {{ {} }}", limited_query))
    }

    /// Compile scope constraint using RelationMap
    fn compile_scope_constraint(
        target: &Target,
        scope: &Scope,
        ctx: &RequestContext,
        registry: &dyn NameResolutionRegistry
    ) -> Result<Option<serde_json::Value>, ToolError> {
        match &scope.r#type {
            ScopeType::CurrentTeam => {
                // CurrentTeam scope is always a direct teamId filter
                Ok(Some(serde_json::json!({ "teamId": ctx.team_id })))
            },
            ScopeType::Building | ScopeType::RealEstate => {
                // Get already-resolved scope ID (resolution happened during normalization)
                let resolved_id = Self::get_scope_id(scope)?
                    .ok_or_else(|| ToolError::InvalidParameters("Expected resolved ID for non-CurrentTeam scope".to_string()))?;

                // Look up relation in RelationMap
                let relation = relation_map::get_relation_path(target, &scope.r#type)
                    .ok_or_else(|| ToolError::InvalidParameters(
                        format!("No relation defined for target {:?} with scope type {:?}", target, scope.r#type)
                    ))?;

                // Generate constraint using relation path
                let constraint = relation_map::compile_scope_constraint(relation, &resolved_id);
                Ok(Some(constraint))
            },
        }
    }

    /// Get scope ID from already-resolved scope (for non-CurrentTeam scopes)
    /// INVARIANT: Scope IDs must be resolved during normalization, never here
    fn get_scope_id(scope: &Scope) -> Result<Option<String>, ToolError> {
        match &scope.r#type {
            ScopeType::CurrentTeam => Ok(None), // CurrentTeam doesn't need ID resolution
            ScopeType::Building => {
                scope.building_id.as_ref()
                    .ok_or_else(|| ToolError::InvalidParameters("Building scope requires resolved building_id".to_string()))
                    .map(|id| Some(id.clone()))
            },
            ScopeType::RealEstate => {
                scope.real_estate_id.as_ref()
                    .ok_or_else(|| ToolError::InvalidParameters("RealEstate scope requires resolved real_estate_id".to_string()))
                    .map(|id| Some(id.clone()))
            },
        }
    }

    /// Compile filters into GraphQL where clause JSON
    fn compile_filters_to_json(filters: &Filters) -> Option<serde_json::Value> {
        let mut filter_obj = serde_json::Map::new();

        // Handle status filter
        if let Some(ref status_str) = filters.status {
            let status_value = match status_str.as_str() {
                "completed" => "COMPLETED",
                "not_completed" => "NOT_COMPLETED",
                _ => return None, // Invalid status, skip filters
            };
            filter_obj.insert("status".to_string(), serde_json::json!(status_value));
        }

        // Handle year range filters
        if let Some(year_min) = filters.year_min {
            filter_obj.insert("yearBuilt".to_string(), serde_json::json!({ "gte": year_min }));
        }
        if let Some(year_max) = filters.year_max {
            // If yearBuilt already exists (from year_min), merge the constraints
            if let Some(existing) = filter_obj.get_mut("yearBuilt") {
                if let serde_json::Value::Object(ref mut obj) = existing {
                    obj.insert("lte".to_string(), serde_json::json!(year_max));
                }
            } else {
                filter_obj.insert("yearBuilt".to_string(), serde_json::json!({ "lte": year_max }));
            }
        }

        // Handle other filters as needed
        // For now, only status and year filters are supported

        if filter_obj.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(filter_obj))
        }
    }
}


impl ToolPort for IntentQueryTool {
    fn name(&self) -> &str {
        "query_intent"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn execute(&self, input: ToolInput, ctx: &crate::runtime::executor::ExecutorContext) -> Result<ToolOutput, ToolError> {
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
        let mut intent = Self::parse_intent(&input.payload)?;

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
            Self::compile_intent_to_graphql(&intent, &request_ctx, &*self.registry)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_eligibility_domain_entities() {
        let tool = IntentQueryTool::new(create_test_registry());

        // Should be eligible - valid intent
        let intent = Intent {
            action: Action::List,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };
        let ctx = ToolEligibilityContext {
            user_message: "find all buildings in the project",
            assistant_message: "",
            explicitly_requested: false,
            persona: "default",
            intent: Some(&intent),
        };
        assert!(tool.is_eligible(&ctx));

        // Should be eligible - valid intent with different entity
        let intent = Intent {
            action: Action::Get,
            target: Target::Component,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };
        let ctx = ToolEligibilityContext {
            user_message: "get component details for measure 123",
            assistant_message: "",
            explicitly_requested: false,
            persona: "default",
            intent: Some(&intent),
        };
        assert!(tool.is_eligible(&ctx));

        // Should be eligible - valid intent
        let intent = Intent {
            action: Action::Aggregate,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: None, // Test case - not resolved
                building_name: Some("X".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: Some(Metric::TotalFloorArea),
            filters: None,
            limit: None,
            group_by: None,
        };
        let ctx = ToolEligibilityContext {
            user_message: "what is the total floor area of building X",
            assistant_message: "",
            explicitly_requested: false,
            persona: "default",
            intent: Some(&intent),
        };
        assert!(tool.is_eligible(&ctx));
    }

    #[test]
    fn test_eligibility_conversational_rejection() {
        let tool = IntentQueryTool::new(create_test_registry());

        // Should be rejected - no intent (conversational)
        let ctx = ToolEligibilityContext {
            user_message: "hello how are you",
            assistant_message: "",
            explicitly_requested: false,
            persona: "default",
            intent: None,
        };
        assert!(!tool.is_eligible(&ctx));

        // Should be rejected - no intent (opinion request)
        let ctx = ToolEligibilityContext {
            user_message: "what do you think about the new plan",
            assistant_message: "",
            explicitly_requested: false,
            persona: "default",
            intent: None,
        };
        assert!(!tool.is_eligible(&ctx));

        // Should be rejected - no intent (tool meta discussion)
        let ctx = ToolEligibilityContext {
            user_message: "tell me about the graphql tool",
            assistant_message: "",
            explicitly_requested: false,
            persona: "default",
            intent: None,
        };
        assert!(!tool.is_eligible(&ctx));
    }

    #[test]
    fn test_parse_intent_valid() {
        let args = json!({
            "intent": {
                "action": "count",
                "target": "component",
                "scope": { "type": "building", "buildingName": "Test Building" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok());
        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Component));
    }

    #[test]
    fn test_parse_intent_count_building_attribute() {
        let args = json!({
            "intent": {
                "action": "count",
                "target": "building",
                "scope": { "type": "current_team" },
                "attribute": "windows"
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Intent parsing should succeed: {:?}", result.err());
        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Building));
        assert_eq!(intent.attribute, Some("windows".to_string()));

        // Test compilation too
        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let compile_result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(compile_result.is_ok(), "Compilation should succeed: {:?}", compile_result.err());
        let query = compile_result.unwrap();
        assert!(query.contains("buildings"));
        assert!(query.contains("windows"));
        assert!(query.contains("_count"));
        assert!(query.contains("team-123"));
        assert!(query.contains("query {"));
    }

    #[test]
    fn test_capability_matrix_supported_intents() {
        let registry = CapabilityRegistry::new();

        // Test supported: count components
        let intent1 = Intent {
            action: Action::Count,
            target: Target::Component,
            attribute: None,
            scope: Scope { r#type: ScopeType::CurrentTeam, building_id: None, building_name: None, real_estate_id: None, real_estate_name: None },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };
        assert!(matches!(registry.supports(&intent1), CapabilityDecision::Supported));

        // Test supported: count building windows
        let intent2 = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: Some("windows".to_string()),
            scope: Scope { r#type: ScopeType::CurrentTeam, building_id: None, building_name: None, real_estate_id: None, real_estate_name: None },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };
        assert!(matches!(registry.supports(&intent2), CapabilityDecision::Supported));

        // Test supported: aggregate building floor area
        let intent3 = Intent {
            action: Action::Aggregate,
            target: Target::Building,
            attribute: Some("floorArea".to_string()),
            scope: Scope { r#type: ScopeType::CurrentTeam, building_id: None, building_name: None, real_estate_id: None, real_estate_name: None },
            metric: Some(Metric::Sum),
            filters: None,
            limit: None,
            group_by: None,
        };
        assert!(matches!(registry.supports(&intent3), CapabilityDecision::Supported));
    }

    #[test]
    fn test_capability_matrix_unsupported_intents() {
        let registry = CapabilityRegistry::new();

        // Test unsupported: aggregate components (not allowed)
        let intent1 = Intent {
            action: Action::Aggregate,
            target: Target::Component,
            attribute: None,
            scope: Scope { r#type: ScopeType::CurrentTeam, building_id: None, building_name: None, real_estate_id: None, real_estate_name: None },
            metric: Some(Metric::Sum), // Components don't support aggregation
            filters: None,
            limit: None,
            group_by: None,
        };
        assert!(matches!(registry.supports(&intent1), CapabilityDecision::Unsupported { .. }));

        // Test unsupported: unknown target
        let intent2 = Intent {
            action: Action::Count,
            target: Target::Project, // Test unsupported combination (not in registry)
            attribute: None,
            scope: Scope { r#type: ScopeType::CurrentTeam, building_id: None, building_name: None, real_estate_id: None, real_estate_name: None },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };
        assert!(matches!(registry.supports(&intent2), CapabilityDecision::Unsupported { .. }));
    }

    #[test]
    fn test_grammar_vs_capability_separation() {
        // Test that grammar allows unknown attributes but capabilities may reject them
        let args = json!({
            "intent": {
                "action": "count",
                "target": "building",
                "scope": { "type": "current_team" },
                "attribute": "unknownAttribute"  // Grammar allows this
            }
        });

        // Grammar parsing should succeed
        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok());
        let intent = result.unwrap();

        // But capability check may reject it (depending on registry)
        let registry = CapabilityRegistry::new();
        let _decision = registry.supports(&intent);
        // This might be supported or not depending on registry configuration
        // The key is that grammar didn't reject it
    }

    fn test_parse_intent_aggregate_building_floor_area() {
        let args = json!({
            "intent": {
                "action": "aggregate",
                "target": "building",
                "scope": { "type": "current_team" },
                "metric": "totalFloorArea"
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Intent parsing should succeed: {:?}", result.err());
        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Aggregate));
        assert!(matches!(intent.target, Target::Building));
        assert!(matches!(intent.scope.r#type, ScopeType::CurrentTeam));
        assert!(matches!(intent.metric.as_ref(), Some(&Metric::TotalFloorArea)));

        // Test compilation too
        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let compile_result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(compile_result.is_ok(), "Compilation should succeed: {:?}", compile_result.err());
        let query = compile_result.unwrap();
        assert!(query.contains("buildings"));
        assert!(query.contains("_sum"));
        assert!(query.contains("totalFloorArea"));
        assert!(query.contains("team-123"));
    }

    #[test]
    fn test_parse_intent_missing_intent() {
        let args = json!({"other_field": "value"});
        assert!(IntentQueryTool::parse_intent(&args).is_err());
    }

    #[test]
    fn test_parse_intent_invalid_format() {
        let args = json!({
            "intent": {
                "action": "invalid_action",
                "target": "component",
                "scope": { "buildingName": "Test" }
            }
        });
        assert!(IntentQueryTool::parse_intent(&args).is_err());
    }

    #[test]
    fn test_parse_intent_invalid_target_window() {
        // Test that nested domain objects like "window" get normalized to valid targets
        let args = json!({
            "intent": {
                "action": "count",
                "target": "window",  // Invalid - should be normalized to "building" + "windows"
                "scope": { "type": "building", "buildingName": "Räven" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Intent with target='window' should be normalized and parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Building)); // Should be normalized
        assert_eq!(intent.attribute, Some("windows".to_string())); // Should get attribute
    }

    #[test]
    fn test_parse_intent_unknown_attribute_passes_grammar() {
        // Test that unknown attributes pass grammar parsing (capabilities decide support)
        let args = json!({
            "intent": {
                "action": "count",
                "target": "building",
                "attribute": "unknownAttribute123",  // Grammar allows any string
                "scope": { "type": "current_team" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Unknown attribute should pass grammar parsing");

        let intent = result.unwrap();
        assert_eq!(intent.attribute, Some("unknownAttribute123".to_string()));

        // But capabilities may reject it
        let registry = CapabilityRegistry::new();
        let _decision = registry.supports(&intent);
        // This might be Unsupported depending on registry configuration
        // The key is that grammar parsing succeeded
    }

    #[test]
    fn test_parse_intent_windows_in_building_raeven() {
        // Test the example from the requirements: "How many windows are there in building Räven?"
        let args = json!({
            "intent": {
                "action": "count",
                "target": "building",      // ✅ Valid root entity
                "attribute": "windows",    // ✅ Physical thing as attribute
                "scope": {
                    "type": "building",
                    "buildingName": "Räven"
                }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Valid intent with windows attribute should parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Building));
        assert_eq!(intent.attribute, Some("windows".to_string()));
        assert!(matches!(intent.scope.r#type, ScopeType::Building));
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
    }

    #[test]
    fn test_parse_intent_doors_in_building_asagard() {
        // Test the example from the requirements: "How many doors are in building Asagård?"
        let args = json!({
            "intent": {
                "action": "count",
                "target": "building",      // ✅ Valid root entity
                "attribute": "doors",      // ✅ Physical thing as attribute
                "scope": {
                    "type": "building",
                    "buildingName": "Asagård"
                }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Valid intent with doors attribute should parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Building));
        assert_eq!(intent.attribute, Some("doors".to_string()));
        assert!(matches!(intent.scope.r#type, ScopeType::Building));
        assert_eq!(intent.scope.building_name, Some("Asagård".to_string()));
    }

    #[test]
    fn test_parse_intent_components_in_building_raeven() {
        // Test the example from the requirements: "How many components are in building Räven?"
        let args = json!({
            "intent": {
                "action": "count",
                "target": "component",     // ✅ Valid root entity
                "scope": {
                    "type": "building",
                    "buildingName": "Räven"
                }
                // ✅ No attribute field - components are the root entity
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Valid intent with components (no attribute) should parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Component));
        assert_eq!(intent.attribute, None); // ✅ No attribute for components
        assert!(matches!(intent.scope.r#type, ScopeType::Building));
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
    }

    #[test]
    fn test_parse_intent_floor_area_aggregation() {
        // Test the example: "What is the total floor area for all my buildings?"
        let args = json!({
            "intent": {
                "action": "aggregate",
                "target": "building",      // ✅ Valid root entity
                "metric": "totalFloorArea", // ✅ Metric for aggregation
                "scope": { "type": "current_team" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Valid intent with floor area aggregation should parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Aggregate));
        assert!(matches!(intent.target, Target::Building));
        assert!(matches!(intent.metric, Some(Metric::TotalFloorArea)));
        assert!(matches!(intent.scope.r#type, ScopeType::CurrentTeam));
    }

    #[test]
    fn test_attribute_normalization_window_to_building() {
        // Test that "window" target gets normalized to "building" + "windows" attribute
        let args = json!({
            "intent": {
                "action": "count",
                "target": "window",  // Invalid - should be normalized
                "scope": { "type": "building", "buildingName": "Räven" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Intent with window target should be normalized and parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Building)); // Should be normalized to Building
        assert_eq!(intent.attribute, Some("windows".to_string())); // Should get windows attribute
        assert!(matches!(intent.scope.r#type, ScopeType::Building));
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
    }

    #[test]
    fn test_attribute_normalization_door_to_building() {
        // Test that "door" target gets normalized to "building" + "doors" attribute
        let args = json!({
            "intent": {
                "action": "list",
                "target": "door",  // Invalid - should be normalized
                "scope": { "type": "current_team" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Intent with door target should be normalized and parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::List));
        assert!(matches!(intent.target, Target::Building)); // Should be normalized to Building
        assert_eq!(intent.attribute, Some("doors".to_string())); // Should get doors attribute
        assert!(matches!(intent.scope.r#type, ScopeType::CurrentTeam));
    }

    #[test]
    fn test_attribute_normalization_condition_to_component() {
        // Test that "condition" target gets normalized to "component" + "condition" attribute
        let args = json!({
            "intent": {
                "action": "aggregate",
                "target": "condition",  // Invalid - should be normalized
                "metric": "avg",
                "scope": { "type": "building", "buildingName": "Test Building" }
            }
        });

        let result = IntentQueryTool::parse_intent(&args);
        assert!(result.is_ok(), "Intent with condition target should be normalized and parse correctly");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Aggregate));
        assert!(matches!(intent.target, Target::Component)); // Should be normalized to Component
        assert_eq!(intent.attribute, Some("condition".to_string())); // Should get condition attribute
        assert!(matches!(intent.metric, Some(Metric::Avg)));
    }

    #[test]
    fn test_normalize_intent_json_utility() {
        // Test the utility function directly
        let invalid_json = json!({
            "action": "count",
            "target": "window",
            "scope": { "type": "building", "buildingName": "Test" }
        });

        let normalized = IntentQueryTool::normalize_intent_json(&invalid_json);
        assert!(normalized.is_some(), "Should normalize window to building + windows");

        let normalized = normalized.unwrap();
        assert_eq!(normalized["target"], "building");
        assert_eq!(normalized["attribute"], "windows");
        assert_eq!(normalized["action"], "count");
    }

    #[test]
    fn test_time_and_status_normalization_measures_not_completed_2025() {
        // Test: "How many measures are not completed in year 2025?"
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "How many measures are not completed in year 2025?";
        IntentQueryTool::normalize_time_and_status_filters(&mut intent, user_message);

        assert!(intent.filters.is_some(), "Should create filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.completed, Some(false), "Should set completed to false");
        assert_eq!(filters.year_min, Some(2025), "Should set year_min to 2025");
        assert_eq!(filters.year_max, Some(2025), "Should set year_max to 2025");
    }

    #[test]
    fn test_time_and_status_normalization_completed_measures_last_year() {
        // Test: "List completed measures from last year"
        let mut intent = Intent {
            action: Action::List,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "List completed measures from last year";
        IntentQueryTool::normalize_time_and_status_filters(&mut intent, user_message);

        assert!(intent.filters.is_some(), "Should create filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.completed, Some(true), "Should set completed to true");
        assert_eq!(filters.year_min, Some(2023), "Should set year_min to 2023 (last year)");
        assert_eq!(filters.year_max, Some(2023), "Should set year_max to 2023 (last year)");
    }

    #[test]
    fn test_time_normalization_year_range() {
        // Test: "between 2020 and 2023"
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "Show buildings between 2020 and 2023";
        IntentQueryTool::normalize_time_and_status_filters(&mut intent, user_message);

        assert!(intent.filters.is_some(), "Should create filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.year_min, Some(2020), "Should set year_min to 2020");
        assert_eq!(filters.year_max, Some(2023), "Should set year_max to 2023");
    }

    #[test]
    fn test_status_normalization_open_measures() {
        // Test: "open measures"
        let mut intent = Intent {
            action: Action::List,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "List open measures";
        IntentQueryTool::normalize_time_and_status_filters(&mut intent, user_message);

        assert!(intent.filters.is_some(), "Should create filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.completed, Some(false), "Should set completed to false for 'open'");
    }

    #[test]
    fn test_normalization_preserves_existing_filters() {
        // Test that existing filters are not overwritten
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: Some(2022), // Pre-existing filter
                year_max: Some(2022), // Pre-existing filter
                completed: Some(true), // Pre-existing filter
                verified: None,
                status: None,
            }),
            limit: None,
            group_by: None,
        };

        let user_message = "How many measures are not completed in year 2025?"; // Would normally set completed=false, year=2025
        IntentQueryTool::normalize_time_and_status_filters(&mut intent, user_message);

        assert!(intent.filters.is_some(), "Should preserve filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.completed, Some(true), "Should NOT overwrite existing completed filter");
        assert_eq!(filters.year_min, Some(2022), "Should NOT overwrite existing year_min filter");
        assert_eq!(filters.year_max, Some(2022), "Should NOT overwrite existing year_max filter");
    }

    #[test]
    fn test_normalize_filters_status_not_completed() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("not completed".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        IntentQueryTool::normalize_filters(&mut intent);

        assert!(intent.filters.is_some(), "Should preserve filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.status, Some("not_completed".to_string()), "Should normalize 'not completed' to 'not_completed'");
    }

    #[test]
    fn test_normalize_filters_status_incomplete() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("incomplete".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        IntentQueryTool::normalize_filters(&mut intent);

        assert!(intent.filters.is_some(), "Should preserve filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.status, Some("not_completed".to_string()), "Should normalize 'incomplete' to 'not_completed'");
    }

    #[test]
    fn test_normalize_filters_status_completed() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("done".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        IntentQueryTool::normalize_filters(&mut intent);

        assert!(intent.filters.is_some(), "Should preserve filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.status, Some("completed".to_string()), "Should normalize 'done' to 'completed'");
    }

    #[test]
    fn test_normalize_filters_status_unknown() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("unknown_status".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        IntentQueryTool::normalize_filters(&mut intent);

        assert!(intent.filters.is_some(), "Should preserve filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.status, Some("unknown_status".to_string()), "Should preserve unknown status values");
    }

    #[test]
    fn test_normalize_filters_idempotent() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("not completed".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        // Call normalize_filters multiple times
        IntentQueryTool::normalize_filters(&mut intent);
        IntentQueryTool::normalize_filters(&mut intent);

        assert!(intent.filters.is_some(), "Should preserve filters");
        let filters = intent.filters.unwrap();
        assert_eq!(filters.status, Some("not_completed".to_string()), "Should be idempotent - same result after multiple calls");
    }

    #[test]
    fn test_compile_intent_count_components() {
        let intent = Intent {
            action: Action::Count,
            target: Target::Component,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(result.is_ok());
        let query = result.unwrap();
        assert!(query.contains("components"));
        assert!(query.contains("_count"));
        assert!(query.contains("buildingId"));
        assert!(query.contains("building-123"));
    }

    #[test]
    fn test_compile_intent_list_buildings() {
        let intent = Intent {
            action: Action::List,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: Some(10),
            group_by: None,
        };

        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(result.is_ok());
        let query = result.unwrap();
        assert!(query.contains("buildings"));
        assert!(query.contains("take: 10"));
        assert!(query.contains("team-123"));
    }

    #[test]
    fn test_compile_intent_aggregate_building_floor_area() {
        let intent = Intent {
            action: Action::Aggregate,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: Some(Metric::TotalFloorArea),
            filters: None,
            limit: None,
            group_by: None,
        };

        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(result.is_ok());

        let query = result.unwrap();
        assert!(query.contains("buildings"));
        assert!(query.contains("_sum"));
        assert!(query.contains("totalFloorArea"));
        assert!(query.contains("team-123"));
        assert!(query.contains("query {"));
    }

    fn test_compile_intent_count_buildings() {
        let intent = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(result.is_ok());

        let query = result.unwrap();
        assert!(query.contains("buildings"));
        assert!(query.contains("_count"));
        assert!(query.contains("team-123"));
        assert!(query.contains("query {"));
    }

    fn test_compile_intent_unsupported_combination() {
        let intent = Intent {
            action: Action::Exists,
            target: Target::Component,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let registry = create_test_registry();
        let result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_scope_id_building() {
        let scope = Scope {
            r#type: ScopeType::Building,
            building_id: Some("building-123".to_string()),
            building_name: Some("Räven".to_string()),
            real_estate_id: None,
            real_estate_name: None,
        };

        let result = IntentQueryTool::get_scope_id(&scope);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some("building-123".to_string()));
    }

    #[test]
    fn test_get_scope_id_building_missing_id() {
        let scope = Scope {
            r#type: ScopeType::Building,
            building_id: None, // Missing required building_id
            building_name: None,
            real_estate_id: None,
            real_estate_name: None,
        };

        let result = IntentQueryTool::get_scope_id(&scope);
        assert!(result.is_err());
    }

    #[test]
    fn test_eligibility_explicit_requests() {
        let tool = IntentQueryTool::new(create_test_registry());

        // Should be eligible when explicitly requested (even with conversational message)
        let intent = Intent {
            action: Action::Count,
            target: Target::Component,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };
        let ctx = ToolEligibilityContext {
            user_message: "hello",
            assistant_message: "",
            explicitly_requested: true,
            persona: "default",
            intent: Some(&intent),
        };
        assert!(tool.is_eligible(&ctx));
    }

    #[test]
    fn test_eligibility_intent_based() {
        let tool = IntentQueryTool::new(create_test_registry());

        // Valid intents that should be eligible
        let valid_intents = vec![
            Intent {
                action: Action::Count,
                target: Target::Component,
                attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: None, // Test building - not in registry
                building_name: Some("Test Building".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
                metric: None,
                filters: None,
                limit: None,
                group_by: None,
            },
            Intent {
                action: Action::List,
                target: Target::Building,
                attribute: None,
                scope: Scope {
                    r#type: ScopeType::CurrentTeam,
                    building_id: None,
                    building_name: None,
                    real_estate_id: None,
                    real_estate_name: None,
                },
                metric: None,
                filters: None,
                limit: Some(10),
                group_by: None,
            },
            Intent {
                action: Action::Aggregate,
                target: Target::Building,
                attribute: None,
                scope: Scope {
                    r#type: ScopeType::CurrentTeam,
                    building_id: None,
                    building_name: None,
                    real_estate_id: None,
                    real_estate_name: None,
                },
                metric: Some(Metric::TotalFloorArea),
                filters: None,
                limit: None,
                group_by: None,
            },
        ];

        // Invalid intents that should NOT be eligible (unsupported combinations)
        let invalid_intents = vec![
            Intent {
                action: Action::Count,
                target: Target::Project, // Unsupported target (not in capability registry)
                attribute: None,
                scope: Scope {
                    r#type: ScopeType::CurrentTeam,
                    building_id: None,
                    building_name: None,
                    real_estate_id: None,
                    real_estate_name: None,
                },
                metric: None,
                filters: None,
                limit: None,
                group_by: None,
            },
        ];

        // Test valid intents are eligible
        for intent in valid_intents {
            let ctx = ToolEligibilityContext {
                user_message: "How many components are in building X?",
                assistant_message: "",
                explicitly_requested: false,
                persona: "test",
                intent: Some(&intent),
            };
            assert!(tool.is_eligible(&ctx), "Intent {:?} should be eligible", intent);
        }

        // Test invalid intents are NOT eligible
        for intent in invalid_intents {
            let ctx = ToolEligibilityContext {
                user_message: "Show me data",
                assistant_message: "",
                explicitly_requested: false,
                persona: "test",
                intent: Some(&intent),
            };
            assert!(!tool.is_eligible(&ctx), "Intent {:?} should NOT be eligible", intent);
        }

        // Test missing intent = not eligible
        let ctx = ToolEligibilityContext {
            user_message: "How many components?",
            assistant_message: "",
            explicitly_requested: false,
            persona: "test",
            intent: None,
        };
        assert!(!tool.is_eligible(&ctx), "Missing intent should NOT be eligible");
    }

    #[test]
    fn test_strict_intent_contract() {
        // Test that malformed intents are rejected
        let invalid_payloads = vec![
            json!({"query": "SELECT * FROM buildings"}), // Raw query string
            json!({"action": "count"}), // Missing required fields
            json!({"action": "count", "target": "unknown"}), // Unknown target
            json!({"action": "invalid", "entity": "building"}), // Unknown action
            json!({"intent": "count components"}), // Natural language string
        ];

        for payload in invalid_payloads {
            let result = IntentQueryTool::parse_intent(&payload);
            assert!(result.is_err(), "Payload {:?} should be rejected", payload);
        }

        // Test that valid intents are accepted
        let valid_payload = json!({
            "intent": {
                "action": "count",
                "target": "component",
                "scope": {"type": "building", "buildingName": "Test Building"}
            }
        });

        let result = IntentQueryTool::parse_intent(&valid_payload);
        assert!(result.is_ok(), "Valid intent should be accepted");

        let intent = result.unwrap();
        assert!(matches!(intent.action, Action::Count));
        assert!(matches!(intent.target, Target::Component));

        // Test invalid scope structures
        let invalid_scopes = vec![
            // Missing type
            json!({"action": "count", "target": "component", "scope": {"buildingName": "test"}}),
            // Building scope without buildingName
            json!({"action": "count", "target": "component", "scope": {"type": "building"}}),
            // RealEstate scope without realEstateName
            json!({"action": "count", "target": "component", "scope": {"type": "realEstate"}}),
            // CurrentTeam with extra fields
            json!({"action": "count", "target": "component", "scope": {"type": "current_team", "buildingName": "test"}}),
            // Invalid organization scope types (these should never be generated)
            json!({"action": "count", "target": "component", "scope": {"type": "organisation", "organisationName": "Pontus"}}),
            json!({"action": "count", "target": "component", "scope": {"type": "organization", "organizationName": "Company"}}),
        ];

        for invalid_payload in invalid_scopes {
            let wrapped_payload = json!({"intent": invalid_payload});
            let result = IntentQueryTool::parse_intent(&wrapped_payload);
            assert!(result.is_err(), "Invalid scope structure should be rejected: {:?}", invalid_payload);
        }

        // Test that semantically incompatible intents pass parsing (grammar validation)
        // but would be rejected by capability matrix
        let semantic_payloads = vec![
            // Component aggregation - passes parsing, rejected by capabilities
            json!({"action": "aggregate", "target": "component", "scope": {"type": "current_team"}, "metric": "totalFloorArea"}),
            // RealEstate with totalFloorArea - passes parsing, rejected by capabilities
            json!({"action": "aggregate", "target": "realEstate", "scope": {"type": "current_team"}, "metric": "totalFloorArea"}),
        ];

        for payload in semantic_payloads {
            let wrapped_payload = json!({"intent": payload});
            let result = IntentQueryTool::parse_intent(&wrapped_payload);
            assert!(result.is_ok(), "Semantic incompatibility should pass grammar validation: {:?}", payload);

            // But should be rejected by capabilities
            let intent = result.unwrap();
            let registry = CapabilityRegistry::new();
            let _decision = registry.supports(&intent);
            assert!(matches!(_decision, CapabilityDecision::Unsupported { .. }),
                "Semantic incompatibility should be rejected by capabilities: {:?}", payload);
        }
    }

    #[test]
    fn test_filter_validation_measure_count_with_status() {
        let intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("not_completed".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        // Should pass validation
        assert!(IntentQueryTool::validate_filters(&intent).is_ok());
    }

    #[test]
    fn test_filter_validation_building_count_no_filters_allowed() {
        let intent = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("completed".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        // Should fail - building count does not allow filters
        assert!(IntentQueryTool::validate_filters(&intent).is_err());
    }

    #[test]
    fn test_filter_validation_no_filters() {
        let intent = Intent {
            action: Action::Count,
            target: Target::Building,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None, // No filters
            limit: None,
            group_by: None,
        };

        // Should pass - no filters is always valid
        assert!(IntentQueryTool::validate_filters(&intent).is_ok());
    }
    #[test]
    fn test_empty_attribute_is_removed() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: Some("".to_string()),
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        // simulate normalization
        if let Some(attr) = &intent.attribute {
            if attr.trim().is_empty() {
                intent.attribute = None;
            }
        }

        assert!(intent.attribute.is_none());
    }

    #[test]
    fn test_status_words_stripped_from_attributes() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: Some("open".to_string()),
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        // Simulate attribute normalization
        if let Some(attr) = &intent.attribute {
            let normalized = attr.trim().to_lowercase();
            if matches!(normalized.as_str(),
                "open" | "closed" | "completed" | "incomplete" |
                "unfinished" | "done" | "not completed" | "not_completed") {
                intent.attribute = None;
            }
        }

        assert!(intent.attribute.is_none());
    }

    #[test]
    fn test_normalization_is_idempotent() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: Some("completed".to_string()),
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("not_completed".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        // Run normalization multiple times
        for _ in 0..3 {
            IntentQueryTool::strip_status_attributes(&mut intent);
            IntentQueryTool::normalize_filters(&mut intent);
        }

        // Should still be valid after multiple normalizations
        assert!(intent.attribute.is_none()); // Status word stripped
        assert!(IntentQueryTool::validate_filters(&intent).is_ok()); // Filters valid
    }

    #[test]
    fn test_canonical_status_conversion_from_completed() {
        let filters = Filters {
            component_type: None,
            building_type: None,
            year_min: None,
            year_max: None,
            completed: Some(true),
            verified: None,
            status: None,
        };

        let canonical = filters.to_canonical_map();
        assert_eq!(canonical.len(), 1);
        assert!(matches!(
            canonical.get(&FilterKey::Status),
            Some(FilterValue::Status(StatusValue::Completed))
        ));
    }

    #[test]
    fn test_canonical_status_conversion_from_status_string() {
        let filters = Filters {
            component_type: None,
            building_type: None,
            year_min: None,
            year_max: None,
            completed: None,
            verified: None,
            status: Some("not_completed".to_string()),
        };

        let canonical = filters.to_canonical_map();
        assert_eq!(canonical.len(), 1);
        assert!(matches!(
            canonical.get(&FilterKey::Status),
            Some(FilterValue::Status(StatusValue::NotCompleted))
        ));
    }

    #[test]
    fn test_invalid_scope_scrubbed_to_current_team() {
        use serde_json::json;

        let mut intent_json = json!({
            "action": "count",
            "target": "measure",
            "scope": {
                "type": "year",
                "value": "2025"
            }
        });

        IntentQueryTool::scrub_invalid_scopes(&mut intent_json);

        let scope = intent_json.get("scope").unwrap().as_object().unwrap();
        assert_eq!(scope.get("type").unwrap().as_str().unwrap(), "current_team");
        assert!(!scope.contains_key("value"));
    }

    #[test]
    fn test_valid_scopes_unchanged_by_scrub() {
        use serde_json::json;

        let mut intent_json = json!({
            "action": "count",
            "target": "measure",
            "scope": {
                "type": "building",
                "buildingName": "Räven"
            }
        });

        let original = intent_json.clone();
        IntentQueryTool::scrub_invalid_scopes(&mut intent_json);

        assert_eq!(intent_json, original);
    }

    fn create_test_registry() -> Box<dyn NameResolutionRegistry> {
        Box::new(InMemoryNameRegistry::new(
            vec![
                ("Räven".to_string(), "building-123".to_string(), "Räven".to_string()),
                ("Björk".to_string(), "building-456".to_string(), "Björk".to_string()),
            ],
            vec![
                ("RealEstate1".to_string(), "real-estate-123".to_string(), "Real Estate 1".to_string()),
            ],
        ))
    }

    #[test]
    fn test_building_scope_normalization() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            attribute: None,
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: None,
                verified: None,
                status: Some("not_completed".to_string()),
            }),
            limit: None,
            group_by: None,
        };

        let user_message = "How many incomplete measures are in building Räven?";

        IntentQueryTool::normalize_explicit_building_scope(&mut intent, user_message, &*create_test_registry());

        // Should have been corrected to building scope
        assert!(matches!(intent.scope.r#type, ScopeType::Building));
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
    }

    #[test]
    fn test_building_scope_normalization_idempotent() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            scope: Scope {
                r#type: ScopeType::Building,
                building_id: Some("building-123".to_string()),
                building_name: Some("Räven".to_string()),
                real_estate_id: None,
                real_estate_name: None,
            },
            attribute: None,
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "How many measures are in building Räven?";

        // Should not change existing building scope
        let original_scope = intent.scope.clone();
        IntentQueryTool::normalize_explicit_building_scope(&mut intent, user_message, &*create_test_registry());

        assert_eq!(intent.scope.r#type, original_scope.r#type);
        assert_eq!(intent.scope.building_name, original_scope.building_name);
    }

    #[test]
    fn test_building_scope_normalization_edge_cases() {
        // Test case 1: "in 2025" should not trigger building extraction
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            attribute: None,
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        IntentQueryTool::normalize_explicit_building_scope(&mut intent, "How many measures were completed in 2025?", &*create_test_registry());
        assert!(matches!(intent.scope.r#type, ScopeType::CurrentTeam));

        // Test case 2: "in progress" should not trigger building extraction
        let mut intent2 = intent.clone();
        IntentQueryTool::normalize_explicit_building_scope(&mut intent2, "List measures in progress", &*create_test_registry());
        assert!(matches!(intent2.scope.r#type, ScopeType::CurrentTeam));

        // Test case 3: lowercase building names should not be accepted
        let mut intent3 = intent.clone();
        IntentQueryTool::normalize_explicit_building_scope(&mut intent3, "How many measures are in building räven?", &*create_test_registry());
        assert!(matches!(intent3.scope.r#type, ScopeType::CurrentTeam));

        // Test case 4: "the building" should not trigger extraction
        let mut intent4 = intent.clone();
        IntentQueryTool::normalize_explicit_building_scope(&mut intent4, "How many measures are in the building?", &*create_test_registry());
        assert!(matches!(intent4.scope.r#type, ScopeType::CurrentTeam));
    }

    #[test]
    fn test_building_name_extraction_stop_words() {
        // Test that extraction stops at stop words
        assert_eq!(
            IntentQueryTool::extract_building_name_from_message("building Räven that has measures"),
            Some("Räven".to_string())
        );

        // Test that extraction includes multiple words up to stop word
        assert_eq!(
            IntentQueryTool::extract_building_name_from_message("in Building Complex A which has"),
            Some("Building Complex A".to_string())
        );
    }

    #[test]
    fn test_building_scope_normalization_measures_in_raeven() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "How many measures are in building Räven?";
        let registry = create_test_registry();

        let result = IntentQueryTool::normalize_explicit_building_scope(&mut intent, user_message, &*registry);
        assert!(result.is_ok(), "Normalization should succeed: {:?}", result.err());

        // Assert scope.type == building
        assert!(matches!(intent.scope.r#type, ScopeType::Building));

        // Assert building_name == "Räven"
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
        assert_eq!(intent.scope.building_id, Some("building-123".to_string()));
    }

    #[test]
    fn test_building_scope_normalization_incomplete_measures_in_raeven() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: Some(Filters {
                component_type: None,
                building_type: None,
                year_min: None,
                year_max: None,
                completed: Some(false),
                verified: None,
                status: None,
            }),
            limit: None,
            group_by: None,
        };

        let user_message = "How many incomplete measures are in building Räven?";
        let registry = create_test_registry();

        let result = IntentQueryTool::normalize_explicit_building_scope(&mut intent, user_message, &*registry);
        assert!(result.is_ok(), "Normalization should succeed: {:?}", result.err());

        // Assert scope.type == building
        assert!(matches!(intent.scope.r#type, ScopeType::Building));

        // Assert building_name == "Räven"
        assert_eq!(intent.scope.building_name, Some("Räven".to_string()));
        assert_eq!(intent.scope.building_id, Some("building-123".to_string()));

        // Test GraphQL compilation
        let ctx = RequestContext {
            team_id: "team-123".to_string(),
            user_id: "user-456".to_string(),
            current_building_id: None,
            current_real_estate_id: None,
        };

        let compile_result = IntentQueryTool::compile_intent_to_graphql(&intent, &ctx, &*registry);
        assert!(compile_result.is_ok(), "Compilation should succeed: {:?}", compile_result.err());

        let query = compile_result.unwrap();
        // Assert GraphQL contains building constraint (buildingId), not teamId
        assert!(query.contains("buildingId"));
        assert!(query.contains("building-123"));
        assert!(!query.contains("teamId")); // Building-scoped queries should NOT have teamId
    }

    #[test]
    fn test_building_scope_normalization_unknown_building_fails() {
        let mut intent = Intent {
            action: Action::Count,
            target: Target::Measure,
            attribute: None,
            scope: Scope {
                r#type: ScopeType::CurrentTeam,
                building_id: None,
                building_name: None,
                real_estate_id: None,
                real_estate_name: None,
            },
            metric: None,
            filters: None,
            limit: None,
            group_by: None,
        };

        let user_message = "How many measures are in building UnknownBuilding?";
        let registry = create_test_registry();

        let result = IntentQueryTool::normalize_explicit_building_scope(&mut intent, user_message, &*registry);
        assert!(result.is_err(), "Normalization should fail for unknown building");
        assert!(result.unwrap_err().to_string().contains("Building 'UnknownBuilding' not found"));
    }
}

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
