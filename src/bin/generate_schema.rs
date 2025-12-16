use std::fs;
use std::path::Path;
use schemars::schema_for;
use serde::{Deserialize, Serialize};

// Define the types here for schema generation
#[derive(schemars::JsonSchema)]
pub struct IntentEnvelope {
    pub intent: Intent,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
pub struct Intent {
    pub action: Action,
    #[serde(alias = "entity")]
    pub target: Target,
    pub scope: Scope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<Metric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<Filters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_by: Option<GroupBy>,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
pub enum Action {
    List,
    Count,
    Get,
    Aggregate,
    Exists,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Target {
    Building,
    Component,
    RealEstate,
    Measure,
    Plan,
    Project,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
pub struct Scope {
    pub r#type: ScopeType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub building_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_estate_name: Option<String>,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScopeType {
    CurrentTeam,
    Building,
    RealEstate,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
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

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Metric {
    Count,
    Sum,
    Avg,
    TotalFloorArea,
    SumCost,
    AvgCondition,
    EnergyCosts,
}

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
pub enum GroupBy {
    Building,
    RealEstate,
    ComponentType,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating intent JSON schema...");

    // Generate schema from the IntentEnvelope type
    let schema = schema_for!(IntentEnvelope);
    let schema_json = serde_json::to_string_pretty(&schema)?;

    // Write to schemas/intent.schema.json
    let schema_path = Path::new("schemas/intent.schema.json");
    fs::create_dir_all(schema_path.parent().unwrap_or(Path::new(".")))?;
    fs::write(schema_path, schema_json)?;

    println!("Generated intent schema at: {}", schema_path.display());
    Ok(())
}
