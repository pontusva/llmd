use jsonschema::JSONSchema;
use once_cell::sync::Lazy;
use serde_json::Value;
use std::error::Error;

#[derive(Debug)]
pub struct IntentSchema {
    schema: JSONSchema,
}

impl IntentSchema {
    /// Create IntentSchema from a JSON Value (used internally for compile-time loading)
    pub fn from_json(schema_value: Value) -> Result<Self, Box<dyn Error>> {
        let schema = JSONSchema::options()
            .with_draft(jsonschema::Draft::Draft7)
            .compile(&schema_value)
            .map_err(|e| format!("Schema compilation failed: {}", e))?;

        Ok(Self { schema })
    }

    /// Validate a JSON value against the schema
    pub fn validate(&self, value: &Value) -> Result<(), String> {
        let mut errors = Vec::new();

        for error in self.schema.validate(value) {
            errors.push(format!("{:?}", error));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("Schema validation failed: {}", errors.join(", ")))
        }
    }
}

// Compile-time embedded schema
static SCHEMA_JSON: &str = include_str!("../../../schemas/intent.schema.json");

// Global singleton IntentSchema instance
static INTENT_SCHEMA: Lazy<IntentSchema> = Lazy::new(|| {
    let schema_value: Value = serde_json::from_str(SCHEMA_JSON)
        .expect("Failed to parse embedded intent schema JSON");

    IntentSchema::from_json(schema_value)
        .expect("Failed to compile embedded intent schema")
});

/// Global accessor for the intent schema validator
pub fn intent_schema() -> &'static IntentSchema {
    &INTENT_SCHEMA
}
