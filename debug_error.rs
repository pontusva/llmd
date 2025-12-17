use serde_json::json;
use crate::tools::graphql::IntentQueryTool;

#[test]
fn debug_error() {
    let invalid_payload = json!({
        "intent": {
            "action": "count",
            "target": "building", 
            "attributes": ["window"],
            "scope": { "type": "building", "building_name": "Test" }
        }
    });

    let intent_value = invalid_payload.as_object().unwrap().get("intent").unwrap();
    let result = IntentQueryTool::validate_intent_shape(intent_value);
    println!("Error: {:?}", result);
}
