use crate::tools::graphql::IntentQueryTool;

#[test]
fn debug_extraction() {
    let result = IntentQueryTool::extract_building_name_from_message("in Building Complex A which has");
    println!("Result: {:?}", result);
    assert_eq!(result, Some("Building Complex A".to_string()));
}
