use model_loader_core::plan::{LoadPlan, LoadStep};

#[test]
fn test_loadplan_wiring() {
    // Create a simple LoadPlan
    let plan = LoadPlan {
        steps: vec![
            LoadStep::LoadConfig { path: "models/test/config.json".to_string() },
            LoadStep::LoadTokenizer { path: "models/test/tokenizer.json".to_string() },
            LoadStep::LoadShard { path: "models/test/model.safetensors".to_string(), index: 0 },
        ],
    };

    // For now, just check that we can create the plan
    // In real usage, this would be passed to AppState::init_from_plan
    assert_eq!(plan.steps.len(), 3);
}
