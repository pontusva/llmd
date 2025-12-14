use axum;
use tracing_subscriber;

use model_loader_core::plan::LoadPlan;

mod model;
mod routes;
mod state;
mod errors;
mod vector_store;
mod storage;
mod llm;
mod prompt_format;
mod system_prompt;
mod persona_memory;
mod llm_registry;
mod executor;

use std::io::{self, Read};

fn read_load_plan_from_stdin() -> Result<LoadPlan, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("failed to read stdin: {e}"))?;

    if input.trim().is_empty() {
        return Err("no LoadPlan provided on stdin".into());
    }

    serde_json::from_str(&input)
        .map_err(|e| format!("invalid LoadPlan JSON: {e}"))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    // --- read LoadPlan from model-loader
    let load_plan = match read_load_plan_from_stdin() {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("❌ Failed to start llmd: {err}");
            eprintln!("Expected LoadPlan JSON on stdin.");
            std::process::exit(1);
        }
    };

    println!("📋 llmd received load plan:");
    for step in &load_plan.steps {
        println!("  {:?}", step);
    }

    // --- initialize app state from LoadPlan
    let state = state::AppState::init_from_plan(load_plan)
        .await
        .expect("failed to initialize AppState from LoadPlan");

    let app = routes::routes().with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    println!("🚀 Inference server running at http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}