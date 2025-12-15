use axum;
use tracing_subscriber;
use std::sync::Arc;
use model_loader_core::plan::LoadPlan;
use crate::toolport::ToolRegistry;
use crate::tools::echo::EchoTool;

mod model;
mod routes;
mod state;
mod errors;
mod vector_store;
mod storage;
mod llm;
mod llm_factory;
mod prompt_format;
mod system_prompt;
mod persona_memory;
mod llm_registry;
mod executor;
mod embedding_decision;
mod toolport;
mod tools {
    pub mod echo;
}

use std::io::{self, Read};

/// Supported LLM backends
#[derive(Clone, Debug)]
enum LlmBackend {
    /// Local models using Candle (requires LoadPlan)
    Candle,
    /// Remote models via Ollama HTTP API (no LoadPlan needed)
    Ollama,
}

/// Get LLM backend from environment variable
fn llm_backend_from_env() -> LlmBackend {
    match std::env::var("LLMD_LLM_BACKEND")
        .unwrap_or_else(|_| "candle".into())
        .to_lowercase()
        .as_str()
    {
        "ollama" => LlmBackend::Ollama,
        _ => LlmBackend::Candle,
    }
}

fn read_load_plan_from_stdin() -> Result<Option<LoadPlan>, String> {
    use std::io::{self, Read, IsTerminal};

    // If stdin is a TTY, nothing is piped
    if io::stdin().is_terminal() {
        return Ok(None);
    }

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("failed to read stdin: {e}"))?;

    if input.trim().is_empty() {
        return Ok(None);
    }

    serde_json::from_str(&input)
        .map(Some)
        .map_err(|e| format!("invalid LoadPlan JSON: {e}"))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    // --- detect LLM backend
    let backend = llm_backend_from_env();
    println!("🔧 Using LLM backend: {:?}", backend);

    // --- initialize tool registry and register all tools
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(EchoTool);
    let tool_registry = Arc::new(tool_registry);

    // --- initialize app state based on backend
    let state = match backend {
        LlmBackend::Candle => {
            // Read LoadPlan from stdin (required for Candle)
            let load_plan = match read_load_plan_from_stdin() {
                Ok(Some(plan)) => plan,
                Ok(None) => {
                    eprintln!("❌ LoadPlan required for Candle backend but none provided on stdin");
                    eprintln!("Pipe model-loader output to llmd or use Ollama backend");
                    std::process::exit(1);
                }
                Err(err) => {
                    eprintln!("❌ Failed to read LoadPlan from stdin: {err}");
                    std::process::exit(1);
                }
            };

            println!("📋 llmd received load plan:");
            for step in &load_plan.steps {
                println!("  {:?}", step);
            }

            state::AppState::init_from_plan(load_plan, tool_registry)
                .await
                .expect("failed to initialize AppState from LoadPlan")
        }
        LlmBackend::Ollama => {
            // No LoadPlan needed for Ollama
            println!("🔗 Initializing Ollama backend (no LoadPlan required)");
            state::AppState::init_remote_llm(tool_registry)
                .await
                .expect("failed to initialize AppState for Ollama")
        }
    };

    let app = routes::routes().with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    println!("🚀 Inference server running at http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}