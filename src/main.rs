use axum;
use tracing_subscriber;
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    // Load model once at startup
    let state = state::AppState::init().await.unwrap();

    let app = routes::routes()
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("🚀 Inference server running at http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}