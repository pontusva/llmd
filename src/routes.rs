use axum::{
    routing::{get, post, delete},
    Router,
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response, sse::{Sse, Event}},
};
use serde::{Deserialize, Serialize};
use crate::state::AppState;
use crate::persona_memory::{IntelligentMemory, MemoryType, MemoryConfig, MemoryMode, MemoryPolicy, EmbeddingModel};
use crate::executor::MemoryUpdateTask;
use uuid::Uuid;
use futures::StreamExt;
use serde_json::json;

#[derive(Deserialize)]
pub struct InferInput {
    pub text: String,
}

#[derive(Serialize)]
pub struct InferOutput {
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct IndexResponse {
    pub id: u64,
}

#[derive(Deserialize)]
pub struct BatchInferRequest {
    pub texts: Vec<String>,
}

#[derive(Serialize)]
pub struct BatchInferResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
pub struct GetResponse {
    pub id: u64,
    pub text: String,
    pub timestamp: u64,
    pub embedding_len: usize,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub count: usize,
}

// OpenAI-compatible Embeddings
#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub input: String,
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingResponseData>,
    pub model: String,
}

#[derive(Serialize)]
pub struct EmbeddingResponseData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

// OpenAI-compatible Models
#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// OpenAI-compatible Chat
#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub system_prompt: Option<String>,
    pub persona: Option<String>,
    pub memory_update: Option<String>, // append | replace | disable
    pub stream: Option<bool>,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Deserialize)]
pub struct IndexRequest {
    pub text: String,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub k: Option<usize>,
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/infer", post(infer_handler))
        .route("/batch_infer", post(batch_infer_handler))
        .route("/index", post(index_handler))
        .route("/get/:id", get(get_handler))
        .route("/delete/:id", delete(delete_handler))
        .route("/search", post(search_handler))
        .route("/health", get(health_handler))
        .route("/v1/embeddings", post(embedding_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_handler))
        .route("/v1/persona/:persona/memory", get(persona_memory_get))
        .route("/v1/persona/:persona/memory/reset", post(persona_memory_reset))
}

// ----------------------
// Correct handler forms
// ----------------------

async fn infer_handler(
    State(state): State<AppState>,
    Json(input): Json<InferInput>,
) -> Json<InferOutput> {
    let result = state.model.infer(&input.text).await.unwrap();
    Json(InferOutput { embedding: result })
}

async fn batch_infer_handler(
    State(state): State<AppState>,
    Json(req): Json<BatchInferRequest>,
) -> impl IntoResponse {
    let embeddings = match state.model.infer_batch(&req.texts).await {
        Ok(embs) => embs,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    (StatusCode::OK, Json(BatchInferResponse { embeddings })).into_response()
}

async fn index_handler(
    State(state): State<AppState>,
    Json(req): Json<IndexRequest>,
) -> impl IntoResponse {
    let emb = match state.model.infer(&req.text).await {
        Ok(e) => e,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    let mut store = state.store.lock().unwrap();
    let id = match store.add(req.text, emb) {
        Ok(i) => i,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    (StatusCode::OK, Json(IndexResponse { id })).into_response()
}

async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let emb = match state.model.infer(&req.query).await {
        Ok(e) => e,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    let store = state.store.lock().unwrap();
    let results = store.search(&emb, req.k.unwrap_or(3));

    (StatusCode::OK, Json(results)).into_response()
}

async fn get_handler(
    Path(id): Path<u64>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let storage = state.storage.lock().unwrap();
    if let Some(entry) = storage.get_entry(id) {
        let resp = GetResponse {
            id: entry.id,
            text: entry.text,
            timestamp: entry.timestamp,
            embedding_len: entry.embedding.len(),
        };
        (StatusCode::OK, Json(resp)).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn embedding_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let text = req.input;
    let model_name = req.model.unwrap_or_else(|| "local-bert".to_string());

    let embedding = match state.model.infer(&text).await {
        Ok(e) => e,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    let data = vec![EmbeddingResponseData {
        object: "embedding".to_string(),
        index: 0,
        embedding,
    }];

    let resp = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: model_name,
    };

    (StatusCode::OK, Json(resp)).into_response()
}

async fn models_handler(State(state): State<AppState>) -> impl IntoResponse {
    let names = state.llms.list_models();
    let data: Vec<ModelInfo> = names
        .into_iter()
        .map(|id| ModelInfo { id, object: "model".to_string() })
        .collect();
    let resp = ModelsResponse {
        object: "list".to_string(),
        data,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    tracing::info!(
        "📥 Incoming ChatCompletionRequest:\n{}",
        serde_json::to_string_pretty(&req).unwrap()
    );
    // Get LLM from registry
    let llm = match state.llms.get(&req.model) {
        Ok(Some(m)) => m,
        Ok(None) => {
            let msg = format!("Unknown model {}", req.model);
            return (StatusCode::BAD_REQUEST, msg).into_response();
        }
        Err(e) => {
            let msg = format!("Failed to load model {}: {}", req.model, e);
            return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response();
        }
    };

    // Convert request messages to LLM format
    let mut user_messages: Vec<crate::llm::ChatMessage> = Vec::new();
    for m in &req.messages {
        user_messages.push(crate::llm::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        });
    }

    // Build generation options
    let mut opts = crate::llm::GenerateOptions::default();
    opts.messages = user_messages.clone(); // Executor will override with full messages
    
    if let Some(v) = req.max_tokens { opts.max_tokens = v; }
    if let Some(v) = req.temperature { opts.temperature = v; }
    if let Some(v) = req.top_p { opts.top_p = v; }
    if let Some(v) = req.top_k { opts.top_k = v; }
    if let Some(v) = req.repetition_penalty { opts.repetition_penalty = v; }


    let id = format!("chatcmpl-{}", Uuid::new_v4());

    if req.stream == Some(true) {
        // Execute streaming request via Executor
        let (token_stream, memory_task) = match state.executor.execute_stream(
            llm,
            user_messages,
            req.persona.as_deref(),
            req.system_prompt.as_deref(),
            req.memory_update.as_deref(),
            opts,
        ).await {
            Ok(result) => result,
            Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
        };

        use futures::stream;

        // Create SSE stream that collects tokens and handles completion
        let executor_clone = state.executor.clone();
        let stream = stream::unfold((token_stream, id.clone(), req.model.clone(), String::new(), memory_task), move |(mut ts, id_val, model_val, mut collected, memory_task_val)| {
            let executor = executor_clone.clone();
            async move {
                match ts.next().await {
                    Some(token) => {
                        // Collect token for memory update
                        collected.push_str(&token);

                        let event_json = json!({
                            "id": id_val,
                            "object": "chat.completion.chunk",
                            "model": model_val,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": token },
                                "finish_reason": null
                            }],
                        });
                        match serde_json::to_string(&event_json) {
                            Ok(json_str) => Some((Event::default().data(format!("{}\n\n", json_str)), (ts, id_val, model_val, collected, memory_task_val))),
                            Err(_) => Some((Event::default().data("[ERROR]\n\n"), (ts, id_val, model_val, collected, memory_task_val))),
                        }
                    }
                    None => {
                        // Stream finished - complete memory update via Executor
                        executor.complete_stream(memory_task_val, &collected).await;
                        Some((Event::default().data("[DONE]\n\n"), (ts, id_val, model_val, collected, MemoryUpdateTask::Disabled)))
                    }
                }
            }
        }).map(Ok::<_, std::convert::Infallible>);

        Sse::new(stream).into_response()
    } else {
        // Execute non-streaming request via Executor
        let reply = match state.executor.execute_chat(
            llm,
            user_messages,
            req.persona.as_deref(),
            req.system_prompt.as_deref(),
            req.memory_update.as_deref(),
            opts,
        ).await {
            Ok(r) => r,
            Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
        };

        let message = ChatMessage {
            role: "assistant".to_string(),
            content: reply,
        };

        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            model: req.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message,
                finish_reason: "stop".to_string(),
            }],
        };

        (StatusCode::OK, Json(resp)).into_response()
    }
}

async fn delete_handler(
    Path(id): Path<u64>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let storage = state.storage.lock().unwrap();
    match storage.delete_entry(id) {
        Ok(deleted) => (StatusCode::OK, Json(DeleteResponse { deleted })).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

async fn health_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let storage = state.storage.lock().unwrap();
    let count = storage.count_entries();
    let resp = HealthResponse {
        status: "ok".to_string(),
        count,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

// GET /v1/persona/:persona/memory
async fn persona_memory_get(
    Path(persona): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let pmem = state.persona_memory.lock().unwrap();
    match pmem.get_memory(MemoryType::Persona, &persona) {
        Ok(mem) => (StatusCode::OK, Json(mem)).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}

// POST /v1/persona/:persona/memory/reset
async fn persona_memory_reset(
    Path(persona): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let pmem = state.persona_memory.lock().unwrap();
    match pmem.set_memory(MemoryType::Persona, &persona, "") {
        Ok(_) => (StatusCode::OK, Json("ok")).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    }
}
