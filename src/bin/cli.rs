use std::io::{self, Write};

#[path = "../model.rs"]
pub mod model;
#[path = "../state.rs"]
pub mod state;
#[path = "../llm.rs"]
pub mod llm;
#[path = "../vector_store.rs"]
pub mod vector_store;
#[path = "../storage.rs"]
pub mod storage;
#[path = "../errors.rs"]
pub mod errors;
#[path = "../prompt_format.rs"]
pub mod prompt_format;
#[path = "../system_prompt.rs"]
pub mod system_prompt;
#[path = "../persona_memory.rs"]
pub mod persona_memory;
#[path = "../llm_registry.rs"]
pub mod llm_registry;

use crate::llm::{ChatMessage, GenerateOptions};
use crate::state::AppState;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use crate::persona_memory::{IntelligentMemory, MemoryConfig, MemoryMode, MemoryPolicy, MemoryType, EmbeddingModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = AppState::init().await.unwrap();
    let pm = Arc::new(Mutex::new(IntelligentMemory::new("persona_memory.sqlite").expect("persona memory db")));
    let convo = Arc::new(Mutex::new(Vec::<ChatMessage>::new()));
    let system_msg = Arc::new(Mutex::new(None::<String>));

    let args: Vec<String> = std::env::args().collect();
    let mut arg_system: Option<String> = None;
    let mut arg_persona: Option<String> = None;
    let mut arg_model: Option<String> = None;
    let mut arg_stream: bool = false;
    let mut arg_memory_mode: Option<String> = None;
    let mut arg_memory_policy: Option<String> = None;
    let mut arg_memory_debug: bool = false;
    let mut arg_memory_wipe: Option<String> = None; // Persona to wipe, or "all" for all personas
    let mut arg_memory_k: Option<usize> = None;
    let mut arg_memory_threshold: Option<f32> = None;
    let mut arg_memory_types: Option<String> = None;
    let mut arg_list_models: bool = false;
    let mut i = 0;
    let mut memory_update: Option<String> = None; // Legacy, keep for compatibility
    while i < args.len() {
        match args[i].as_str() {
            "--system" => {
                if i + 1 < args.len() {
                    arg_system = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--persona" => {
                if i + 1 < args.len() {
                    arg_persona = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    arg_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--stream" => {
                arg_stream = true;
            }
            "--memory-update" => {
                if i + 1 < args.len() {
                    memory_update = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--memory-mode" => {
                if i + 1 < args.len() {
                    arg_memory_mode = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--memory-policy" => {
                if i + 1 < args.len() {
                    arg_memory_policy = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--memory-debug" => {
                arg_memory_debug = true;
            }
            "--memory-k" => {
                if i + 1 < args.len() {
                    arg_memory_k = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--memory-threshold" => {
                if i + 1 < args.len() {
                    arg_memory_threshold = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--memory-types" => {
                if i + 1 < args.len() {
                    arg_memory_types = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--list-models" => {
                arg_list_models = true;
            }
            "--memory-wipe" => {
                if i + 1 < args.len() {
                    arg_memory_wipe = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    arg_memory_wipe = Some("all".to_string()); // Default to wiping all if no persona specified
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Handle --list-models option
    if arg_list_models {
        let state = AppState::init().await?;
        let models = state.llms.list_models();
        println!("📋 Available LLM models:");
        for model in models {
            println!("  • {}", model);
        }
        return Ok(());
    }

    // Create memory config from parsed CLI args
    let memory_mode = match arg_memory_mode.as_deref() {
        Some("off") => MemoryMode::Off,
        Some("read") => MemoryMode::Read,
        Some("write") => MemoryMode::Write,
        Some("readwrite") | None => MemoryMode::ReadWrite, // Default for chat
        _ => MemoryMode::ReadWrite,
    };

    let memory_policy = match arg_memory_policy.as_deref() {
        Some("append") => MemoryPolicy::Append,
        Some("replace") => MemoryPolicy::Replace,
        Some("auto") | None => MemoryPolicy::Auto, // Default
        _ => MemoryPolicy::Auto,
    };

    // Parse memory types
    let memory_types: Vec<MemoryType> = arg_memory_types
        .as_ref()
        .map(|s| s.split(',').filter_map(|t| match t.trim() {
            "persona" => Some(MemoryType::Persona),
            "conversation" => Some(MemoryType::Conversation),
            "fact" => Some(MemoryType::Fact),
            _ => None,
        }).collect())
        .unwrap_or_else(|| vec![MemoryType::Persona, MemoryType::Conversation]);

    let memory_config = MemoryConfig {
        mode: memory_mode,
        policy: memory_policy,
        debug: arg_memory_debug,
        vector_threshold: arg_memory_threshold.unwrap_or(0.78),
        vector_top_k: arg_memory_k.unwrap_or(3),
        vector_types: memory_types,
    };

    // Handle memory wipe if requested
    if let Some(persona_to_wipe) = arg_memory_wipe {
        let persona = if persona_to_wipe == "all" {
            None
        } else {
            Some(persona_to_wipe.as_str())
        };

        match pm.lock().await.reset_memory(persona) {
            Ok(_) => {
                if let Some(persona_name) = persona {
                    println!("Memory wiped for persona: {}", persona_name);
                } else {
                    println!("All memory wiped");
                }
                std::process::exit(0); // Exit successfully after wiping
            }
            Err(e) => {
                eprintln!("Failed to wipe memory: {}", e);
                std::process::exit(1); // Exit with error
            }
        }
    }

    // Choose and load model BEFORE starting REPL
    let model_id = if let Some(m) = arg_model.clone() {
        m
    } else {
        let list = state.llms.list_models();
        if list.is_empty() {
            eprintln!("No chat models available.");
            return Ok(());
        }
        list[0].clone()
    };

    // Load model synchronously before starting REPL
    println!("🤖 Loading LLM: {}", model_id);
    let llm = match state.llms.get(&model_id) {
        Ok(Some(m)) => m,
        Ok(None) => {
            eprintln!("Unknown model {}", model_id);
            return Ok(());
        }
        Err(e) => {
            eprintln!("Failed to load model {}: {}", model_id, e);
            return Ok(());
        }
    };
    println!("🧠 Streaming LLM — type /exit to quit");

    let stdin = io::stdin();

    loop {
        print!("> ");
        let _ = io::stdout().flush();

        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() {
            continue;
        }
        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }

        if input == "/exit" {
            return Ok(());
        }
        if input == "/clear" {
            convo.lock().await.clear();
            print!("\x1b[2J\x1b[H");
            let _ = io::stdout().flush();
            continue;
        }
        if let Some(rest) = input.strip_prefix("/system ") {
            *system_msg.lock().await = Some(rest.trim().to_string());
            println!("(system set)");
            continue;
        }

        {
            let mut c = convo.lock().await;
            c.push(ChatMessage {
                role: "user".to_string(),
                content: input.clone(),
            });
        }

        let messages = {
            let mut msgs: Vec<ChatMessage> = Vec::new();
            msgs.extend(convo.lock().await.clone());
            msgs
        };

        // Build system + persona memory + convo messages
        let persona_name = arg_persona.clone().unwrap_or_else(|| "default".to_string());
        let system_text = state
            .system_prompt
            .get_system_prompt(system_msg.lock().await.as_deref().or(arg_system.as_deref()), arg_persona.as_deref());

        let memory_context = {
            let guard = pm.lock().await;

            // Get keyword-based memory context
            let user_context = convo.lock().await
                .iter()
                .filter(|msg| msg.role == "user")
                .map(|msg| msg.content.as_str())
                .collect::<Vec<&str>>()
                .join(" ");

            let keyword_memory = guard.build_retrieved_memory_context(&persona_name, &user_context, &memory_config).unwrap_or_default();

            // Get vector-based memory context using current user input
            if memory_config.debug {
                println!("[MEMORY] Embeddings available: {}, Vector types: {:?}", state.embeddings_available, memory_config.vector_types);
            }
            let vector_memory = if !memory_config.vector_types.is_empty() && state.embeddings_available {
                match state.model.infer(&input).await {
                    Ok(embedding) => {
                        guard.build_vector_memory_context(&persona_name, &embedding, &memory_config).unwrap_or_default()
                    }
                    Err(e) => {
                        if memory_config.debug {
                            println!("[MEMORY] Failed to compute embedding for vector search: {}", e);
                        }
                        String::new()
                    }
                }
            } else {
                if !state.embeddings_available && memory_config.debug && !memory_config.vector_types.is_empty() {
                    println!("[MEMORY] Vector memory disabled - embedding model not available");
                }
                String::new()
            };

            // Combine both types of memory
            if keyword_memory.is_empty() && vector_memory.is_empty() {
                String::new()
            } else if keyword_memory.is_empty() {
                vector_memory
            } else if vector_memory.is_empty() {
                keyword_memory
            } else {
                format!("{}\n\n{}", keyword_memory, vector_memory)
            }
        };

        // Build messages with integrated memory context
        let mut prompt_messages = state.system_prompt.build_chat_messages(
            &messages,
            system_msg.lock().await.as_deref().or(arg_system.as_deref()),
            arg_persona.as_deref(),
            Some(&memory_context),
        );


        let mut opts = GenerateOptions::default();
        opts.messages = prompt_messages;

        let last_user = input.clone();

        // Use new streaming method for proper message handling
        let stream_result = llm.stream_generate(opts).await;

        match stream_result {
            Ok(mut stream) => {
                let mut collected = String::new();
                print!("\x1b[96m");
                while let Some(tok) = stream.next().await {
                    print!("{}", tok);
                    let _ = io::stdout().flush();
                    collected.push_str(&tok);
                }
                print!("\x1b[0m\n");

                {
                    let mut c = convo.lock().await;
                    c.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: collected.trim().to_string(),
                    });
                }

                // Update memory with intelligent heuristics and embeddings
                let status_note = if memory_config.mode.should_write() {
                    if memory_config.debug {
                        println!("[MEMORY] Processing memory write for user input '{}' ({} chars)",
                                &last_user[..last_user.len().min(100)],
                                last_user.len());
                    }

                    // Compute embedding from the SAME content that will be stored (embedding invariant)
                    // Predict the memory type and extract content that will be stored
                    let predicted_memory_type = if memory_config.policy == crate::persona_memory::MemoryPolicy::Auto {
                        if crate::persona_memory::IntelligentMemory::is_identity_content(&last_user) {
                            crate::persona_memory::MemoryType::Persona
                        } else {
                            crate::persona_memory::MemoryType::Conversation
                        }
                    } else {
                        crate::persona_memory::MemoryType::Conversation // Default for CLI
                    };

                    let content_to_embed = crate::persona_memory::IntelligentMemory::extract_memory_content(
                        predicted_memory_type, &last_user, "");

                    let embedding = if state.embeddings_available {
                        if memory_config.debug {
                            println!("[MEMORY] Computing embedding from predicted stored content '{}' ({} chars)...",
                                    &content_to_embed[..content_to_embed.len().min(50)],
                                    content_to_embed.len());
                        }
                        match state.model.infer(&content_to_embed).await {
                            Ok(emb) => {
                                if memory_config.debug {
                                    println!("[MEMORY] User input embedding computed successfully ({} dims)", emb.len());
                                }
                                Some(emb)
                            }
                            Err(e) => {
                                if memory_config.debug {
                                    println!("[MEMORY] Failed to compute embedding for storage: {}", e);
                                }
                                None
                            }
                        }
                    } else {
                        if memory_config.debug {
                            println!("[MEMORY] Skipping embedding storage - embedding model not available");
                        }
                        None
                    };

                    let pmem = pm.lock().await;
                    match pmem.update_memory_with_embedding_sync(MemoryType::Conversation, &persona_name, &last_user, &collected.trim(), &memory_config, embedding.as_deref()) {
                        Ok(memory_was_written) => {
                            if memory_was_written && memory_config.debug {
                                Some("[memory updated]".to_string())
                            } else {
                                None
                            }
                        }
                        Err(e) => {
                            if memory_config.debug {
                                println!("[MEMORY] Memory update failed: {}", e);
                            }
                            Some("[memory update failed]".to_string())
                        }
                    }
                } else {
                    None
                };
                if let Some(note) = status_note {
                    println!("{}", note);
                }
            }
            Err(err) => {
                eprintln!("Error: {}", err);
            }
        }
    }
}

