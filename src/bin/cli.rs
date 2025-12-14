use std::io::{self, Write};
use reqwest::Client;
use serde_json::json;
use tokio_stream::StreamExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

#[derive(Debug)]
enum StreamMsg {
    Token(String),
    Done,
    Error(String),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::new();
    let base_url = "http://localhost:3000";

    let args: Vec<String> = std::env::args().collect();
    let mut arg_model: Option<String> = None;
    let mut arg_persona: Option<String> = None;
    let mut arg_list_models: bool = false;
    let mut i = 1; // Skip program name

    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    arg_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--persona" => {
                if i + 1 < args.len() {
                    arg_persona = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--list-models" => {
                arg_list_models = true;
            }
            _ => {}
        }
        i += 1;
    }

    // Handle --list-models option
    if arg_list_models {
        let response = client
            .get(format!("{}/v1/models", base_url))
            .send()
            .await?;

        if response.status().is_success() {
            let models: serde_json::Value = response.json().await?;
            println!("📋 Available LLM models:");
            if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
                for model in data {
                    if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                        println!("  • {}", id);
                    }
                }
            }
        } else {
            eprintln!("❌ Failed to fetch models: {}", response.status());
        }
        return Ok(());
    }

    // Interactive chat mode
    let model = if let Some(m) = arg_model {
        m
    } else {
        // Query available models and use the first one
        let response = client
            .get(format!("{}/v1/models", base_url))
            .send()
            .await?;

        if response.status().is_success() {
            let models: serde_json::Value = response.json().await?;
            if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
                if let Some(first_model) = data.first() {
                    if let Some(id) = first_model.get("id").and_then(|i| i.as_str()) {
                        id.to_string()
                    } else {
                        "loaded-model".to_string()
                    }
                } else {
                    "loaded-model".to_string()
                }
            } else {
                "loaded-model".to_string()
            }
        } else {
            eprintln!("⚠️  Could not fetch models, using default");
            "loaded-model".to_string()
        }
    };

    let persona = arg_persona.unwrap_or_else(|| "minimal".to_string());

    println!("🧠 Streaming LLM — type /exit to quit");
    println!("🤖 Using LLM: {}", model);
    println!("👤 Using persona: {}", persona);

    let mut stdin = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();

    // Channel-based streaming: generation task sends control messages to input loop for synchronized output
    async fn stream_generation(
        response: reqwest::Response,
        msg_tx: mpsc::Sender<StreamMsg>,
    ) -> anyhow::Result<()> {
        let mut stream = response.bytes_stream();

        // Stream and send tokens over channel (don't write to stdout directly)
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);

                    // Parse SSE format and extract content
                    for line in text.lines() {
                        // Check for explicit [DONE] markers first
                        if line == "data: [DONE]" || line == "[DONE]" {
                            // ✅ Generation complete - send Done and stop parsing
                            let _ = msg_tx.send(StreamMsg::Done).await;
                            return Ok(());
                        }

                        // Parse JSON chunks for semantic completion detection
                        if line.starts_with("data: ") {
                            let json_str = if line.starts_with("data: data: ") {
                                &line[11..] // Remove double "data: " prefix
                            } else if line.starts_with("data: ") {
                                &line[6..] // Remove "data: " prefix
                            } else {
                                continue; // Skip lines that don't start with data:
                            };

                            if let Ok(sse_data) = serde_json::from_str::<serde_json::Value>(json_str) {
                                // Check for finish_reason indicating completion
                                if let Some(choices) = sse_data.get("choices").and_then(|c| c.as_array()) {
                                    if let Some(choice) = choices.get(0) {
                                        // ✅ If finish_reason exists and is not null, generation is complete
                                        if let Some(finish_reason) = choice.get("finish_reason") {
                                            if !finish_reason.is_null() {
                                                // Send any remaining content token first
                                                if let Some(delta) = choice.get("delta") {
                                                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                        let _ = msg_tx.send(StreamMsg::Token(content.to_string())).await;
                                                    }
                                                }
                                                // Then send Done and stop parsing
                                                let _ = msg_tx.send(StreamMsg::Done).await;
                                                return Ok(());
                                            }
                                        }

                                        // Extract and send content tokens
                                        if let Some(delta) = choice.get("delta") {
                                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                // Send token over channel instead of writing directly to stdout
                                                if msg_tx.send(StreamMsg::Token(content.to_string())).await.is_err() {
                                                    // Receiver was dropped, stop streaming
                                                    return Ok(());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = msg_tx.send(StreamMsg::Error(format!("❌ Stream error: {}", e))).await;
                    return Err(e.into());
                }
            }
        }

        // If we reach here without detecting semantic completion, it's an error
        // The stream should have sent [DONE] or finish_reason before closing
        let _ = msg_tx.send(StreamMsg::Error("Stream ended without completion marker".to_string())).await;
        let _ = msg_tx.send(StreamMsg::Done).await;

        Ok(())
    }

    loop {
        // Print prompt and wait for user input
        stdout.write_all(b"> ").await?;
        stdout.flush().await?;

        let mut input = String::new();
        if stdin.read_line(&mut input).await.is_err() {
            continue;
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" {
            break;
        }

        // Create a channel for streaming messages from generation task to input loop
        let (msg_tx, mut msg_rx) = mpsc::channel::<StreamMsg>(100);

        // Clone data for the generation task
        let client_clone = client.clone();
        let base_url_clone = base_url.to_string();
        let model_clone = model.clone();
        let persona_clone = persona.clone();

        // Spawn generation task
        let generation_handle = tokio::spawn(async move {
            let payload = json!({
                "model": model_clone,
                "messages": [{"role": "user", "content": input}],
                "stream": true,
                "persona": persona_clone,
                "memory_update": "disable"
            });

            match client_clone
                .post(format!("{}/v1/chat/completions", base_url_clone))
                .header("Content-Type", "application/json")
                .body(payload.to_string())
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        // Stream messages over channel
                        stream_generation(response, msg_tx).await
                    } else {
                        let error_msg = format!("❌ Request failed: {}", response.status());
                        let _ = msg_tx.send(StreamMsg::Error(error_msg)).await;
                        let _ = msg_tx.send(StreamMsg::Done).await; // Always send Done after error
                        Err(anyhow::anyhow!("Request failed"))
                    }
                }
                Err(e) => {
                    let error_msg = format!("❌ HTTP error: {}", e);
                    let _ = msg_tx.send(StreamMsg::Error(error_msg)).await;
                    let _ = msg_tx.send(StreamMsg::Done).await; // Always send Done after error
                    Err(e.into())
                }
            }
        });

        // Input loop receives and processes messages from generation task
        // This ensures only ONE task writes to stdout and input loop waits for completion
        while let Some(msg) = msg_rx.recv().await {
            match msg {
                StreamMsg::Token(token) => {
                    stdout.write_all(token.as_bytes()).await?;
                    stdout.flush().await?;
                }
                StreamMsg::Error(error_msg) => {
                    stdout.write_all(error_msg.as_bytes()).await?;
                    stdout.flush().await?;
                }
                StreamMsg::Done => {
                    // Streaming finished - break and show next prompt
                    break;
                }
            }
        }

        // Wait for generation task to complete (important for cleanup)
        let _ = generation_handle.await;

        // Add newline after response completes, then loop back for next prompt
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
}
