use std::io::{self, Write};
use reqwest::Client;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::new();
    let base_url = "http://localhost:3000";

    let args: Vec<String> = std::env::args().collect();
    let mut arg_model: Option<String> = None;
    let mut arg_persona: Option<String> = None;
    let mut arg_list_models: bool = false;
    let mut memory_enabled: bool = false; // Default: memory disabled
    let mut compile_mode: bool = false; // Default: chat mode
    let mut i = 1; // Skip program name

    while i < args.len() {
        match args[i].as_str() {
            "--compile" => {
                compile_mode = true;
            }
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
            "--memory" => {
                memory_enabled = true;
            }
            "--no-memory" => {
                memory_enabled = false; // Explicit disable (already default)
            }
            _ => {}
        }
        i += 1;
    }

    // Handle --list-models option
    if arg_list_models {
        match client.get(format!("{}/v1/models", base_url)).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<serde_json::Value>().await {
                        Ok(models) => {
                            println!("📋 Available LLM models:");
                            if let Some(data) = models.get("data").and_then(|d| d.as_array()) {
                                for model in data {
                                    if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                                        println!("  • {}", id);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to parse server response: {}", e);
                            eprintln!("💡 Make sure the llmd server is running on {}", base_url);
                        }
                    }
                } else {
                    eprintln!("❌ Server returned error: {}", response.status());
                    eprintln!("💡 Make sure the llmd server is running on {}", base_url);
                }
            }
            Err(e) => {
                eprintln!("❌ Cannot connect to server: {}", e);
                eprintln!("💡 Make sure the llmd server is running on {}", base_url);
                eprintln!("💡 Start it with: echo '{{}}' | cargo run");
            }
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
            eprintln!("❌ Server returned error: {}", response.status());
            eprintln!("💡 Make sure the llmd server is running on {}", base_url);
            std::process::exit(1);
        }
    };

    let persona = arg_persona.unwrap_or_else(|| "minimal".to_string());

    // Quick server connectivity check
    println!("🔍 Checking server connectivity...");
    match client.get(format!("{}/health", base_url)).send().await {
        Ok(response) if response.status().is_success() => {
            println!("✅ Server is running on {}", base_url);
        }
        Ok(response) => {
            eprintln!("❌ Server returned status: {}", response.status());
            eprintln!("💡 Make sure the llmd server is running correctly on {}", base_url);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ Cannot connect to server: {}", e);
            eprintln!("💡 Make sure the llmd server is running on {}", base_url);
            eprintln!("💡 Start it with: echo '{{}}' | cargo run");
            std::process::exit(1);
        }
    }

    if compile_mode {
        println!("🧠 Intent compiler mode — type /exit to quit");
        println!("🧠 Using LLM: {}", model);
    } else {
        println!("🤖 Chat mode (tools disabled) — type /exit to quit");
        println!("🧠 Using LLM: {}", model);
        println!("👤 Using persona: {}", persona);
        println!("💾 Memory: {}", if memory_enabled { "enabled" } else { "disabled" });
    }

    let mut stdin = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();

    // Simple non-streaming chat loop

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

        if compile_mode {
            // Compile mode: use /v1/compile endpoint
            let payload = json!({
                "model": model.clone(),
                "input": input
            });

            match client
                .post(format!("{}/v1/compile", base_url))
                .header("Content-Type", "application/json")
                .body(payload.to_string())
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();

                    if status.is_success() {
                        match serde_json::from_str::<serde_json::Value>(&body) {
                            Ok(json) => {
                                if json.get("object") == Some(&serde_json::json!("compile.result")) {
                                    if let Some(query) = json
                                        .get("output")
                                        .and_then(|o| o.get("data"))
                                        .and_then(|d| d.get("compiled_query"))
                                        .and_then(|q| q.as_str())
                                    {
                                stdout
                                    .write_all("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n".as_bytes())
                                    .await?;
                                stdout
                                    .write_all("📦 Compiled GraphQL Query\n".as_bytes())
                                    .await?;
                                stdout
                                    .write_all("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".as_bytes())
                                    .await?;
                                stdout.write_all(query.as_bytes()).await?;
                                stdout.write_all("\n\n".as_bytes()).await?;
                                stdout.flush().await?;
                                    } else {
                                        // Fallback if structure is unexpected
                                        stdout.write_all(body.as_bytes()).await?;
                                        stdout.write_all(b"\n").await?;
                                    }
                                } else {
                                    // Not a compile result, print raw
                                    stdout.write_all(body.as_bytes()).await?;
                                    stdout.write_all(b"\n").await?;
                                }
                            }
                            Err(_) => {
                                // Not JSON, print raw
                                stdout.write_all(body.as_bytes()).await?;
                                stdout.write_all(b"\n").await?;
                            }
                        }
                        stdout.flush().await?;
                    } else {
                        stdout.write_all(
                            format!("❌ Compile failed ({}):\n{}\n", status, body).as_bytes()
                        ).await?;
                        stdout.flush().await?;
                    }
                }
                Err(e) => {
                    stdout.write_all(format!("❌ HTTP error: {}\n", e).as_bytes()).await?;
                    stdout.flush().await?;
                }
            }
        } else {
            // Chat mode: use /v1/chat/completions endpoint
            let memory_update = if memory_enabled {
                serde_json::Value::Null
            } else {
                serde_json::Value::String("disable".to_string())
            };

            let payload = json!({
                "model": model.clone(),
                "messages": [{"role": "user", "content": input}],
                "stream": false,
                "persona": persona.clone(),
                "memory_update": memory_update
            });

            match client
                .post(format!("{}/v1/chat/completions", base_url))
                .header("Content-Type", "application/json")
                .body(payload.to_string())
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        match response.json::<serde_json::Value>().await {
                            Ok(json_response) => {
                                if let Some(choices) = json_response.get("choices").and_then(|c| c.as_array()) {
                                    if let Some(choice) = choices.get(0) {
                                        if let Some(message) = choice.get("message") {
                                            if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                                                // Print the full response
                                                stdout.write_all(content.as_bytes()).await?;
                                                stdout.write_all(b"\n").await?;
                                                stdout.flush().await?;
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                stdout.write_all(format!("❌ Failed to parse response: {}\n", e).as_bytes()).await?;
                                stdout.flush().await?;
                            }
                        }
                    } else {
                        stdout.write_all(format!("❌ Request failed: {}\n", response.status()).as_bytes()).await?;
                        stdout.flush().await?;
                    }
                }
                Err(e) => {
                    stdout.write_all(format!("❌ HTTP error: {}\n", e).as_bytes()).await?;
                    stdout.flush().await?;
                }
            }
        }
    }

    Ok(())
}
