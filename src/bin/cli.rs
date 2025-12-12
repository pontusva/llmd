use std::io::{self, Write};
use reqwest::Client;
use serde_json::json;
use tokio_stream::StreamExt;

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

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        let _ = stdout.flush();

        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            continue;
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" {
            break;
        }

        // Make HTTP request to llmd server
        let payload = json!({
            "model": model,
            "messages": [{"role": "user", "content": input}],
            "stream": true,
            "persona": persona,
            "memory_update": "disable"
        });

        let response = client
            .post(format!("{}/v1/chat/completions", base_url))
            .header("Content-Type", "application/json")
            .body(payload.to_string())
            .send()
            .await?;

        if response.status().is_success() {
            let mut stream = response.bytes_stream();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        // Parse SSE format and extract content
                        for line in text.lines() {
                            if line.starts_with("data: ") && line != "data: [DONE]" {
                                if let Ok(sse_data) = serde_json::from_str::<serde_json::Value>(&line[6..]) {
                                    if let Some(choices) = sse_data.get("choices").and_then(|c| c.as_array()) {
                                        if let Some(choice) = choices.get(0) {
                                            if let Some(delta) = choice.get("delta") {
                                                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                                    print!("{}", content);
                                                    let _ = stdout.flush();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("❌ Stream error: {}", e),
                }
            }
            println!(); // New line after response
        } else {
            eprintln!("❌ Request failed: {}", response.status());
            if let Ok(error_text) = response.text().await {
                eprintln!("Response: {}", error_text);
            }
        }
    }

    Ok(())
}
