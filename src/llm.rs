use async_trait::async_trait;
use anyhow::Result;
use std::sync::Arc;
use candle::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, LlamaConfig, Config as LlamaInnerConfig, Cache};
use candle_transformers::models::mistral::{Model as Mistral, Config as MistralConfig};
use candle_transformers::models::phi::{Model as Phi, Config as PhiConfig};
use std::sync::Mutex;

#[derive(Clone)]
pub enum ModelType {
    Llama(Arc<Llama>),
    Mistral(Arc<Mutex<Mistral>>),
    Phi(Arc<Mutex<Phi>>),
}

#[derive(Clone)]
pub enum ConfigType {
    Llama(Arc<LlamaInnerConfig>),
    Mistral(Arc<MistralConfig>),
    Phi(Arc<PhiConfig>),
}
use tokenizers::Tokenizer;
use serde_json;
use std::fs;
use std::path::Path;
use tracing::info;
use serde::{Deserialize, Serialize};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[async_trait]
pub trait LlmModelTrait: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<String>;
    async fn generate_with_options(&self, options: GenerateOptions) -> Result<String>;
    #[allow(dead_code)]
    async fn stream(&self, options: GenerateOptions) -> Result<ReceiverStream<String>>;
    async fn stream_generate(&self, options: GenerateOptions) -> Result<ReceiverStream<String>>;
}

#[cfg(test)]
#[allow(dead_code)]
pub struct StubLlm;

#[cfg(test)]
#[allow(dead_code)]
impl StubLlm {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
#[async_trait]
impl LlmModelTrait for StubLlm {
    async fn generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("Echo: {}", prompt))
    }

    async fn generate_with_options(&self, options: GenerateOptions) -> Result<String> {
        let prompt = CandleLlm::format_chatml(&options.messages);
        self.generate(&prompt).await
    }

    async fn stream(&self, options: GenerateOptions) -> Result<ReceiverStream<String>> {
        let text = self.generate_with_options(options).await?;
        let (tx, rx) = mpsc::channel(16);
        tokio::spawn(async move {
            for token in text.split_whitespace() {
                if tx.send(token.to_string()).await.is_err() {
                    break;
                }
            }
        });
        Ok(ReceiverStream::new(rx))
    }

    async fn stream_generate(&self, options: GenerateOptions) -> Result<ReceiverStream<String>> {
        let (tx, rx) = mpsc::channel(16);

        tokio::spawn(async move {
            // Emit deterministic fake tokens for testing streaming
            let fake_tokens = vec![
                "Test", " ", "token", " ", "1", " ", "Test", " ", "token", " ", "2", " ", "Test", " ", "token", " ", "3"
            ];

            for token in fake_tokens {
                // Small delay to simulate real streaming
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

                if tx.send(token.to_string()).await.is_err() {
                    break; // Receiver dropped
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }
}

pub struct CandleLlm {
    device: Device,
    tokenizer: Tokenizer,
    model: ModelType,
    config: ConfigType,
    dtype: DType,
    eos_token: Option<u32>,
}

impl CandleLlm {
    pub fn new(model_dir: &str) -> anyhow::Result<Self> {
        tracing::info!("Loading CandleLlm from directory: {}", model_dir);
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);

        let tokenizer = load_tokenizer(model_dir)?;

        let eos_token = tokenizer.get_vocab(true).get("</s>").copied()
            .or_else(|| tokenizer.get_vocab(true).get("<eos>").copied());

        let config_path = format!("{}/config.json", model_dir);
        // Read the raw config to determine model type
        let config_content = std::fs::read(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_slice(&config_content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

        // Check model type to determine which loader to use
        let model_type = config_json.get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");

        // Support sharded safetensors
        let weight_paths = load_safetensor_paths(model_dir)?;
        let weight_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();

        // Use F32 for broad device compatibility (Metal BF16 matmul is unsupported).
        let dtype = DType::F32;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_refs, dtype, &device)? };

        match model_type {
            "mistral" => {
                let config: MistralConfig = serde_json::from_slice(&config_content)?;
                let model = Mistral::new(&config, vb)?;
                Ok(Self {
                    device,
                    tokenizer,
                    model: ModelType::Mistral(Arc::new(Mutex::new(model))),
                    config: ConfigType::Mistral(Arc::new(config)),
                    dtype,
                    eos_token,
                })
            }
            "phi" => {
                let config: PhiConfig = serde_json::from_slice(&config_content)?;
                let model = Phi::new(&config, vb)?;
                Ok(Self {
                    device,
                    tokenizer,
                    model: ModelType::Phi(Arc::new(Mutex::new(model))),
                    config: ConfigType::Phi(Arc::new(config)),
                    dtype,
                    eos_token,
                })
            }
            _ => {
                // Default to Llama for backward compatibility
                let config_raw: LlamaConfig = serde_json::from_slice(&config_content)?;
                let config = config_raw.into_config(false);
                let model = Llama::load(vb, &config)?;
                Ok(Self {
                    device,
                    tokenizer,
                    model: ModelType::Llama(Arc::new(model)),
                    config: ConfigType::Llama(Arc::new(config)),
                    dtype,
                    eos_token,
                })
            }
        }
    }

    fn generate_blocking(
        model: ModelType,
        config: ConfigType,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
    ) -> anyhow::Result<String> {
        match (model, config) {
            (ModelType::Llama(model), ConfigType::Llama(config)) => {
                Self::generate_blocking_llama(
                    model,
                    config,
                    tokenizer,
                    device,
                    dtype,
                    eos_token,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                )
            }
            (ModelType::Mistral(model), ConfigType::Mistral(config)) => {
                Self::generate_blocking_mistral(
                    model,
                    config,
                    tokenizer,
                    device,
                    dtype,
                    eos_token,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                )
            }
            _ => Err(anyhow::anyhow!("Model and config type mismatch")),
        }
    }

    fn generate_blocking_llama(
        model: Arc<Llama>,
        config: Arc<LlamaInnerConfig>,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
    ) -> anyhow::Result<String> {
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        let mut cache = Cache::new(true, dtype, &config, &device)?;
        let mut position = 0usize;

        info!("llama: starting generation, prompt_tokens={}", tokens.len());

        let mut rng = rand::thread_rng();
        let mut logits_buf: Vec<f32> = Vec::new();
        let mut indices_buf: Vec<usize> = Vec::new();

        for _ in 0..max_tokens {
            let input_tokens: Vec<u32> = if position == 0 {
                tokens.clone()
            } else {
                vec![*tokens.last().unwrap()]
            };

            let input = Tensor::new(input_tokens.as_slice(), &device)?.unsqueeze(0)?;

            let logits = model.forward(&input, position, &mut cache)?;

            let mut logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;

            // Repetition penalty
            if repetition_penalty > 1.0 {
                for &t in tokens.iter() {
                    if let Some(v) = logits_vec.get_mut(t as usize) {
                        *v /= repetition_penalty;
                    }
                }
            }

            let next = if temperature == 0.0 {
                logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                // temperature scaling
                if (temperature - 1.0).abs() > f32::EPSILON {
                    for v in logits_vec.iter_mut() {
                        *v /= temperature;
                    }
                }

                // top-k
                let vocab = logits_vec.len();
                let k = if top_k == 0 || top_k > vocab { vocab } else { top_k };
                indices_buf.clear();
                indices_buf.extend(0..vocab);
                indices_buf.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
                    logits_vec[b]
                        .partial_cmp(&logits_vec[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                indices_buf.truncate(k);

                // top-p on reduced set
                indices_buf.sort_by(|&a, &b| {
                    logits_vec[b]
                        .partial_cmp(&logits_vec[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                logits_buf.clear();
                for &idx in &indices_buf {
                    logits_buf.push(logits_vec[idx]);
                }

                // softmax with max trick
                if let Some(max_logit) = logits_buf.iter().cloned().reduce(f32::max) {
                    let mut sum = 0.0f32;
                    for v in logits_buf.iter_mut() {
                        *v = (*v - max_logit).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in logits_buf.iter_mut() {
                            *v /= sum;
                        }
                    }
                }

                // nucleus filtering
                let mut cumulative = 0.0f32;
                let mut cutoff = logits_buf.len();
                for (i, p) in logits_buf.iter().enumerate() {
                    cumulative += *p;
                    if cumulative >= top_p {
                        cutoff = i + 1;
                        break;
                    }
                }
                if cutoff == 0 {
                    cutoff = 1;
                } else if cutoff > logits_buf.len() {
                    cutoff = logits_buf.len();
                }
                logits_buf.truncate(cutoff);
                indices_buf.truncate(cutoff);

                // renormalize
                let mut sum = logits_buf.iter().sum::<f32>();
                if sum <= 0.0 {
                    sum = 1.0;
                }
                for v in logits_buf.iter_mut() {
                    *v /= sum;
                }

                let dist = WeightedIndex::new(&logits_buf)
                    .unwrap_or_else(|_| WeightedIndex::new(vec![1.0]).unwrap());
                let sample_idx = dist.sample(&mut rng);
                indices_buf.get(sample_idx).cloned().unwrap_or(0) as u32
            };

            tokens.push(next);

            position += input_tokens.len();

            if let Some(eos) = eos_token {
                if next == eos {
                    break;
                }
            }
        }

        let generated = &tokens[prompt_len..];
        let text = tokenizer.decode(generated, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
        info!("llm: finished, total_tokens={}", tokens.len());
        Ok(text)
    }

    fn generate_blocking_mistral(
        model: Arc<Mutex<Mistral>>,
        config: Arc<MistralConfig>,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
    ) -> anyhow::Result<String> {
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        info!("mistral: starting generation, prompt_tokens={}", tokens.len());

        let mut rng = rand::thread_rng();
        let mut seqlen_offset = 0usize;

        for _ in 0..max_tokens {
            let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.lock().unwrap().forward(&input, seqlen_offset)?;
            let logits = logits.squeeze(0)?.to_vec1::<f32>()?;
            seqlen_offset += tokens.len();

            // Apply repetition penalty
            let mut next_token_logits = logits.clone();
            if repetition_penalty > 1.0 {
                for &t in tokens.iter() {
                    if let Some(v) = next_token_logits.get_mut(t as usize) {
                        *v /= repetition_penalty;
                    }
                }
            }

            // Apply temperature
            let next_token_logits = if temperature > 0.0 {
                next_token_logits.iter().map(|&x| x / temperature).collect::<Vec<_>>()
            } else {
                next_token_logits
            };

            // Sample from the distribution
            let next_token = if top_k > 1 {
                // Top-k sampling
                let mut logits_with_indices: Vec<(f32, usize)> = next_token_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &logit)| (logit, i))
                    .collect();

                logits_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                logits_with_indices.truncate(top_k);

                let top_k_logits: Vec<f32> = logits_with_indices.iter().map(|(logit, _)| *logit).collect();
                let top_k_indices: Vec<usize> = logits_with_indices.iter().map(|(_, idx)| *idx).collect();

                let dist = rand::distributions::WeightedIndex::new(&top_k_logits)?;
                let sampled_idx = top_k_indices[dist.sample(&mut rng)];
                sampled_idx as u32
            } else {
                // Greedy decoding
                next_token_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            // Check for EOS token
            if Some(next_token) == eos_token {
                break;
            }
        }

        // Decode the generated tokens
        let generated_tokens = &tokens[prompt_len..];
        let text = tokenizer.decode(generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

        info!("mistral: finished, total_tokens={}", tokens.len());
        Ok(text)
    }

    fn stream_generate_blocking(
        model: ModelType,
        config: ConfigType,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        tx: mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        match (model, config) {
            (ModelType::Llama(model), ConfigType::Llama(config)) => {
                Self::stream_generate_blocking_llama(
                    model,
                    config,
                    tokenizer,
                    device,
                    dtype,
                    eos_token,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    tx,
                )
            }
            (ModelType::Mistral(model), ConfigType::Mistral(config)) => {
                Self::stream_generate_blocking_mistral(
                    model,
                    config,
                    tokenizer,
                    device,
                    dtype,
                    eos_token,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    tx,
                )
            }
            (ModelType::Phi(model), ConfigType::Phi(config)) => {
                Self::stream_generate_blocking_phi(
                    model,
                    config,
                    tokenizer,
                    device,
                    dtype,
                    eos_token,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    tx,
                )
            }
            _ => Err(anyhow::anyhow!("Model and config type mismatch")),
        }
    }

    fn stream_generate_blocking_llama(
        model: Arc<Llama>,
        config: Arc<LlamaInnerConfig>,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        tx: mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        let mut cache = Cache::new(true, dtype, &config, &device)?;
        let mut position = 0usize;

        info!("llm: starting streaming generation, prompt_tokens={}", tokens.len());

        let mut rng = rand::thread_rng();
        let mut logits_buf: Vec<f32> = Vec::new();
        let mut indices_buf: Vec<usize> = Vec::new();

        for _ in 0..max_tokens {
            let input_tokens: Vec<u32> = if position == 0 {
                tokens.clone()
            } else {
                vec![*tokens.last().unwrap()]
            };

            let input = Tensor::new(input_tokens.as_slice(), &device)?.unsqueeze(0)?;

            let logits = model.forward(&input, position, &mut cache)?;

            let mut logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;

            // Repetition penalty
            if repetition_penalty > 1.0 {
                for &t in tokens.iter() {
                    if let Some(v) = logits_vec.get_mut(t as usize) {
                        *v /= repetition_penalty;
                    }
                }
            }

            let next = if temperature == 0.0 {
                logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            } else {
                // temperature scaling
                if (temperature - 1.0).abs() > f32::EPSILON {
                    for v in logits_vec.iter_mut() {
                        *v /= temperature;
                    }
                }

                // top-k
                let vocab = logits_vec.len();
                let k = if top_k == 0 || top_k > vocab { vocab } else { top_k };
                indices_buf.clear();
                indices_buf.extend(0..vocab);
                indices_buf.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
                    logits_vec[b]
                        .partial_cmp(&logits_vec[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                indices_buf.truncate(k);

                // top-p on reduced set
                indices_buf.sort_by(|&a, &b| {
                    logits_vec[b]
                        .partial_cmp(&logits_vec[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                logits_buf.clear();
                for &idx in &indices_buf {
                    logits_buf.push(logits_vec[idx]);
                }

                // softmax with max trick
                if let Some(max_logit) = logits_buf.iter().cloned().reduce(f32::max) {
                    let mut sum = 0.0f32;
                    for v in logits_buf.iter_mut() {
                        *v = (*v - max_logit).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in logits_buf.iter_mut() {
                            *v /= sum;
                        }
                    }
                }

                // nucleus filtering
                let mut cumulative = 0.0f32;
                let mut cutoff = logits_buf.len();
                for (i, p) in logits_buf.iter().enumerate() {
                    cumulative += *p;
                    if cumulative >= top_p {
                        cutoff = i + 1;
                        break;
                    }
                }
                if cutoff == 0 {
                    cutoff = 1;
                } else if cutoff > logits_buf.len() {
                    cutoff = logits_buf.len();
                }
                logits_buf.truncate(cutoff);
                indices_buf.truncate(cutoff);

                // renormalize
                let mut sum = logits_buf.iter().sum::<f32>();
                if sum <= 0.0 {
                    sum = 1.0;
                }
                for v in logits_buf.iter_mut() {
                    *v /= sum;
                }

                let dist = WeightedIndex::new(&logits_buf)
                    .unwrap_or_else(|_| WeightedIndex::new(vec![1.0]).unwrap());
                let sample_idx = dist.sample(&mut rng);
                indices_buf.get(sample_idx).cloned().unwrap_or(0) as u32
            };

            // Check for EOS token before processing
            if let Some(eos) = eos_token {
                if next == eos {
                    break;
                }
            }

            tokens.push(next);

            position += input_tokens.len();

            // Send the new token as it gets generated
            if let Ok(new_text) = tokenizer.decode(&[next], true) {
                if tx.blocking_send(new_text).is_err() {
                    // Receiver was dropped, stop generation
                    break;
                }
            }
        }

        info!("llm: finished streaming, total_tokens={}", tokens.len());
        Ok(())
    }

    fn stream_generate_blocking_mistral(
        model: Arc<Mutex<Mistral>>,
        config: Arc<MistralConfig>,
        tokenizer: Tokenizer,
        device: Device,
        dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        tx: mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        info!("mistral: starting streaming generation, prompt_tokens={}", tokens.len());

        let mut rng = rand::thread_rng();
        let mut seqlen_offset = 0usize;

        for _ in 0..max_tokens {
            let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.lock().unwrap().forward(&input, seqlen_offset)?;
            let logits = logits.squeeze(0)?.to_vec1::<f32>()?;
            seqlen_offset += tokens.len();

            // Apply repetition penalty
            let mut next_token_logits = logits.clone();
            if repetition_penalty > 1.0 {
                for &t in tokens.iter() {
                    if let Some(v) = next_token_logits.get_mut(t as usize) {
                        *v /= repetition_penalty;
                    }
                }
            }

            // Apply temperature
            let next_token_logits = if temperature > 0.0 {
                next_token_logits.iter().map(|&x| x / temperature).collect::<Vec<_>>()
            } else {
                next_token_logits
            };

            // Sample from the distribution
            let next_token = if top_k > 1 {
                // Top-k sampling
                let mut logits_with_indices: Vec<(f32, usize)> = next_token_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &logit)| (logit, i))
                    .collect();

                logits_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                logits_with_indices.truncate(top_k);

                let top_k_logits: Vec<f32> = logits_with_indices.iter().map(|(logit, _)| *logit).collect();
                let top_k_indices: Vec<usize> = logits_with_indices.iter().map(|(_, idx)| *idx).collect();

                let dist = rand::distributions::WeightedIndex::new(&top_k_logits)?;
                let sampled_idx = top_k_indices[dist.sample(&mut rng)];
                sampled_idx as u32
            } else {
                // Greedy decoding
                next_token_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            // Check for EOS token
            if Some(next_token) == eos_token {
                break;
            }

            // Send the new token as it gets generated
            if let Ok(new_text) = tokenizer.decode(&[next_token], true) {
                let _ = tx.send(new_text);
            }
        }

        info!("mistral: finished streaming, total_tokens={}", tokens.len());
        Ok(())
    }

    fn stream_generate_blocking_phi(
        model: Arc<Mutex<Phi>>,
        _config: Arc<PhiConfig>,
        tokenizer: Tokenizer,
        device: Device,
        _dtype: DType,
        eos_token: Option<u32>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        tx: mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        info!("phi: starting streaming generation, prompt_tokens={}", tokens.len());

        let mut rng = rand::thread_rng();

        for _ in 0..max_tokens {
            let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = model.lock().unwrap().forward(&input)?;
            let logits = logits.squeeze(0)?.to_vec1::<f32>()?;

            // Apply repetition penalty
            let mut next_token_logits = logits.clone();
            if repetition_penalty > 1.0 {
                for &t in tokens.iter() {
                    if let Some(v) = next_token_logits.get_mut(t as usize) {
                        *v /= repetition_penalty;
                    }
                }
            }

            // Apply temperature
            let next_token_logits = if temperature > 0.0 {
                next_token_logits.iter().map(|&x| x / temperature).collect::<Vec<_>>()
            } else {
                next_token_logits
            };

            // Sample from the distribution
            let next_token = if top_k > 1 {
                // Top-k sampling
                let mut logits_with_indices: Vec<(f32, usize)> = next_token_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &logit)| (logit, i))
                    .collect();

                logits_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                logits_with_indices.truncate(top_k);

                let top_k_logits: Vec<f32> = logits_with_indices.iter().map(|(logit, _)| *logit).collect();
                let top_k_indices: Vec<usize> = logits_with_indices.iter().map(|(_, idx)| *idx).collect();

                let dist = rand::distributions::WeightedIndex::new(&top_k_logits)?;
                let sampled_idx = top_k_indices[dist.sample(&mut rng)];
                sampled_idx as u32
            } else {
                // Greedy decoding
                next_token_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            // Check for EOS token
            if Some(next_token) == eos_token {
                break;
            }

            // Send the new token as it gets generated
            if let Ok(new_text) = tokenizer.decode(&[next_token], true) {
                let _ = tx.send(new_text);
            }
        }

        info!("phi: finished streaming, total_tokens={}", tokens.len());
        Ok(())
    }
}

#[async_trait]
impl LlmModelTrait for CandleLlm {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let mut opts = GenerateOptions::default();
        opts.messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        self.generate_with_options(opts).await
    }

    async fn generate_with_options(&self, options: GenerateOptions) -> Result<String> {
        let prompt = CandleLlm::format_chatml(&options.messages);

        let model = self.model.clone();
        let config = self.config.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let dtype = self.dtype;
        let eos = self.eos_token;
        let max_tokens = options.max_tokens;
        let temperature = options.temperature;
        let top_p = options.top_p;
        let top_k = options.top_k;
        let repetition_penalty = options.repetition_penalty;

        let output = tokio::task::spawn_blocking(move || {
            CandleLlm::generate_blocking(
                model,
                config,
                tokenizer,
                device,
                dtype,
                eos,
                &prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            )
        }).await??;

        Ok(output)
    }

    async fn stream(&self, options: GenerateOptions) -> Result<ReceiverStream<String>> {
        let prompt = CandleLlm::format_chatml(&options.messages);

        let model = self.model.clone();
        let config = self.config.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let dtype = self.dtype;
        let eos = self.eos_token;
        let max_tokens = options.max_tokens;
        let temperature = options.temperature;
        let top_p = options.top_p;
        let top_k = options.top_k;
        let repetition_penalty = options.repetition_penalty;

        let (tx, rx) = mpsc::channel::<String>(32);
        tokio::task::spawn_blocking(move || {
            let result = CandleLlm::generate_blocking(
                model,
                config,
                tokenizer,
                device,
                dtype,
                eos,
                &prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            );

            if let Ok(text) = result {
                for token in text.split_whitespace() {
                    if tx.blocking_send(token.to_string()).is_err() {
                        break;
                    }
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    async fn stream_generate(&self, options: GenerateOptions) -> Result<ReceiverStream<String>> {
        let (tx, rx) = mpsc::channel::<String>(32);

        let model = self.model.clone();
        let config = self.config.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let dtype = self.dtype;
        let eos_token = self.eos_token;
        let prompt = CandleLlm::format_chatml(&options.messages);
        let max_tokens = options.max_tokens;
        let temperature = options.temperature;
        let top_p = options.top_p;
        let top_k = options.top_k;
        let repetition_penalty = options.repetition_penalty;

        tokio::task::spawn_blocking(move || {
            let result = CandleLlm::stream_generate_blocking(
                model, config, tokenizer, device, dtype, eos_token, &prompt,
                max_tokens, temperature, top_p, top_k, repetition_penalty, tx
            );
            if let Err(e) = result {
                eprintln!("Streaming error: {}", e);
            }
        });

        Ok(ReceiverStream::new(rx))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            messages: vec![],
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
        }
    }
}

impl CandleLlm {
    pub fn format_chatml(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for m in messages {
            let role = match m.role.as_str() {
                "system" => "system",
                "assistant" => "assistant",
                _ => "user",
            };
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            out.push_str(&m.content);
            out.push('\n');
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    #[allow(dead_code)]
    pub fn apply_chat_template(messages: &[ChatMessage]) -> String {
        CandleLlm::format_chatml(messages)
    }
}

fn load_safetensor_paths(dir: &str) -> anyhow::Result<Vec<String>> {
    let mut files = vec![];

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                files.push(path.to_string_lossy().to_string());
            }
        }
    }

    if files.is_empty() {
        return Err(anyhow::anyhow!("No safetensor files found in {}", dir));
    }

    files.sort();
    Ok(files)
}

fn load_tokenizer(model_dir: &str) -> anyhow::Result<Tokenizer> {
    let json_path = format!("{}/tokenizer.json", model_dir);
    if Path::new(&json_path).exists() {
        let tok = Tokenizer::from_file(&json_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {}", e))?;
        return Ok(tok);
    }

    let sp_path = format!("{}/tokenizer.model", model_dir);
    if Path::new(&sp_path).exists() {
        // The current build of tokenizers cannot load binary tokenizer.model on this setup.
        return Err(anyhow::anyhow!(
            "tokenizer.model found at {}, but this build of tokenizers cannot parse it. Please use tokenizer.json instead.",
            sp_path
        ));
    }

    Err(anyhow::anyhow!(
        "No tokenizer.json or tokenizer.model found in {}",
        model_dir
    ))
}

