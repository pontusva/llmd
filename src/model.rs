use anyhow::Result;
use candle::{DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::Tokenizer;
use std::fs;
use crate::persona_memory::EmbeddingModel;

pub struct Model {
    model: Option<BertModel>,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl Model {
    pub async fn load(path: &str) -> Result<Self> {
        let device = Device::Cpu;

        // ------------------------------
        // Load tokenizer.json
        // ------------------------------
        let tokenizer_path = format!("{}/tokenizer.json", path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // ------------------------------
        // Load config.json
        // ------------------------------
        let config_path = format!("{}/config.json", path);
        let config: BertConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        // ------------------------------
        // Load real model weights
        // ------------------------------
        let weight_paths = load_safetensor_paths(path)?;
        let weight_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_refs, DType::F32, &device)? };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model: Some(model),
            tokenizer: Some(tokenizer),
            device,
        })
    }

    pub fn dummy() -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: Device::Cpu,
        }
    }

    pub async fn infer(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Embedding model not available - vector memory features are disabled")
        })?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Embedding tokenizer not available - vector memory features are disabled")
        })?;

        // Tokenize text
        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        let ids = encoding.get_ids();
        let token_type_ids = encoding.get_type_ids();

        // Convert tokens → Tensor
        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_types = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;

        // Run BERT forward
        let output = model.forward(&input_ids, &token_types, None)?;

        // Extract CLS embedding — output[0, 0, :]
        let cls = output.i((0, 0))?;

        // Convert to Vec<f32>
        let mut emb = cls.to_vec1::<f32>()?;

        // ---- Correct L2 normalization ----
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();

        if norm > 1e-8 {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }
        // ----------------------------------

        Ok(emb)
    }

    pub async fn infer_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.model.is_none() {
            return Err(anyhow::anyhow!("Embedding model not available - vector memory features are disabled"));
        }

        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let emb = self.infer(text).await?;
            embeddings.push(emb);
        }

        Ok(embeddings)
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for Model {
    async fn compute_embedding(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.infer(text).await
    }
}

fn load_safetensor_paths(dir: &str) -> Result<Vec<String>> {
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

    // Ensure shards load in correct order: 1, 2, 3...
    files.sort();
    Ok(files)
}