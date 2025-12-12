use std::{collections::HashMap, fs};
use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use tracing::{info, warn};
use crate::llm::{LlmModelTrait, CandleLlm};

#[derive(Clone)]
pub struct ModelMetadata {
    pub path: String,
}

pub struct LlmRegistry {
    pub models: HashMap<String, ModelMetadata>,
    loaded_models: Mutex<HashMap<String, Arc<dyn LlmModelTrait>>>,
}

impl LlmRegistry {
    pub async fn discover_models(base_dir: &str) -> Result<Self> {
        let mut models: HashMap<String, ModelMetadata> = HashMap::new();
        let base = Path::new(base_dir);
        if !base.exists() {
            warn!("LLM base dir not found: {}", base_dir);
            return Ok(Self {
                models,
                loaded_models: Mutex::new(HashMap::new()),
            });
        }

        for entry in fs::read_dir(base)? {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    warn!("Skipping entry read error: {}", e);
                    continue;
                }
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let id = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };


            // Check for required files (fast validation only)
            let has_config = path.join("config.json").exists();
            let has_tokenizer = path.join("tokenizer.json").exists() || path.join("tokenizer.model").exists();
            let has_weights = fs::read_dir(&path)
                .map(|mut it| it.any(|f| f.ok().map_or(false, |f| f.path().extension().map_or(false, |e| e == "safetensors"))))
                .unwrap_or(false);

            if !has_config || !has_tokenizer || !has_weights {
                warn!("Skipping model {}: missing config/tokenizer/weights", id);
                continue;
            }

            // Just store metadata, don't load the model yet
            models.insert(id, ModelMetadata {
                path: path.to_string_lossy().to_string(),
            });
        }

        Ok(Self {
            models,
            loaded_models: Mutex::new(HashMap::new()),
        })
    }

    pub fn get(&self, id: &str) -> Result<Option<Arc<dyn LlmModelTrait>>> {
        if let Some(metadata) = self.models.get(id) {
            let mut loaded = self.loaded_models.lock().unwrap();
            if let Some(model) = loaded.get(id) {
                return Ok(Some(model.clone()));
            }

            // Lazy load the model
            match CandleLlm::new(&metadata.path) {
                Ok(model) => {
                    let model_arc = Arc::new(model);
                    loaded.insert(id.to_string(), model_arc.clone());
                    Ok(Some(model_arc))
                }
                Err(e) => {
                    warn!("Failed to load model {}: {}", id, e);
                    Err(e)
                }
            }
        } else {
            Ok(None)
        }
    }

    pub fn list_models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.models.keys().cloned().collect();
        v.sort();
        v
    }
}

