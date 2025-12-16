use std::time::{SystemTime, UNIX_EPOCH};
use serde::Serialize;
use hnsw_rs::prelude::*;
use crate::memory::storage::Storage;
use anyhow::Result;

#[derive(Clone, Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub text: String,
    pub score: f32,
}

pub struct VectorStore {
    pub hnsw: Hnsw<'static, f32, DistCosine>,
    pub dim: usize,
    pub storage: Storage,
    ef_search: usize,
}

impl VectorStore {
    pub fn new(dim: usize, storage: Storage) -> Self {
        let max_nb_connections = 32;
        let ef_construct = 200;
        let max_elements = 100_000;
        let max_layer = 16;
        let ef_search = 128;

        let hnsw = Hnsw::<f32, DistCosine>::new(
            max_nb_connections,
            max_elements,
            max_layer,
            ef_construct,
            DistCosine {},
        );

        Self {
            hnsw,
            dim,
            storage,
            ef_search,
        }
    }

    pub fn rebuild_index(&mut self) {
        let max_nb_connections = 32;
        let ef_construct = 200;
        let max_elements = 100_000;
        let max_layer = 16;

        self.hnsw = Hnsw::<f32, DistCosine>::new(
            max_nb_connections,
            max_elements,
            max_layer,
            ef_construct,
            DistCosine {},
        );

        let entries = self.storage.load_all_entries();
        for entry in entries {
            // Skip entries with mismatched dimensions
            if entry.embedding.len() != self.dim {
                continue;
            }
            self.hnsw.insert((entry.embedding.as_slice(), entry.id as usize));
        }
    }

    pub fn add(&mut self, text: String, embedding: Vec<f32>) -> Result<u64> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let id = self.storage.insert_entry(&text, timestamp, &embedding)?; 

        if embedding.len() == self.dim {
            self.hnsw.insert((embedding.as_slice(), id as usize));
        }

        Ok(id)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let neighbours = self.hnsw.search(query, k, self.ef_search);

        neighbours.into_iter().filter_map(|n| {
            let id = n.d_id as u64;
            let entry = self.storage.get_entry(id)?;
            let score = 1.0 - n.distance;
            Some(SearchResult {
                id,
                text: entry.text,
                score,
            })
        }).collect()
    }
}