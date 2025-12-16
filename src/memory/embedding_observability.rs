use std::sync::atomic::{AtomicU64, Ordering};

/// Embedding observability and metrics
#[derive(Debug)]
pub struct EmbeddingStats {
    /// Total number of decisions made
    pub decisions_total: AtomicU64,
    /// Total number of allowed decisions
    pub allowed_total: AtomicU64,
    /// Total number of denied decisions
    pub denied_total: AtomicU64,
    /// Total number of actual embeddings computed
    pub embedded_total: AtomicU64,
}

impl EmbeddingStats {
    /// Create new embedding stats with zero counters
    pub fn new() -> Self {
        Self {
            decisions_total: AtomicU64::new(0),
            allowed_total: AtomicU64::new(0),
            denied_total: AtomicU64::new(0),
            embedded_total: AtomicU64::new(0),
        }
    }

    /// Get current stats snapshot
    pub fn snapshot(&self) -> EmbeddingStatsSnapshot {
        EmbeddingStatsSnapshot {
            decisions_total: self.decisions_total.load(Ordering::Relaxed),
            allowed_total: self.allowed_total.load(Ordering::Relaxed),
            denied_total: self.denied_total.load(Ordering::Relaxed),
            embedded_total: self.embedded_total.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of embedding stats for reading
#[derive(Debug, Clone)]
pub struct EmbeddingStatsSnapshot {
    pub decisions_total: u64,
    pub allowed_total: u64,
    pub denied_total: u64,
    pub embedded_total: u64,
}
