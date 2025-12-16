use rusqlite::{params, Connection, OptionalExtension};

#[derive(Clone)]
pub struct StoredEntry {
    pub id: u64,
    pub text: String,
    pub embedding: Vec<f32>,
    pub timestamp: u64,
}

pub struct Storage {
    pub conn: Connection,
}

impl Storage {
    pub fn new(path: &str) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS persona_memory (
                persona TEXT PRIMARY KEY,
                memory TEXT NOT NULL
            );",
            [],
        )?;
        Ok(Self { conn })
    }

    pub fn insert_entry(&self, text: &str, timestamp: u64, embedding: &[f32]) -> rusqlite::Result<u64> {
        let blob = f32_to_blob(embedding);
        self.conn.execute(
            "INSERT INTO entries (text, timestamp, embedding) VALUES (?1, ?2, ?3)",
            params![text, timestamp as i64, blob],
        )?;
        Ok(self.conn.last_insert_rowid() as u64)
    }

    pub fn get_entry(&self, id: u64) -> Option<StoredEntry> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, timestamp, embedding FROM entries WHERE id = ?1",
        ).ok()?;

        stmt.query_row(params![id as i64], |row| {
            let blob: Vec<u8> = row.get(3)?;
            Ok(StoredEntry {
                id: row.get::<_, i64>(0)? as u64,
                text: row.get(1)?,
                timestamp: row.get::<_, i64>(2)? as u64,
                embedding: blob_to_f32(&blob)?,
            })
        }).optional().ok().flatten()
    }

    pub fn load_all_entries(&self) -> Vec<StoredEntry> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, text, timestamp, embedding FROM entries ORDER BY id ASC",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let iter = stmt.query_map([], |row| {
            let blob: Vec<u8> = row.get(3)?;
            Ok(StoredEntry {
                id: row.get::<_, i64>(0)? as u64,
                text: row.get(1)?,
                timestamp: row.get::<_, i64>(2)? as u64,
                embedding: blob_to_f32(&blob)?,
            })
        });

        match iter {
            Ok(rows) => rows.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn delete_entry(&self, id: u64) -> rusqlite::Result<bool> {
        let affected = self.conn.execute("DELETE FROM entries WHERE id = ?1", params![id as i64])?;
        Ok(affected > 0)
    }

    pub fn count_entries(&self) -> usize {
        let mut stmt = match self.conn.prepare("SELECT COUNT(*) FROM entries") {
            Ok(s) => s,
            Err(_) => return 0,
        };
        stmt.query_row([], |row| row.get::<_, i64>(0)).map(|v| v as usize).unwrap_or(0)
    }
}

fn f32_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for v in embedding {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    blob
}

fn blob_to_f32(blob: &[u8]) -> rusqlite::Result<Vec<f32>> {
    if blob.len() % 4 != 0 {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            blob.len(),
            rusqlite::types::Type::Blob,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid f32 blob length")),
        ));
    }

    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

