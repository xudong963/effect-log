use async_trait::async_trait;
use tokio_rusqlite::Connection;

use crate::error::StorageError;
use crate::types::*;

use super::{EffectStore, WalEntry};

/// SQLite-backed storage for the write-ahead log.
///
/// Uses WAL mode with `PRAGMA synchronous = NORMAL` for a balance of
/// durability and throughput. Each record is JSON-serialized into a TEXT column.
pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    /// Open (or create) a SQLite database at the given path.
    pub async fn open(path: &str) -> Result<Self, StorageError> {
        let conn = Connection::open(path)
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))?;

        conn.call(|conn| {
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA foreign_keys = ON;

                 CREATE TABLE IF NOT EXISTS intents (
                     effect_id TEXT PRIMARY KEY,
                     execution_id TEXT NOT NULL,
                     sequence_number INTEGER NOT NULL,
                     tool_name TEXT NOT NULL,
                     data TEXT NOT NULL,
                     UNIQUE(execution_id, sequence_number)
                 );

                 CREATE TABLE IF NOT EXISTS completions (
                     effect_id TEXT PRIMARY KEY,
                     data TEXT NOT NULL,
                     FOREIGN KEY (effect_id) REFERENCES intents(effect_id)
                 );

                 CREATE INDEX IF NOT EXISTS idx_intents_exec
                     ON intents(execution_id, sequence_number);",
            )?;
            Ok(())
        })
        .await
        .map_err(|e| StorageError::Sqlite(e.to_string()))?;

        Ok(Self { conn })
    }

    /// Open an in-memory SQLite database (for testing).
    pub async fn open_in_memory() -> Result<Self, StorageError> {
        Self::open(":memory:").await
    }
}

#[async_trait]
impl EffectStore for SqliteStore {
    async fn write_intent(&self, record: &IntentRecord) -> Result<(), StorageError> {
        let effect_id = record.effect_id.to_string();
        let execution_id = record.cursor.execution_id.clone();
        let sequence_number = record.cursor.sequence_number as i64;
        let tool_name = record.tool_name.clone();
        let data = serde_json::to_string(record)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO intents (effect_id, execution_id, sequence_number, tool_name, data)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![effect_id, execution_id, sequence_number, tool_name, data],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))
    }

    async fn write_completion(&self, record: &CompletionRecord) -> Result<(), StorageError> {
        let effect_id = record.effect_id.to_string();
        let data = serde_json::to_string(record)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT OR REPLACE INTO completions (effect_id, data) VALUES (?1, ?2)",
                    rusqlite::params![effect_id, data],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))
    }

    async fn load_execution(&self, execution_id: &str) -> Result<Vec<WalEntry>, StorageError> {
        let execution_id = execution_id.to_string();

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT i.data, c.data
                     FROM intents i
                     LEFT JOIN completions c ON i.effect_id = c.effect_id
                     WHERE i.execution_id = ?1
                     ORDER BY i.sequence_number ASC",
                )?;

                let rows = stmt.query_map(rusqlite::params![execution_id], |row| {
                    let intent_data: String = row.get(0)?;
                    let completion_data: Option<String> = row.get(1)?;
                    Ok((intent_data, completion_data))
                })?;

                let mut entries = Vec::new();
                for row in rows {
                    let (intent_json, completion_json) = row?;
                    let intent: IntentRecord = serde_json::from_str(&intent_json)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                    entries.push(WalEntry::Intent(intent));

                    if let Some(cj) = completion_json {
                        let completion: CompletionRecord = serde_json::from_str(&cj)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                        entries.push(WalEntry::Completion(completion));
                    }
                }

                Ok(entries)
            })
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))
    }

    async fn lookup_intent(
        &self,
        execution_id: &str,
        sequence_number: u64,
    ) -> Result<Option<IntentRecord>, StorageError> {
        let execution_id = execution_id.to_string();
        let seq = sequence_number as i64;

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT data FROM intents
                     WHERE execution_id = ?1 AND sequence_number = ?2",
                )?;

                let result = stmt
                    .query_row(rusqlite::params![execution_id, seq], |row| {
                        let data: String = row.get(0)?;
                        Ok(data)
                    })
                    .optional()?;

                match result {
                    Some(data) => {
                        let record: IntentRecord = serde_json::from_str(&data)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                        Ok(Some(record))
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))
    }

    async fn get_completion(
        &self,
        effect_id: &EffectId,
    ) -> Result<Option<CompletionRecord>, StorageError> {
        let id = effect_id.to_string();

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare("SELECT data FROM completions WHERE effect_id = ?1")?;

                let result = stmt
                    .query_row(rusqlite::params![id], |row| {
                        let data: String = row.get(0)?;
                        Ok(data)
                    })
                    .optional()?;

                match result {
                    Some(data) => {
                        let record: CompletionRecord = serde_json::from_str(&data)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                        Ok(Some(record))
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))
    }

    async fn lookup_by_idempotency_key(
        &self,
        execution_id: &str,
        key: &str,
    ) -> Result<Option<IntentRecord>, StorageError> {
        let execution_id = execution_id.to_string();
        let key = key.to_string();

        self.conn
            .call(move |conn| {
                // We need to search the JSON data for the idempotency_key field
                let mut stmt = conn.prepare(
                    "SELECT data FROM intents
                     WHERE execution_id = ?1
                     AND json_extract(data, '$.idempotency_key') = ?2
                     LIMIT 1",
                )?;

                let result = stmt
                    .query_row(rusqlite::params![execution_id, key], |row| {
                        let data: String = row.get(0)?;
                        Ok(data)
                    })
                    .optional()?;

                match result {
                    Some(data) => {
                        let record: IntentRecord = serde_json::from_str(&data)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                        Ok(Some(record))
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(|e| StorageError::Sqlite(e.to_string()))
    }
}

// Import the optional() extension
use rusqlite::OptionalExtension;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_intent(execution_id: &str, seq: u64, tool: &str) -> IntentRecord {
        IntentRecord {
            effect_id: EffectId::new(),
            tool_call_id: format!("call-{seq}"),
            tool_name: tool.into(),
            effect_kind: EffectKind::ReadOnly,
            input: serde_json::json!({"seq": seq}),
            idempotency_key: None,
            impact_scope: None,
            timestamp: Utc::now(),
            cursor: ExecutionCursor {
                execution_id: execution_id.into(),
                sequence_number: seq,
                parent_effect_id: None,
            },
        }
    }

    fn make_completion(effect_id: EffectId) -> CompletionRecord {
        CompletionRecord {
            effect_id,
            outcome: Outcome::Success,
            sealed_response: serde_json::json!({"ok": true}),
            version_fingerprint: None,
            has_irreversible_change: false,
            compensation_info: None,
            completed_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn sqlite_write_and_read_intent() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        let intent = make_intent("exec-1", 1, "read_file");

        store.write_intent(&intent).await.unwrap();

        let found = store.lookup_intent("exec-1", 1).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().tool_name, "read_file");
    }

    #[tokio::test]
    async fn sqlite_write_and_read_completion() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        let intent = make_intent("exec-1", 1, "read_file");
        let effect_id = intent.effect_id;

        store.write_intent(&intent).await.unwrap();
        store
            .write_completion(&make_completion(effect_id))
            .await
            .unwrap();

        let found = store.get_completion(&effect_id).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().outcome, Outcome::Success);
    }

    #[tokio::test]
    async fn sqlite_load_execution_ordered() {
        let store = SqliteStore::open_in_memory().await.unwrap();

        let i1 = make_intent("exec-1", 1, "tool_a");
        let i2 = make_intent("exec-1", 2, "tool_b");
        let i3 = make_intent("exec-1", 3, "tool_c");

        // Write out of order
        store.write_intent(&i3).await.unwrap();
        store.write_intent(&i1).await.unwrap();
        store.write_intent(&i2).await.unwrap();

        store
            .write_completion(&make_completion(i1.effect_id))
            .await
            .unwrap();

        let entries = store.load_execution("exec-1").await.unwrap();
        // i1 + completion + i2 + i3 = 4
        assert_eq!(entries.len(), 4);

        match &entries[0] {
            WalEntry::Intent(i) => assert_eq!(i.cursor.sequence_number, 1),
            _ => panic!("expected intent"),
        }
        match &entries[1] {
            WalEntry::Completion(c) => assert_eq!(c.effect_id, i1.effect_id),
            _ => panic!("expected completion"),
        }
    }

    #[tokio::test]
    async fn sqlite_lookup_missing_returns_none() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        assert!(store.lookup_intent("nope", 1).await.unwrap().is_none());
        assert!(store
            .get_completion(&EffectId::new())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn sqlite_separate_executions_isolated() {
        let store = SqliteStore::open_in_memory().await.unwrap();

        let i1 = make_intent("exec-1", 1, "tool_a");
        let i2 = make_intent("exec-2", 1, "tool_b");

        store.write_intent(&i1).await.unwrap();
        store.write_intent(&i2).await.unwrap();

        let entries = store.load_execution("exec-1").await.unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn sqlite_duplicate_intent_rejected() {
        let store = SqliteStore::open_in_memory().await.unwrap();
        let intent = make_intent("exec-1", 1, "read_file");

        store.write_intent(&intent).await.unwrap();
        let result = store.write_intent(&intent).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn sqlite_file_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let path_str = path.to_str().unwrap().to_string();

        let intent = make_intent("exec-1", 1, "send_email");
        let effect_id = intent.effect_id;

        // Write
        {
            let store = SqliteStore::open(&path_str).await.unwrap();
            store.write_intent(&intent).await.unwrap();
            store
                .write_completion(&make_completion(effect_id))
                .await
                .unwrap();
        }

        // Read from fresh connection
        {
            let store = SqliteStore::open(&path_str).await.unwrap();
            let found = store.lookup_intent("exec-1", 1).await.unwrap();
            assert!(found.is_some());
            let found_c = store.get_completion(&effect_id).await.unwrap();
            assert!(found_c.is_some());
        }
    }
}
