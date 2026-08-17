//! Servicio de memoria local trazable para Aethel.
//!
//! Expone un protocolo JSONL para que un supervisor persistente invoque memoria,
//! recuperación y consolidación sin mantener estados opacos en el dashboard.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;
#[cfg(unix)]
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind { Episodic, Semantic, SleepReplay }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub session_id: String,
    pub kind: MemoryKind,
    pub embedding: Vec<f32>,
    pub salience: f32,
    pub source_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    pub created_step: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Recall { pub record: MemoryRecord, pub cosine_similarity: f32, pub retrieval_score: f32 }

#[derive(Clone, Debug, Serialize)]
pub struct CitableRecall {
    pub id: String,
    pub source_sha256: String,
    pub source_uri: Option<String>,
    pub content: String,
    pub cosine_similarity: f32,
    pub retrieval_score: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConsolidationReport { pub retained: usize, pub evicted: usize, pub replay_candidates: usize }

#[derive(Debug)]
pub struct MemoryStore { dimension: usize, capacity: usize, records: VecDeque<MemoryRecord> }

impl MemoryStore {
    pub fn new(dimension: usize, capacity: usize) -> Result<Self, String> {
        if dimension == 0 || capacity == 0 { return Err("dimension y capacity deben ser mayores que cero".into()); }
        Ok(Self { dimension, capacity, records: VecDeque::new() })
    }

    pub fn restore_jsonl(dimension: usize, capacity: usize, path: &Path) -> Result<Self, String> {
        let mut store = Self::new(dimension, capacity)?;
        if !path.exists() { return Ok(store); }
        let content = fs::read_to_string(path).map_err(|error| format!("no se pudo leer snapshot: {error}"))?;
        for line in content.lines().filter(|line| !line.trim().is_empty()) {
            let record: MemoryRecord = serde_json::from_str(line).map_err(|error| format!("snapshot JSONL inválido: {error}"))?;
            store.insert(record)?;
        }
        Ok(store)
    }

    pub fn insert(&mut self, record: MemoryRecord) -> Result<Option<MemoryRecord>, String> {
        self.validate(&record)?;
        let evicted = if self.records.len() == self.capacity { self.records.pop_front() } else { None };
        self.records.push_back(record);
        Ok(evicted)
    }

    pub fn retrieve(&self, query: &[f32], top_k: usize) -> Result<Vec<Recall>, String> {
        self.validate_vector(query)?;
        let query_norm = l2_norm(query);
        if query_norm == 0.0 { return Err("el vector de consulta no puede tener norma cero".into()); }
        let mut recalls: Vec<Recall> = self.records.iter().cloned().map(|record| {
            let similarity = dot(&record.embedding, query) / (l2_norm(&record.embedding) * query_norm);
            Recall { retrieval_score: similarity * record.salience, cosine_similarity: similarity, record }
        }).collect();
        recalls.sort_by(|a, b| b.retrieval_score.partial_cmp(&a.retrieval_score).unwrap_or(Ordering::Equal));
        recalls.truncate(top_k);
        Ok(recalls)
    }

    pub fn retrieve_context(&self, query: &[f32], top_k: usize, max_chars: usize) -> Result<Vec<CitableRecall>, String> {
        if max_chars == 0 { return Err("max_chars debe ser mayor que cero".into()); }
        self.retrieve(query, top_k).map(|recalls| recalls.into_iter().filter_map(|recall| {
            let content = recall.record.content.as_ref()?.chars().take(max_chars).collect::<String>();
            if content.is_empty() { return None; }
            Some(CitableRecall {
                id: recall.record.id,
                source_sha256: recall.record.source_sha256,
                source_uri: recall.record.source_uri,
                content,
                cosine_similarity: recall.cosine_similarity,
                retrieval_score: recall.retrieval_score,
            })
        }).collect())
    }

    pub fn consolidate(&mut self, replay_limit: usize) -> ConsolidationReport {
        let before = self.records.len();
        let mut ranked: Vec<MemoryRecord> = self.records.drain(..).collect();
        ranked.sort_by(|a, b| b.salience.partial_cmp(&a.salience).unwrap_or(Ordering::Equal).then_with(|| b.created_step.cmp(&a.created_step)));
        let replay_candidates = ranked.len().min(replay_limit);
        ranked.truncate(self.capacity);
        ranked.sort_by_key(|record| record.created_step);
        self.records = ranked.into();
        ConsolidationReport { retained: self.records.len(), evicted: before.saturating_sub(self.records.len()), replay_candidates }
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn snapshot_jsonl(&self) -> Result<String, String> {
        self.records.iter().map(|record| serde_json::to_string(record).map_err(|error| error.to_string())).collect::<Result<Vec<_>, _>>().map(|lines| lines.join("\n"))
    }

    fn validate(&self, record: &MemoryRecord) -> Result<(), String> {
        if record.id.is_empty() || record.session_id.is_empty() || record.source_sha256.is_empty() { return Err("id, session_id y source_sha256 son obligatorios para trazabilidad".into()); }
        if record.source_uri.as_ref().is_some_and(|value| value.trim().is_empty()) { return Err("source_uri no puede ser vacío".into()); }
        if record.content.as_ref().is_some_and(|value| value.trim().is_empty() || value.chars().count() > 32_768) { return Err("content debe tener entre 1 y 32768 caracteres".into()); }
        if !(0.0..=1.0).contains(&record.salience) { return Err("salience debe estar entre 0 y 1".into()); }
        self.validate_vector(&record.embedding)
    }
    fn validate_vector(&self, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dimension || vector.iter().any(|value| !value.is_finite()) { return Err("embedding con dimensión o valores no válidos".into()); }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MemoryService { store: MemoryStore, snapshot_path: Option<PathBuf> }

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ServiceCommand {
    Health,
    Remember { record: MemoryRecord },
    Retrieve { query: Vec<f32>, top_k: usize },
    RetrieveContext { query: Vec<f32>, top_k: usize, max_chars: usize },
    Sleep { replay_limit: usize },
    Snapshot,
}

#[derive(Debug, Serialize)]
pub struct ServiceResponse { pub ok: bool, pub result: Option<Value>, pub error: Option<String> }

impl MemoryService {
    pub fn new(store: MemoryStore, snapshot_path: Option<PathBuf>) -> Self { Self { store, snapshot_path } }
    pub fn from_snapshot(dimension: usize, capacity: usize, snapshot_path: Option<PathBuf>) -> Result<Self, String> {
        let store = match &snapshot_path { Some(path) => MemoryStore::restore_jsonl(dimension, capacity, path)?, None => MemoryStore::new(dimension, capacity)? };
        Ok(Self::new(store, snapshot_path))
    }
    pub fn handle(&mut self, command: ServiceCommand) -> Result<Value, String> {
        match command {
            ServiceCommand::Health => Ok(json!({"service":"aethel-memory-rust", "status":"READY", "records":self.store.len(), "persistence":self.snapshot_path.is_some()})),
            ServiceCommand::Remember { record } => {
                let evicted = self.store.insert(record)?;
                self.persist()?;
                Ok(json!({"stored":true, "records":self.store.len(), "evicted_id":evicted.map(|item| item.id)}))
            }
            ServiceCommand::Retrieve { query, top_k } => Ok(json!({"recalls":self.store.retrieve(&query, top_k)?})),
            ServiceCommand::RetrieveContext { query, top_k, max_chars } => Ok(json!({"context":self.store.retrieve_context(&query, top_k, max_chars)?})),
            ServiceCommand::Sleep { replay_limit } => {
                let report = self.store.consolidate(replay_limit);
                self.persist()?;
                Ok(json!({"consolidation":report}))
            }
            ServiceCommand::Snapshot => Ok(json!({"jsonl":self.store.snapshot_jsonl()?})),
        }
    }
    fn persist(&self) -> Result<(), String> {
        let Some(path) = &self.snapshot_path else { return Ok(()); };
        if let Some(parent) = path.parent() { fs::create_dir_all(parent).map_err(|error| format!("no se pudo crear directorio: {error}"))?; }
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, self.store.snapshot_jsonl()?).map_err(|error| format!("no se pudo escribir snapshot: {error}"))?;
        fs::rename(&temporary, path).map_err(|error| format!("no se pudo publicar snapshot atómico: {error}"))
    }
}

pub fn run_jsonl<R: BufRead, W: Write>(service: &mut MemoryService, reader: R, mut writer: W) -> Result<(), String> {
    for line in reader.lines() {
        let response = match line.map_err(|error| error.to_string()).and_then(|text| serde_json::from_str::<ServiceCommand>(&text).map_err(|error| error.to_string())).and_then(|command| service.handle(command)) {
            Ok(result) => ServiceResponse { ok: true, result: Some(result), error: None },
            Err(error) => ServiceResponse { ok: false, result: None, error: Some(error) },
        };
        let encoded = serde_json::to_string(&response).map_err(|error| error.to_string())?;
        writeln!(writer, "{encoded}").map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Atiende una conexión local por socket Unix sin abrir un puerto público.
#[cfg(unix)]
pub fn serve_unix_socket_once(service: &mut MemoryService, listener: &UnixListener) -> Result<(), String> {
    let (stream, _) = listener.accept().map_err(|error| format!("no se pudo aceptar socket: {error}"))?;
    let reader = BufReader::new(stream.try_clone().map_err(|error| format!("no se pudo clonar socket: {error}"))?);
    run_jsonl(service, reader, stream)
}

/// Sirve el protocolo JSONL secuencialmente; un supervisor reinicia el proceso si falla.
#[cfg(unix)]
pub fn run_unix_socket(service: &mut MemoryService, socket_path: &Path) -> Result<(), String> {
    if socket_path.exists() {
        let metadata = fs::symlink_metadata(socket_path).map_err(|error| format!("no se pudo inspeccionar socket: {error}"))?;
        if !metadata.file_type().is_socket() {
            return Err(format!("la ruta de socket ya existe y no es un socket: {}", socket_path.display()));
        }
        fs::remove_file(socket_path).map_err(|error| format!("no se pudo eliminar socket obsoleto: {error}"))?;
    }
    if let Some(parent) = socket_path.parent() { fs::create_dir_all(parent).map_err(|error| format!("no se pudo crear directorio de socket: {error}"))?; }
    let listener = UnixListener::bind(socket_path).map_err(|error| format!("no se pudo abrir socket local: {error}"))?;
    loop { serve_unix_socket_once(service, &listener)?; }
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() }
fn l2_norm(vector: &[f32]) -> f32 { dot(vector, vector).sqrt() }

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn record(id: &str, embedding: Vec<f32>, salience: f32, step: u64) -> MemoryRecord {
        MemoryRecord { id: id.into(), session_id: "session-a".into(), kind: MemoryKind::Episodic, embedding, salience, source_sha256: "abc123".into(), source_uri: None, content: None, created_step: step }
    }

    #[test]
    fn retrieve_is_ranked_and_traceable() {
        let mut store = MemoryStore::new(3, 3).unwrap();
        store.insert(record("near", vec![1.0, 0.0, 0.0], 1.0, 1)).unwrap();
        store.insert(record("far", vec![0.0, 1.0, 0.0], 1.0, 2)).unwrap();
        let recalls = store.retrieve(&[0.9, 0.1, 0.0], 1).unwrap();
        assert_eq!(recalls[0].record.id, "near");
        assert!(recalls[0].cosine_similarity > 0.9);
        assert!(store.snapshot_jsonl().unwrap().contains("source_sha256"));
    }

    #[test]
    fn jsonl_service_persists_restores_and_consolidates() {
        let path = std::env::temp_dir().join(format!("aethel-memory-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&path);
        let mut service = MemoryService::from_snapshot(2, 2, Some(path.clone())).unwrap();
        let commands = concat!(
            "{\"op\":\"remember\",\"record\":{\"id\":\"one\",\"session_id\":\"s\",\"kind\":\"semantic\",\"embedding\":[1.0,0.0],\"salience\":0.9,\"source_sha256\":\"hash\",\"created_step\":1}}\n",
            "{\"op\":\"retrieve\",\"query\":[1.0,0.0],\"top_k\":1}\n",
            "{\"op\":\"sleep\",\"replay_limit\":1}\n"
        );
        let mut output = Vec::new();
        run_jsonl(&mut service, Cursor::new(commands), &mut output).unwrap();
        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("\"ok\":true"));
        assert!(result.contains("\"one\""));
        assert!(path.exists());
        let restored = MemoryService::from_snapshot(2, 2, Some(path.clone())).unwrap();
        assert_eq!(restored.store.len(), 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn retrieve_context_returns_content_with_auditable_source() {
        let mut store = MemoryStore::new(2, 2).unwrap();
        let mut item = record("source-1", vec![1.0, 0.0], 0.9, 1);
        item.kind = MemoryKind::Semantic;
        item.source_sha256 = "sha256:source-1".into();
        item.source_uri = Some("https://example.invalid/source-1".into());
        item.content = Some("Contexto técnico aprobado para recuperación.".into());
        store.insert(item).unwrap();
        let context = store.retrieve_context(&[1.0, 0.0], 1, 16).unwrap();
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].source_sha256, "sha256:source-1");
        assert_eq!(context[0].content, "Contexto técnico");
        assert_eq!(context[0].source_uri.as_deref(), Some("https://example.invalid/source-1"));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_serves_health_without_network_port() {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::{UnixListener, UnixStream};
        use std::thread;
        let path = std::env::temp_dir().join(format!("aethel-memory-{}.sock", std::process::id()));
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let worker = thread::spawn(move || {
            let mut service = MemoryService::from_snapshot(2, 2, None).unwrap();
            serve_unix_socket_once(&mut service, &listener).unwrap();
        });
        let mut stream = UnixStream::connect(&path).unwrap();
        writeln!(stream, "{{\"op\":\"health\"}}").unwrap();
        let mut response = String::new();
        BufReader::new(stream).read_line(&mut response).unwrap();
        assert!(response.contains("\"READY\""));
        worker.join().unwrap();
        fs::remove_file(path).unwrap();
    }
}
