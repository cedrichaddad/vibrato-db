use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fmt::Write as FmtWrite;
use std::os::raw::{c_char, c_int, c_uchar, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use rand::RngCore;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use vibrato_core::metadata::VectorMetadata;

use super::model::AuditEvent;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    Query,
    Ingest,
    Admin,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Query => "query",
            Role::Ingest => "ingest",
            Role::Admin => "admin",
        }
    }

    pub fn parse_csv(value: &str) -> Vec<Role> {
        value
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter_map(|s| match s.as_str() {
                "query" => Some(Role::Query),
                "ingest" => Some(Role::Ingest),
                "admin" => Some(Role::Admin),
                _ => None,
            })
            .collect()
    }

    pub fn to_csv(roles: &[Role]) -> String {
        roles.iter().map(Role::as_str).collect::<Vec<_>>().join(",")
    }
}

#[derive(Debug, Clone)]
pub struct CollectionRecord {
    pub id: String,
    pub name: String,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct ApiKeyRecord {
    pub id: String,
    pub key_hash: String,
    pub roles: Vec<Role>,
    pub revoked: bool,
}

#[derive(Debug, Clone)]
pub struct ApiKeyCreateResult {
    pub id: String,
    pub token: String,
}

#[derive(Debug, Clone)]
pub struct WalIngestResult {
    pub vector_id: usize,
    pub created: bool,
    pub lsn: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub lsn: u64,
    pub vector_id: usize,
    pub vector: Vec<f32>,
    pub metadata: VectorMetadata,
}

#[derive(Debug, Clone)]
pub struct SegmentRecord {
    pub id: String,
    pub collection_id: String,
    pub level: i64,
    pub path: PathBuf,
    pub row_count: usize,
    pub vector_id_start: usize,
    pub vector_id_end: usize,
    pub created_lsn: u64,
    pub state: String,
}

#[derive(Debug, Clone)]
pub struct CheckpointJobRecord {
    pub id: String,
    pub collection_id: String,
    pub state: String,
    pub start_lsn: Option<u64>,
    pub end_lsn: Option<u64>,
    pub details: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct CompactionJobRecord {
    pub id: String,
    pub collection_id: String,
    pub state: String,
    pub details: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct CatalogOptions {
    pub read_timeout_ms: u64,
    pub wal_autocheckpoint_pages: u32,
}

impl Default for CatalogOptions {
    fn default() -> Self {
        Self {
            read_timeout_ms: 5_000,
            wal_autocheckpoint_pages: 1000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FilterRow {
    pub vector_id: usize,
    pub bpm: f32,
    pub tag_ids: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct QuarantineUsage {
    pub files: usize,
    pub bytes: u64,
}

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_INTERRUPT: c_int = 9;

const SQLITE_INTEGER: c_int = 1;
const SQLITE_FLOAT: c_int = 2;
const SQLITE_TEXT: c_int = 3;
const SQLITE_BLOB: c_int = 4;
const SQLITE_NULL: c_int = 5;

const SQLITE_OPEN_READONLY: c_int = 0x0000_0001;
const SQLITE_OPEN_READWRITE: c_int = 0x0000_0002;
const SQLITE_OPEN_CREATE: c_int = 0x0000_0004;
const SQLITE_OPEN_FULLMUTEX: c_int = 0x0001_0000;

type Sqlite3 = c_void;
type Sqlite3Stmt = c_void;

trait FromSqlRow: Sized {
    /// # Safety
    /// `stmt` must be positioned on a valid SQLITE_ROW and follow the expected select column order.
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self>;
}

#[link(name = "sqlite3")]
unsafe extern "C" {
    fn sqlite3_open_v2(
        filename: *const c_char,
        pp_db: *mut *mut Sqlite3,
        flags: c_int,
        z_vfs: *const c_char,
    ) -> c_int;
    fn sqlite3_close(db: *mut Sqlite3) -> c_int;
    fn sqlite3_errmsg(db: *mut Sqlite3) -> *const c_char;
    fn sqlite3_busy_timeout(db: *mut Sqlite3, ms: c_int) -> c_int;
    fn sqlite3_prepare_v2(
        db: *mut Sqlite3,
        sql: *const c_char,
        nbytes: c_int,
        pp_stmt: *mut *mut Sqlite3Stmt,
        pz_tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_step(stmt: *mut Sqlite3Stmt) -> c_int;
    fn sqlite3_finalize(stmt: *mut Sqlite3Stmt) -> c_int;
    fn sqlite3_column_count(stmt: *mut Sqlite3Stmt) -> c_int;
    fn sqlite3_column_name(stmt: *mut Sqlite3Stmt, i_col: c_int) -> *const c_char;
    fn sqlite3_column_type(stmt: *mut Sqlite3Stmt, i_col: c_int) -> c_int;
    fn sqlite3_column_int64(stmt: *mut Sqlite3Stmt, i_col: c_int) -> i64;
    fn sqlite3_column_double(stmt: *mut Sqlite3Stmt, i_col: c_int) -> f64;
    fn sqlite3_column_text(stmt: *mut Sqlite3Stmt, i_col: c_int) -> *const c_uchar;
    fn sqlite3_column_blob(stmt: *mut Sqlite3Stmt, i_col: c_int) -> *const c_void;
    fn sqlite3_column_bytes(stmt: *mut Sqlite3Stmt, i_col: c_int) -> c_int;
    fn sqlite3_exec(
        db: *mut Sqlite3,
        sql: *const c_char,
        callback: Option<
            unsafe extern "C" fn(
                data: *mut c_void,
                cols: c_int,
                values: *mut *mut c_char,
                names: *mut *mut c_char,
            ) -> c_int,
        >,
        data: *mut c_void,
        errmsg: *mut *mut c_char,
    ) -> c_int;
    fn sqlite3_free(ptr: *mut c_void);
    fn sqlite3_progress_handler(
        db: *mut Sqlite3,
        n_ops: c_int,
        callback: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
        data: *mut c_void,
    );
}

struct RawSqliteConnection {
    db: *mut Sqlite3,
}

unsafe impl Send for RawSqliteConnection {}

impl Drop for RawSqliteConnection {
    fn drop(&mut self) {
        if !self.db.is_null() {
            unsafe {
                let _ = sqlite3_close(self.db);
            }
            self.db = std::ptr::null_mut();
        }
    }
}

struct TimeoutCtx {
    start: Instant,
    timeout: Duration,
    timed_out: bool,
}

impl RawSqliteConnection {
    fn open(path: &Path, readonly: bool) -> Result<Self> {
        let c_path = CString::new(path.to_string_lossy().as_bytes())?;
        let mut db: *mut Sqlite3 = std::ptr::null_mut();
        let flags = if readonly {
            SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX
        } else {
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        };
        let rc = unsafe { sqlite3_open_v2(c_path.as_ptr(), &mut db, flags, std::ptr::null()) };
        if rc != SQLITE_OK || db.is_null() {
            let msg = if !db.is_null() {
                unsafe { c_ptr_to_string(sqlite3_errmsg(db)) }
            } else {
                "sqlite open failed".to_string()
            };
            if !db.is_null() {
                unsafe {
                    let _ = sqlite3_close(db);
                }
            }
            return Err(anyhow!("sqlite open failed: {}", msg));
        }
        let conn = Self { db };
        conn.exec("PRAGMA foreign_keys=ON;")?;
        conn.exec("PRAGMA temp_store=MEMORY;")?;
        unsafe {
            let _ = sqlite3_busy_timeout(conn.db, 30000);
        }
        Ok(conn)
    }

    fn configure_writer(&self, wal_autocheckpoint_pages: u32) -> Result<()> {
        self.exec("PRAGMA journal_mode=WAL;")?;
        self.exec("PRAGMA synchronous=NORMAL;")?;
        self.exec(&format!(
            "PRAGMA wal_autocheckpoint={};",
            wal_autocheckpoint_pages
        ))?;
        // 30 GB mmap — zero-copy reads bypass syscall boundary on 64-bit
        self.exec("PRAGMA mmap_size=30000000000;")?;
        // 64 MB page cache — keeps B-Tree internal nodes in RAM
        self.exec("PRAGMA cache_size=-64000;")?;
        Ok(())
    }

    fn exec(&self, sql: &str) -> Result<()> {
        let c_sql = CString::new(sql)?;
        let mut err_ptr: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            sqlite3_exec(
                self.db,
                c_sql.as_ptr(),
                None,
                std::ptr::null_mut(),
                &mut err_ptr,
            )
        };
        if rc != SQLITE_OK {
            let msg = if !err_ptr.is_null() {
                let s = c_ptr_to_string(err_ptr as *const c_char);
                unsafe { sqlite3_free(err_ptr as *mut c_void) };
                s
            } else {
                unsafe { c_ptr_to_string(sqlite3_errmsg(self.db)) }
            };
            return Err(anyhow!("sqlite exec failed: {}", msg));
        }
        Ok(())
    }

    fn query_json(&self, sql: &str, timeout_ms: Option<u64>) -> Result<Vec<Value>> {
        let c_sql = CString::new(sql)?;
        let mut rows = Vec::new();
        let mut tail = c_sql.as_ptr();

        let mut timeout_ctx = timeout_ms.map(|ms| TimeoutCtx {
            start: Instant::now(),
            timeout: Duration::from_millis(ms.max(1)),
            timed_out: false,
        });
        if let Some(ctx) = timeout_ctx.as_mut() {
            unsafe {
                sqlite3_progress_handler(
                    self.db,
                    100,
                    Some(progress_timeout_callback),
                    ctx as *mut TimeoutCtx as *mut c_void,
                );
            }
        }

        let query_result = (|| -> Result<()> {
            loop {
                let mut stmt: *mut Sqlite3Stmt = std::ptr::null_mut();
                let mut next_tail: *const c_char = std::ptr::null();
                let rc = unsafe {
                    sqlite3_prepare_v2(self.db, tail, -1, &mut stmt, &mut next_tail as *mut _)
                };
                if rc != SQLITE_OK {
                    return Err(anyhow!("sqlite prepare failed: {}", unsafe {
                        c_ptr_to_string(sqlite3_errmsg(self.db))
                    }));
                }

                tail = next_tail;
                if stmt.is_null() {
                    if tail.is_null() || unsafe { *tail } == 0 {
                        break;
                    }
                    continue;
                }

                let step_result = loop {
                    let step_rc = unsafe { sqlite3_step(stmt) };
                    if step_rc == SQLITE_ROW {
                        rows.push(statement_row_to_json(stmt));
                        continue;
                    }
                    if step_rc == SQLITE_DONE {
                        break Ok(());
                    }
                    if step_rc == SQLITE_INTERRUPT
                        && timeout_ctx.as_ref().map(|c| c.timed_out).unwrap_or(false)
                    {
                        break Err(anyhow!(
                            "catalog_read_timeout: sqlite exceeded {}ms",
                            timeout_ms.unwrap_or(0)
                        ));
                    }
                    break Err(anyhow!("sqlite step failed: {}", unsafe {
                        c_ptr_to_string(sqlite3_errmsg(self.db))
                    }));
                };

                unsafe {
                    let _ = sqlite3_finalize(stmt);
                }
                step_result?;

                if tail.is_null() || unsafe { *tail } == 0 {
                    break;
                }
            }
            Ok(())
        })();

        unsafe {
            sqlite3_progress_handler(self.db, 0, None, std::ptr::null_mut());
        }

        query_result?;
        Ok(rows)
    }

    fn query_rows<T: FromSqlRow>(&self, sql: &str, timeout_ms: Option<u64>) -> Result<Vec<T>> {
        let c_sql = CString::new(sql)?;
        let mut rows = Vec::new();
        let mut tail = c_sql.as_ptr();

        let mut timeout_ctx = timeout_ms.map(|ms| TimeoutCtx {
            start: Instant::now(),
            timeout: Duration::from_millis(ms.max(1)),
            timed_out: false,
        });
        if let Some(ctx) = timeout_ctx.as_mut() {
            unsafe {
                sqlite3_progress_handler(
                    self.db,
                    100,
                    Some(progress_timeout_callback),
                    ctx as *mut TimeoutCtx as *mut c_void,
                );
            }
        }

        let query_result = (|| -> Result<()> {
            loop {
                let mut stmt: *mut Sqlite3Stmt = std::ptr::null_mut();
                let mut next_tail: *const c_char = std::ptr::null();
                let rc = unsafe {
                    sqlite3_prepare_v2(self.db, tail, -1, &mut stmt, &mut next_tail as *mut _)
                };
                if rc != SQLITE_OK {
                    return Err(anyhow!("sqlite prepare failed: {}", unsafe {
                        c_ptr_to_string(sqlite3_errmsg(self.db))
                    }));
                }
                tail = next_tail;

                if stmt.is_null() {
                    if tail.is_null() || unsafe { *tail } == 0 {
                        break;
                    }
                    continue;
                }

                let step_result = loop {
                    let step_rc = unsafe { sqlite3_step(stmt) };
                    if step_rc == SQLITE_ROW {
                        rows.push(unsafe { T::from_row(stmt)? });
                        continue;
                    }
                    if step_rc == SQLITE_DONE {
                        break Ok(());
                    }
                    if step_rc == SQLITE_INTERRUPT
                        && timeout_ctx.as_ref().map(|c| c.timed_out).unwrap_or(false)
                    {
                        break Err(anyhow!(
                            "catalog_read_timeout: sqlite exceeded {}ms",
                            timeout_ms.unwrap_or(0)
                        ));
                    }
                    break Err(anyhow!("sqlite step failed: {}", unsafe {
                        c_ptr_to_string(sqlite3_errmsg(self.db))
                    }));
                };

                unsafe {
                    let _ = sqlite3_finalize(stmt);
                }
                step_result?;

                if tail.is_null() || unsafe { *tail } == 0 {
                    break;
                }
            }
            Ok(())
        })();

        unsafe {
            sqlite3_progress_handler(self.db, 0, None, std::ptr::null_mut());
        }

        query_result?;
        Ok(rows)
    }
}

unsafe extern "C" fn progress_timeout_callback(data: *mut c_void) -> c_int {
    if data.is_null() {
        return 0;
    }
    let ctx = &mut *(data as *mut TimeoutCtx);
    if ctx.start.elapsed() >= ctx.timeout {
        ctx.timed_out = true;
        1
    } else {
        0
    }
}

fn statement_row_to_json(stmt: *mut Sqlite3Stmt) -> Value {
    let cols = unsafe { sqlite3_column_count(stmt) }.max(0) as usize;
    let mut map = serde_json::Map::with_capacity(cols);
    for i in 0..cols {
        let name = unsafe { c_ptr_to_string(sqlite3_column_name(stmt, i as c_int)) };
        let col_type = unsafe { sqlite3_column_type(stmt, i as c_int) };
        let value = match col_type {
            SQLITE_INTEGER => Value::from(unsafe { sqlite3_column_int64(stmt, i as c_int) }),
            SQLITE_FLOAT => {
                serde_json::Number::from_f64(unsafe { sqlite3_column_double(stmt, i as c_int) })
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }
            SQLITE_TEXT => {
                let ptr = unsafe { sqlite3_column_text(stmt, i as c_int) };
                if ptr.is_null() {
                    Value::Null
                } else {
                    let s = unsafe { CStr::from_ptr(ptr as *const c_char) }
                        .to_string_lossy()
                        .to_string();
                    Value::String(s)
                }
            }
            SQLITE_BLOB => {
                let ptr = unsafe { sqlite3_column_blob(stmt, i as c_int) };
                let len = unsafe { sqlite3_column_bytes(stmt, i as c_int) }.max(0) as usize;
                if ptr.is_null() || len == 0 {
                    Value::String(String::new())
                } else {
                    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
                    Value::String(hex_string(bytes))
                }
            }
            SQLITE_NULL => Value::Null,
            _ => Value::Null,
        };
        map.insert(name, value);
    }
    Value::Object(map)
}

unsafe fn column_i64(stmt: *mut Sqlite3Stmt, idx: c_int) -> i64 {
    sqlite3_column_int64(stmt, idx)
}

unsafe fn column_f64(stmt: *mut Sqlite3Stmt, idx: c_int) -> f64 {
    sqlite3_column_double(stmt, idx)
}

unsafe fn column_text_bytes<'a>(stmt: *mut Sqlite3Stmt, idx: c_int) -> &'a [u8] {
    let ptr = sqlite3_column_text(stmt, idx);
    let len = sqlite3_column_bytes(stmt, idx).max(0) as usize;
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr as *const u8, len)
    }
}

unsafe fn column_text_string(stmt: *mut Sqlite3Stmt, idx: c_int) -> String {
    String::from_utf8_lossy(column_text_bytes(stmt, idx)).to_string()
}

fn c_ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
}

fn is_read_only_sql(sql: &str) -> bool {
    let trimmed = sql.trim_start();
    if trimmed.is_empty() {
        return true;
    }
    let first = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    matches!(first.as_str(), "SELECT" | "PRAGMA" | "WITH")
}

pub trait CatalogStore: Send + Sync {
    fn ensure_collection(&self, name: &str, dim: usize) -> Result<CollectionRecord>;
    fn get_collection(&self, name: &str) -> Result<Option<CollectionRecord>>;

    fn count_api_keys(&self) -> Result<usize>;
    fn create_api_key(
        &self,
        name: &str,
        roles: &[Role],
        pepper: &str,
    ) -> Result<ApiKeyCreateResult>;
    fn revoke_api_key(&self, id: &str) -> Result<()>;
    fn lookup_api_key(&self, id: &str) -> Result<Option<ApiKeyRecord>>;

    fn next_vector_id(&self, collection_id: &str) -> Result<usize>;
    fn ingest_wal(
        &self,
        collection_id: &str,
        vector_id: usize,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult>;
    fn ingest_wal_atomic(
        &self,
        collection_id: &str,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult>;

    fn pending_wal(&self, collection_id: &str, limit: usize) -> Result<Vec<WalEntry>>;
    fn pending_wal_after_lsn(
        &self,
        collection_id: &str,
        lsn_exclusive: u64,
    ) -> Result<Vec<WalEntry>>;
    fn mark_wal_checkpointed(
        &self,
        collection_id: &str,
        start_lsn: u64,
        end_lsn: u64,
    ) -> Result<()>;
    fn count_wal_pending(&self, collection_id: &str) -> Result<usize>;

    fn insert_segment(&self, record: &SegmentRecord) -> Result<()>;
    fn update_segment_state(&self, segment_id: &str, state: &str) -> Result<()>;
    fn list_segments_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<SegmentRecord>>;
    fn get_segment(&self, segment_id: &str) -> Result<Option<SegmentRecord>>;
    fn list_known_segment_paths(&self, collection_id: &str) -> Result<Vec<PathBuf>>;

    fn upsert_checkpoint_job(&self, job: &CheckpointJobRecord) -> Result<()>;
    fn update_checkpoint_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()>;
    fn checkpoint_mark_pending_activate(
        &self,
        collection_id: &str,
        segment_id: &str,
        job_id: &str,
        start_lsn: u64,
        end_lsn: u64,
        details: Value,
    ) -> Result<()>;
    fn checkpoint_activate(&self, segment_id: &str, job_id: &str, details: Value) -> Result<()>;
    fn list_checkpoint_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CheckpointJobRecord>>;

    fn upsert_compaction_job(&self, job: &CompactionJobRecord) -> Result<()>;
    fn update_compaction_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()>;
    fn compaction_mark_pending_activate(
        &self,
        output_segment_id: &str,
        job_id: &str,
        details: Value,
    ) -> Result<()>;
    fn compaction_activate(
        &self,
        output_segment_id: &str,
        input_segment_ids: &[String],
        job_id: &str,
        details: Value,
    ) -> Result<()>;
    fn list_compaction_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CompactionJobRecord>>;

    fn upsert_metadata(
        &self,
        collection_id: &str,
        vector_id: usize,
        metadata: &VectorMetadata,
    ) -> Result<()>;
    fn fetch_metadata(
        &self,
        collection_id: &str,
        ids: &[usize],
    ) -> Result<HashMap<usize, VectorMetadata>>;
    fn fetch_all_metadata(&self, collection_id: &str) -> Result<Vec<(usize, VectorMetadata)>>;
    fn fetch_tag_dictionary(&self, collection_id: &str) -> Result<HashMap<String, u32>>;
    fn fetch_filter_rows(&self, collection_id: &str) -> Result<Vec<FilterRow>>;

    fn insert_orphan_file(
        &self,
        collection_id: &str,
        original_path: &Path,
        quarantine_path: &Path,
        reason: &str,
    ) -> Result<()>;
    fn quarantine_usage(&self) -> Result<QuarantineUsage>;
    fn enforce_quarantine_cap(&self, max_files: usize, max_bytes: u64) -> Result<usize>;
    fn quarantine_gc(&self, ttl_secs: i64) -> Result<Vec<PathBuf>>;

    fn total_vectors(&self, collection_id: &str) -> Result<usize>;
    fn vacuum_into(&self, output_path: &Path) -> Result<()>;
    fn audit_event(
        &self,
        request_id: &str,
        api_key_id: Option<&str>,
        endpoint: &str,
        action: &str,
        status_code: u16,
        latency_ms: f64,
        details: Value,
    ) -> Result<()>;
}

pub struct SqliteCatalog {
    path: PathBuf,
    options: CatalogOptions,
    writer: std::sync::Mutex<RawSqliteConnection>,
    readers: Vec<std::sync::Mutex<RawSqliteConnection>>,
    next_reader: AtomicUsize,
    read_timeout_total: AtomicU64,
    quarantine_evictions_total: AtomicU64,
}

impl SqliteCatalog {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_options(path, CatalogOptions::default())
    }

    pub fn open_with_options(path: &Path, options: CatalogOptions) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let writer = RawSqliteConnection::open(path, false)?;
        writer.configure_writer(options.wal_autocheckpoint_pages)?;

        let mut catalog = Self {
            path: path.to_path_buf(),
            options,
            writer: std::sync::Mutex::new(writer),
            readers: Vec::new(),
            next_reader: AtomicUsize::new(0),
            read_timeout_total: AtomicU64::new(0),
            quarantine_evictions_total: AtomicU64::new(0),
        };
        catalog.apply_migrations()?;
        catalog.init_readers(3)?;
        Ok(catalog)
    }

    fn exec(&self, sql: &str) -> Result<()> {
        let guard = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.exec(sql)
    }

    fn query_json(&self, sql: &str) -> Result<Vec<Value>> {
        let is_read = is_read_only_sql(sql);
        let timeout = if is_read {
            Some(self.options.read_timeout_ms.max(1))
        } else {
            None
        };

        let result = if is_read && !self.readers.is_empty() {
            let idx = self.next_reader.fetch_add(1, AtomicOrdering::Relaxed) % self.readers.len();
            let guard = self.readers[idx]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.query_json(sql, timeout)
        } else {
            let guard = self
                .writer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.query_json(sql, timeout)
        };

        if result
            .as_ref()
            .err()
            .map(|e| e.to_string().contains("catalog_read_timeout"))
            .unwrap_or(false)
        {
            self.read_timeout_total
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        result
    }

    fn query_rows<T: FromSqlRow>(&self, sql: &str) -> Result<Vec<T>> {
        let is_read = is_read_only_sql(sql);
        let timeout = if is_read {
            Some(self.options.read_timeout_ms.max(1))
        } else {
            None
        };

        let result = if is_read && !self.readers.is_empty() {
            let idx = self.next_reader.fetch_add(1, AtomicOrdering::Relaxed) % self.readers.len();
            let guard = self.readers[idx]
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.query_rows(sql, timeout)
        } else {
            let guard = self
                .writer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.query_rows(sql, timeout)
        };

        if result
            .as_ref()
            .err()
            .map(|e| e.to_string().contains("catalog_read_timeout"))
            .unwrap_or(false)
        {
            self.read_timeout_total
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        result
    }

    fn init_readers(&mut self, count: usize) -> Result<()> {
        self.readers.clear();
        for _ in 0..count {
            self.readers
                .push(std::sync::Mutex::new(RawSqliteConnection::open(
                    &self.path, true,
                )?));
        }
        Ok(())
    }

    pub fn read_timeout_total(&self) -> u64 {
        self.read_timeout_total.load(AtomicOrdering::Relaxed)
    }

    pub fn quarantine_evictions_total(&self) -> u64 {
        self.quarantine_evictions_total
            .load(AtomicOrdering::Relaxed)
    }

    pub fn sqlite_wal_bytes(&self) -> u64 {
        let wal_path = PathBuf::from(format!("{}-wal", self.path.to_string_lossy()));
        std::fs::metadata(wal_path).map(|m| m.len()).unwrap_or(0)
    }

    fn apply_migrations(&self) -> Result<()> {
        let migration_sql = [
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL, checksum TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, dim INTEGER NOT NULL, created_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS segments (id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, level INTEGER NOT NULL, path TEXT NOT NULL UNIQUE, row_count INTEGER NOT NULL, vector_id_start INTEGER NOT NULL, vector_id_end INTEGER NOT NULL, created_lsn INTEGER NOT NULL, state TEXT NOT NULL, created_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS wal_entries (lsn INTEGER PRIMARY KEY AUTOINCREMENT, collection_id TEXT NOT NULL, vector_id INTEGER NOT NULL, vector_json TEXT NOT NULL, metadata_json TEXT NOT NULL, idempotency_key TEXT, checkpointed_at INTEGER, created_at INTEGER NOT NULL)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_idempotency ON wal_entries(collection_id, idempotency_key) WHERE idempotency_key IS NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_vector_id ON wal_entries(collection_id, vector_id)",
            "CREATE INDEX IF NOT EXISTS idx_wal_pending_by_collection_lsn ON wal_entries(collection_id, checkpointed_at, lsn)",
            "CREATE TABLE IF NOT EXISTS vector_metadata (collection_id TEXT NOT NULL, vector_id INTEGER NOT NULL, source_file TEXT NOT NULL, start_time_ms INTEGER NOT NULL, duration_ms INTEGER NOT NULL, bpm REAL NOT NULL, tags_json TEXT NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY(collection_id, vector_id))",
            "CREATE INDEX IF NOT EXISTS idx_vector_metadata_collection_vector ON vector_metadata(collection_id, vector_id)",
            "CREATE TABLE IF NOT EXISTS vector_id_counters (collection_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS tag_ids (collection_id TEXT NOT NULL, id INTEGER NOT NULL, tag TEXT NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY(collection_id, id), UNIQUE(collection_id, tag))",
            "CREATE INDEX IF NOT EXISTS idx_tag_ids_collection_tag ON tag_ids(collection_id, tag)",
            "CREATE TABLE IF NOT EXISTS vector_tags (collection_id TEXT NOT NULL, vector_id INTEGER NOT NULL, tag_id INTEGER NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY(collection_id, vector_id, tag_id))",
            "CREATE INDEX IF NOT EXISTS idx_vector_tags_collection_vector ON vector_tags(collection_id, vector_id)",
            "CREATE INDEX IF NOT EXISTS idx_vector_tags_collection_tag ON vector_tags(collection_id, tag_id)",
            "CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL, roles TEXT NOT NULL, created_at INTEGER NOT NULL, revoked_at INTEGER)",
            "CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, request_id TEXT NOT NULL, api_key_id TEXT, endpoint TEXT NOT NULL, action TEXT NOT NULL, status_code INTEGER NOT NULL, latency_ms REAL NOT NULL, client_ip TEXT, details_json TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS checkpoint_jobs (id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, state TEXT NOT NULL, start_lsn INTEGER, end_lsn INTEGER, details_json TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS compaction_jobs (id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, state TEXT NOT NULL, details_json TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS orphan_files (id INTEGER PRIMARY KEY AUTOINCREMENT, collection_id TEXT NOT NULL, original_path TEXT NOT NULL, quarantine_path TEXT NOT NULL, reason TEXT NOT NULL, size_bytes INTEGER NOT NULL DEFAULT 0, quarantined_at INTEGER NOT NULL, deleted_at INTEGER, deleted_reason TEXT)",
        ];

        for sql in migration_sql {
            self.exec(sql)?;
        }
        self.migrate_vector_metadata_pk()?;
        self.ensure_column("orphan_files", "size_bytes", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("orphan_files", "deleted_reason", "TEXT")?;
        self.ensure_column("api_keys", "hash_version", "INTEGER NOT NULL DEFAULT 1")?;
        self.exec(
            "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
             SELECT collection_id, COALESCE(MAX(vector_id) + 1, 0)
             FROM vector_metadata
             GROUP BY collection_id;",
        )?;
        self.backfill_vector_tags_if_needed()?;
        self.exec(&format!(
            "INSERT OR REPLACE INTO schema_migrations(version, applied_at, checksum) VALUES (2, {}, '{}');",
            now_unix_ts(),
            sql_quote("v2_1")
        ))?;
        Ok(())
    }

    pub fn execute_sql(&self, sql: &str) -> Result<()> {
        self.exec(sql)
    }

    fn ensure_column(&self, table: &str, column: &str, decl: &str) -> Result<()> {
        let rows = self.query_json(&format!("PRAGMA table_info({});", table))?;
        let exists = rows.iter().any(|row| {
            row.get("name")
                .and_then(Value::as_str)
                .map(|name| name == column)
                .unwrap_or(false)
        });
        if exists {
            return Ok(());
        }
        self.exec(&format!(
            "ALTER TABLE {} ADD COLUMN {} {};",
            table, column, decl
        ))
    }

    fn migrate_vector_metadata_pk(&self) -> Result<()> {
        let rows = self.query_json("PRAGMA table_info(vector_metadata);")?;
        if rows.is_empty() {
            return Ok(());
        }

        let mut has_collection_pk = false;
        let mut has_vector_pk = false;
        for row in &rows {
            let name = row.get("name").and_then(Value::as_str).unwrap_or_default();
            let pk = row.get("pk").and_then(Value::as_i64).unwrap_or_default();
            if pk > 0 && name == "collection_id" {
                has_collection_pk = true;
            }
            if pk > 0 && name == "vector_id" {
                has_vector_pk = true;
            }
        }
        if has_collection_pk && has_vector_pk {
            return Ok(());
        }

        self.exec(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS vector_metadata_v2 (
               collection_id TEXT NOT NULL,
               vector_id INTEGER NOT NULL,
               source_file TEXT NOT NULL,
               start_time_ms INTEGER NOT NULL,
               duration_ms INTEGER NOT NULL,
               bpm REAL NOT NULL,
               tags_json TEXT NOT NULL,
               created_at INTEGER NOT NULL,
               PRIMARY KEY(collection_id, vector_id)
             );
             INSERT OR REPLACE INTO vector_metadata_v2(
               collection_id, vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at
             )
             SELECT collection_id, vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at
             FROM vector_metadata;
             DROP TABLE vector_metadata;
             ALTER TABLE vector_metadata_v2 RENAME TO vector_metadata;
             COMMIT;",
        )
    }

    fn backfill_vector_tags_if_needed(&self) -> Result<()> {
        let rows = self.query_json("SELECT COUNT(*) AS n FROM vector_tags;")?;
        let existing = rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default();
        if existing > 0 {
            return Ok(());
        }

        let collections = self.query_json("SELECT id FROM collections ORDER BY id ASC;")?;
        for row in collections {
            let collection_id = row["id"].as_str().unwrap_or_default();
            if collection_id.is_empty() {
                continue;
            }
            let metadata_rows = self.fetch_all_metadata(collection_id)?;
            for (vector_id, metadata) in metadata_rows {
                self.upsert_metadata(collection_id, vector_id, &metadata)?;
            }
        }
        Ok(())
    }
}

impl CatalogStore for SqliteCatalog {
    fn ensure_collection(&self, name: &str, dim: usize) -> Result<CollectionRecord> {
        if let Some(c) = self.get_collection(name)? {
            if c.dim != dim {
                return Err(anyhow!(
                    "collection '{}' exists with dim {}, expected {}",
                    name,
                    c.dim,
                    dim
                ));
            }
            return Ok(c);
        }

        let id = random_id("col");
        self.exec(&format!(
            "INSERT INTO collections(id, name, dim, created_at) VALUES ('{}', '{}', {}, {});",
            sql_quote(&id),
            sql_quote(name),
            dim,
            now_unix_ts()
        ))?;

        Ok(CollectionRecord {
            id,
            name: name.to_string(),
            dim,
        })
    }

    fn get_collection(&self, name: &str) -> Result<Option<CollectionRecord>> {
        let rows = self.query_json(&format!(
            "SELECT id, name, dim FROM collections WHERE name='{}' LIMIT 1;",
            sql_quote(name)
        ))?;

        let Some(row) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(CollectionRecord {
            id: row["id"].as_str().unwrap_or_default().to_string(),
            name: row["name"].as_str().unwrap_or_default().to_string(),
            dim: row["dim"].as_i64().unwrap_or_default() as usize,
        }))
    }

    fn count_api_keys(&self) -> Result<usize> {
        let rows =
            self.query_json("SELECT COUNT(*) AS n FROM api_keys WHERE revoked_at IS NULL;")?;
        Ok(rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default() as usize)
    }

    fn create_api_key(
        &self,
        name: &str,
        roles: &[Role],
        pepper: &str,
    ) -> Result<ApiKeyCreateResult> {
        let id = random_id("vbk");
        let mut secret_bytes = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut secret_bytes);
        let secret = hex_string(&secret_bytes);
        let token = format!("{}.{}", id, secret);
        let hash = hash_secret(pepper, &secret);

        self.exec(&format!(
            "INSERT INTO api_keys(id, name, key_hash, roles, created_at, revoked_at, hash_version) VALUES ('{}', '{}', '{}', '{}', {}, NULL, 2);",
            sql_quote(&id),
            sql_quote(name),
            sql_quote(&hash),
            sql_quote(&Role::to_csv(roles)),
            now_unix_ts()
        ))?;

        Ok(ApiKeyCreateResult { id, token })
    }

    fn revoke_api_key(&self, id: &str) -> Result<()> {
        self.exec(&format!(
            "UPDATE api_keys SET revoked_at = {} WHERE id = '{}';",
            now_unix_ts(),
            sql_quote(id)
        ))
    }

    fn lookup_api_key(&self, id: &str) -> Result<Option<ApiKeyRecord>> {
        let rows = self.query_json(&format!(
            "SELECT id, key_hash, roles, revoked_at FROM api_keys WHERE id='{}' LIMIT 1;",
            sql_quote(id)
        ))?;

        let Some(row) = rows.first() else {
            return Ok(None);
        };

        Ok(Some(ApiKeyRecord {
            id: row["id"].as_str().unwrap_or_default().to_string(),
            key_hash: row["key_hash"].as_str().unwrap_or_default().to_string(),
            roles: Role::parse_csv(row["roles"].as_str().unwrap_or_default()),
            revoked: !row["revoked_at"].is_null(),
        }))
    }

    fn next_vector_id(&self, collection_id: &str) -> Result<usize> {
        let collection = sql_quote(collection_id);
        let rows = self.query_json(&format!(
            "BEGIN IMMEDIATE;
             INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
             SELECT '{collection}', COALESCE(MAX(vector_id) + 1, 0)
             FROM vector_metadata
             WHERE collection_id='{collection}';
             SELECT next_id AS next_id FROM vector_id_counters WHERE collection_id='{collection}' LIMIT 1;
             COMMIT;"
        ))?;
        let next_id = rows
            .first()
            .and_then(|r| r["next_id"].as_i64())
            .unwrap_or(0)
            .max(0);
        Ok(next_id as usize)
    }

    fn ingest_wal(
        &self,
        collection_id: &str,
        vector_id: usize,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
        self.ingest_wal_internal(
            collection_id,
            Some(vector_id),
            vector,
            metadata,
            idempotency_key,
        )
    }

    fn ingest_wal_atomic(
        &self,
        collection_id: &str,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
        self.ingest_wal_internal(collection_id, None, vector, metadata, idempotency_key)
    }

    fn pending_wal(&self, collection_id: &str, limit: usize) -> Result<Vec<WalEntry>> {
        self.query_wal(&format!(
            "SELECT lsn, vector_id, vector_json, metadata_json FROM wal_entries WHERE collection_id='{}' AND checkpointed_at IS NULL ORDER BY lsn ASC LIMIT {};",
            sql_quote(collection_id),
            limit
        ))
    }

    fn pending_wal_after_lsn(
        &self,
        collection_id: &str,
        lsn_exclusive: u64,
    ) -> Result<Vec<WalEntry>> {
        self.query_wal(&format!(
            "SELECT lsn, vector_id, vector_json, metadata_json FROM wal_entries WHERE collection_id='{}' AND checkpointed_at IS NULL AND lsn > {} ORDER BY lsn ASC;",
            sql_quote(collection_id),
            lsn_exclusive
        ))
    }

    fn mark_wal_checkpointed(
        &self,
        collection_id: &str,
        start_lsn: u64,
        end_lsn: u64,
    ) -> Result<()> {
        self.exec(&format!(
            "UPDATE wal_entries SET checkpointed_at = {} WHERE collection_id='{}' AND lsn >= {} AND lsn <= {};",
            now_unix_ts(),
            sql_quote(collection_id),
            start_lsn,
            end_lsn
        ))
    }

    fn count_wal_pending(&self, collection_id: &str) -> Result<usize> {
        let rows = self.query_json(&format!(
            "SELECT COUNT(*) AS n FROM wal_entries WHERE collection_id='{}' AND checkpointed_at IS NULL;",
            sql_quote(collection_id)
        ))?;
        Ok(rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default() as usize)
    }

    fn insert_segment(&self, record: &SegmentRecord) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO segments(id, collection_id, level, path, row_count, vector_id_start, vector_id_end, created_lsn, state, created_at) VALUES ('{}', '{}', {}, '{}', {}, {}, {}, {}, '{}', {});",
            sql_quote(&record.id),
            sql_quote(&record.collection_id),
            record.level,
            sql_quote(&record.path.to_string_lossy()),
            record.row_count,
            record.vector_id_start,
            record.vector_id_end,
            record.created_lsn,
            sql_quote(&record.state),
            now_unix_ts(),
        ))
    }

    fn update_segment_state(&self, segment_id: &str, state: &str) -> Result<()> {
        self.exec(&format!(
            "UPDATE segments SET state='{}' WHERE id='{}';",
            sql_quote(state),
            sql_quote(segment_id)
        ))
    }

    fn list_segments_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<SegmentRecord>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }

        let in_clause = states
            .iter()
            .map(|s| format!("'{}'", sql_quote(s)))
            .collect::<Vec<_>>()
            .join(",");

        self.query_rows::<SegmentRecord>(&format!(
            "SELECT id, collection_id, level, path, row_count, vector_id_start, vector_id_end, created_lsn, state FROM segments WHERE collection_id='{}' AND state IN ({}) ORDER BY level ASC, vector_id_start ASC;",
            sql_quote(collection_id),
            in_clause
        ))
    }

    fn list_known_segment_paths(&self, collection_id: &str) -> Result<Vec<PathBuf>> {
        let rows = self.query_json(&format!(
            "SELECT path FROM segments WHERE collection_id='{}';",
            sql_quote(collection_id)
        ))?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r["path"].as_str().map(PathBuf::from))
            .collect())
    }

    fn get_segment(&self, segment_id: &str) -> Result<Option<SegmentRecord>> {
        let rows = self.query_rows::<SegmentRecord>(&format!(
            "SELECT id, collection_id, level, path, row_count, vector_id_start, vector_id_end, created_lsn, state FROM segments WHERE id='{}' LIMIT 1;",
            sql_quote(segment_id)
        ))?;
        Ok(rows.into_iter().next())
    }

    fn upsert_checkpoint_job(&self, job: &CheckpointJobRecord) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO checkpoint_jobs(id, collection_id, state, start_lsn, end_lsn, details_json, created_at, updated_at)
             VALUES ('{}', '{}', '{}', {}, {}, '{}', {}, {});",
            sql_quote(&job.id),
            sql_quote(&job.collection_id),
            sql_quote(&job.state),
            job.start_lsn
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NULL".to_string()),
            job.end_lsn
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NULL".to_string()),
            sql_quote(&job.details.to_string()),
            job.created_at,
            job.updated_at,
        ))
    }

    fn update_checkpoint_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()> {
        self.exec(&format!(
            "UPDATE checkpoint_jobs SET state='{}', details_json='{}', updated_at={} WHERE id='{}';",
            sql_quote(state),
            sql_quote(&details.to_string()),
            now_unix_ts(),
            sql_quote(job_id)
        ))
    }

    fn checkpoint_mark_pending_activate(
        &self,
        collection_id: &str,
        segment_id: &str,
        job_id: &str,
        start_lsn: u64,
        end_lsn: u64,
        details: Value,
    ) -> Result<()> {
        let now = now_unix_ts();
        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='pending_activate' WHERE id='{}';
             UPDATE wal_entries SET checkpointed_at = {} WHERE collection_id='{}' AND lsn >= {} AND lsn <= {};
             UPDATE checkpoint_jobs SET state='pending_activate', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(segment_id),
            now,
            sql_quote(collection_id),
            start_lsn,
            end_lsn,
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn checkpoint_activate(&self, segment_id: &str, job_id: &str, details: Value) -> Result<()> {
        let now = now_unix_ts();
        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='active' WHERE id='{}';
             UPDATE checkpoint_jobs SET state='completed', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(segment_id),
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn list_checkpoint_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CheckpointJobRecord>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let in_clause = states
            .iter()
            .map(|s| format!("'{}'", sql_quote(s)))
            .collect::<Vec<_>>()
            .join(",");

        let rows = self.query_json(&format!(
            "SELECT id, collection_id, state, start_lsn, end_lsn, details_json, created_at, updated_at
             FROM checkpoint_jobs WHERE collection_id='{}' AND state IN ({}) ORDER BY updated_at ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let details = serde_json::from_str(row["details_json"].as_str().unwrap_or("{}"))
                .unwrap_or_else(|_| json!({}));
            out.push(CheckpointJobRecord {
                id: row["id"].as_str().unwrap_or_default().to_string(),
                collection_id: row["collection_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                state: row["state"].as_str().unwrap_or_default().to_string(),
                start_lsn: row["start_lsn"].as_i64().map(|v| v as u64),
                end_lsn: row["end_lsn"].as_i64().map(|v| v as u64),
                details,
                created_at: row["created_at"].as_i64().unwrap_or_default(),
                updated_at: row["updated_at"].as_i64().unwrap_or_default(),
            });
        }
        Ok(out)
    }

    fn upsert_compaction_job(&self, job: &CompactionJobRecord) -> Result<()> {
        self.exec(&format!(
            "INSERT OR REPLACE INTO compaction_jobs(id, collection_id, state, details_json, created_at, updated_at)
             VALUES ('{}', '{}', '{}', '{}', {}, {});",
            sql_quote(&job.id),
            sql_quote(&job.collection_id),
            sql_quote(&job.state),
            sql_quote(&job.details.to_string()),
            job.created_at,
            job.updated_at,
        ))
    }

    fn update_compaction_job_state(&self, job_id: &str, state: &str, details: Value) -> Result<()> {
        self.exec(&format!(
            "UPDATE compaction_jobs SET state='{}', details_json='{}', updated_at={} WHERE id='{}';",
            sql_quote(state),
            sql_quote(&details.to_string()),
            now_unix_ts(),
            sql_quote(job_id)
        ))
    }

    fn compaction_mark_pending_activate(
        &self,
        output_segment_id: &str,
        job_id: &str,
        details: Value,
    ) -> Result<()> {
        let now = now_unix_ts();
        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='pending_activate' WHERE id='{}';
             UPDATE compaction_jobs SET state='pending_activate', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(output_segment_id),
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn compaction_activate(
        &self,
        output_segment_id: &str,
        input_segment_ids: &[String],
        job_id: &str,
        details: Value,
    ) -> Result<()> {
        let now = now_unix_ts();
        let input_update = if input_segment_ids.is_empty() {
            String::new()
        } else {
            let in_clause = input_segment_ids
                .iter()
                .map(|id| format!("'{}'", sql_quote(id)))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "UPDATE segments SET state='obsolete' WHERE id IN ({});",
                in_clause
            )
        };

        self.exec(&format!(
            "BEGIN IMMEDIATE;
             UPDATE segments SET state='active' WHERE id='{}';
             {}
             UPDATE compaction_jobs SET state='completed', details_json='{}', updated_at={} WHERE id='{}';
             COMMIT;",
            sql_quote(output_segment_id),
            input_update,
            sql_quote(&details.to_string()),
            now,
            sql_quote(job_id)
        ))
    }

    fn list_compaction_jobs_by_state(
        &self,
        collection_id: &str,
        states: &[&str],
    ) -> Result<Vec<CompactionJobRecord>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let in_clause = states
            .iter()
            .map(|s| format!("'{}'", sql_quote(s)))
            .collect::<Vec<_>>()
            .join(",");
        let rows = self.query_json(&format!(
            "SELECT id, collection_id, state, details_json, created_at, updated_at
             FROM compaction_jobs WHERE collection_id='{}' AND state IN ({}) ORDER BY updated_at ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let details = serde_json::from_str(row["details_json"].as_str().unwrap_or("{}"))
                .unwrap_or_else(|_| json!({}));
            out.push(CompactionJobRecord {
                id: row["id"].as_str().unwrap_or_default().to_string(),
                collection_id: row["collection_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                state: row["state"].as_str().unwrap_or_default().to_string(),
                details,
                created_at: row["created_at"].as_i64().unwrap_or_default(),
                updated_at: row["updated_at"].as_i64().unwrap_or_default(),
            });
        }
        Ok(out)
    }

    fn upsert_metadata(
        &self,
        collection_id: &str,
        vector_id: usize,
        metadata: &VectorMetadata,
    ) -> Result<()> {
        let now = now_unix_ts();
        let collection = sql_quote(collection_id);
        let mut tags = metadata
            .tags
            .iter()
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>();
        tags.sort();
        tags.dedup();

        let mut tag_sql = format!(
            "DELETE FROM vector_tags WHERE collection_id='{}' AND vector_id={};",
            collection, vector_id
        );
        for tag in &tags {
            let tag_q = sql_quote(tag);
            tag_sql.push_str(&format!(
                "INSERT INTO tag_ids(collection_id, id, tag, created_at)
                 SELECT '{collection}',
                        COALESCE((SELECT MAX(id) + 1 FROM tag_ids WHERE collection_id='{collection}'), 1),
                        '{tag_q}',
                        {now}
                 WHERE NOT EXISTS (
                    SELECT 1 FROM tag_ids WHERE collection_id='{collection}' AND tag='{tag_q}'
                 );
                 INSERT OR IGNORE INTO vector_tags(collection_id, vector_id, tag_id, created_at)
                 SELECT '{collection}', {vector_id}, id, {now}
                 FROM tag_ids
                 WHERE collection_id='{collection}' AND tag='{tag_q}'
                 LIMIT 1;"
            ));
        }

        self.exec(&format!(
            "BEGIN IMMEDIATE;
             INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
             SELECT '{}', COALESCE(MAX(vector_id) + 1, 0) FROM vector_metadata WHERE collection_id='{}';
             UPDATE vector_id_counters
             SET next_id = CASE WHEN next_id <= {} THEN {} ELSE next_id END
             WHERE collection_id='{}';
             INSERT OR REPLACE INTO vector_metadata(collection_id, vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at)
             VALUES ('{}', {}, '{}', {}, {}, {}, '{}', {});
             {}
             COMMIT;",
            collection,
            collection,
            vector_id,
            vector_id + 1,
            collection,
            collection,
            vector_id,
            sql_quote(&metadata.source_file),
            metadata.start_time_ms,
            metadata.duration_ms,
            metadata.bpm,
            sql_quote(&serde_json::to_string(&tags)?),
            now,
            tag_sql,
        ))
    }

    fn fetch_metadata(
        &self,
        collection_id: &str,
        ids: &[usize],
    ) -> Result<HashMap<usize, VectorMetadata>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut unique_ids = ids.to_vec();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        let in_clause = unique_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let rows = self.query_rows::<MetadataRow>(&format!(
            "SELECT vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json
             FROM vector_metadata
             WHERE collection_id='{}' AND vector_id IN ({})
             ORDER BY vector_id ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            out.insert(row.vector_id, row.metadata);
        }
        Ok(out)
    }

    fn fetch_all_metadata(&self, collection_id: &str) -> Result<Vec<(usize, VectorMetadata)>> {
        let rows = self.query_rows::<MetadataRow>(&format!(
            "SELECT vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json
             FROM vector_metadata
             WHERE collection_id='{}'
             ORDER BY vector_id ASC;",
            sql_quote(collection_id)
        ))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push((row.vector_id, row.metadata));
        }
        Ok(out)
    }

    fn fetch_tag_dictionary(&self, collection_id: &str) -> Result<HashMap<String, u32>> {
        let rows = self.query_rows::<TagDictionaryRow>(&format!(
            "SELECT id, tag
             FROM tag_ids
             WHERE collection_id='{}'
             ORDER BY id ASC;",
            sql_quote(collection_id)
        ))?;
        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            if !row.tag.is_empty() {
                out.insert(row.tag, row.id);
            }
        }
        Ok(out)
    }

    fn fetch_filter_rows(&self, collection_id: &str) -> Result<Vec<FilterRow>> {
        self.query_rows::<FilterRow>(&format!(
            "SELECT vm.vector_id AS vector_id, vm.bpm AS bpm, COALESCE(GROUP_CONCAT(vt.tag_id), '') AS tag_ids
             FROM vector_metadata vm
             LEFT JOIN vector_tags vt
               ON vt.collection_id = vm.collection_id
              AND vt.vector_id = vm.vector_id
             WHERE vm.collection_id='{}'
             GROUP BY vm.vector_id, vm.bpm
             ORDER BY vm.vector_id ASC;",
            sql_quote(collection_id)
        ))
    }

    fn insert_orphan_file(
        &self,
        collection_id: &str,
        original_path: &Path,
        quarantine_path: &Path,
        reason: &str,
    ) -> Result<()> {
        let size_bytes = std::fs::metadata(quarantine_path)
            .map(|m| m.len())
            .unwrap_or(0);
        self.exec(&format!(
            "INSERT INTO orphan_files(collection_id, original_path, quarantine_path, reason, size_bytes, quarantined_at, deleted_at, deleted_reason) VALUES ('{}', '{}', '{}', '{}', {}, {}, NULL, NULL);",
            sql_quote(collection_id),
            sql_quote(&original_path.to_string_lossy()),
            sql_quote(&quarantine_path.to_string_lossy()),
            sql_quote(reason),
            size_bytes,
            now_unix_ts()
        ))
    }

    fn quarantine_usage(&self) -> Result<QuarantineUsage> {
        let rows = self.query_json(
            "SELECT COUNT(*) AS files, COALESCE(SUM(size_bytes), 0) AS bytes
             FROM orphan_files
             WHERE deleted_at IS NULL;",
        )?;
        let files = rows
            .first()
            .and_then(|r| r["files"].as_i64())
            .unwrap_or_default()
            .max(0) as usize;
        let bytes = rows
            .first()
            .and_then(|r| r["bytes"].as_i64())
            .unwrap_or_default()
            .max(0) as u64;
        Ok(QuarantineUsage { files, bytes })
    }

    fn enforce_quarantine_cap(&self, max_files: usize, max_bytes: u64) -> Result<usize> {
        let rows = self.query_json(
            "SELECT id, quarantine_path, COALESCE(size_bytes, 0) AS size_bytes
             FROM orphan_files
             WHERE deleted_at IS NULL
             ORDER BY quarantined_at ASC, id ASC;",
        )?;
        if rows.is_empty() {
            return Ok(0);
        }

        let mut tracked = rows
            .into_iter()
            .map(|row| {
                (
                    row["id"].as_i64().unwrap_or_default(),
                    PathBuf::from(row["quarantine_path"].as_str().unwrap_or_default()),
                    row["size_bytes"].as_i64().unwrap_or_default().max(0) as u64,
                )
            })
            .collect::<Vec<_>>();
        let mut files = tracked.len();
        let mut bytes = tracked.iter().map(|(_, _, sz)| *sz).sum::<u64>();
        let mut evicted = 0usize;

        while files > max_files || bytes > max_bytes {
            let Some((id, path, size)) = tracked.first().cloned() else {
                break;
            };
            tracked.remove(0);
            if path.exists() {
                let _ = std::fs::remove_file(&path);
            }
            self.exec(&format!(
                "UPDATE orphan_files
                 SET deleted_at = {}, deleted_reason = 'cap_evict'
                 WHERE id = {};",
                now_unix_ts(),
                id
            ))?;
            files = files.saturating_sub(1);
            bytes = bytes.saturating_sub(size);
            evicted += 1;
        }

        if evicted > 0 {
            self.quarantine_evictions_total
                .fetch_add(evicted as u64, AtomicOrdering::Relaxed);
        }
        Ok(evicted)
    }

    fn quarantine_gc(&self, ttl_secs: i64) -> Result<Vec<PathBuf>> {
        let cutoff = now_unix_ts() - ttl_secs;
        let rows = self.query_json(&format!(
            "SELECT id, quarantine_path FROM orphan_files WHERE deleted_at IS NULL AND quarantined_at < {};",
            cutoff
        ))?;

        let mut deleted = Vec::new();
        for row in rows {
            let id = row["id"].as_i64().unwrap_or_default();
            let path = PathBuf::from(row["quarantine_path"].as_str().unwrap_or_default());
            if path.exists() {
                let _ = std::fs::remove_file(&path);
            }
            self.exec(&format!(
                "UPDATE orphan_files SET deleted_at = {}, deleted_reason = 'ttl_gc' WHERE id = {};",
                now_unix_ts(),
                id
            ))?;
            deleted.push(path);
        }
        Ok(deleted)
    }

    fn total_vectors(&self, collection_id: &str) -> Result<usize> {
        let rows = self.query_json(&format!(
            "SELECT COUNT(*) AS n FROM vector_metadata WHERE collection_id='{}';",
            sql_quote(collection_id)
        ))?;
        Ok(rows
            .first()
            .and_then(|r| r["n"].as_i64())
            .unwrap_or_default() as usize)
    }

    fn vacuum_into(&self, output_path: &Path) -> Result<()> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if output_path.exists() {
            std::fs::remove_file(output_path)
                .with_context(|| format!("removing stale snapshot {:?}", output_path))?;
        }
        self.exec(&format!(
            "VACUUM INTO '{}';",
            sql_quote(&output_path.to_string_lossy())
        ))
        .with_context(|| format!("vacuum into {:?}", output_path))
    }

    fn audit_event(
        &self,
        request_id: &str,
        api_key_id: Option<&str>,
        endpoint: &str,
        action: &str,
        status_code: u16,
        latency_ms: f64,
        details: Value,
    ) -> Result<()> {
        self.exec(&format!(
            "INSERT INTO audit_events(ts, request_id, api_key_id, endpoint, action, status_code, latency_ms, client_ip, details_json) VALUES ({}, '{}', {}, '{}', '{}', {}, {}, NULL, '{}');",
            now_unix_ts(),
            sql_quote(request_id),
            api_key_id
                .map(|v| format!("'{}'", sql_quote(v)))
                .unwrap_or_else(|| "NULL".to_string()),
            sql_quote(endpoint),
            sql_quote(action),
            status_code,
            latency_ms,
            sql_quote(&details.to_string())
        ))
    }
}

impl SqliteCatalog {
    /// Range-based metadata fetch: single SQL query using BETWEEN instead of
    /// an unbounded IN(...) clause. Extra rows outside the exact ID set are
    /// harmless — callers look up by ID from the returned HashMap.
    pub fn fetch_metadata_range(
        &self,
        collection_id: &str,
        start_id: usize,
        end_id: usize,
    ) -> Result<HashMap<usize, VectorMetadata>> {
        if start_id > end_id {
            return Ok(HashMap::new());
        }
        let rows = self.query_rows::<MetadataRow>(&format!(
            "SELECT vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json
             FROM vector_metadata
             WHERE collection_id='{}' AND vector_id >= {} AND vector_id <= {}
             ORDER BY vector_id ASC;",
            sql_quote(collection_id),
            start_id,
            end_id
        ))?;

        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            out.insert(row.vector_id, row.metadata);
        }
        Ok(out)
    }

    pub fn audit_events_batch(&self, events: &[AuditEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let now = now_unix_ts();
        let mut sql = String::with_capacity(events.len() * 200 + 50);
        sql.push_str("BEGIN IMMEDIATE;");
        sql.push_str("INSERT INTO audit_events(ts, request_id, api_key_id, endpoint, action, status_code, latency_ms, client_ip, details_json) VALUES ");

        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('(');
            sql.push_str(&now.to_string());
            sql.push_str(", '");
            sql.push_str(&sql_quote(&event.request_id));
            sql.push_str("', ");
            if let Some(ak) = &event.api_key_id {
                sql.push('\'');
                sql.push_str(&sql_quote(ak));
                sql.push('\'');
            } else {
                sql.push_str("NULL");
            }
            sql.push_str(", '");
            sql.push_str(&sql_quote(&event.endpoint));
            sql.push_str("', '");
            sql.push_str(&sql_quote(&event.action));
            sql.push_str("', ");
            sql.push_str(&event.status_code.to_string());
            sql.push_str(", ");
            sql.push_str(&event.latency_ms.to_string());
            sql.push_str(", NULL, '");
            sql.push_str(&sql_quote(&event.details.to_string()));
            sql.push_str("')");
        }
        sql.push_str("; COMMIT;");
        self.exec(&sql)
    }
}

#[derive(Debug)]
struct MetadataRow {
    vector_id: usize,
    metadata: VectorMetadata,
}

#[derive(Debug)]
struct TagDictionaryRow {
    id: u32,
    tag: String,
}

impl FromSqlRow for WalEntry {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        let lsn = column_i64(stmt, 0).max(0) as u64;
        let vector_id = column_i64(stmt, 1).max(0) as usize;
        let vector: Vec<f32> = serde_json::from_slice(column_text_bytes(stmt, 2))?;
        let metadata: VectorMetadata = serde_json::from_slice(column_text_bytes(stmt, 3))?;
        Ok(Self {
            lsn,
            vector_id,
            vector,
            metadata,
        })
    }
}

impl FromSqlRow for WalIngestResult {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        let vector_id = column_i64(stmt, 0).max(0) as usize;
        let created = column_i64(stmt, 1) != 0;
        let lsn = match sqlite3_column_type(stmt, 2) {
            SQLITE_NULL => None,
            _ => Some(column_i64(stmt, 2).max(0) as u64),
        };
        Ok(Self {
            vector_id,
            created,
            lsn,
        })
    }
}

impl FromSqlRow for SegmentRecord {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        Ok(Self {
            id: column_text_string(stmt, 0),
            collection_id: column_text_string(stmt, 1),
            level: column_i64(stmt, 2),
            path: PathBuf::from(column_text_string(stmt, 3)),
            row_count: column_i64(stmt, 4).max(0) as usize,
            vector_id_start: column_i64(stmt, 5).max(0) as usize,
            vector_id_end: column_i64(stmt, 6).max(0) as usize,
            created_lsn: column_i64(stmt, 7).max(0) as u64,
            state: column_text_string(stmt, 8),
        })
    }
}

impl FromSqlRow for MetadataRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        let vector_id = column_i64(stmt, 0).max(0) as usize;
        let source_file = column_text_string(stmt, 1);
        let start_time_ms = column_i64(stmt, 2).max(0) as u32;
        let duration_ms = column_i64(stmt, 3).max(0) as u16;
        let bpm = column_f64(stmt, 4) as f32;
        let tags: Vec<String> = serde_json::from_slice(column_text_bytes(stmt, 5))?;
        Ok(Self {
            vector_id,
            metadata: VectorMetadata {
                source_file,
                start_time_ms,
                duration_ms,
                bpm,
                tags,
            },
        })
    }
}

impl FromSqlRow for TagDictionaryRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        Ok(Self {
            id: column_i64(stmt, 0).max(0) as u32,
            tag: column_text_string(stmt, 1),
        })
    }
}

impl FromSqlRow for FilterRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        let vector_id = column_i64(stmt, 0).max(0) as usize;
        let bpm = column_f64(stmt, 1) as f32;
        let tag_ids = column_text_string(stmt, 2)
            .split(',')
            .filter_map(|v| {
                let t = v.trim();
                if t.is_empty() {
                    None
                } else {
                    t.parse::<u32>().ok()
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            vector_id,
            bpm,
            tag_ids,
        })
    }
}

impl SqliteCatalog {
    fn query_wal(&self, sql: &str) -> Result<Vec<WalEntry>> {
        self.query_rows::<WalEntry>(sql)
    }

    /// Pure SQL builder for a single ingest row. Produces SQL that:
    /// - Sets up temp tables (_ingest_res, _ingest_alloc)
    /// - Handles idempotency lookup
    /// - Allocates vector ID
    /// - Inserts into wal_entries + vector_metadata + tags
    /// - Selects the result row
    ///
    /// Does NOT include BEGIN/COMMIT — the caller manages the transaction.
    /// Uses `write!` on a pre-allocated buffer to avoid per-call heap allocations.
    fn build_ingest_row_sql(
        sql: &mut String,
        collection_id: &str,
        vector_id: Option<usize>,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<()> {
        let normalized_key = idempotency_key.and_then(|k| {
            let trimmed = k.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

        let now = now_unix_ts();
        let collection = sql_quote(collection_id);
        let vector_json = sql_quote(&serde_json::to_string(vector)?);
        let source_file = sql_quote(&metadata.source_file);

        let key_sql = normalized_key
            .as_ref()
            .map(|k| format!("'{}'", sql_quote(k)))
            .unwrap_or_else(|| "NULL".to_string());

        // Temp table setup + clear
        write!(
            sql,
            "CREATE TEMP TABLE IF NOT EXISTS _ingest_res(\
               vector_id INTEGER NOT NULL, created INTEGER NOT NULL, lsn INTEGER);\
             CREATE TEMP TABLE IF NOT EXISTS _ingest_alloc(\
               vector_id INTEGER NOT NULL);\
             DELETE FROM _ingest_res;\
             DELETE FROM _ingest_alloc;"
        )
        .unwrap();

        // Idempotency lookup
        if let Some(key) = &normalized_key {
            let key_q = sql_quote(key);
            write!(
                sql,
                "INSERT INTO _ingest_res(vector_id, created, lsn) \
                 SELECT vector_id, 0, NULL \
                 FROM wal_entries \
                 WHERE collection_id='{}' AND idempotency_key='{}' \
                 LIMIT 1;",
                collection, key_q
            )
            .unwrap();
        }

        // Normalize tags
        let mut normalized_tags = metadata
            .tags
            .iter()
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>();
        normalized_tags.sort();
        normalized_tags.dedup();

        let mut normalized_metadata = metadata.clone();
        normalized_metadata.tags = normalized_tags.clone();
        let metadata_json = sql_quote(&serde_json::to_string(&normalized_metadata)?);
        let tags_json = sql_quote(&serde_json::to_string(&normalized_tags)?);

        // ID allocation
        write!(
            sql,
            "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id) \
             SELECT '{collection}', COALESCE(MAX(vector_id) + 1, 0) \
             FROM vector_metadata \
             WHERE collection_id='{collection}';"
        )
        .unwrap();

        match vector_id {
            Some(v) => {
                let next = v + 1;
                write!(
                    sql,
                    "INSERT INTO _ingest_alloc(vector_id) \
                     SELECT {v} \
                     WHERE NOT EXISTS (SELECT 1 FROM _ingest_res);\
                     UPDATE vector_id_counters \
                     SET next_id = CASE WHEN next_id <= {v} THEN {next} ELSE next_id END \
                     WHERE collection_id='{collection}' AND NOT EXISTS (SELECT 1 FROM _ingest_res);"
                )
                .unwrap();
            }
            None => {
                write!(
                    sql,
                    "UPDATE vector_id_counters \
                     SET next_id = next_id + 1 \
                     WHERE collection_id='{collection}' AND NOT EXISTS (SELECT 1 FROM _ingest_res);\
                     INSERT INTO _ingest_alloc(vector_id) \
                     SELECT next_id - 1 \
                     FROM vector_id_counters \
                     WHERE collection_id='{collection}' AND NOT EXISTS (SELECT 1 FROM _ingest_res);"
                )
                .unwrap();
            }
        }

        // WAL insert
        write!(
            sql,
            "INSERT INTO wal_entries(collection_id, vector_id, vector_json, metadata_json, \
             idempotency_key, checkpointed_at, created_at) \
             SELECT '{collection}', vector_id, '{vector_json}', '{metadata_json}', \
             {key_sql}, NULL, {now} \
             FROM _ingest_alloc \
             WHERE NOT EXISTS (SELECT 1 FROM _ingest_res) \
             LIMIT 1;"
        )
        .unwrap();

        // Result capture
        write!(
            sql,
            "INSERT INTO _ingest_res(vector_id, created, lsn) \
             SELECT vector_id, 1, lsn \
             FROM wal_entries \
             WHERE lsn = last_insert_rowid() \
               AND changes() > 0 \
               AND NOT EXISTS (SELECT 1 FROM _ingest_res);"
        )
        .unwrap();

        // Metadata insert
        write!(
            sql,
            "INSERT OR REPLACE INTO vector_metadata(\
             collection_id, vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at) \
             SELECT '{collection}', vector_id, '{source_file}', {start_time}, {duration}, {bpm}, '{tags_json}', {now} \
             FROM _ingest_res \
             WHERE created = 1;",
            start_time = metadata.start_time_ms,
            duration = metadata.duration_ms,
            bpm = metadata.bpm
        ).unwrap();

        // Tag inserts
        if !normalized_tags.is_empty() {
            write!(
                sql,
                "DELETE FROM vector_tags \
                 WHERE collection_id='{collection}' \
                   AND vector_id IN (SELECT vector_id FROM _ingest_res WHERE created = 1);"
            )
            .unwrap();
            for tag in &normalized_tags {
                let tag_q = sql_quote(tag);
                write!(
                    sql,
                    "INSERT INTO tag_ids(collection_id, id, tag, created_at) \
                     SELECT '{collection}', \
                            COALESCE((SELECT MAX(id) + 1 FROM tag_ids WHERE collection_id='{collection}'), 1), \
                            '{tag_q}', {now} \
                     WHERE NOT EXISTS ( \
                       SELECT 1 FROM tag_ids WHERE collection_id='{collection}' AND tag='{tag_q}' \
                     );\
                     INSERT OR IGNORE INTO vector_tags(collection_id, vector_id, tag_id, created_at) \
                     SELECT '{collection}', vector_id, \
                            (SELECT id FROM tag_ids WHERE collection_id='{collection}' AND tag='{tag_q}' LIMIT 1), \
                            {now} \
                     FROM _ingest_res \
                     WHERE created = 1;"
                ).unwrap();
            }
        }

        // Result row
        sql.push_str("SELECT vector_id, created, lsn FROM _ingest_res LIMIT 1;");

        Ok(())
    }

    fn ingest_wal_internal(
        &self,
        collection_id: &str,
        vector_id: Option<usize>,
        vector: &[f32],
        metadata: &VectorMetadata,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
        let mut sql = String::with_capacity(4096);
        sql.push_str("BEGIN IMMEDIATE;");
        Self::build_ingest_row_sql(
            &mut sql,
            collection_id,
            vector_id,
            vector,
            metadata,
            idempotency_key,
        )?;
        sql.push_str("COMMIT;");

        let rows = self.query_rows::<WalIngestResult>(&sql)?;
        let Some(row) = rows.first() else {
            return Err(anyhow!("ingest transaction produced no result row"));
        };

        Ok(row.clone())
    }

    /// Batch ingest: single BEGIN...COMMIT wrapping N row insertions.
    /// Holds the writer lock for the entire batch to amortize transaction cost.
    /// Preserves idempotency per-row.
    pub fn ingest_wal_batch(
        &self,
        collection_id: &str,
        entries: &[(Vec<f32>, VectorMetadata, Option<String>)],
    ) -> Result<Vec<WalIngestResult>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        #[derive(Debug)]
        struct PreparedBatchEntry {
            vector_json: String,
            metadata: VectorMetadata,
            metadata_json: String,
            tags_json: String,
            idempotency_key: Option<String>,
        }

        fn normalize_idempotency_key(key: Option<&str>) -> Option<String> {
            key.and_then(|k| {
                let trimmed = k.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
        }

        // Keep serialization/normalization outside the writer lock to avoid
        // inflating lock hold times under concurrent ingest load.
        let prepared_entries = entries
            .iter()
            .map(
                |(vector, metadata, idempotency_key)| -> Result<PreparedBatchEntry> {
                    let mut normalized_metadata = metadata.clone();
                    let mut tags = normalized_metadata
                        .tags
                        .iter()
                        .map(|t| t.trim().to_ascii_lowercase())
                        .filter(|t| !t.is_empty())
                        .collect::<Vec<_>>();
                    tags.sort();
                    tags.dedup();
                    normalized_metadata.tags = tags;

                    Ok(PreparedBatchEntry {
                        vector_json: serde_json::to_string(vector)?,
                        metadata_json: serde_json::to_string(&normalized_metadata)?,
                        tags_json: serde_json::to_string(&normalized_metadata.tags)?,
                        metadata: normalized_metadata,
                        idempotency_key: normalize_idempotency_key(idempotency_key.as_deref()),
                    })
                },
            )
            .collect::<Result<Vec<_>>>()?;

        let guard = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        guard.exec("BEGIN IMMEDIATE;")?;

        let tx_result = (|| -> Result<Vec<WalIngestResult>> {
            const BATCH_CHUNK_ROWS: usize = 10_000;

            #[derive(Debug)]
            struct PendingInsert<'a> {
                vector_id: usize,
                entry: &'a PreparedBatchEntry,
            }

            let collection = sql_quote(collection_id);
            let mut results = Vec::with_capacity(prepared_entries.len());

            guard.exec(&format!(
                "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
                 SELECT '{collection}', COALESCE(MAX(vector_id) + 1, 0)
                 FROM vector_metadata
                 WHERE collection_id='{collection}';"
            ))?;

            for chunk in prepared_entries.chunks(BATCH_CHUNK_ROWS) {
                let now = now_unix_ts();
                let normalized_keys = chunk
                    .iter()
                    .map(|entry| entry.idempotency_key.clone())
                    .collect::<Vec<_>>();

                let mut unique_lookup_keys = Vec::new();
                let mut seen_lookup = HashSet::new();
                for key in normalized_keys.iter().flatten() {
                    if seen_lookup.insert(key.clone()) {
                        unique_lookup_keys.push(key.clone());
                    }
                }

                let mut existing_by_key: HashMap<String, usize> = HashMap::new();
                if !unique_lookup_keys.is_empty() {
                    let in_clause = unique_lookup_keys
                        .iter()
                        .map(|k| format!("'{}'", sql_quote(k)))
                        .collect::<Vec<_>>()
                        .join(",");
                    let rows = guard.query_json(
                        &format!(
                            "SELECT idempotency_key, vector_id
                             FROM wal_entries
                             WHERE collection_id='{}'
                               AND idempotency_key IN ({});
                            ",
                            collection, in_clause
                        ),
                        None,
                    )?;
                    for row in rows {
                        let key = row["idempotency_key"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string();
                        if key.is_empty() {
                            continue;
                        }
                        let vector_id = row["vector_id"].as_i64().unwrap_or(0).max(0) as usize;
                        existing_by_key.insert(key, vector_id);
                    }
                }

                let mut seen_new_keys = HashSet::new();
                let mut new_row_count = 0usize;
                for key in &normalized_keys {
                    match key {
                        Some(k) => {
                            if existing_by_key.contains_key(k) || seen_new_keys.contains(k) {
                                continue;
                            }
                            seen_new_keys.insert(k.clone());
                            new_row_count += 1;
                        }
                        None => {
                            new_row_count += 1;
                        }
                    }
                }

                let mut next_allocated_id = 0usize;
                if new_row_count > 0 {
                    let rows = guard.query_json(
                        &format!(
                            "SELECT next_id AS next_id
                             FROM vector_id_counters
                             WHERE collection_id='{}'
                             LIMIT 1;",
                            collection
                        ),
                        None,
                    )?;
                    next_allocated_id = rows
                        .first()
                        .and_then(|row| row["next_id"].as_i64())
                        .unwrap_or(0)
                        .max(0) as usize;
                    guard.exec(&format!(
                        "UPDATE vector_id_counters
                         SET next_id = next_id + {}
                         WHERE collection_id='{}';",
                        new_row_count, collection
                    ))?;
                }

                let mut pending = Vec::with_capacity(new_row_count);
                let mut assigned_new_by_key: HashMap<String, usize> = HashMap::new();

                for (entry, key) in chunk.iter().zip(normalized_keys.iter()) {
                    let mut created = true;
                    let vector_id = if let Some(k) = key {
                        if let Some(existing_id) = existing_by_key.get(k) {
                            created = false;
                            *existing_id
                        } else if let Some(existing_id) = assigned_new_by_key.get(k) {
                            created = false;
                            *existing_id
                        } else {
                            let allocated = next_allocated_id;
                            next_allocated_id += 1;
                            assigned_new_by_key.insert(k.clone(), allocated);
                            allocated
                        }
                    } else {
                        let allocated = next_allocated_id;
                        next_allocated_id += 1;
                        allocated
                    };

                    results.push(WalIngestResult {
                        vector_id,
                        created,
                        lsn: None,
                    });

                    if !created {
                        continue;
                    }

                    pending.push(PendingInsert { vector_id, entry });
                }

                if pending.is_empty() {
                    continue;
                }

                let mut unique_tags = HashSet::new();
                let mut vector_tag_pairs = Vec::new();
                for row in &pending {
                    for tag in &row.entry.metadata.tags {
                        unique_tags.insert(tag.clone());
                        vector_tag_pairs.push((row.vector_id, tag.clone()));
                    }
                }

                let estimated_capacity = pending.len() * 2048
                    + unique_tags.len() * 128
                    + vector_tag_pairs.len() * 96
                    + 2048;
                let mut sql = String::with_capacity(estimated_capacity);

                sql.push_str(
                    "INSERT INTO wal_entries(collection_id, vector_id, vector_json, metadata_json, idempotency_key, checkpointed_at, created_at) VALUES ",
                );
                for (idx, row) in pending.iter().enumerate() {
                    if idx > 0 {
                        sql.push(',');
                    }
                    write!(
                        sql,
                        "('{}', {}, '{}', '{}', {}, NULL, {})",
                        collection,
                        row.vector_id,
                        sql_quote(&row.entry.vector_json),
                        sql_quote(&row.entry.metadata_json),
                        row.entry
                            .idempotency_key
                            .as_ref()
                            .map(|k| format!("'{}'", sql_quote(k)))
                            .unwrap_or_else(|| "NULL".to_string()),
                        now,
                    )
                    .unwrap();
                }
                sql.push(';');

                sql.push_str("INSERT OR REPLACE INTO vector_metadata(collection_id, vector_id, source_file, start_time_ms, duration_ms, bpm, tags_json, created_at) VALUES ");
                for (idx, row) in pending.iter().enumerate() {
                    if idx > 0 {
                        sql.push(',');
                    }
                    write!(
                        sql,
                        "('{}', {}, '{}', {}, {}, {}, '{}', {})",
                        collection,
                        row.vector_id,
                        sql_quote(&row.entry.metadata.source_file),
                        row.entry.metadata.start_time_ms,
                        row.entry.metadata.duration_ms,
                        row.entry.metadata.bpm,
                        sql_quote(&row.entry.tags_json),
                        now,
                    )
                    .unwrap();
                }
                sql.push(';');

                let mut ordered_tags = unique_tags.into_iter().collect::<Vec<_>>();
                ordered_tags.sort();
                for tag in ordered_tags {
                    write!(
                        sql,
                        "INSERT INTO tag_ids(collection_id, id, tag, created_at)
                         SELECT '{collection}',
                                COALESCE((SELECT MAX(id) + 1 FROM tag_ids WHERE collection_id='{collection}'), 1),
                                '{}',
                                {}
                         WHERE NOT EXISTS (
                             SELECT 1 FROM tag_ids WHERE collection_id='{collection}' AND tag='{}'
                         );",
                        sql_quote(&tag),
                        now,
                        sql_quote(&tag),
                    )
                    .unwrap();
                }

                if !vector_tag_pairs.is_empty() {
                    sql.push_str("INSERT OR IGNORE INTO vector_tags(collection_id, vector_id, tag_id, created_at) VALUES ");
                    for (idx, (vector_id, tag)) in vector_tag_pairs.iter().enumerate() {
                        if idx > 0 {
                            sql.push(',');
                        }
                        write!(
                            sql,
                            "('{}', {}, (SELECT id FROM tag_ids WHERE collection_id='{}' AND tag='{}' LIMIT 1), {})",
                            collection,
                            vector_id,
                            collection,
                            sql_quote(tag),
                            now,
                        )
                        .unwrap();
                    }
                    sql.push(';');
                }

                guard.exec(&sql)?;
            }

            Ok(results)
        })();

        match tx_result {
            Ok(results) => {
                guard.exec("COMMIT;")?;
                Ok(results)
            }
            Err(err) => {
                let _ = guard.exec("ROLLBACK;");
                Err(err)
            }
        }
    }
}

fn now_unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn sql_quote(value: &str) -> String {
    value.replace('\'', "''")
}

fn random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("{}_{}{}", prefix, now_unix_ts(), hex_string(&bytes))
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn hash_secret(pepper: &str, secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(pepper.as_bytes());
    hasher.update(b":");
    hasher.update(secret.as_bytes());
    let digest = hasher.finalize();
    hex_string(&digest)
}

pub fn verify_token_hash(pepper: &str, secret: &str, expected_hash: &str) -> bool {
    // Backward-compatible verification for pre-v2.1 short hashes.
    if expected_hash.len() == 16 {
        let legacy = legacy_hash(pepper, secret);
        return constant_time_eq(legacy.as_bytes(), expected_hash.as_bytes());
    }
    let actual = hash_secret(pepper, secret);
    constant_time_eq(actual.as_bytes(), expected_hash.as_bytes())
}

pub fn parse_token(header_value: &str) -> Option<(String, String)> {
    let mut parts = header_value.split_whitespace();
    let scheme = parts.next()?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let (id, secret) = token.split_once('.')?;
    if !id.starts_with("vbk_") || secret.is_empty() {
        return None;
    }
    Some((id.to_string(), secret.to_string()))
}

pub fn audit_error_payload(code: &'static str, message: impl Into<String>) -> Value {
    json!({"code": code, "message": message.into()})
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn legacy_hash(pepper: &str, secret: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    pepper.hash(&mut hasher);
    secret.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_token_requires_bearer_and_vbk_prefix() {
        let parsed = parse_token("Bearer vbk_abc123.secretxyz");
        assert_eq!(
            parsed,
            Some(("vbk_abc123".to_string(), "secretxyz".to_string()))
        );
        let parsed_lower = parse_token("bearer vbk_abc123.secretxyz");
        assert_eq!(
            parsed_lower,
            Some(("vbk_abc123".to_string(), "secretxyz".to_string()))
        );

        assert!(parse_token("vbk_abc123.secretxyz").is_none());
        assert!(parse_token("Bearer abc123.secretxyz").is_none());
        assert!(parse_token("Bearer vbk_abc123.").is_none());
        assert!(parse_token("Bearer vbk_abc123.secretxyz trailing").is_none());
    }

    #[test]
    fn verify_token_hash_supports_current_and_legacy_formats() {
        let pepper = "pepper";
        let secret = "super-secret";

        let current = hash_secret(pepper, secret);
        assert!(verify_token_hash(pepper, secret, &current));
        assert!(!verify_token_hash(pepper, "wrong", &current));

        let legacy = legacy_hash(pepper, secret);
        assert_eq!(legacy.len(), 16);
        assert!(verify_token_hash(pepper, secret, &legacy));
        assert!(!verify_token_hash(pepper, "wrong", &legacy));
    }

    #[test]
    fn constant_time_eq_requires_equal_length_and_content() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
