use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uchar, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use vibrato_core::metadata::{VectorMetadata, VectorMetadataV3};
use xxhash_rust::xxh3::xxh3_64;

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
    pub metadata: Option<VectorMetadataV3>,
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub lsn: u64,
    pub vector_id: usize,
    pub vector: Vec<f32>,
    pub metadata: VectorMetadataV3,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestMetadataV3Input {
    pub entity_id: u64,
    pub sequence_ts: u64,
    pub tags: Vec<String>,
    pub payload: Vec<u8>,
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
    pub max_tag_registry_size: usize,
}

impl Default for CatalogOptions {
    fn default() -> Self {
        Self {
            read_timeout_ms: 5_000,
            wal_autocheckpoint_pages: 1000,
            max_tag_registry_size: 500_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FilterRow {
    pub vector_id: usize,
    pub tag_ids: Vec<u32>,
}

#[derive(Default)]
struct TagRegistryCache {
    forward: HashMap<String, u32>,
    reverse: HashMap<String, String>,
    collection_counts: HashMap<String, usize>,
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
    fn sqlite3_reset(stmt: *mut Sqlite3Stmt) -> c_int;
    fn sqlite3_clear_bindings(stmt: *mut Sqlite3Stmt) -> c_int;
    fn sqlite3_bind_null(stmt: *mut Sqlite3Stmt, idx: c_int) -> c_int;
    fn sqlite3_bind_int64(stmt: *mut Sqlite3Stmt, idx: c_int, value: i64) -> c_int;
    fn sqlite3_bind_double(stmt: *mut Sqlite3Stmt, idx: c_int, value: f64) -> c_int;
    fn sqlite3_bind_text(
        stmt: *mut Sqlite3Stmt,
        idx: c_int,
        value: *const c_char,
        nbytes: c_int,
        destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> c_int;
    fn sqlite3_bind_blob(
        stmt: *mut Sqlite3Stmt,
        idx: c_int,
        value: *const c_void,
        nbytes: c_int,
        destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> c_int;
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

#[inline]
fn sqlite_transient_destructor() -> unsafe extern "C" fn(*mut c_void) {
    unsafe { std::mem::transmute::<isize, unsafe extern "C" fn(*mut c_void)>(-1) }
}

struct PreparedStmt {
    db: *mut Sqlite3,
    stmt: *mut Sqlite3Stmt,
}

impl PreparedStmt {
    fn new(conn: &RawSqliteConnection, sql: &str) -> Result<Self> {
        let c_sql = CString::new(sql)?;
        let mut stmt: *mut Sqlite3Stmt = std::ptr::null_mut();
        let mut tail: *const c_char = std::ptr::null();
        let rc = unsafe {
            sqlite3_prepare_v2(conn.db, c_sql.as_ptr(), -1, &mut stmt, &mut tail as *mut _)
        };
        if rc != SQLITE_OK || stmt.is_null() {
            return Err(anyhow!("sqlite prepare failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(conn.db))
            }));
        }
        Ok(Self { db: conn.db, stmt })
    }

    #[inline]
    fn reset(&mut self) -> Result<()> {
        let rc_reset = unsafe { sqlite3_reset(self.stmt) };
        if rc_reset != SQLITE_OK {
            return Err(anyhow!("sqlite reset failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        let rc_clear = unsafe { sqlite3_clear_bindings(self.stmt) };
        if rc_clear != SQLITE_OK {
            return Err(anyhow!("sqlite clear_bindings failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
    }

    #[inline]
    fn bind_null(&mut self, idx: c_int) -> Result<()> {
        let rc = unsafe { sqlite3_bind_null(self.stmt, idx) };
        if rc != SQLITE_OK {
            return Err(anyhow!("sqlite bind_null failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
    }

    #[inline]
    fn bind_i64(&mut self, idx: c_int, value: i64) -> Result<()> {
        let rc = unsafe { sqlite3_bind_int64(self.stmt, idx, value) };
        if rc != SQLITE_OK {
            return Err(anyhow!("sqlite bind_int64 failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
    }

    #[inline]
    fn bind_f64(&mut self, idx: c_int, value: f64) -> Result<()> {
        let rc = unsafe { sqlite3_bind_double(self.stmt, idx, value) };
        if rc != SQLITE_OK {
            return Err(anyhow!("sqlite bind_double failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
    }

    #[inline]
    fn bind_text(&mut self, idx: c_int, value: &str) -> Result<()> {
        let bytes = value.as_bytes();
        let ptr = if bytes.is_empty() {
            std::ptr::null()
        } else {
            bytes.as_ptr() as *const c_char
        };
        let rc = unsafe {
            sqlite3_bind_text(
                self.stmt,
                idx,
                ptr,
                bytes.len() as c_int,
                Some(sqlite_transient_destructor()),
            )
        };
        if rc != SQLITE_OK {
            return Err(anyhow!("sqlite bind_text failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
    }

    #[inline]
    fn bind_blob(&mut self, idx: c_int, value: &[u8]) -> Result<()> {
        let ptr = if value.is_empty() {
            std::ptr::null()
        } else {
            value.as_ptr() as *const c_void
        };
        let rc = unsafe {
            sqlite3_bind_blob(
                self.stmt,
                idx,
                ptr,
                value.len() as c_int,
                Some(sqlite_transient_destructor()),
            )
        };
        if rc != SQLITE_OK {
            return Err(anyhow!("sqlite bind_blob failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
    }

    #[inline]
    fn bind_optional_text(&mut self, idx: c_int, value: Option<&str>) -> Result<()> {
        if let Some(v) = value {
            self.bind_text(idx, v)
        } else {
            self.bind_null(idx)
        }
    }

    #[inline]
    fn step_done(&mut self) -> Result<()> {
        let rc = unsafe { sqlite3_step(self.stmt) };
        if rc == SQLITE_DONE {
            return Ok(());
        }
        Err(anyhow!("sqlite step failed: {}", unsafe {
            c_ptr_to_string(sqlite3_errmsg(self.db))
        }))
    }

    #[inline]
    fn step_row(&mut self) -> Result<bool> {
        let rc = unsafe { sqlite3_step(self.stmt) };
        match rc {
            SQLITE_ROW => Ok(true),
            SQLITE_DONE => Ok(false),
            _ => Err(anyhow!("sqlite step failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            })),
        }
    }

    #[inline]
    fn column_i64(&self, idx: c_int) -> i64 {
        unsafe { sqlite3_column_int64(self.stmt, idx) }
    }

    #[inline]
    fn column_text(&self, idx: c_int) -> String {
        unsafe { column_text_string(self.stmt, idx) }
    }
}

impl Drop for PreparedStmt {
    fn drop(&mut self) {
        if !self.stmt.is_null() {
            unsafe {
                let _ = sqlite3_finalize(self.stmt);
            }
            self.stmt = std::ptr::null_mut();
        }
    }
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
        conn.set_busy_timeout(30_000)?;
        Ok(conn)
    }

    fn set_busy_timeout(&self, ms: u64) -> Result<()> {
        let clamped = ms.max(1).min(c_int::MAX as u64) as c_int;
        let rc = unsafe { sqlite3_busy_timeout(self.db, clamped) };
        if rc != SQLITE_OK {
            return Err(anyhow!("sqlite busy_timeout failed: {}", unsafe {
                c_ptr_to_string(sqlite3_errmsg(self.db))
            }));
        }
        Ok(())
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
                        let decoded = unsafe { T::from_row(stmt) };
                        match decoded {
                            Ok(row) => rows.push(row),
                            Err(err) => break Err(err),
                        }
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

unsafe fn column_text_bytes<'a>(stmt: *mut Sqlite3Stmt, idx: c_int) -> &'a [u8] {
    let ptr = sqlite3_column_text(stmt, idx);
    let len = sqlite3_column_bytes(stmt, idx).max(0) as usize;
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr as *const u8, len)
    }
}

unsafe fn column_blob_bytes<'a>(stmt: *mut Sqlite3Stmt, idx: c_int) -> &'a [u8] {
    let ptr = sqlite3_column_blob(stmt, idx);
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
        metadata: &IngestMetadataV3Input,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult>;
    fn ingest_wal_atomic(
        &self,
        collection_id: &str,
        vector: &[f32],
        metadata: &IngestMetadataV3Input,
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
        metadata: &VectorMetadataV3,
    ) -> Result<()>;
    fn fetch_metadata(
        &self,
        collection_id: &str,
        ids: &[usize],
    ) -> Result<HashMap<usize, VectorMetadataV3>>;
    fn fetch_all_metadata(&self, collection_id: &str) -> Result<Vec<(usize, VectorMetadataV3)>>;
    fn fetch_tag_dictionary(&self, collection_id: &str) -> Result<HashMap<String, u32>>;
    fn fetch_filter_rows(&self, collection_id: &str) -> Result<Vec<FilterRow>>;
    fn resolve_tag_ids_readonly(&self, collection_id: &str, tags: &[String]) -> Result<Vec<u32>>;
    fn resolve_tag_texts(&self, collection_id: &str, ids: &[u32]) -> Result<HashMap<u32, String>>;

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
    tag_registry_cache: RwLock<TagRegistryCache>,
    next_tag_id: AtomicU32,
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
            tag_registry_cache: RwLock::new(TagRegistryCache::default()),
            next_tag_id: AtomicU32::new(1),
        };
        catalog.apply_migrations()?;
        catalog.load_tag_registry_cache()?;
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
            let conn = RawSqliteConnection::open(&self.path, true)?;
            // Bound lock wait on read handles to the catalog read timeout so
            // lock contention cannot stall reads for the full writer busy timeout.
            conn.set_busy_timeout(self.options.read_timeout_ms.max(1))?;
            self.readers.push(std::sync::Mutex::new(conn));
        }
        Ok(())
    }

    fn tag_cache_key(collection_id: &str, tag: &str) -> String {
        format!("{}\u{1f}{}", collection_id, tag)
    }

    fn tag_id_cache_key(collection_id: &str, tag_id: u32) -> String {
        format!("{}\u{1f}{}", collection_id, tag_id)
    }

    fn cache_insert_tag(
        cache: &mut TagRegistryCache,
        collection_id: &str,
        tag_id: u32,
        tag_text: &str,
    ) -> bool {
        let key = Self::tag_cache_key(collection_id, tag_text);
        let is_new = !cache.forward.contains_key(&key);
        cache.forward.insert(key, tag_id);
        let id_key = Self::tag_id_cache_key(collection_id, tag_id);
        cache.reverse.insert(id_key, tag_text.to_string());
        if is_new {
            *cache
                .collection_counts
                .entry(collection_id.to_string())
                .or_insert(0) += 1;
        }
        is_new
    }

    fn load_tag_registry_cache(&self) -> Result<()> {
        let rows = self.query_rows::<TagRegistryRow>(
            "SELECT collection_id, tag_id, tag_text FROM tag_registry ORDER BY collection_id ASC, tag_id ASC;",
        )?;

        let mut cache = TagRegistryCache {
            forward: HashMap::with_capacity(rows.len()),
            reverse: HashMap::with_capacity(rows.len()),
            collection_counts: HashMap::new(),
        };
        let mut max_tag_id = 0u32;
        for row in rows {
            max_tag_id = max_tag_id.max(row.tag_id);
            Self::cache_insert_tag(&mut cache, &row.collection_id, row.tag_id, &row.tag_text);
        }
        *self
            .tag_registry_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = cache;
        self.next_tag_id
            .store(max_tag_id.saturating_add(1), AtomicOrdering::Relaxed);
        Ok(())
    }

    fn normalize_unique_tags(tags: &[String]) -> Vec<String> {
        let mut out = tags
            .iter()
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>();
        out.sort();
        out.dedup();
        out
    }

    fn resolve_tag_ids_for_batch(
        &self,
        collection_id: &str,
        unique_tags: &[String],
    ) -> Result<(HashMap<String, u32>, Vec<(u32, String)>)> {
        let mut resolved = HashMap::with_capacity(unique_tags.len());
        let unique_tags = Self::normalize_unique_tags(unique_tags);
        let mut missing = Vec::new();

        {
            let cache = self
                .tag_registry_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for tag in &unique_tags {
                let key = Self::tag_cache_key(collection_id, tag);
                if let Some(id) = cache.forward.get(&key).copied() {
                    resolved.insert(tag.clone(), id);
                } else {
                    missing.push(tag.clone());
                }
            }
        }

        if missing.is_empty() {
            return Ok((resolved, Vec::new()));
        }

        let mut new_rows = Vec::new();
        let mut cache = self
            .tag_registry_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut still_missing = Vec::new();
        for tag in &missing {
            let key = Self::tag_cache_key(collection_id, tag);
            if let Some(id) = cache.forward.get(&key).copied() {
                resolved.insert(tag.clone(), id);
            } else {
                still_missing.push(tag.clone());
            }
        }

        let current_collection_count = cache
            .collection_counts
            .get(collection_id)
            .copied()
            .unwrap_or(0);
        if current_collection_count.saturating_add(still_missing.len())
            > self.options.max_tag_registry_size
        {
            return Err(anyhow!(
                "tag_registry_overflow: limit={} collection_id={} current={} incoming_new={}",
                self.options.max_tag_registry_size,
                collection_id,
                current_collection_count,
                still_missing.len()
            ));
        }

        for tag in still_missing {
            let allocated = self.next_tag_id.fetch_add(1, AtomicOrdering::Relaxed);
            Self::cache_insert_tag(&mut cache, collection_id, allocated, &tag);
            resolved.insert(tag.clone(), allocated);
            new_rows.push((allocated, tag));
        }

        Ok((resolved, new_rows))
    }

    fn commit_tag_rows_to_cache(&self, collection_id: &str, new_rows: &[(u32, String)]) {
        if new_rows.is_empty() {
            return;
        }

        let mut max_seen = 0u32;
        let mut cache = self
            .tag_registry_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (tag_id, tag_text) in new_rows {
            max_seen = max_seen.max(*tag_id);
            Self::cache_insert_tag(&mut cache, collection_id, *tag_id, tag_text);
        }
        drop(cache);

        if max_seen > 0 {
            let desired = max_seen.saturating_add(1);
            loop {
                let current = self.next_tag_id.load(AtomicOrdering::Relaxed);
                if current >= desired {
                    break;
                }
                if self
                    .next_tag_id
                    .compare_exchange(
                        current,
                        desired,
                        AtomicOrdering::Relaxed,
                        AtomicOrdering::Relaxed,
                    )
                    .is_ok()
                {
                    break;
                }
            }
        }
    }

    fn resolve_tag_ids_readonly_internal(
        &self,
        collection_id: &str,
        tags: &[String],
    ) -> Result<Vec<u32>> {
        let normalized = Self::normalize_unique_tags(tags);
        if normalized.is_empty() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::with_capacity(normalized.len());
        let mut missing = Vec::new();
        {
            let cache = self
                .tag_registry_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for tag in &normalized {
                let key = Self::tag_cache_key(collection_id, tag);
                if let Some(id) = cache.forward.get(&key).copied() {
                    ids.push(id);
                } else {
                    missing.push(tag.clone());
                }
            }
        }
        if missing.is_empty() {
            ids.sort_unstable();
            ids.dedup();
            return Ok(ids);
        }

        let in_clause = missing
            .iter()
            .map(|t| format!("'{}'", sql_quote(t)))
            .collect::<Vec<_>>()
            .join(",");
        let rows = self.query_rows::<TagDictionaryRow>(&format!(
            "SELECT tag_id AS id, tag_text AS tag
             FROM tag_registry
             WHERE collection_id='{}' AND tag_text IN ({})
             ORDER BY tag_id ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;
        if !rows.is_empty() {
            let mut cache = self
                .tag_registry_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for row in rows {
                Self::cache_insert_tag(&mut cache, collection_id, row.id, &row.tag);
                ids.push(row.id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    fn resolve_tag_texts_internal(
        &self,
        collection_id: &str,
        ids: &[u32],
    ) -> Result<HashMap<u32, String>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut out = HashMap::with_capacity(ids.len());
        let mut missing = Vec::new();
        {
            let cache = self
                .tag_registry_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for id in ids {
                let key = Self::tag_id_cache_key(collection_id, *id);
                if let Some(tag) = cache.reverse.get(&key) {
                    out.insert(*id, tag.clone());
                } else {
                    missing.push(*id);
                }
            }
        }

        if !missing.is_empty() {
            let in_clause = missing
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let rows = self.query_rows::<TagDictionaryRow>(&format!(
                "SELECT tag_id AS id, tag_text AS tag
                 FROM tag_registry
                 WHERE collection_id='{}' AND tag_id IN ({})
                 ORDER BY tag_id ASC;",
                sql_quote(collection_id),
                in_clause
            ))?;
            if !rows.is_empty() {
                let mut cache = self
                    .tag_registry_cache
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for row in rows {
                    Self::cache_insert_tag(&mut cache, collection_id, row.id, &row.tag);
                    out.insert(row.id, row.tag);
                }
            }
        }

        Ok(out)
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
            "CREATE TABLE IF NOT EXISTS wal_entries (lsn INTEGER PRIMARY KEY AUTOINCREMENT, collection_id TEXT NOT NULL, vector_id INTEGER NOT NULL, vector_json TEXT, metadata_json TEXT, schema_version INTEGER NOT NULL DEFAULT 2, vector_blob BLOB, metadata_blob BLOB, idempotency_key TEXT, checkpointed_at INTEGER, created_at INTEGER NOT NULL)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_idempotency ON wal_entries(collection_id, idempotency_key) WHERE idempotency_key IS NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_vector_id ON wal_entries(collection_id, vector_id)",
            "CREATE INDEX IF NOT EXISTS idx_wal_pending_by_collection_lsn ON wal_entries(collection_id, checkpointed_at, lsn)",
            "CREATE TABLE IF NOT EXISTS vector_metadata (collection_id TEXT NOT NULL, vector_id INTEGER NOT NULL, schema_version INTEGER NOT NULL DEFAULT 3, entity_id INTEGER NOT NULL DEFAULT 0, sequence_ts INTEGER NOT NULL DEFAULT 0, payload_blob BLOB NOT NULL DEFAULT X'', tags_blob BLOB NOT NULL DEFAULT X'', created_at INTEGER NOT NULL, PRIMARY KEY(collection_id, vector_id))",
            "CREATE INDEX IF NOT EXISTS idx_vector_metadata_collection_vector ON vector_metadata(collection_id, vector_id)",
            "CREATE TABLE IF NOT EXISTS vector_id_counters (collection_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS tag_ids (collection_id TEXT NOT NULL, id INTEGER NOT NULL, tag TEXT NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY(collection_id, id), UNIQUE(collection_id, tag))",
            "CREATE INDEX IF NOT EXISTS idx_tag_ids_collection_tag ON tag_ids(collection_id, tag)",
            "CREATE TABLE IF NOT EXISTS tag_registry (collection_id TEXT NOT NULL, tag_id INTEGER NOT NULL, tag_text TEXT NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY(collection_id, tag_id), UNIQUE(collection_id, tag_text))",
            "CREATE INDEX IF NOT EXISTS idx_tag_registry_lookup ON tag_registry(collection_id, tag_text)",
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
        self.ensure_column(
            "vector_metadata",
            "schema_version",
            "INTEGER NOT NULL DEFAULT 3",
        )?;
        self.ensure_column("vector_metadata", "entity_id", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column(
            "vector_metadata",
            "sequence_ts",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "vector_metadata",
            "payload_blob",
            "BLOB NOT NULL DEFAULT X''",
        )?;
        self.ensure_column("vector_metadata", "tags_blob", "BLOB NOT NULL DEFAULT X''")?;
        self.ensure_column(
            "wal_entries",
            "schema_version",
            "INTEGER NOT NULL DEFAULT 2",
        )?;
        self.ensure_column("wal_entries", "vector_blob", "BLOB")?;
        self.ensure_column("wal_entries", "metadata_blob", "BLOB")?;
        self.migrate_wal_entries_v3_layout()?;
        self.ensure_column("orphan_files", "size_bytes", "INTEGER NOT NULL DEFAULT 0")?;
        self.ensure_column("orphan_files", "deleted_reason", "TEXT")?;
        self.ensure_column("api_keys", "hash_version", "INTEGER NOT NULL DEFAULT 1")?;
        self.exec(
            "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
             SELECT collection_id, COALESCE(MAX(vector_id) + 1, 0)
             FROM vector_metadata
             GROUP BY collection_id;",
        )?;
        self.exec(
            "INSERT OR IGNORE INTO tag_registry(collection_id, tag_id, tag_text, created_at)
             SELECT collection_id, id, tag, COALESCE(created_at, 0)
             FROM tag_ids;",
        )?;
        self.backfill_vector_tags_if_needed()?;
        self.backfill_vector_metadata_tags_blob_if_needed()?;
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
        let mut has_entity_id = false;
        let mut has_sequence_ts = false;
        let mut has_payload_blob = false;
        let mut has_tags_blob = false;
        for row in &rows {
            let name = row.get("name").and_then(Value::as_str).unwrap_or_default();
            let pk = row.get("pk").and_then(Value::as_i64).unwrap_or_default();
            if pk > 0 && name == "collection_id" {
                has_collection_pk = true;
            }
            if pk > 0 && name == "vector_id" {
                has_vector_pk = true;
            }
            if name == "entity_id" {
                has_entity_id = true;
            } else if name == "sequence_ts" {
                has_sequence_ts = true;
            } else if name == "payload_blob" {
                has_payload_blob = true;
            } else if name == "tags_blob" {
                has_tags_blob = true;
            }
        }
        if has_collection_pk
            && has_vector_pk
            && has_entity_id
            && has_sequence_ts
            && has_payload_blob
            && has_tags_blob
        {
            return Ok(());
        }

        let entity_expr = if has_entity_id { "entity_id" } else { "0" };
        let sequence_expr = if has_sequence_ts {
            "sequence_ts"
        } else if rows.iter().any(|r| {
            r.get("name")
                .and_then(Value::as_str)
                .map(|n| n == "start_time_ms")
                .unwrap_or(false)
        }) {
            "start_time_ms"
        } else {
            "0"
        };
        let payload_expr = if has_payload_blob {
            "payload_blob"
        } else {
            "X''"
        };
        let tags_expr = if has_tags_blob { "tags_blob" } else { "X''" };

        self.exec(
            &format!(
                "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS vector_metadata_v3 (
               collection_id TEXT NOT NULL,
               vector_id INTEGER NOT NULL,
               schema_version INTEGER NOT NULL DEFAULT 3,
               entity_id INTEGER NOT NULL DEFAULT 0,
               sequence_ts INTEGER NOT NULL DEFAULT 0,
               payload_blob BLOB NOT NULL DEFAULT X'',
               tags_blob BLOB NOT NULL DEFAULT X'',
               created_at INTEGER NOT NULL,
               PRIMARY KEY(collection_id, vector_id)
             );
             INSERT OR REPLACE INTO vector_metadata_v3(
               collection_id, vector_id, schema_version, entity_id, sequence_ts, payload_blob, tags_blob, created_at
             )
             SELECT collection_id, vector_id, 3, {entity_expr}, {sequence_expr}, {payload_expr}, {tags_expr}, created_at
             FROM vector_metadata;
             DROP TABLE vector_metadata;
             ALTER TABLE vector_metadata_v3 RENAME TO vector_metadata;
             COMMIT;"
            ),
        )
    }

    fn migrate_wal_entries_v3_layout(&self) -> Result<()> {
        let rows = self.query_json("PRAGMA table_info(wal_entries);")?;
        if rows.is_empty() {
            return Ok(());
        }

        let mut vector_json_notnull = false;
        let mut metadata_json_notnull = false;
        for row in &rows {
            let name = row.get("name").and_then(Value::as_str).unwrap_or_default();
            let notnull = row
                .get("notnull")
                .and_then(Value::as_i64)
                .unwrap_or_default()
                != 0;
            if name == "vector_json" {
                vector_json_notnull = notnull;
            } else if name == "metadata_json" {
                metadata_json_notnull = notnull;
            }
        }
        if !vector_json_notnull && !metadata_json_notnull {
            return Ok(());
        }

        self.exec(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS wal_entries_v3 (
               lsn INTEGER PRIMARY KEY AUTOINCREMENT,
               collection_id TEXT NOT NULL,
               vector_id INTEGER NOT NULL,
               vector_json TEXT,
               metadata_json TEXT,
               schema_version INTEGER NOT NULL DEFAULT 2,
               vector_blob BLOB,
               metadata_blob BLOB,
               idempotency_key TEXT,
               checkpointed_at INTEGER,
               created_at INTEGER NOT NULL
             );
             INSERT INTO wal_entries_v3(
               lsn, collection_id, vector_id, vector_json, metadata_json,
               schema_version, vector_blob, metadata_blob, idempotency_key, checkpointed_at, created_at
             )
             SELECT
               lsn, collection_id, vector_id, vector_json, metadata_json,
               COALESCE(schema_version, 2), vector_blob, metadata_blob, idempotency_key, checkpointed_at, created_at
             FROM wal_entries;
             DROP TABLE wal_entries;
             ALTER TABLE wal_entries_v3 RENAME TO wal_entries;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_idempotency
               ON wal_entries(collection_id, idempotency_key)
               WHERE idempotency_key IS NOT NULL;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_vector_id
               ON wal_entries(collection_id, vector_id);
             CREATE INDEX IF NOT EXISTS idx_wal_pending_by_collection_lsn
               ON wal_entries(collection_id, checkpointed_at, lsn);
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

    fn backfill_vector_metadata_tags_blob_if_needed(&self) -> Result<()> {
        let rows = self.query_rows::<VectorTagBackfillRow>(
            "SELECT collection_id, vector_id, tag_id
             FROM vector_tags
             ORDER BY collection_id ASC, vector_id ASC, tag_id ASC;",
        )?;
        if rows.is_empty() {
            return Ok(());
        }

        let guard = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.exec("BEGIN IMMEDIATE;")?;

        let tx_result = (|| -> Result<()> {
            let mut update_stmt = PreparedStmt::new(
                &guard,
                "UPDATE vector_metadata
                 SET tags_blob = ?1
                 WHERE collection_id = ?2
                   AND vector_id = ?3
                   AND (tags_blob IS NULL OR length(tags_blob) = 0);",
            )?;

            let mut i = 0usize;
            while i < rows.len() {
                let collection_id = rows[i].collection_id.clone();
                let vector_id = rows[i].vector_id;
                let mut tag_ids = Vec::new();
                while i < rows.len()
                    && rows[i].collection_id == collection_id
                    && rows[i].vector_id == vector_id
                {
                    tag_ids.push(rows[i].tag_id);
                    i += 1;
                }

                let tags_blob = encode_tags_blob(&tag_ids);
                update_stmt.reset()?;
                update_stmt.bind_blob(1, &tags_blob)?;
                update_stmt.bind_text(2, &collection_id)?;
                update_stmt.bind_i64(3, vector_id as i64)?;
                update_stmt.step_done()?;
            }
            Ok(())
        })();

        match tx_result {
            Ok(()) => guard.exec("COMMIT;"),
            Err(err) => {
                let _ = guard.exec("ROLLBACK;");
                Err(err)
            }
        }
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
        metadata: &IngestMetadataV3Input,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
        self.ingest_wal_explicit(collection_id, vector_id, vector, metadata, idempotency_key)
    }

    fn ingest_wal_atomic(
        &self,
        collection_id: &str,
        vector: &[f32],
        metadata: &IngestMetadataV3Input,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
        let rows = self.ingest_wal_batch(
            collection_id,
            &[(
                vector.to_vec(),
                metadata.clone(),
                idempotency_key.map(|v| v.to_string()),
            )],
        )?;
        rows.into_iter()
            .next()
            .ok_or_else(|| anyhow!("ingest transaction produced no result row"))
    }

    fn pending_wal(&self, collection_id: &str, limit: usize) -> Result<Vec<WalEntry>> {
        self.query_wal(&format!(
            "SELECT lsn, vector_id, schema_version, vector_blob, metadata_blob, vector_json, metadata_json
             FROM wal_entries
             WHERE collection_id='{}' AND checkpointed_at IS NULL
             ORDER BY lsn ASC LIMIT {};",
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
            "SELECT lsn, vector_id, schema_version, vector_blob, metadata_blob, vector_json, metadata_json
             FROM wal_entries
             WHERE collection_id='{}' AND checkpointed_at IS NULL AND lsn > {}
             ORDER BY lsn ASC;",
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
        metadata: &VectorMetadataV3,
    ) -> Result<()> {
        let now = now_unix_ts();
        let mut sorted_tags = metadata.tags.clone();
        sorted_tags.sort_unstable();
        sorted_tags.dedup();
        let tags_blob = encode_tags_blob(&sorted_tags);

        let guard = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.exec("BEGIN IMMEDIATE;")?;

        let tx_result = (|| -> Result<()> {
            let mut ensure_counter_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
                 SELECT ?1, COALESCE(MAX(vector_id) + 1, 0)
                 FROM vector_metadata
                 WHERE collection_id = ?2;",
            )?;
            ensure_counter_stmt.bind_text(1, collection_id)?;
            ensure_counter_stmt.bind_text(2, collection_id)?;
            ensure_counter_stmt.step_done()?;

            let mut update_counter_stmt = PreparedStmt::new(
                &guard,
                "UPDATE vector_id_counters
                 SET next_id = CASE WHEN next_id <= ?1 THEN ?2 ELSE next_id END
                 WHERE collection_id = ?3;",
            )?;
            update_counter_stmt.bind_i64(1, vector_id as i64)?;
            update_counter_stmt.bind_i64(2, vector_id.saturating_add(1) as i64)?;
            update_counter_stmt.bind_text(3, collection_id)?;
            update_counter_stmt.step_done()?;

            let mut metadata_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR REPLACE INTO vector_metadata(
                   collection_id, vector_id, schema_version, entity_id, sequence_ts, payload_blob, tags_blob, created_at
                 ) VALUES (?1, ?2, 3, ?3, ?4, ?5, ?6, ?7);",
            )?;
            metadata_insert_stmt.bind_text(1, collection_id)?;
            metadata_insert_stmt.bind_i64(2, vector_id as i64)?;
            metadata_insert_stmt.bind_i64(3, u64_to_i64_bits(metadata.entity_id))?;
            metadata_insert_stmt.bind_i64(4, metadata.sequence_ts.min(i64::MAX as u64) as i64)?;
            metadata_insert_stmt.bind_blob(5, &metadata.payload)?;
            metadata_insert_stmt.bind_blob(6, &tags_blob)?;
            metadata_insert_stmt.bind_i64(7, now)?;
            metadata_insert_stmt.step_done()?;

            let mut clear_tags_stmt = PreparedStmt::new(
                &guard,
                "DELETE FROM vector_tags WHERE collection_id = ?1 AND vector_id = ?2;",
            )?;
            clear_tags_stmt.bind_text(1, collection_id)?;
            clear_tags_stmt.bind_i64(2, vector_id as i64)?;
            clear_tags_stmt.step_done()?;

            let mut vector_tags_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO vector_tags(collection_id, vector_id, tag_id, created_at)
                 VALUES (?1, ?2, ?3, ?4);",
            )?;
            for tag_id in &sorted_tags {
                vector_tags_insert_stmt.reset()?;
                vector_tags_insert_stmt.bind_text(1, collection_id)?;
                vector_tags_insert_stmt.bind_i64(2, vector_id as i64)?;
                vector_tags_insert_stmt.bind_i64(3, *tag_id as i64)?;
                vector_tags_insert_stmt.bind_i64(4, now)?;
                vector_tags_insert_stmt.step_done()?;
            }

            Ok(())
        })();

        match tx_result {
            Ok(()) => guard.exec("COMMIT;"),
            Err(err) => {
                let _ = guard.exec("ROLLBACK;");
                Err(err)
            }
        }
    }

    fn fetch_metadata(
        &self,
        collection_id: &str,
        ids: &[usize],
    ) -> Result<HashMap<usize, VectorMetadataV3>> {
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
            "SELECT vm.vector_id, vm.schema_version, vm.entity_id, vm.sequence_ts, vm.payload_blob, vm.tags_blob
             FROM vector_metadata vm
             WHERE vm.collection_id='{}' AND vm.vector_id IN ({})
             ORDER BY vm.vector_id ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;

        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            out.insert(row.vector_id, row.metadata);
        }
        self.merge_legacy_tags_for_ids(collection_id, &unique_ids, &mut out)?;
        Ok(out)
    }

    fn fetch_all_metadata(&self, collection_id: &str) -> Result<Vec<(usize, VectorMetadataV3)>> {
        let rows = self.query_rows::<MetadataRow>(&format!(
            "SELECT vm.vector_id, vm.schema_version, vm.entity_id, vm.sequence_ts, vm.payload_blob, vm.tags_blob
             FROM vector_metadata vm
             WHERE vm.collection_id='{}'
             ORDER BY vm.vector_id ASC;",
            sql_quote(collection_id)
        ))?;

        let mut map = HashMap::with_capacity(rows.len());
        for row in rows {
            map.insert(row.vector_id, row.metadata);
        }
        self.merge_legacy_tags_for_collection(collection_id, &mut map)?;
        let mut out = map.into_iter().collect::<Vec<_>>();
        out.sort_by_key(|(id, _)| *id);
        Ok(out)
    }

    fn fetch_tag_dictionary(&self, collection_id: &str) -> Result<HashMap<String, u32>> {
        let rows = self.query_rows::<TagDictionaryRow>(&format!(
            "SELECT tag_id AS id, tag_text AS tag
             FROM tag_registry
             WHERE collection_id='{}'
             ORDER BY tag_id ASC;",
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
            "SELECT vm.vector_id AS vector_id, COALESCE(GROUP_CONCAT(vt.tag_id), '') AS tag_ids
             FROM vector_metadata vm
             LEFT JOIN vector_tags vt
               ON vt.collection_id = vm.collection_id
              AND vt.vector_id = vm.vector_id
             WHERE vm.collection_id='{}'
             GROUP BY vm.vector_id
             ORDER BY vm.vector_id ASC;",
            sql_quote(collection_id)
        ))
    }

    fn resolve_tag_ids_readonly(&self, collection_id: &str, tags: &[String]) -> Result<Vec<u32>> {
        self.resolve_tag_ids_readonly_internal(collection_id, tags)
    }

    fn resolve_tag_texts(&self, collection_id: &str, ids: &[u32]) -> Result<HashMap<u32, String>> {
        self.resolve_tag_texts_internal(collection_id, ids)
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
    ) -> Result<HashMap<usize, VectorMetadataV3>> {
        if start_id > end_id {
            return Ok(HashMap::new());
        }
        let rows = self.query_rows::<MetadataRow>(&format!(
            "SELECT vm.vector_id, vm.schema_version, vm.entity_id, vm.sequence_ts, vm.payload_blob, vm.tags_blob
             FROM vector_metadata vm
             WHERE vm.collection_id='{}' AND vm.vector_id >= {} AND vm.vector_id <= {}
             ORDER BY vm.vector_id ASC;",
            sql_quote(collection_id),
            start_id,
            end_id
        ))?;

        let mut out = HashMap::with_capacity(rows.len());
        for row in rows {
            out.insert(row.vector_id, row.metadata);
        }
        self.merge_legacy_tags_for_range(collection_id, start_id, end_id, &mut out)?;
        Ok(out)
    }

    fn apply_legacy_tag_rows(
        out: &mut HashMap<usize, VectorMetadataV3>,
        rows: &[VectorTagPairRow],
    ) {
        if rows.is_empty() {
            return;
        }
        let mut touched_ids = HashSet::new();
        for row in rows {
            if let Some(meta) = out.get_mut(&row.vector_id) {
                meta.tags.push(row.tag_id);
                touched_ids.insert(row.vector_id);
            }
        }
        for vector_id in touched_ids {
            if let Some(meta) = out.get_mut(&vector_id) {
                meta.tags.sort_unstable();
                meta.tags.dedup();
            }
        }
    }

    fn merge_legacy_tags_for_ids(
        &self,
        collection_id: &str,
        ids: &[usize],
        out: &mut HashMap<usize, VectorMetadataV3>,
    ) -> Result<()> {
        let missing = ids
            .iter()
            .copied()
            .filter(|id| out.get(id).map(|m| m.tags.is_empty()).unwrap_or(false))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        let in_clause = missing
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let rows = self.query_rows::<VectorTagPairRow>(&format!(
            "SELECT vector_id, tag_id
             FROM vector_tags
             WHERE collection_id='{}' AND vector_id IN ({})
             ORDER BY vector_id ASC, tag_id ASC;",
            sql_quote(collection_id),
            in_clause
        ))?;
        Self::apply_legacy_tag_rows(out, &rows);
        Ok(())
    }

    fn merge_legacy_tags_for_range(
        &self,
        collection_id: &str,
        start_id: usize,
        end_id: usize,
        out: &mut HashMap<usize, VectorMetadataV3>,
    ) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let rows = self.query_rows::<VectorTagPairRow>(&format!(
            "SELECT vector_id, tag_id
             FROM vector_tags
             WHERE collection_id='{}' AND vector_id >= {} AND vector_id <= {}
             ORDER BY vector_id ASC, tag_id ASC;",
            sql_quote(collection_id),
            start_id,
            end_id
        ))?;
        Self::apply_legacy_tag_rows(out, &rows);
        Ok(())
    }

    fn merge_legacy_tags_for_collection(
        &self,
        collection_id: &str,
        out: &mut HashMap<usize, VectorMetadataV3>,
    ) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let rows = self.query_rows::<VectorTagPairRow>(&format!(
            "SELECT vector_id, tag_id
             FROM vector_tags
             WHERE collection_id='{}'
             ORDER BY vector_id ASC, tag_id ASC;",
            sql_quote(collection_id)
        ))?;
        Self::apply_legacy_tag_rows(out, &rows);
        Ok(())
    }

    pub fn audit_events_batch(&self, events: &[AuditEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let now = now_unix_ts();
        let guard = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.exec("BEGIN IMMEDIATE;")?;

        let tx_result = (|| -> Result<()> {
            let mut insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT INTO audit_events(
                   ts,
                   request_id,
                   api_key_id,
                   endpoint,
                   action,
                   status_code,
                   latency_ms,
                   client_ip,
                   details_json
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8);",
            )?;

            for event in events {
                insert_stmt.reset()?;
                insert_stmt.bind_i64(1, now)?;
                insert_stmt.bind_text(2, &event.request_id)?;
                insert_stmt.bind_optional_text(3, event.api_key_id.as_deref())?;
                insert_stmt.bind_text(4, &event.endpoint)?;
                insert_stmt.bind_text(5, &event.action)?;
                insert_stmt.bind_i64(6, event.status_code as i64)?;
                insert_stmt.bind_f64(7, event.latency_ms)?;
                insert_stmt.bind_text(8, &event.details.to_string())?;
                insert_stmt.step_done()?;
            }
            Ok(())
        })();

        match tx_result {
            Ok(()) => guard.exec("COMMIT;"),
            Err(err) => {
                let _ = guard.exec("ROLLBACK;");
                Err(err)
            }
        }
    }
}

#[derive(Debug)]
struct MetadataRow {
    vector_id: usize,
    metadata: VectorMetadataV3,
}

#[derive(Debug)]
struct TagDictionaryRow {
    id: u32,
    tag: String,
}

#[derive(Debug)]
struct TagRegistryRow {
    collection_id: String,
    tag_id: u32,
    tag_text: String,
}

#[derive(Debug)]
struct VectorTagBackfillRow {
    collection_id: String,
    vector_id: usize,
    tag_id: u32,
}

#[derive(Debug)]
struct VectorTagPairRow {
    vector_id: usize,
    tag_id: u32,
}

impl FromSqlRow for WalEntry {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        let lsn = column_i64(stmt, 0).max(0) as u64;
        let vector_id = column_i64(stmt, 1).max(0) as usize;
        let schema_version = column_i64(stmt, 2).max(0) as u32;
        let vector_blob = column_blob_bytes(stmt, 3);
        let metadata_blob = column_blob_bytes(stmt, 4);
        let vector_json = column_text_bytes(stmt, 5);
        let metadata_json = column_text_bytes(stmt, 6);

        let (vector, metadata) = if schema_version >= 3 && !vector_blob.is_empty() {
            (
                decode_vector_blob(vector_blob)?,
                decode_metadata_blob(metadata_blob)?,
            )
        } else {
            let legacy: VectorMetadata = serde_json::from_slice(metadata_json)?;
            (
                serde_json::from_slice(vector_json)?,
                legacy_metadata_to_v3(&legacy),
            )
        };
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
            metadata: None,
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
        let schema_version = column_i64(stmt, 1).max(0) as u32;
        let entity_id = i64_to_u64_bits(column_i64(stmt, 2));
        let sequence_ts = column_i64(stmt, 3).max(0) as u64;
        let payload_blob = column_blob_bytes(stmt, 4).to_vec();
        let tags_blob = column_blob_bytes(stmt, 5);
        let tag_ids = decode_tags_blob(tags_blob)?;
        let metadata = if schema_version >= 3 {
            VectorMetadataV3 {
                entity_id,
                sequence_ts,
                tags: tag_ids,
                payload: payload_blob,
            }
        } else {
            VectorMetadataV3 {
                entity_id,
                sequence_ts,
                tags: tag_ids,
                payload: payload_blob,
            }
        };
        Ok(Self {
            vector_id,
            metadata,
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

impl FromSqlRow for TagRegistryRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        Ok(Self {
            collection_id: column_text_string(stmt, 0),
            tag_id: column_i64(stmt, 1).max(0) as u32,
            tag_text: column_text_string(stmt, 2),
        })
    }
}

impl FromSqlRow for FilterRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        let vector_id = column_i64(stmt, 0).max(0) as usize;
        let tag_ids = column_text_string(stmt, 1)
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
        Ok(Self { vector_id, tag_ids })
    }
}

impl FromSqlRow for VectorTagBackfillRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        Ok(Self {
            collection_id: column_text_string(stmt, 0),
            vector_id: column_i64(stmt, 1).max(0) as usize,
            tag_id: column_i64(stmt, 2).max(0) as u32,
        })
    }
}

impl FromSqlRow for VectorTagPairRow {
    unsafe fn from_row(stmt: *mut Sqlite3Stmt) -> Result<Self> {
        Ok(Self {
            vector_id: column_i64(stmt, 0).max(0) as usize,
            tag_id: column_i64(stmt, 1).max(0) as u32,
        })
    }
}

impl SqliteCatalog {
    fn query_wal(&self, sql: &str) -> Result<Vec<WalEntry>> {
        self.query_rows::<WalEntry>(sql)
    }

    fn lookup_existing_idempotency_keys(
        &self,
        conn: &RawSqliteConnection,
        collection_id: &str,
        keys: &[&str],
    ) -> Result<HashMap<String, usize>> {
        if keys.is_empty() {
            return Ok(HashMap::new());
        }

        let mut out = HashMap::with_capacity(keys.len());
        // Keep parameter count below SQLite bind limits with safe chunking.
        const LOOKUP_CHUNK_SIZE: usize = 500;
        for chunk in keys.chunks(LOOKUP_CHUNK_SIZE) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let sql = format!(
                "SELECT idempotency_key, vector_id
                 FROM wal_entries
                 WHERE collection_id = ?1
                   AND idempotency_key IN ({placeholders});"
            );
            let mut lookup_stmt = PreparedStmt::new(conn, &sql)?;
            lookup_stmt.bind_text(1, collection_id)?;
            for (i, key) in chunk.iter().enumerate() {
                lookup_stmt.bind_text((i + 2) as c_int, key)?;
            }
            while lookup_stmt.step_row()? {
                out.insert(
                    lookup_stmt.column_text(0),
                    lookup_stmt.column_i64(1).max(0) as usize,
                );
            }
        }

        Ok(out)
    }

    fn ingest_wal_explicit(
        &self,
        collection_id: &str,
        vector_id: usize,
        vector: &[f32],
        metadata: &IngestMetadataV3Input,
        idempotency_key: Option<&str>,
    ) -> Result<WalIngestResult> {
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

        let normalized_tags = Self::normalize_unique_tags(&metadata.tags);
        let vector_blob = encode_vector_blob(vector);
        let normalized_key = normalize_idempotency_key(idempotency_key);

        let guard = self
            .writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.exec("BEGIN IMMEDIATE;")?;

        let tx_result = (|| -> Result<(WalIngestResult, Vec<(u32, String)>)> {
            let now = now_unix_ts();

            let mut ensure_counter_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
                 SELECT ?1, COALESCE(MAX(vector_id) + 1, 0)
                 FROM vector_metadata
                 WHERE collection_id = ?2;",
            )?;
            ensure_counter_stmt.bind_text(1, collection_id)?;
            ensure_counter_stmt.bind_text(2, collection_id)?;
            ensure_counter_stmt.step_done()?;

            if let Some(key) = normalized_key.as_deref() {
                let mut lookup_stmt = PreparedStmt::new(
                    &guard,
                    "SELECT vector_id
                     FROM wal_entries
                     WHERE collection_id = ?1
                       AND idempotency_key = ?2
                     LIMIT 1;",
                )?;
                lookup_stmt.bind_text(1, collection_id)?;
                lookup_stmt.bind_text(2, key)?;
                if lookup_stmt.step_row()? {
                    return Ok((
                        WalIngestResult {
                            vector_id: lookup_stmt.column_i64(0).max(0) as usize,
                            created: false,
                            lsn: None,
                            metadata: None,
                        },
                        Vec::new(),
                    ));
                }
            }

            let mut update_counter_stmt = PreparedStmt::new(
                &guard,
                "UPDATE vector_id_counters
                 SET next_id = CASE WHEN next_id <= ?1 THEN ?2 ELSE next_id END
                 WHERE collection_id = ?3;",
            )?;
            update_counter_stmt.bind_i64(1, vector_id as i64)?;
            update_counter_stmt.bind_i64(2, vector_id.saturating_add(1) as i64)?;
            update_counter_stmt.bind_text(3, collection_id)?;
            update_counter_stmt.step_done()?;

            let (tag_ids, new_tag_rows) =
                self.resolve_tag_ids_for_batch(collection_id, &normalized_tags)?;
            let mut ordered_tag_ids = Vec::with_capacity(normalized_tags.len());
            for tag in &normalized_tags {
                if let Some(id) = tag_ids.get(tag) {
                    ordered_tag_ids.push(*id);
                }
            }
            ordered_tag_ids.sort_unstable();
            ordered_tag_ids.dedup();

            let metadata_v3 = VectorMetadataV3 {
                entity_id: metadata.entity_id,
                sequence_ts: metadata.sequence_ts,
                tags: ordered_tag_ids.clone(),
                payload: metadata.payload.clone(),
            };
            let metadata_blob = encode_metadata_blob(&metadata_v3)?;
            let tags_blob = encode_tags_blob(&ordered_tag_ids);

            let mut wal_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT INTO wal_entries(
                   collection_id,
                   vector_id,
                   schema_version,
                   vector_blob,
                   metadata_blob,
                   vector_json,
                   metadata_json,
                   idempotency_key,
                   checkpointed_at,
                   created_at
                 ) VALUES (?1, ?2, 3, ?3, ?4, NULL, NULL, ?5, NULL, ?6)
                 RETURNING lsn;",
            )?;
            wal_insert_stmt.bind_text(1, collection_id)?;
            wal_insert_stmt.bind_i64(2, vector_id as i64)?;
            wal_insert_stmt.bind_blob(3, &vector_blob)?;
            wal_insert_stmt.bind_blob(4, &metadata_blob)?;
            wal_insert_stmt.bind_optional_text(5, normalized_key.as_deref())?;
            wal_insert_stmt.bind_i64(6, now)?;
            let lsn = if wal_insert_stmt.step_row()? {
                wal_insert_stmt.column_i64(0).max(0) as u64
            } else {
                return Err(anyhow!("wal insert returned no row for explicit ingest"));
            };

            let mut metadata_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR REPLACE INTO vector_metadata(
                   collection_id,
                   vector_id,
                   schema_version,
                   entity_id,
                   sequence_ts,
                   payload_blob,
                   tags_blob,
                   created_at
                 ) VALUES (?1, ?2, 3, ?3, ?4, ?5, ?6, ?7);",
            )?;
            metadata_insert_stmt.bind_text(1, collection_id)?;
            metadata_insert_stmt.bind_i64(2, vector_id as i64)?;
            metadata_insert_stmt.bind_i64(3, u64_to_i64_bits(metadata.entity_id))?;
            metadata_insert_stmt.bind_i64(4, metadata.sequence_ts.min(i64::MAX as u64) as i64)?;
            metadata_insert_stmt.bind_blob(5, &metadata.payload)?;
            metadata_insert_stmt.bind_blob(6, &tags_blob)?;
            metadata_insert_stmt.bind_i64(7, now)?;
            metadata_insert_stmt.step_done()?;

            let mut tag_registry_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO tag_registry(collection_id, tag_id, tag_text, created_at)
                 VALUES (?1, ?2, ?3, ?4);",
            )?;
            let mut tag_ids_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO tag_ids(collection_id, id, tag, created_at)
                 VALUES (?1, ?2, ?3, ?4);",
            )?;
            for (tag_id, tag_text) in &new_tag_rows {
                tag_registry_insert_stmt.reset()?;
                tag_registry_insert_stmt.bind_text(1, collection_id)?;
                tag_registry_insert_stmt.bind_i64(2, *tag_id as i64)?;
                tag_registry_insert_stmt.bind_text(3, tag_text)?;
                tag_registry_insert_stmt.bind_i64(4, now)?;
                tag_registry_insert_stmt.step_done()?;

                tag_ids_insert_stmt.reset()?;
                tag_ids_insert_stmt.bind_text(1, collection_id)?;
                tag_ids_insert_stmt.bind_i64(2, *tag_id as i64)?;
                tag_ids_insert_stmt.bind_text(3, tag_text)?;
                tag_ids_insert_stmt.bind_i64(4, now)?;
                tag_ids_insert_stmt.step_done()?;
            }

            if !ordered_tag_ids.is_empty() {
                let mut vector_tags_insert_stmt = PreparedStmt::new(
                    &guard,
                    "INSERT OR IGNORE INTO vector_tags(collection_id, vector_id, tag_id, created_at)
                     VALUES (?1, ?2, ?3, ?4);",
                )?;
                for tag_id in &ordered_tag_ids {
                    vector_tags_insert_stmt.reset()?;
                    vector_tags_insert_stmt.bind_text(1, collection_id)?;
                    vector_tags_insert_stmt.bind_i64(2, vector_id as i64)?;
                    vector_tags_insert_stmt.bind_i64(3, *tag_id as i64)?;
                    vector_tags_insert_stmt.bind_i64(4, now)?;
                    vector_tags_insert_stmt.step_done()?;
                }
            }

            Ok((
                WalIngestResult {
                    vector_id,
                    created: true,
                    lsn: Some(lsn),
                    metadata: Some(metadata_v3),
                },
                new_tag_rows,
            ))
        })();

        match tx_result {
            Ok((result, staged_tag_rows)) => {
                guard.exec("COMMIT;")?;
                self.commit_tag_rows_to_cache(collection_id, &staged_tag_rows);
                Ok(result)
            }
            Err(err) => {
                let _ = guard.exec("ROLLBACK;");
                let _ = self.load_tag_registry_cache();
                Err(err)
            }
        }
    }

    /// Batch ingest: single BEGIN...COMMIT wrapping N row insertions.
    /// Holds the writer lock for the entire batch to amortize transaction cost.
    /// Preserves idempotency per-row.
    pub fn ingest_wal_batch(
        &self,
        collection_id: &str,
        entries: &[(Vec<f32>, IngestMetadataV3Input, Option<String>)],
    ) -> Result<Vec<WalIngestResult>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        #[derive(Debug)]
        struct PreparedBatchEntry {
            vector_blob: Vec<u8>,
            metadata: IngestMetadataV3Input,
            normalized_tags: Vec<String>,
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
                    Ok(PreparedBatchEntry {
                        vector_blob: encode_vector_blob(vector),
                        metadata: metadata.clone(),
                        normalized_tags: Self::normalize_unique_tags(&metadata.tags),
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

        let tx_result = (|| -> Result<(Vec<WalIngestResult>, Vec<(u32, String)>)> {
            const BATCH_CHUNK_ROWS: usize = 10_000;

            #[derive(Debug)]
            struct PendingInsert<'a> {
                vector_id: usize,
                entry: &'a PreparedBatchEntry,
                result_index: usize,
            }

            let mut results = Vec::with_capacity(prepared_entries.len());
            let mut staged_tag_rows = Vec::new();

            let mut ensure_counter_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO vector_id_counters(collection_id, next_id)
                 SELECT ?1, COALESCE(MAX(vector_id) + 1, 0)
                 FROM vector_metadata
                 WHERE collection_id = ?2;",
            )?;
            ensure_counter_stmt.bind_text(1, collection_id)?;
            ensure_counter_stmt.bind_text(2, collection_id)?;
            ensure_counter_stmt.step_done()?;

            let mut wal_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT INTO wal_entries(
                   collection_id,
                   vector_id,
                   schema_version,
                   vector_blob,
                   metadata_blob,
                   vector_json,
                   metadata_json,
                   idempotency_key,
                   checkpointed_at,
                   created_at
                 ) VALUES (?1, ?2, 3, ?3, ?4, NULL, NULL, ?5, NULL, ?6);",
            )?;
            let mut metadata_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR REPLACE INTO vector_metadata(
                   collection_id,
                   vector_id,
                   schema_version,
                   entity_id,
                   sequence_ts,
                   payload_blob,
                   tags_blob,
                   created_at
                 ) VALUES (?1, ?2, 3, ?3, ?4, ?5, ?6, ?7);",
            )?;
            let mut tag_registry_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO tag_registry(collection_id, tag_id, tag_text, created_at)
                 VALUES (?1, ?2, ?3, ?4);",
            )?;
            let mut tag_ids_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO tag_ids(collection_id, id, tag, created_at)
                 VALUES (?1, ?2, ?3, ?4);",
            )?;
            let mut vector_tags_insert_stmt = PreparedStmt::new(
                &guard,
                "INSERT OR IGNORE INTO vector_tags(collection_id, vector_id, tag_id, created_at)
                 VALUES (?1, ?2, ?3, ?4);",
            )?;
            let mut next_id_select_stmt = PreparedStmt::new(
                &guard,
                "SELECT next_id
                 FROM vector_id_counters
                 WHERE collection_id = ?1
                 LIMIT 1;",
            )?;
            let mut next_id_update_stmt = PreparedStmt::new(
                &guard,
                "UPDATE vector_id_counters
                 SET next_id = next_id + ?1
                 WHERE collection_id = ?2;",
            )?;

            for chunk in prepared_entries.chunks(BATCH_CHUNK_ROWS) {
                let now = now_unix_ts();
                let normalized_keys = chunk
                    .iter()
                    .map(|entry| entry.idempotency_key.as_deref())
                    .collect::<Vec<_>>();

                let mut unique_lookup_keys = Vec::new();
                let mut seen_lookup = HashSet::new();
                for key in normalized_keys.iter().flatten().copied() {
                    if seen_lookup.insert(key) {
                        unique_lookup_keys.push(key);
                    }
                }

                let existing_by_key = self.lookup_existing_idempotency_keys(
                    &guard,
                    collection_id,
                    &unique_lookup_keys,
                )?;

                let mut seen_new_keys = HashSet::new();
                let mut new_row_count = 0usize;
                for key in &normalized_keys {
                    match key {
                        Some(k) => {
                            if existing_by_key.contains_key(*k) || seen_new_keys.contains(k) {
                                continue;
                            }
                            seen_new_keys.insert(*k);
                            new_row_count += 1;
                        }
                        None => {
                            new_row_count += 1;
                        }
                    }
                }

                let mut next_allocated_id = 0usize;
                if new_row_count > 0 {
                    next_id_select_stmt.reset()?;
                    next_id_select_stmt.bind_text(1, collection_id)?;
                    if next_id_select_stmt.step_row()? {
                        next_allocated_id = next_id_select_stmt.column_i64(0).max(0) as usize;
                    } else {
                        return Err(anyhow!(
                            "missing vector_id_counters row for collection '{}'",
                            collection_id
                        ));
                    }
                    next_id_update_stmt.reset()?;
                    next_id_update_stmt.bind_i64(1, new_row_count as i64)?;
                    next_id_update_stmt.bind_text(2, collection_id)?;
                    next_id_update_stmt.step_done()?;
                }

                let mut pending = Vec::with_capacity(new_row_count);
                let mut assigned_new_by_key: HashMap<&str, usize> = HashMap::new();

                for (entry, key) in chunk.iter().zip(normalized_keys.iter()) {
                    let mut created = true;
                    let vector_id = if let Some(k) = key {
                        if let Some(existing_id) = existing_by_key.get(*k) {
                            created = false;
                            *existing_id
                        } else if let Some(existing_id) = assigned_new_by_key.get(k) {
                            created = false;
                            *existing_id
                        } else {
                            let allocated = next_allocated_id;
                            next_allocated_id += 1;
                            assigned_new_by_key.insert(*k, allocated);
                            allocated
                        }
                    } else {
                        let allocated = next_allocated_id;
                        next_allocated_id += 1;
                        allocated
                    };

                    let result_index = results.len();
                    results.push(WalIngestResult {
                        vector_id,
                        created,
                        lsn: None,
                        metadata: None,
                    });

                    if !created {
                        continue;
                    }

                    pending.push(PendingInsert {
                        vector_id,
                        entry,
                        result_index,
                    });
                }

                if pending.is_empty() {
                    continue;
                }

                let mut unique_tags = HashSet::new();
                for row in &pending {
                    for tag in &row.entry.normalized_tags {
                        unique_tags.insert(tag.clone());
                    }
                }
                let mut ordered_tags = unique_tags.into_iter().collect::<Vec<_>>();
                ordered_tags.sort();
                let (tag_ids, new_tag_rows) =
                    self.resolve_tag_ids_for_batch(collection_id, &ordered_tags)?;

                let mut vector_tag_pairs: Vec<(usize, u32)> = Vec::new();
                for row in &pending {
                    for tag in &row.entry.normalized_tags {
                        if let Some(tag_id) = tag_ids.get(tag) {
                            vector_tag_pairs.push((row.vector_id, *tag_id));
                        }
                    }
                }

                for row in &pending {
                    wal_insert_stmt.reset()?;
                    wal_insert_stmt.bind_text(1, collection_id)?;
                    wal_insert_stmt.bind_i64(2, row.vector_id as i64)?;
                    wal_insert_stmt.bind_blob(3, &row.entry.vector_blob)?;
                    let mut ordered_tag_ids = Vec::with_capacity(row.entry.normalized_tags.len());
                    for tag in &row.entry.normalized_tags {
                        if let Some(id) = tag_ids.get(tag) {
                            ordered_tag_ids.push(*id);
                        }
                    }
                    ordered_tag_ids.sort_unstable();
                    ordered_tag_ids.dedup();
                    let metadata_v3 = VectorMetadataV3 {
                        entity_id: row.entry.metadata.entity_id,
                        sequence_ts: row.entry.metadata.sequence_ts,
                        tags: ordered_tag_ids,
                        payload: row.entry.metadata.payload.clone(),
                    };
                    let metadata_blob = encode_metadata_blob(&metadata_v3)?;
                    wal_insert_stmt.bind_blob(4, &metadata_blob)?;
                    wal_insert_stmt.bind_optional_text(5, row.entry.idempotency_key.as_deref())?;
                    wal_insert_stmt.bind_i64(6, now)?;
                    wal_insert_stmt.step_done()?;

                    metadata_insert_stmt.reset()?;
                    metadata_insert_stmt.bind_text(1, collection_id)?;
                    metadata_insert_stmt.bind_i64(2, row.vector_id as i64)?;
                    metadata_insert_stmt
                        .bind_i64(3, u64_to_i64_bits(row.entry.metadata.entity_id))?;
                    metadata_insert_stmt.bind_i64(
                        4,
                        row.entry.metadata.sequence_ts.min(i64::MAX as u64) as i64,
                    )?;
                    metadata_insert_stmt.bind_blob(5, &row.entry.metadata.payload)?;
                    let tags_blob = encode_tags_blob(&metadata_v3.tags);
                    metadata_insert_stmt.bind_blob(6, &tags_blob)?;
                    metadata_insert_stmt.bind_i64(7, now)?;
                    metadata_insert_stmt.step_done()?;

                    if let Some(result) = results.get_mut(row.result_index) {
                        result.metadata = Some(metadata_v3);
                    }
                }

                for (tag_id, tag_text) in &new_tag_rows {
                    tag_registry_insert_stmt.reset()?;
                    tag_registry_insert_stmt.bind_text(1, collection_id)?;
                    tag_registry_insert_stmt.bind_i64(2, *tag_id as i64)?;
                    tag_registry_insert_stmt.bind_text(3, tag_text)?;
                    tag_registry_insert_stmt.bind_i64(4, now)?;
                    tag_registry_insert_stmt.step_done()?;

                    // Backward-compatible mirror while v2 readers still exist.
                    tag_ids_insert_stmt.reset()?;
                    tag_ids_insert_stmt.bind_text(1, collection_id)?;
                    tag_ids_insert_stmt.bind_i64(2, *tag_id as i64)?;
                    tag_ids_insert_stmt.bind_text(3, tag_text)?;
                    tag_ids_insert_stmt.bind_i64(4, now)?;
                    tag_ids_insert_stmt.step_done()?;
                }

                for (vector_id, tag_id) in &vector_tag_pairs {
                    vector_tags_insert_stmt.reset()?;
                    vector_tags_insert_stmt.bind_text(1, collection_id)?;
                    vector_tags_insert_stmt.bind_i64(2, *vector_id as i64)?;
                    vector_tags_insert_stmt.bind_i64(3, *tag_id as i64)?;
                    vector_tags_insert_stmt.bind_i64(4, now)?;
                    vector_tags_insert_stmt.step_done()?;
                }
                staged_tag_rows.extend(new_tag_rows.into_iter());
            }

            Ok((results, staged_tag_rows))
        })();

        match tx_result {
            Ok((results, staged_tag_rows)) => {
                guard.exec("COMMIT;")?;
                self.commit_tag_rows_to_cache(collection_id, &staged_tag_rows);
                Ok(results)
            }
            Err(err) => {
                let _ = guard.exec("ROLLBACK;");
                let _ = self.load_tag_registry_cache();
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

fn encode_vector_blob(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decode_vector_blob(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "invalid vector blob length: expected multiple of 4, got {}",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn encode_tags_blob(tag_ids: &[u32]) -> Vec<u8> {
    let mut sorted = tag_ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut out = Vec::with_capacity(sorted.len() * std::mem::size_of::<u32>());
    for tag_id in sorted {
        out.extend_from_slice(&tag_id.to_le_bytes());
    }
    out
}

fn decode_tags_blob(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.len() % std::mem::size_of::<u32>() != 0 {
        return Err(anyhow!(
            "invalid tags_blob length: expected multiple of 4, got {}",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / std::mem::size_of::<u32>());
    for chunk in bytes.chunks_exact(std::mem::size_of::<u32>()) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[inline]
fn u64_to_i64_bits(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

#[inline]
fn i64_to_u64_bits(value: i64) -> u64 {
    u64::from_ne_bytes(value.to_ne_bytes())
}

fn encode_metadata_blob(metadata: &VectorMetadataV3) -> Result<Vec<u8>> {
    Ok(rmp_serde::to_vec(metadata)?)
}

fn decode_metadata_blob(bytes: &[u8]) -> Result<VectorMetadataV3> {
    Ok(rmp_serde::from_slice(bytes)?)
}

fn legacy_metadata_to_v3(metadata: &VectorMetadata) -> VectorMetadataV3 {
    let entity_id = if metadata.source_file.is_empty() {
        0
    } else {
        xxh3_64(metadata.source_file.as_bytes())
    };
    let payload = rmp_serde::to_vec(metadata).unwrap_or_default();
    VectorMetadataV3 {
        entity_id,
        sequence_ts: metadata.start_time_ms as u64,
        tags: Vec::new(),
        payload,
    }
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
    use tempfile::tempdir;

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

    #[test]
    fn tag_registry_cap_is_enforced_per_collection() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("catalog.sqlite3");
        let catalog = SqliteCatalog::open_with_options(
            &path,
            CatalogOptions {
                max_tag_registry_size: 2,
                ..CatalogOptions::default()
            },
        )
        .expect("open catalog");

        catalog
            .ensure_collection("collection_a", 4)
            .expect("ensure collection a");
        catalog
            .ensure_collection("collection_b", 4)
            .expect("ensure collection b");

        let vector = vec![0.1f32, 0.2, 0.3, 0.4];
        let mk_meta = |tag: &str| IngestMetadataV3Input {
            entity_id: 1,
            sequence_ts: 1,
            tags: vec![tag.to_string()],
            payload: Vec::new(),
        };

        catalog
            .ingest_wal_batch(
                "collection_a",
                &[
                    (vector.clone(), mk_meta("a_one"), None),
                    (vector.clone(), mk_meta("a_two"), None),
                ],
            )
            .expect("ingest collection a tags");

        // Different collection should maintain an independent cap budget.
        catalog
            .ingest_wal_batch(
                "collection_b",
                &[
                    (vector.clone(), mk_meta("b_one"), None),
                    (vector.clone(), mk_meta("b_two"), None),
                ],
            )
            .expect("ingest collection b tags");

        let overflow = catalog
            .ingest_wal_batch(
                "collection_a",
                &[(vector.clone(), mk_meta("a_three"), None)],
            )
            .expect_err("expected collection_a overflow");
        let overflow_msg = overflow.to_string();
        assert!(overflow_msg.contains("tag_registry_overflow"));
        assert!(overflow_msg.contains("collection_id=collection_a"));

        let dict_a = catalog
            .fetch_tag_dictionary("collection_a")
            .expect("dictionary a");
        let dict_b = catalog
            .fetch_tag_dictionary("collection_b")
            .expect("dictionary b");
        assert_eq!(dict_a.len(), 2);
        assert_eq!(dict_b.len(), 2);
    }

    #[test]
    fn audit_events_batch_inserts_all_rows() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("catalog.sqlite3");
        let catalog = SqliteCatalog::open(&path).expect("open catalog");

        let events = vec![
            AuditEvent {
                request_id: "req_1".to_string(),
                api_key_id: Some("vbk_1".to_string()),
                endpoint: "/v3/query".to_string(),
                action: "query".to_string(),
                status_code: 200,
                latency_ms: 1.5,
                details: serde_json::json!({"ok": true}),
            },
            AuditEvent {
                request_id: "req_2".to_string(),
                api_key_id: None,
                endpoint: "/v3/ingest".to_string(),
                action: "ingest".to_string(),
                status_code: 202,
                latency_ms: 2.25,
                details: serde_json::json!({"rows": 2}),
            },
        ];
        catalog
            .audit_events_batch(&events)
            .expect("batch insert succeeds");

        let rows = catalog
            .query_json(
                "SELECT request_id, api_key_id, endpoint, action, status_code
                 FROM audit_events
                 ORDER BY rowid ASC;",
            )
            .expect("query audit events");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["request_id"], serde_json::json!("req_1"));
        assert_eq!(rows[0]["api_key_id"], serde_json::json!("vbk_1"));
        assert_eq!(rows[1]["request_id"], serde_json::json!("req_2"));
        assert_eq!(rows[1]["api_key_id"], serde_json::Value::Null);
    }
}
