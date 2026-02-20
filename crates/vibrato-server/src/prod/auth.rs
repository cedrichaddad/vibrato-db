use axum::http::HeaderMap;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::catalog::{parse_token, verify_token_hash, ApiKeyRecord, CatalogStore, Role};

#[derive(Debug)]
pub enum AuthError {
    Missing,
    Invalid,
    Forbidden,
    Internal,
}

const API_KEY_CACHE_TTL: Duration = Duration::from_secs(30);
const API_KEY_CACHE_MAX: usize = 4096;

#[derive(Debug, Clone)]
struct CachedKey {
    record: ApiKeyRecord,
    expires_at: Instant,
}

fn key_cache() -> &'static Mutex<HashMap<String, CachedKey>> {
    static CACHE: OnceLock<Mutex<HashMap<String, CachedKey>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn authorize(
    catalog: &dyn CatalogStore,
    headers: &HeaderMap,
    required: Option<Role>,
    pepper: &str,
) -> Result<Option<String>, AuthError> {
    let Some(required_role) = required else {
        return Ok(None);
    };

    let auth = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or(AuthError::Missing)?;

    let (key_id, secret) = parse_token(auth).ok_or(AuthError::Invalid)?;

    let now = Instant::now();
    let cached = {
        let mut guard = key_cache().lock().map_err(|_| AuthError::Internal)?;
        if let Some(entry) = guard.get(&key_id) {
            if entry.expires_at > now {
                Some(entry.record.clone())
            } else {
                guard.remove(&key_id);
                None
            }
        } else {
            None
        }
    };

    let key = if let Some(cached_key) = cached {
        cached_key
    } else {
        let Some(fetched) = catalog
            .lookup_api_key(&key_id)
            .map_err(|_| AuthError::Internal)?
        else {
            return Err(AuthError::Invalid);
        };
        let mut guard = key_cache().lock().map_err(|_| AuthError::Internal)?;
        if guard.len() >= API_KEY_CACHE_MAX {
            guard.clear();
        }
        guard.insert(
            key_id.clone(),
            CachedKey {
                record: fetched.clone(),
                expires_at: now + API_KEY_CACHE_TTL,
            },
        );
        fetched
    };

    if key.revoked {
        return Err(AuthError::Forbidden);
    }

    if !verify_token_hash(pepper, &secret, &key.key_hash) {
        return Err(AuthError::Invalid);
    }

    let has_role = key
        .roles
        .iter()
        .any(|r| r == &Role::Admin || r == &required_role);
    if !has_role {
        return Err(AuthError::Forbidden);
    }

    Ok(Some(key.id))
}
