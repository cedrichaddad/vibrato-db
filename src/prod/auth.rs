use axum::http::HeaderMap;

use super::catalog::{parse_token, verify_token_hash, CatalogStore, Role};

#[derive(Debug)]
pub enum AuthError {
    Missing,
    Invalid,
    Forbidden,
    Internal,
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

    let Some(key) = catalog
        .lookup_api_key(&key_id)
        .map_err(|_| AuthError::Internal)?
    else {
        return Err(AuthError::Invalid);
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
