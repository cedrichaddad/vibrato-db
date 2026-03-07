//! Thread-local visited set pool for HNSW search.
//!
//! Default implementation uses an epoch array to avoid O(n) clear cost per query:
//! - `is_visited(id)` is a single array read/compare
//! - `visit(id)` is a single array write
//! - `clear()` increments epoch instead of zeroing memory

use std::cell::RefCell;

use rustc_hash::FxHashSet;

thread_local! {
    /// Thread-local pool of epoch-backed visited sets.
    static VISITED_POOL: RefCell<VisitedPoolInner> = RefCell::new(VisitedPoolInner::new());
}

/// Common visited-set contract for HNSW traversals.
///
/// The hot path can be monomorphized over this trait for edge/server-specific
/// implementations without virtual dispatch.
pub trait VisitedSet {
    /// Ensure capacity for visiting `max_id` and clear previous marks.
    fn prepare_for(&mut self, max_id: usize);

    /// Check if `id` has already been visited in the current search.
    fn is_visited(&self, id: usize) -> bool;

    /// Mark `id` as visited in the current search.
    fn visit(&mut self, id: usize);

    /// Reset visited state for the next search.
    fn clear(&mut self);
}

/// Epoch-backed visited state.
struct EpochVisited {
    epochs: Vec<u32>,
    current_epoch: u32,
}

impl EpochVisited {
    fn with_capacity(required_len: usize) -> Self {
        Self {
            epochs: vec![0; required_len.max(1024)],
            current_epoch: 1,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.epochs.len()
    }

    #[inline(always)]
    fn ensure_len(&mut self, required_len: usize) {
        if required_len <= self.epochs.len() {
            return;
        }
        let new_len = required_len
            .checked_next_power_of_two()
            .unwrap_or(required_len)
            .max(1024);
        self.epochs.resize(new_len, 0);
    }

    #[inline(always)]
    fn clear_epoch(&mut self) {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        if self.current_epoch == 0 {
            // Extremely rare overflow path: reset epochs and restart epoch numbering.
            self.epochs.fill(0);
            self.current_epoch = 1;
        }
    }
}

impl VisitedSet for EpochVisited {
    #[inline(always)]
    fn prepare_for(&mut self, max_id: usize) {
        self.ensure_len(max_id.saturating_add(1).max(1));
        self.clear_epoch();
    }

    #[inline(always)]
    fn is_visited(&self, id: usize) -> bool {
        debug_assert!(id < self.epochs.len());
        // SAFETY: HNSW pre-sizes visited storage by max node ID before search.
        unsafe { *self.epochs.get_unchecked(id) == self.current_epoch }
    }

    #[inline(always)]
    fn visit(&mut self, id: usize) {
        debug_assert!(id < self.epochs.len());
        // SAFETY: HNSW pre-sizes visited storage by max node ID before search.
        unsafe {
            *self.epochs.get_unchecked_mut(id) = self.current_epoch;
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.clear_epoch();
    }
}

/// Hash-set backed visited state for memory-constrained edge modes.
#[allow(dead_code)]
#[derive(Default)]
pub struct HashVisitedSet {
    visited: FxHashSet<usize>,
}

impl VisitedSet for HashVisitedSet {
    #[inline(always)]
    fn prepare_for(&mut self, max_id: usize) {
        let reserve = max_id.saturating_add(1).max(1).min(4096);
        self.visited.clear();
        if self.visited.capacity() < reserve {
            self.visited.reserve(reserve - self.visited.capacity());
        }
    }

    #[inline(always)]
    fn is_visited(&self, id: usize) -> bool {
        self.visited.contains(&id)
    }

    #[inline(always)]
    fn visit(&mut self, id: usize) {
        self.visited.insert(id);
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.visited.clear();
    }
}

/// Inner pool state.
struct VisitedPoolInner {
    sets: Vec<EpochVisited>,
}

impl VisitedPoolInner {
    fn new() -> Self {
        Self {
            sets: Vec::with_capacity(4),
        }
    }
}

/// Handle to a borrowed visited set from the pool.
pub struct PooledVisitedSet {
    set: EpochVisited,
}

impl PooledVisitedSet {
    #[allow(dead_code)]
    #[inline(always)]
    fn len(&self) -> usize {
        self.set.len()
    }

    #[inline(always)]
    pub fn is_visited(&self, id: usize) -> bool {
        self.set.is_visited(id)
    }

    #[inline(always)]
    pub fn visit(&mut self, id: usize) {
        self.set.visit(id);
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.set.clear();
    }
}

impl VisitedSet for PooledVisitedSet {
    #[inline(always)]
    fn prepare_for(&mut self, max_id: usize) {
        self.set.prepare_for(max_id);
    }

    #[inline(always)]
    fn is_visited(&self, id: usize) -> bool {
        self.set.is_visited(id)
    }

    #[inline(always)]
    fn visit(&mut self, id: usize) {
        self.set.visit(id);
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.set.clear();
    }
}

/// Pool for managing visited sets.
pub struct VisitedPool;

impl VisitedPool {
    /// Get a visited set sized for `max_id`.
    pub fn get(max_id: usize) -> PooledVisitedSet {
        let required_len = max_id.saturating_add(1).max(1024);
        VISITED_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let mut set = if let Some(idx) = pool.sets.iter().position(|s| s.len() >= required_len)
            {
                pool.sets.swap_remove(idx)
            } else if !pool.sets.is_empty() {
                // Grow and reuse the largest pooled buffer instead of allocating a new one.
                let largest_idx = pool
                    .sets
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, s)| s.len())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let mut existing = pool.sets.swap_remove(largest_idx);
                existing.ensure_len(required_len);
                existing
            } else {
                EpochVisited::with_capacity(required_len)
            };
            set.prepare_for(max_id);
            PooledVisitedSet { set }
        })
    }

    /// Return a visited set to the pool for reuse.
    pub fn put(visited: PooledVisitedSet) {
        VISITED_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.sets.len() < 4 {
                pool.sets.push(visited.set);
            }
        });
    }
}

/// RAII guard for automatic pool return.
pub struct VisitedGuard {
    set: Option<PooledVisitedSet>,
}

impl VisitedGuard {
    /// Borrow a visited set from the pool, prepared for `max_id`.
    pub fn new(max_id: usize) -> Self {
        Self {
            set: Some(VisitedPool::get(max_id)),
        }
    }

    #[inline(always)]
    fn set_ref_fast(&self) -> &PooledVisitedSet {
        debug_assert!(self.set.is_some());
        // SAFETY: `VisitedGuard` always contains `Some` until `drop` begins.
        unsafe { self.set.as_ref().unwrap_unchecked() }
    }

    #[inline(always)]
    fn set_mut_fast(&mut self) -> &mut PooledVisitedSet {
        debug_assert!(self.set.is_some());
        // SAFETY: `VisitedGuard` always contains `Some` until `drop` begins.
        unsafe { self.set.as_mut().unwrap_unchecked() }
    }

    /// Re-prepare this visited set for a (possibly larger) max node ID.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn prepare_for(&mut self, max_id: usize) {
        self.set_mut().prepare_for(max_id);
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn is_visited(&self, id: usize) -> bool {
        self.set_ref_fast().is_visited(id)
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn visit(&mut self, id: usize) {
        self.set_mut_fast().visit(id);
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.set_mut_fast().clear();
    }

    /// Access the visited set.
    #[inline(always)]
    pub fn set(&self) -> &PooledVisitedSet {
        self.set.as_ref().expect("visited set missing")
    }

    /// Access the visited set mutably.
    #[inline(always)]
    pub fn set_mut(&mut self) -> &mut PooledVisitedSet {
        self.set.as_mut().expect("visited set missing")
    }

    /// Current backing length (for tests/diagnostics).
    #[allow(dead_code)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.set().len()
    }
}

impl Drop for VisitedGuard {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            VisitedPool::put(set);
        }
    }
}

impl VisitedSet for VisitedGuard {
    #[inline(always)]
    fn prepare_for(&mut self, max_id: usize) {
        self.set_mut().prepare_for(max_id);
    }

    #[inline(always)]
    fn is_visited(&self, id: usize) -> bool {
        self.set_ref_fast().is_visited(id)
    }

    #[inline(always)]
    fn visit(&mut self, id: usize) {
        self.set_mut_fast().visit(id);
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.set_mut_fast().clear();
    }
}

impl std::ops::Deref for VisitedGuard {
    type Target = PooledVisitedSet;

    fn deref(&self) -> &Self::Target {
        self.set()
    }
}

impl std::ops::DerefMut for VisitedGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.set_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_basic() {
        let mut guard = VisitedGuard::new(100);

        assert!(!guard.is_visited(0));
        assert!(!guard.is_visited(50));

        guard.visit(0);
        guard.visit(50);

        assert!(guard.is_visited(0));
        assert!(guard.is_visited(50));
        assert!(!guard.is_visited(25));
    }

    #[test]
    fn test_visited_reuse() {
        {
            let mut guard = VisitedGuard::new(100);
            guard.visit(42);
            assert!(guard.is_visited(42));
        }

        {
            let guard = VisitedGuard::new(100);
            assert!(!guard.is_visited(42));
        }
    }

    #[test]
    fn test_visited_large_capacity() {
        let mut guard = VisitedGuard::new(99_999);

        guard.visit(0);
        guard.visit(99_999);

        assert!(guard.is_visited(0));
        assert!(guard.is_visited(99_999));
        assert!(!guard.is_visited(50_000));
    }

    #[test]
    fn test_hash_visited_set_trait_impl() {
        let mut visited = HashVisitedSet::default();
        visited.prepare_for(512);
        assert!(!visited.is_visited(7));
        visited.visit(7);
        assert!(visited.is_visited(7));
        visited.clear();
        assert!(!visited.is_visited(7));
    }

    #[test]
    fn test_visited_prepare_grows_once() {
        let mut guard = VisitedGuard::new(16);
        let before = guard.len();
        guard.prepare_for(4096);
        assert!(guard.len() >= 4097);
        assert!(guard.len() >= before);
        guard.visit(4096);
        assert!(guard.is_visited(4096));
    }
}
