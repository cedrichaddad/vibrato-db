//! Thread-local visited set pool for HNSW search.
//!
//! Uses an epoch array to avoid O(n) clear cost per query:
//! - `is_visited(id)` is a single array read/compare
//! - `visit(id)` is a single array write
//! - `clear()` increments epoch instead of zeroing memory

use std::cell::RefCell;

thread_local! {
    /// Thread-local pool of visited sets
    static VISITED_POOL: RefCell<VisitedPoolInner> = RefCell::new(VisitedPoolInner::new());
}

/// Epoch-backed visited state.
struct EpochVisited {
    epochs: Vec<u32>,
    current_epoch: u32,
}

impl EpochVisited {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            epochs: vec![0; capacity.max(1024)],
            current_epoch: 1,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.epochs.len()
    }

    #[inline(always)]
    fn ensure_capacity_for(&mut self, id: usize) {
        if id < self.epochs.len() {
            return;
        }
        let required = id.saturating_add(1);
        let new_len = required
            .checked_next_power_of_two()
            .unwrap_or(required)
            .max(1024);
        self.epochs.resize(new_len, 0);
    }

    #[inline(always)]
    fn is_visited(&self, id: usize) -> bool {
        id < self.epochs.len() && self.epochs[id] == self.current_epoch
    }

    #[inline(always)]
    fn visit(&mut self, id: usize) {
        self.ensure_capacity_for(id);
        self.epochs[id] = self.current_epoch;
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        if self.current_epoch == 0 {
            // Extremely rare overflow path: reset epochs and restart epoch numbering.
            self.epochs.fill(0);
            self.current_epoch = 1;
        }
    }
}

/// Inner pool state
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
pub struct VisitedSet {
    set: EpochVisited,
}

impl VisitedSet {
    /// Check if a node has been visited.
    #[inline(always)]
    pub fn is_visited(&self, id: usize) -> bool {
        self.set.is_visited(id)
    }

    /// Mark a node as visited.
    #[inline(always)]
    pub fn visit(&mut self, id: usize) {
        self.set.visit(id);
    }

    /// Clear visited marks by advancing epoch.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.set.clear();
    }
}

/// Pool for managing visited sets.
pub struct VisitedPool;

impl VisitedPool {
    /// Get a visited set with at least the specified capacity.
    pub fn get(capacity: usize) -> VisitedSet {
        VISITED_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let mut set = if let Some(idx) = pool.sets.iter().position(|s| s.len() >= capacity) {
                pool.sets.swap_remove(idx)
            } else {
                EpochVisited::with_capacity(capacity)
            };
            set.clear();
            VisitedSet { set }
        })
    }

    /// Return a visited set to the pool for reuse.
    pub fn put(visited: VisitedSet) {
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
    set: Option<VisitedSet>,
}

impl VisitedGuard {
    /// Borrow a visited set from the pool.
    pub fn new(capacity: usize) -> Self {
        Self {
            set: Some(VisitedPool::get(capacity)),
        }
    }

    /// Access the visited set.
    #[inline(always)]
    pub fn set(&self) -> &VisitedSet {
        self.set.as_ref().expect("visited set missing")
    }

    /// Access the visited set mutably.
    #[inline(always)]
    pub fn set_mut(&mut self) -> &mut VisitedSet {
        self.set.as_mut().expect("visited set missing")
    }
}

impl Drop for VisitedGuard {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            VisitedPool::put(set);
        }
    }
}

impl std::ops::Deref for VisitedGuard {
    type Target = VisitedSet;

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
        // First use
        {
            let mut guard = VisitedGuard::new(100);
            guard.visit(42);
            assert!(guard.is_visited(42));
        }

        // Second use - should get a cleared set
        {
            let guard = VisitedGuard::new(100);
            assert!(!guard.is_visited(42)); // Should be cleared
        }
    }

    #[test]
    fn test_visited_large_capacity() {
        let mut guard = VisitedGuard::new(100_000);

        guard.visit(0);
        guard.visit(99_999);

        assert!(guard.is_visited(0));
        assert!(guard.is_visited(99_999));
        assert!(!guard.is_visited(50_000));
    }

    #[test]
    fn test_visited_grows_for_sparse_ids() {
        let mut guard = VisitedGuard::new(16);
        guard.visit(1028);
        assert!(guard.is_visited(1028));
        assert!(!guard.is_visited(1029));
    }
}
