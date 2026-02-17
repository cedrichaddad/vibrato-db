//! Thread-local visited set pool for HNSW search
//!
//! Using a BitSet instead of HashSet avoids hashing overhead in the hot path.
//! The pool reuses buffers across queries to reduce allocation pressure.

use fixedbitset::FixedBitSet;
use std::cell::RefCell;

thread_local! {
    /// Thread-local pool of visited sets
    static VISITED_POOL: RefCell<VisitedPoolInner> = RefCell::new(VisitedPoolInner::new());
}

/// Inner pool state
struct VisitedPoolInner {
    /// Reusable bitsets (different sizes for different graph sizes)
    sets: Vec<FixedBitSet>,
}

impl VisitedPoolInner {
    fn new() -> Self {
        Self {
            sets: Vec::with_capacity(4),
        }
    }
}

/// Handle to a borrowed visited set from the pool
pub struct VisitedSet {
    set: FixedBitSet,
}

impl VisitedSet {
    /// Check if a node has been visited
    #[inline(always)]
    pub fn is_visited(&self, id: usize) -> bool {
        if id >= self.set.len() {
            return false;
        }
        self.set.contains(id)
    }

    /// Mark a node as visited
    #[inline(always)]
    pub fn visit(&mut self, id: usize) {
        if id >= self.set.len() {
            // HNSW can be traversed with sparse/global IDs (e.g. sharded indexes),
            // so grow on demand when IDs exceed the initial node-count estimate.
            let current = self.set.len().max(1);
            let target = id.saturating_add(1);
            let required = current.max(target);
            let new_len = required.checked_next_power_of_two().unwrap_or(required);
            self.set.grow(new_len);
        }
        self.set.insert(id);
    }

    /// Clear all visited marks
    pub fn clear(&mut self) {
        self.set.clear();
    }
}

/// Pool for managing visited sets
///
/// Provides thread-local allocation-free visited set access.
pub struct VisitedPool;

impl VisitedPool {
    /// Get a visited set with at least the specified capacity
    ///
    /// The returned set is cleared and ready for use.
    pub fn get(capacity: usize) -> VisitedSet {
        VISITED_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();

            // Try to find a reusable set of sufficient size
            let set = if let Some(idx) = pool.sets.iter().position(|s| s.len() >= capacity) {
                let mut set = pool.sets.swap_remove(idx);
                set.clear();
                set
            } else {
                // Create new set with some headroom
                FixedBitSet::with_capacity(capacity.max(1024))
            };

            VisitedSet { set }
        })
    }

    /// Return a visited set to the pool for reuse
    pub fn put(visited: VisitedSet) {
        VISITED_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            // Only keep a few sets in the pool
            if pool.sets.len() < 4 {
                pool.sets.push(visited.set);
            }
            // Otherwise let it drop
        });
    }
}

/// RAII guard for automatic pool return
pub struct VisitedGuard {
    set: Option<VisitedSet>,
}

impl VisitedGuard {
    /// Borrow a visited set from the pool
    pub fn new(capacity: usize) -> Self {
        Self {
            set: Some(VisitedPool::get(capacity)),
        }
    }

    /// Access the visited set
    #[inline(always)]
    pub fn set(&self) -> &VisitedSet {
        self.set.as_ref().unwrap()
    }

    /// Access the visited set mutably
    #[inline(always)]
    pub fn set_mut(&mut self) -> &mut VisitedSet {
        self.set.as_mut().unwrap()
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
