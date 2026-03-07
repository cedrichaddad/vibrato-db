use std::collections::HashMap;
use std::ops::RangeBounds;
use std::ops::{Bound, RangeInclusive};

use roaring::RoaringBitmap;

use super::model::QueryFilter;

#[derive(Debug, Clone, Default)]
pub struct BitmapSet {
    inner: RoaringBitmap,
}

impl BitmapSet {
    pub fn with_capacity(_capacity: usize) -> Self {
        Self::default()
    }

    pub fn grow(&mut self, _new_len: usize) {}

    #[inline]
    fn id_to_bitmap(id: usize) -> u32 {
        u32::try_from(id)
            .unwrap_or_else(|_| panic!("data integrity fault: bitmap id overflow id={id}"))
    }

    pub fn insert(&mut self, id: usize) {
        let id32 = Self::id_to_bitmap(id);
        self.inner.insert(id32);
    }

    pub fn contains(&self, id: usize) -> bool {
        let id32 = Self::id_to_bitmap(id);
        self.inner.contains(id32)
    }

    pub fn count_ones<R>(&self, range: R) -> usize
    where
        R: RangeBounds<usize>,
    {
        let start = match range.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => v.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => v.saturating_sub(1),
            Bound::Unbounded => u32::MAX as usize,
        };
        if start > end {
            return 0;
        }
        let start_u32 = match u32::try_from(start) {
            Ok(v) => v,
            Err(_) => return 0,
        };
        let end_u32 = u32::try_from(end).unwrap_or(u32::MAX);
        self.inner
            .range_cardinality(RangeInclusive::new(start_u32, end_u32)) as usize
    }

    pub fn cardinality(&self) -> usize {
        self.inner.len() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn ones(&self) -> Vec<usize> {
        self.inner.iter().map(|id| id as usize).collect()
    }

    pub fn iter_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.inner.iter().map(|id| id as usize)
    }

    pub fn union_with(&mut self, other: &Self) {
        self.inner |= &other.inner;
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let mut inner = self.inner.clone();
        inner &= &other.inner;
        Self { inner }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FilterIndex {
    pub tag_ids: HashMap<String, u32>,
    pub tag_bitmaps: HashMap<u32, BitmapSet>,
}

impl FilterIndex {
    pub fn with_dictionary(tag_ids: HashMap<String, u32>) -> Self {
        let mut normalized = HashMap::with_capacity(tag_ids.len());
        for (tag, id) in tag_ids {
            let key = tag.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            normalized.insert(key, id);
        }
        Self {
            tag_ids: normalized,
            tag_bitmaps: HashMap::new(),
        }
    }

    pub fn add_with_tag_ids(&mut self, vector_id: usize, tag_ids: &[u32]) {
        for tag_id in tag_ids {
            self.tag_bitmaps
                .entry(*tag_id)
                .or_default()
                .insert(vector_id);
        }
    }

    pub fn build_allow_set(&self, filter: &QueryFilter) -> Option<BitmapSet> {
        let mut allow: Option<BitmapSet> = None;

        if !filter.tags_all.is_empty() {
            for tag in &filter.tags_all {
                let t = tag.trim().to_ascii_lowercase();
                let bm = self
                    .tag_ids
                    .get(&t)
                    .and_then(|id| self.tag_bitmaps.get(id))
                    .cloned()
                    .unwrap_or_default();
                allow = Some(match allow {
                    Some(curr) => curr.intersect(&bm),
                    None => bm,
                });
            }
        }

        if !filter.tags_any.is_empty() {
            let mut any_union = BitmapSet::default();
            for tag in &filter.tags_any {
                let t = tag.trim().to_ascii_lowercase();
                if let Some(tag_id) = self.tag_ids.get(&t) {
                    if let Some(bm) = self.tag_bitmaps.get(tag_id) {
                        any_union.union_with(bm);
                    }
                }
            }
            allow = Some(match allow {
                Some(curr) => curr.intersect(&any_union),
                None => any_union,
            });
        }

        allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roaring_bitmap_sparse_dense_union_and_intersect() {
        let mut a = BitmapSet::default();
        let mut b = BitmapSet::default();
        for i in (0..10_000usize).step_by(3) {
            a.insert(i);
        }
        for i in (0..10_000usize).step_by(5) {
            b.insert(i);
        }

        let inter = a.intersect(&b);
        for id in inter.ones() {
            assert!(id % 3 == 0 && id % 5 == 0);
        }

        let mut union = a.clone();
        union.union_with(&b);
        assert!(union.contains(0));
        assert!(union.contains(3));
        assert!(union.contains(5));
        assert!(union.contains(9999));
    }

    #[test]
    fn roaring_bitmap_handles_multiple_high_ranges() {
        let mut bm = BitmapSet::default();
        bm.insert(1);
        bm.insert(65_537);
        bm.insert(131_073);

        assert!(bm.contains(1));
        assert!(bm.contains(65_537));
        assert!(bm.contains(131_073));
        assert_eq!(bm.cardinality(), 3);
    }

    #[test]
    fn filter_combines_tags_with_expected_set_algebra() {
        let mut index = FilterIndex::with_dictionary(HashMap::from([
            ("drums".to_string(), 1u32),
            ("snare".to_string(), 2u32),
            ("bass".to_string(), 3u32),
        ]));

        index.add_with_tag_ids(10, &[1, 2]);
        index.add_with_tag_ids(11, &[1]);
        index.add_with_tag_ids(12, &[3]);

        let filter = QueryFilter {
            tags_any: vec!["drums".to_string(), "bass".to_string()],
            tags_all: vec!["drums".to_string()],
        };
        let allow = index
            .build_allow_set(&filter)
            .expect("expected allow set for tags filter");

        assert!(allow.contains(10));
        assert!(allow.contains(11));
        assert!(!allow.contains(12));
    }
}
