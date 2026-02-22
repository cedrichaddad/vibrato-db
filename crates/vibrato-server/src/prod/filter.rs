use std::collections::{BTreeMap, HashMap};
use std::ops::RangeBounds;
use std::ops::{Bound, RangeInclusive};

use roaring::RoaringBitmap;
use vibrato_core::metadata::VectorMetadata;

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
    fn id_to_bitmap(id: usize) -> Option<u32> {
        u32::try_from(id).ok()
    }

    pub fn insert(&mut self, id: usize) {
        if let Some(id32) = Self::id_to_bitmap(id) {
            self.inner.insert(id32);
        }
    }

    pub fn contains(&self, id: usize) -> bool {
        Self::id_to_bitmap(id)
            .map(|id32| self.inner.contains(id32))
            .unwrap_or(false)
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
    pub next_tag_id: u32,
    pub bpm_buckets: BTreeMap<i32, BitmapSet>,
    pub bpm_values: HashMap<usize, f32>,
}

impl FilterIndex {
    pub fn with_dictionary(tag_ids: HashMap<String, u32>) -> Self {
        let mut normalized = HashMap::with_capacity(tag_ids.len());
        let mut next = 0u32;
        for (tag, id) in tag_ids {
            let key = tag.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            normalized.insert(key, id);
            next = next.max(id.saturating_add(1));
        }
        Self {
            tag_ids: normalized,
            tag_bitmaps: HashMap::new(),
            next_tag_id: next,
            bpm_buckets: BTreeMap::new(),
            bpm_values: HashMap::new(),
        }
    }

    fn intern_tag(&mut self, tag: &str) -> u32 {
        if let Some(existing) = self.tag_ids.get(tag) {
            return *existing;
        }
        let id = self.next_tag_id;
        self.next_tag_id = self.next_tag_id.saturating_add(1);
        self.tag_ids.insert(tag.to_string(), id);
        id
    }

    pub fn add(&mut self, vector_id: usize, metadata: &VectorMetadata) {
        for tag in &metadata.tags {
            let key = tag.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            let tag_id = self.intern_tag(&key);
            self.tag_bitmaps
                .entry(tag_id)
                .or_default()
                .insert(vector_id);
        }

        let bpm_bucket = metadata.bpm.floor() as i32;
        self.bpm_buckets
            .entry(bpm_bucket)
            .or_default()
            .insert(vector_id);
        self.bpm_values.insert(vector_id, metadata.bpm);
    }

    pub fn add_with_tag_ids(&mut self, vector_id: usize, bpm: f32, tag_ids: &[u32]) {
        for tag_id in tag_ids {
            self.next_tag_id = self.next_tag_id.max(tag_id.saturating_add(1));
            self.tag_bitmaps
                .entry(*tag_id)
                .or_default()
                .insert(vector_id);
        }

        let bpm_bucket = bpm.floor() as i32;
        self.bpm_buckets
            .entry(bpm_bucket)
            .or_default()
            .insert(vector_id);
        self.bpm_values.insert(vector_id, bpm);
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

        if filter.bpm_gte.is_some() || filter.bpm_lte.is_some() {
            let lower = filter.bpm_gte.unwrap_or(f32::MIN);
            let upper = filter.bpm_lte.unwrap_or(f32::MAX);
            let gte = lower.floor() as i32;
            let lte = upper.ceil() as i32;

            let mut bpm_union = BitmapSet::default();
            for (_, bitmap) in self.bpm_buckets.range(gte..=lte) {
                bpm_union.union_with(bitmap);
            }

            let mut bpm_exact = BitmapSet::default();
            for id in bpm_union.iter_ids() {
                if let Some(bpm) = self.bpm_values.get(&id) {
                    if *bpm >= lower && *bpm <= upper {
                        bpm_exact.insert(id);
                    }
                }
            }

            allow = Some(match allow {
                Some(curr) => curr.intersect(&bpm_exact),
                None => bpm_exact,
            });
        }

        allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vibrato_core::metadata::VectorMetadata;

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
    fn bpm_range_uses_exact_boundaries_after_bucket_prune() {
        let mut index = FilterIndex::default();
        let meta_119_9 = VectorMetadata {
            source_file: "a.wav".to_string(),
            start_time_ms: 0,
            duration_ms: 100,
            bpm: 119.9,
            tags: vec!["drums".to_string()],
        };
        let meta_120_2 = VectorMetadata {
            source_file: "b.wav".to_string(),
            start_time_ms: 0,
            duration_ms: 100,
            bpm: 120.2,
            tags: vec!["drums".to_string()],
        };

        index.add(1, &meta_119_9);
        index.add(2, &meta_120_2);

        let filter = QueryFilter {
            bpm_gte: Some(120.0),
            bpm_lte: Some(120.0),
            ..Default::default()
        };
        let allow = index
            .build_allow_set(&filter)
            .expect("expected allow set for bpm filter");

        assert!(!allow.contains(1), "119.9 must be excluded by exact bounds");
        assert!(!allow.contains(2), "120.2 must be excluded by exact bounds");
    }

    #[test]
    fn filter_combines_tags_and_bpm_with_expected_set_algebra() {
        let mut index = FilterIndex::default();

        index.add(
            10,
            &VectorMetadata {
                source_file: "kick.wav".to_string(),
                start_time_ms: 0,
                duration_ms: 100,
                bpm: 120.0,
                tags: vec!["Drums".to_string(), "kick".to_string()],
            },
        );
        index.add(
            11,
            &VectorMetadata {
                source_file: "snare.wav".to_string(),
                start_time_ms: 0,
                duration_ms: 100,
                bpm: 124.0,
                tags: vec!["drums".to_string(), "snare".to_string()],
            },
        );
        index.add(
            12,
            &VectorMetadata {
                source_file: "bass.wav".to_string(),
                start_time_ms: 0,
                duration_ms: 100,
                bpm: 124.0,
                tags: vec!["bass".to_string()],
            },
        );

        let filter = QueryFilter {
            tags_all: vec!["drums".to_string()],
            tags_any: vec!["snare".to_string(), "kick".to_string()],
            bpm_gte: Some(123.0),
            bpm_lte: Some(125.0),
        };
        let allow = index
            .build_allow_set(&filter)
            .expect("expected allow set for combined filter");

        assert!(!allow.contains(10), "kick should be excluded by bpm range");
        assert!(allow.contains(11), "snare should pass all clauses");
        assert!(!allow.contains(12), "bass should be excluded by tags_all");
    }

    #[test]
    fn unknown_tags_any_returns_empty_allow_set() {
        let mut index = FilterIndex::default();
        index.add(
            7,
            &VectorMetadata {
                source_file: "known.wav".to_string(),
                start_time_ms: 0,
                duration_ms: 100,
                bpm: 120.0,
                tags: vec!["drums".to_string()],
            },
        );

        let filter = QueryFilter {
            tags_any: vec!["missing".to_string()],
            ..Default::default()
        };
        let allow = index
            .build_allow_set(&filter)
            .expect("tags_any should produce an allow set");
        assert_eq!(allow.cardinality(), 0);
    }

    #[test]
    fn dictionary_normalization_is_lowercase_and_trimmed() {
        let mut dict = HashMap::new();
        dict.insert(" Drums ".to_string(), 3u32);
        let index = FilterIndex::with_dictionary(dict);
        assert_eq!(index.tag_ids.get("drums"), Some(&3u32));
    }
}
