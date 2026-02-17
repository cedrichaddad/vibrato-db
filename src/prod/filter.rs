use std::collections::{BTreeMap, HashMap};
use std::ops::RangeBounds;

use vibrato_core::metadata::VectorMetadata;

use super::model::QueryFilter;

const ARRAY_TO_BITMAP_THRESHOLD: usize = 4096;
const CONTAINER_BITS: usize = 1 << 16;
const BITMAP_WORDS: usize = CONTAINER_BITS / 64;

#[derive(Debug, Clone)]
enum Container {
    Array(Vec<u16>),
    Bitmap {
        bits: Box<[u64; BITMAP_WORDS]>,
        cardinality: usize,
    },
}

impl Container {
    fn empty() -> Self {
        Self::Array(Vec::new())
    }

    fn cardinality(&self) -> usize {
        match self {
            Self::Array(values) => values.len(),
            Self::Bitmap { cardinality, .. } => *cardinality,
        }
    }

    fn contains(&self, value: u16) -> bool {
        match self {
            Self::Array(values) => values.binary_search(&value).is_ok(),
            Self::Bitmap { bits, .. } => {
                let word = value as usize / 64;
                let mask = 1u64 << (value as usize % 64);
                (bits[word] & mask) != 0
            }
        }
    }

    fn insert(&mut self, value: u16) {
        match self {
            Self::Array(values) => match values.binary_search(&value) {
                Ok(_) => {}
                Err(pos) => {
                    values.insert(pos, value);
                    if values.len() > ARRAY_TO_BITMAP_THRESHOLD {
                        let mut bits = Box::new([0u64; BITMAP_WORDS]);
                        for v in values.iter().copied() {
                            let word = v as usize / 64;
                            let bit = 1u64 << (v as usize % 64);
                            bits[word] |= bit;
                        }
                        *self = Self::Bitmap {
                            bits,
                            cardinality: values.len(),
                        };
                    }
                }
            },
            Self::Bitmap { bits, cardinality } => {
                let word = value as usize / 64;
                let mask = 1u64 << (value as usize % 64);
                if (bits[word] & mask) == 0 {
                    bits[word] |= mask;
                    *cardinality += 1;
                }
            }
        }
    }

    fn values(&self) -> Vec<u16> {
        match self {
            Self::Array(values) => values.clone(),
            Self::Bitmap { bits, .. } => {
                let mut out = Vec::new();
                for (word_idx, word) in bits.iter().enumerate() {
                    let mut w = *word;
                    while w != 0 {
                        let tz = w.trailing_zeros() as usize;
                        out.push((word_idx * 64 + tz) as u16);
                        w &= w - 1;
                    }
                }
                out
            }
        }
    }
}

fn union_container(a: &Container, b: &Container) -> Container {
    match (a, b) {
        (Container::Array(av), Container::Array(bv)) => {
            let mut out = Vec::with_capacity(av.len() + bv.len());
            let mut i = 0usize;
            let mut j = 0usize;
            while i < av.len() && j < bv.len() {
                match av[i].cmp(&bv[j]) {
                    std::cmp::Ordering::Less => {
                        out.push(av[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        out.push(bv[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        out.push(av[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            out.extend_from_slice(&av[i..]);
            out.extend_from_slice(&bv[j..]);
            dedup_sorted(&mut out);
            if out.len() > ARRAY_TO_BITMAP_THRESHOLD {
                let mut bits = Box::new([0u64; BITMAP_WORDS]);
                for v in out.iter().copied() {
                    let word = v as usize / 64;
                    let bit = 1u64 << (v as usize % 64);
                    bits[word] |= bit;
                }
                Container::Bitmap {
                    bits,
                    cardinality: out.len(),
                }
            } else {
                Container::Array(out)
            }
        }
        (Container::Bitmap { bits, cardinality }, Container::Array(values))
        | (Container::Array(values), Container::Bitmap { bits, cardinality }) => {
            let mut out_bits = bits.clone();
            let mut out_cardinality = *cardinality;
            for v in values.iter().copied() {
                let word = v as usize / 64;
                let mask = 1u64 << (v as usize % 64);
                if (out_bits[word] & mask) == 0 {
                    out_bits[word] |= mask;
                    out_cardinality += 1;
                }
            }
            Container::Bitmap {
                bits: out_bits,
                cardinality: out_cardinality,
            }
        }
        (
            Container::Bitmap {
                bits: a_bits,
                cardinality: _,
            },
            Container::Bitmap {
                bits: b_bits,
                cardinality: _,
            },
        ) => {
            let mut out_bits = Box::new([0u64; BITMAP_WORDS]);
            let mut out_cardinality = 0usize;
            for i in 0..BITMAP_WORDS {
                out_bits[i] = a_bits[i] | b_bits[i];
                out_cardinality += out_bits[i].count_ones() as usize;
            }
            Container::Bitmap {
                bits: out_bits,
                cardinality: out_cardinality,
            }
        }
    }
}

fn intersect_container(a: &Container, b: &Container) -> Container {
    match (a, b) {
        (Container::Array(av), Container::Array(bv)) => {
            let mut out = Vec::with_capacity(av.len().min(bv.len()));
            let mut i = 0usize;
            let mut j = 0usize;
            while i < av.len() && j < bv.len() {
                match av[i].cmp(&bv[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        out.push(av[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            Container::Array(out)
        }
        (Container::Bitmap { bits, .. }, Container::Array(values))
        | (Container::Array(values), Container::Bitmap { bits, .. }) => {
            let mut out = Vec::new();
            for v in values.iter().copied() {
                let word = v as usize / 64;
                let mask = 1u64 << (v as usize % 64);
                if (bits[word] & mask) != 0 {
                    out.push(v);
                }
            }
            Container::Array(out)
        }
        (
            Container::Bitmap {
                bits: a_bits,
                cardinality: _,
            },
            Container::Bitmap {
                bits: b_bits,
                cardinality: _,
            },
        ) => {
            let mut out_bits = Box::new([0u64; BITMAP_WORDS]);
            let mut out_cardinality = 0usize;
            for i in 0..BITMAP_WORDS {
                out_bits[i] = a_bits[i] & b_bits[i];
                out_cardinality += out_bits[i].count_ones() as usize;
            }
            if out_cardinality <= ARRAY_TO_BITMAP_THRESHOLD {
                let mut values = Vec::with_capacity(out_cardinality);
                for (word_idx, word) in out_bits.iter().enumerate() {
                    let mut w = *word;
                    while w != 0 {
                        let tz = w.trailing_zeros() as usize;
                        values.push((word_idx * 64 + tz) as u16);
                        w &= w - 1;
                    }
                }
                Container::Array(values)
            } else {
                Container::Bitmap {
                    bits: out_bits,
                    cardinality: out_cardinality,
                }
            }
        }
    }
}

fn dedup_sorted(values: &mut Vec<u16>) {
    if values.is_empty() {
        return;
    }
    let mut out_idx = 1usize;
    for i in 1..values.len() {
        if values[i] != values[out_idx - 1] {
            values[out_idx] = values[i];
            out_idx += 1;
        }
    }
    values.truncate(out_idx);
}

#[derive(Debug, Clone, Default)]
pub struct BitmapSet {
    containers: BTreeMap<u16, Container>,
}

impl BitmapSet {
    pub fn with_capacity(_capacity: usize) -> Self {
        Self::default()
    }

    pub fn grow(&mut self, _new_len: usize) {}

    pub fn insert(&mut self, id: usize) {
        let value = id as u32;
        let high = (value >> 16) as u16;
        let low = (value & 0xFFFF) as u16;
        self.containers
            .entry(high)
            .or_insert_with(Container::empty)
            .insert(low);
    }

    pub fn contains(&self, id: usize) -> bool {
        let value = id as u32;
        let high = (value >> 16) as u16;
        let low = (value & 0xFFFF) as u16;
        self.containers
            .get(&high)
            .map(|c| c.contains(low))
            .unwrap_or(false)
    }

    pub fn count_ones<R>(&self, _range: R) -> usize
    where
        R: RangeBounds<usize>,
    {
        self.cardinality()
    }

    pub fn cardinality(&self) -> usize {
        self.containers.values().map(Container::cardinality).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    pub fn ones(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.cardinality());
        for (high, container) in &self.containers {
            for low in container.values() {
                out.push(((*high as usize) << 16) | low as usize);
            }
        }
        out
    }

    pub fn union_with(&mut self, other: &Self) {
        for (high, other_container) in &other.containers {
            match self.containers.get(high) {
                Some(this_container) => {
                    let merged = union_container(this_container, other_container);
                    self.containers.insert(*high, merged);
                }
                None => {
                    self.containers.insert(*high, other_container.clone());
                }
            }
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let mut out = Self::default();
        for (high, this_container) in &self.containers {
            if let Some(other_container) = other.containers.get(high) {
                let inter = intersect_container(this_container, other_container);
                if inter.cardinality() > 0 {
                    out.containers.insert(*high, inter);
                }
            }
        }
        out
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
            let entry = self.tag_bitmaps.entry(tag_id).or_default();
            entry.insert(vector_id);
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
            for id in bpm_union.ones() {
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
    fn roaring_like_bitmap_sparse_dense_union_and_intersect() {
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
    fn roaring_like_bitmap_handles_multiple_high_containers() {
        let mut bm = BitmapSet::default();
        bm.insert(1);
        bm.insert(65_537); // different high-16 container
        bm.insert(131_073); // another container

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
