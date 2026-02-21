//! `vibrato-midas`: constrained DTW primitives for quantitative time-series search.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MidasError {
    #[error("query sequence must not be empty")]
    EmptyQuery,
    #[error("candidate sequence must not be empty")]
    EmptyCandidate,
    #[error("sakoe-chiba band {band} exceeds configured maximum {max_band}")]
    InvalidBand { band: usize, max_band: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidasMatch {
    pub index: usize,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug)]
struct HeapItem {
    index: usize,
    distance: f32,
}

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn sq_l2(a: f32, b: f32) -> f32 {
    let d = a - b;
    d * d
}

/// Constrained DTW distance with Sakoe-Chiba band and optional early abandonment.
///
/// If `early_abandon_at` is set, the function returns `f32::INFINITY` as soon as
/// the current row minimum exceeds that threshold.
pub fn constrained_dtw_distance(
    query: &[f32],
    candidate: &[f32],
    sakoe_chiba_band: usize,
    early_abandon_at: Option<f32>,
) -> Result<f32, MidasError> {
    if query.is_empty() {
        return Err(MidasError::EmptyQuery);
    }
    if candidate.is_empty() {
        return Err(MidasError::EmptyCandidate);
    }

    let n = query.len();
    let m = candidate.len();
    let w = sakoe_chiba_band;
    if n.abs_diff(m) > w {
        return Ok(f32::INFINITY);
    }

    let mut prev = vec![f32::INFINITY; m + 1];
    let mut curr = vec![f32::INFINITY; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        let j_start = i.saturating_sub(w).max(1);
        let j_end = (i + w).min(m);

        curr.fill(f32::INFINITY);
        let mut row_min = f32::INFINITY;
        let qv = query[i - 1];
        for j in j_start..=j_end {
            let cv = candidate[j - 1];
            let cost = sq_l2(qv, cv);
            let best_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            let v = cost + best_prev;
            curr[j] = v;
            row_min = row_min.min(v);
        }

        if let Some(threshold) = early_abandon_at {
            if row_min > threshold {
                return Ok(f32::INFINITY);
            }
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    Ok(prev[m])
}

/// Search a query against multiple candidate sequences and return top-k nearest matches.
///
/// Uses Sakoe-Chiba constrained DTW and early abandonment tied to the current worst
/// top-k distance.
pub fn midas_fractal_search(
    query: &[f32],
    candidates: &[Vec<f32>],
    top_k: usize,
    sakoe_chiba_band: usize,
    max_band: usize,
) -> Result<Vec<MidasMatch>, MidasError> {
    if sakoe_chiba_band > max_band {
        return Err(MidasError::InvalidBand {
            band: sakoe_chiba_band,
            max_band,
        });
    }
    if top_k == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(top_k + 1);
    for (idx, candidate) in candidates.iter().enumerate() {
        let cutoff = heap.peek().map(|item| item.distance);
        let distance = constrained_dtw_distance(query, candidate, sakoe_chiba_band, cutoff)?;
        if !distance.is_finite() {
            continue;
        }

        if heap.len() < top_k {
            heap.push(HeapItem {
                index: idx,
                distance,
            });
            continue;
        }

        let should_insert = heap
            .peek()
            .map(|worst| distance < worst.distance)
            .unwrap_or(true);
        if should_insert {
            heap.pop();
            heap.push(HeapItem {
                index: idx,
                distance,
            });
        }
    }

    let mut out = heap
        .into_iter()
        .map(|item| MidasMatch {
            index: item.index,
            distance: item.distance,
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_identity_is_zero() {
        let query = [1.0f32, 2.0, 3.0, 4.0];
        let dist = constrained_dtw_distance(&query, &query, 1, None).expect("dtw");
        assert!(dist.abs() < 1e-6, "expected near-zero distance, got {dist}");
    }

    #[test]
    fn dtw_returns_infinity_when_outside_band() {
        let query = [1.0f32, 2.0, 3.0, 4.0];
        let candidate = [1.0f32, 2.0];
        let dist = constrained_dtw_distance(&query, &candidate, 1, None).expect("dtw");
        assert!(!dist.is_finite());
    }

    #[test]
    fn invalid_band_is_rejected() {
        let query = [1.0f32, 2.0, 3.0];
        let candidates = vec![vec![1.0f32, 2.0, 3.0]];
        let err = midas_fractal_search(&query, &candidates, 3, 12, 8).expect_err("invalid band");
        assert_eq!(
            err,
            MidasError::InvalidBand {
                band: 12,
                max_band: 8
            }
        );
    }

    #[test]
    fn top_k_results_are_sorted_and_best_match_first() {
        let query = [1.0f32, 2.0, 3.0, 4.0];
        let candidates = vec![
            vec![2.0f32, 3.0, 4.0, 5.0],
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![0.5f32, 2.0, 3.5, 5.0],
        ];

        let results = midas_fractal_search(&query, &candidates, 2, 2, 8).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 1);
        assert!(results[0].distance <= results[1].distance);
    }
}
