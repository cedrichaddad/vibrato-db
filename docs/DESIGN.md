# Vibrato Design Notes (v2.1 RC1)

## Immutable ID Contract

Vibrato treats `vector_id` as an append-only timeline coordinate:

1. IDs are strictly monotonic per collection.
2. IDs are never reused.
3. Compaction must preserve ID-space geometry.

This contract is required by sequence-alignment (`identify` / `search_subsequence`), where
`start_id + offset` is assumed to map to adjacent frames in time.

## Segment Merge Invariant

During checkpoint/compaction, merged output must preserve dense ID space from `min_id..=max_id`.

- If a source gap exists, Vibrato writes a tombstone slot (zero vector + tombstone metadata).
- Overlapping source ID ranges are rejected as integrity errors.

This guarantees output segments keep `row_count == vector_id_end - vector_id_start + 1`, which is
required for mmap accessor correctness and restart determinism.

## Identify Hole Guard

`vibrato-core::HNSW::search_subsequence_with_predicate` supports an explicit ID-validity predicate.
Production callers can reject deleted/tombstoned/hole IDs during subsequence verification, avoiding
false alignments across broken sequences.

