from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import requests

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.flight as paf
except Exception:  # pragma: no cover - import guarded at runtime
    pa = None
    pc = None
    paf = None

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None


DEFAULT_FLIGHT_CHUNK_ROWS = 4096
DEFAULT_FLIGHT_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass
class IngestResult:
    accepted: int
    created: int


class VibratoClient:
    """Vibrato V3 client with Flight-first ingest and HTTP query/identify APIs."""

    def __init__(
        self,
        http_url: str = "http://127.0.0.1:8080",
        *,
        api_key: str,
        flight_url: str | None = None,
        timeout_s: float = 30.0,
        flight_chunk_rows: int = DEFAULT_FLIGHT_CHUNK_ROWS,
        flight_chunk_bytes: int = DEFAULT_FLIGHT_CHUNK_BYTES,
    ) -> None:
        self.http_url = http_url.rstrip("/")
        self.api_key = api_key
        self.flight_url = flight_url
        self.timeout_s = timeout_s
        self.flight_chunk_rows = max(1, int(flight_chunk_rows))
        self.flight_chunk_bytes = max(1, int(flight_chunk_bytes))
        self._session = requests.Session()

    def query(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        ef: int = 50,
        include_metadata: bool = True,
        query_filter: Mapping[str, Any] | None = None,
        search_tier: str = "active",
    ) -> dict[str, Any]:
        """Run `/v3/query`."""
        payload = {
            "vector": [float(x) for x in vector],
            "k": int(k),
            "ef": int(ef),
            "include_metadata": bool(include_metadata),
            "filter": query_filter,
            "search_tier": search_tier,
        }
        return self._post_json("/v3/query", payload)["data"]

    def identify(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        k: int = 5,
        ef: int = 100,
        include_metadata: bool = True,
        search_tier: str = "all",
    ) -> dict[str, Any]:
        """Run `/v3/identify`."""
        payload = {
            "vectors": [[float(x) for x in row] for row in vectors],
            "k": int(k),
            "ef": int(ef),
            "include_metadata": bool(include_metadata),
            "search_tier": search_tier,
        }
        return self._post_json("/v3/identify", payload)["data"]

    def ingest(
        self,
        data: Any,
        *,
        entity_ids: Sequence[int] | None = None,
        sequence_ts: Sequence[int] | None = None,
        payloads: Sequence[bytes | str] | None = None,
        tags: Sequence[Sequence[str]] | None = None,
        idempotency_keys: Sequence[str | None] | None = None,
        prefer_flight: bool = True,
    ) -> IngestResult:
        """Ingest vectors from numpy/polars/pyarrow with Flight-first transport."""
        if prefer_flight and self.flight_url is not None:
            table = self._as_arrow_table(
                data,
                entity_ids=entity_ids,
                sequence_ts=sequence_ts,
                payloads=payloads,
                tags=tags,
                idempotency_keys=idempotency_keys,
            )
            return self._ingest_via_flight(table)
        return self._ingest_via_http(
            data,
            entity_ids=entity_ids,
            sequence_ts=sequence_ts,
            payloads=payloads,
            tags=tags,
            idempotency_keys=idempotency_keys,
        )

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._session.post(
            f"{self.http_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def _as_arrow_table(
        self,
        data: Any,
        *,
        entity_ids: Sequence[int] | None,
        sequence_ts: Sequence[int] | None,
        payloads: Sequence[bytes | str] | None,
        tags: Sequence[Sequence[str]] | None,
        idempotency_keys: Sequence[str | None] | None,
    ) -> "pa.Table":
        if pa is None:
            raise RuntimeError("pyarrow is required for Flight ingest")

        if isinstance(data, np.ndarray):
            return self._table_from_numpy(
                data,
                entity_ids=entity_ids,
                sequence_ts=sequence_ts,
                payloads=payloads,
                tags=tags,
                idempotency_keys=idempotency_keys,
            )
        if pa is not None and isinstance(data, pa.RecordBatch):
            return self._normalize_arrow_table(pa.Table.from_batches([data]))
        if pa is not None and isinstance(data, pa.Table):
            return self._normalize_arrow_table(data)
        if pl is not None and isinstance(data, pl.DataFrame):
            return self._normalize_arrow_table(data.to_arrow())
        raise TypeError(
            "unsupported ingest data type; expected numpy.ndarray, pyarrow.Table/RecordBatch, or polars.DataFrame"
        )

    def _table_from_numpy(
        self,
        vectors: np.ndarray,
        *,
        entity_ids: Sequence[int] | None,
        sequence_ts: Sequence[int] | None,
        payloads: Sequence[bytes | str] | None,
        tags: Sequence[Sequence[str]] | None,
        idempotency_keys: Sequence[str | None] | None,
    ) -> "pa.Table":
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected vectors shape (N, D), got {arr.shape}")
        nrows, dim = arr.shape
        flat = pa.array(arr.reshape(-1), type=pa.float32())
        vector_col = pa.FixedSizeListArray.from_arrays(flat, dim)
        entity_col = pa.array(
            list(entity_ids) if entity_ids is not None else list(range(nrows)),
            type=pa.uint64(),
        )
        seq_col = pa.array(
            list(sequence_ts) if sequence_ts is not None else list(range(nrows)),
            type=pa.uint64(),
        )
        payload_src = payloads if payloads is not None else [b""] * nrows
        payload_col = pa.array(
            [
                p if isinstance(p, (bytes, bytearray)) else str(p).encode("utf-8")
                for p in payload_src
            ],
            type=pa.binary(),
        )
        tags_src = tags if tags is not None else [[] for _ in range(nrows)]
        tags_col = self._dictionary_encode_tags(pa.array(tags_src, type=pa.list_(pa.utf8())))
        cols: dict[str, pa.Array] = {
            "vector": vector_col,
            "entity_id": entity_col,
            "sequence_ts": seq_col,
            "payload": payload_col,
            "tags": tags_col,
        }
        if idempotency_keys is not None:
            cols["idempotency_key"] = pa.array(idempotency_keys, type=pa.utf8())
        return pa.table(cols)

    def _normalize_arrow_table(self, table: "pa.Table") -> "pa.Table":
        if pa is None:
            raise RuntimeError("pyarrow is required for Flight ingest")
        nrows = table.num_rows
        names = set(table.column_names)
        cols: dict[str, pa.Array] = {}

        if "vector" not in names:
            raise ValueError("missing required 'vector' column")
        vector_col = table.column("vector").combine_chunks()
        vtype = vector_col.type
        if pa.types.is_fixed_size_list(vtype):
            if not pa.types.is_float32(vtype.value_type):
                vector_col = pc.cast(vector_col, pa.list_(pa.float32(), vtype.list_size))
        elif pa.types.is_list(vtype):
            lengths = pc.list_value_length(vector_col).to_pylist()
            if not lengths:
                raise ValueError("empty vector batch")
            dim = lengths[0]
            if dim is None or any(x != dim for x in lengths):
                raise ValueError("vector list column must have a fixed per-row dimension")
            values = pc.cast(vector_col.values, pa.float32())
            vector_col = pa.FixedSizeListArray.from_arrays(values, dim)
        else:
            raise ValueError(
                "column 'vector' must be FixedSizeList<Float32> or List<Float32>"
            )
        cols["vector"] = vector_col

        if "entity_id" in names:
            cols["entity_id"] = pc.cast(table.column("entity_id").combine_chunks(), pa.uint64())
        else:
            cols["entity_id"] = pa.array(list(range(nrows)), type=pa.uint64())
        if "sequence_ts" in names:
            cols["sequence_ts"] = pc.cast(table.column("sequence_ts").combine_chunks(), pa.uint64())
        else:
            cols["sequence_ts"] = pa.array(list(range(nrows)), type=pa.uint64())

        if "payload" in names:
            payload_col = table.column("payload").combine_chunks()
            if pa.types.is_binary(payload_col.type):
                cols["payload"] = payload_col
            elif pa.types.is_string(payload_col.type):
                cols["payload"] = pc.utf8_encode(payload_col)
            else:
                raise ValueError("column 'payload' must be Binary or Utf8")
        else:
            cols["payload"] = pa.array([b""] * nrows, type=pa.binary())

        if "tags" in names:
            tags_col = table.column("tags").combine_chunks()
            if pa.types.is_list(tags_col.type) and pa.types.is_dictionary(
                tags_col.type.value_type
            ):
                cols["tags"] = tags_col
            elif pa.types.is_list(tags_col.type) and pa.types.is_string(
                tags_col.type.value_type
            ):
                cols["tags"] = self._dictionary_encode_tags(tags_col)
            else:
                raise ValueError(
                    "column 'tags' must be List<Utf8> or List<Dictionary<*, Utf8>>"
                )
        else:
            cols["tags"] = self._dictionary_encode_tags(
                pa.array([[] for _ in range(nrows)], type=pa.list_(pa.utf8()))
            )

        if "idempotency_key" in names:
            cols["idempotency_key"] = pc.cast(
                table.column("idempotency_key").combine_chunks(), pa.utf8()
            )
        return pa.table(cols)

    def _dictionary_encode_tags(self, tags_col: "pa.Array") -> "pa.Array":
        if pa is None or pc is None:
            raise RuntimeError("pyarrow is required for Flight ingest")
        if not pa.types.is_list(tags_col.type):
            raise ValueError("tags column must be a List array")
        if pa.types.is_dictionary(tags_col.type.value_type):
            return tags_col
        if not pa.types.is_string(tags_col.type.value_type):
            raise ValueError("tags list value type must be Utf8")
        encoded_values = pc.dictionary_encode(tags_col.values)
        return pa.ListArray.from_arrays(tags_col.offsets, encoded_values)

    def _iter_table_chunks(self, table: "pa.Table") -> Iterable["pa.RecordBatch"]:
        if pa is None:
            raise RuntimeError("pyarrow is required for Flight ingest")
        if table.num_rows == 0:
            return
        total_rows = table.num_rows
        max_rows = max(1, self.flight_chunk_rows)
        max_bytes = max(1, self.flight_chunk_bytes)
        start = 0

        while start < total_rows:
            rows = min(max_rows, total_rows - start)
            batch = None
            while rows > 0:
                candidate = table.slice(start, rows).to_batches(max_chunksize=rows)[0]
                if candidate.nbytes <= max_bytes:
                    batch = candidate
                    break
                if rows == 1:
                    raise ValueError(
                        f"single row exceeds Flight chunk byte budget: "
                        f"row_bytes={candidate.nbytes} limit={max_bytes}"
                    )
                rows //= 2

            if batch is None:
                raise ValueError("failed to build Flight chunk under byte budget")

            yield batch
            start += rows

    def _ingest_via_flight(self, table: "pa.Table") -> IngestResult:
        if paf is None:
            raise RuntimeError("pyarrow.flight is required for Flight ingest")
        if self.flight_url is None:
            raise RuntimeError("flight_url is required for Flight ingest")
        if table.num_rows == 0:
            return IngestResult(accepted=0, created=0)

        client = paf.FlightClient(self.flight_url)
        options = paf.FlightCallOptions(
            headers=[(b"authorization", f"Bearer {self.api_key}".encode("utf-8"))]
        )
        accepted = 0
        created = 0

        descriptor = paf.FlightDescriptor.for_path("v3", "vectors")
        writer, reader = client.do_put(descriptor, table.schema, options=options)
        try:
            for batch in self._iter_table_chunks(table):
                writer.write_batch(batch)
        finally:
            # Ensure the stream is closed even when write_batch() raises.
            writer.done_writing()

        while True:
            metadata = reader.read()
            if metadata is None:
                break
            payload = json.loads(metadata.to_pybytes().decode("utf-8"))
            accepted += int(payload.get("accepted", 0))
            created += int(payload.get("created", 0))
        return IngestResult(accepted=accepted, created=created)

    def _ingest_via_http(
        self,
        data: Any,
        *,
        entity_ids: Sequence[int] | None,
        sequence_ts: Sequence[int] | None,
        payloads: Sequence[bytes | str] | None,
        tags: Sequence[Sequence[str]] | None,
        idempotency_keys: Sequence[str | None] | None,
    ) -> IngestResult:
        vectors = np.asarray(data, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("HTTP ingest fallback expects a 2D numpy array")
        nrows = vectors.shape[0]
        entity = list(entity_ids) if entity_ids is not None else list(range(nrows))
        seq = list(sequence_ts) if sequence_ts is not None else list(range(nrows))
        payload = list(payloads) if payloads is not None else [b""] * nrows
        tags_rows = list(tags) if tags is not None else [[] for _ in range(nrows)]
        idem = list(idempotency_keys) if idempotency_keys is not None else [None] * nrows

        rows = []
        for i in range(nrows):
            raw_payload = payload[i]
            payload_bytes = (
                raw_payload
                if isinstance(raw_payload, (bytes, bytearray))
                else str(raw_payload).encode("utf-8")
            )
            rows.append(
                {
                    "vector": vectors[i].astype(np.float32).tolist(),
                    "metadata": {
                        "entity_id": int(entity[i]),
                        "sequence_ts": int(seq[i]),
                        "tags": [str(t) for t in tags_rows[i]],
                        "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
                    },
                    "idempotency_key": idem[i],
                }
            )
        response = self._post_json("/v3/vectors/batch", {"vectors": rows})["data"]
        results = response.get("results", [])
        created = sum(1 for r in results if r.get("created"))
        return IngestResult(accepted=len(results), created=created)
