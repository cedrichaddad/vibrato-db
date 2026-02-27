# Vibrato-DB Python Package

Python SDK for Vibrato V3.

## Installation

```bash
pip install vibrato-db
```

## Usage

### Flight-first ingest (zero-config for numpy / polars / pyarrow)

```python
import numpy as np
from vibrato import VibratoClient

client = VibratoClient(
    http_url="http://127.0.0.1:8080",
    flight_url="grpc://127.0.0.1:8815",
    api_key="YOUR_API_KEY",
)

vectors = np.random.randn(1024, 128).astype("float32")
result = client.ingest(vectors)
print(result)
```

### Query

```python
query = vectors[0]
resp = client.query(query, k=10, ef=50)
print(resp["results"][0])
```

### Legacy local `.vdb` writer

```bash
# Ingest audio files to .vdb format
python ingest.py --input ./audio/ --output data.vdb

# Test with synthetic random vectors
python test_writer.py --output test.vdb --count 1000 --dim 128
```
