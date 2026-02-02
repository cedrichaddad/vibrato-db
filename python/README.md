# Vibrato-DB Python Package

Python ingestion pipeline for audio-to-vector conversion.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Ingest audio files to .vdb format
python ingest.py --input ./audio/ --output data.vdb

# Test with synthetic random vectors
python test_writer.py --output test.vdb --count 1000 --dim 128
```
