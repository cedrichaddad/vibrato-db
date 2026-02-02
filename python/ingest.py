#!/usr/bin/env python3
"""
Vibrato-DB Audio Ingestion CLI

Converts audio files to vector embeddings and writes them to .vdb format.

Usage:
    python ingest.py --input ./audio/ --output data.vdb
"""

import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

from vibrato.writer import VDBWriter, l2_normalize
from vibrato.audio import load_audio, extract_frames, compute_mel_spectrogram
from vibrato.embeddings import VGGishEncoder, RandomEncoder


@click.command()
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing audio files or single audio file",
)
@click.option(
    "--output", "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output .vdb file path",
)
@click.option(
    "--dim", "-d",
    default=128,
    type=int,
    help="Embedding dimension (default: 128)",
)
@click.option(
    "--frame-duration",
    default=0.96,
    type=float,
    help="Frame duration in seconds (default: 0.96)",
)
@click.option(
    "--sample-rate", "-sr",
    default=16000,
    type=int,
    help="Target sample rate (default: 16000)",
)
@click.option(
    "--use-vggish/--no-vggish",
    default=True,
    help="Use VGGish model (falls back to random if unavailable)",
)
@click.option(
    "--random-seed",
    default=None,
    type=int,
    help="Random seed for reproducible testing",
)
@click.option(
    "--extensions",
    default="mp3,wav,flac,ogg,m4a",
    help="Comma-separated list of audio file extensions",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(
    input_path: str,
    output_path: str,
    dim: int,
    frame_duration: float,
    sample_rate: int,
    use_vggish: bool,
    random_seed: Optional[int],
    extensions: str,
    verbose: bool,
):
    """Ingest audio files and create a .vdb vector database."""
    
    input_dir = Path(input_path)
    output_file = Path(output_path)
    ext_list = [f".{e.strip().lower()}" for e in extensions.split(",")]

    # Collect audio files
    if input_dir.is_file():
        audio_files = [input_dir]
    else:
        audio_files = [
            f for f in input_dir.rglob("*")
            if f.suffix.lower() in ext_list
        ]

    if not audio_files:
        click.echo(f"No audio files found in {input_path}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(audio_files)} audio files")

    # Initialize encoder
    if use_vggish:
        encoder = VGGishEncoder(use_vggish=True, embedding_dim=dim)
    else:
        encoder = RandomEncoder(embedding_dim=dim, seed=random_seed)
    
    # Create writer
    with VDBWriter(output_file, dimensions=dim) as writer:
        total_vectors = 0
        
        for i, audio_path in enumerate(audio_files):
            if verbose:
                click.echo(f"[{i+1}/{len(audio_files)}] Processing {audio_path.name}...")
            
            try:
                # Load and process audio
                audio, sr = load_audio(str(audio_path), sr=sample_rate)
                frames = extract_frames(audio, sr=sr, frame_duration=frame_duration)
                
                if verbose:
                    click.echo(f"  Extracted {len(frames)} frames")
                
                # Compute embeddings
                for frame in frames:
                    embedding = encoder.encode(frame, normalize=True)
                    writer.write(embedding)
                    total_vectors += 1
                    
            except Exception as e:
                click.echo(f"  WARNING: Failed to process {audio_path}: {e}", err=True)
                continue

        click.echo(f"Wrote {total_vectors} vectors to {output_file}")


if __name__ == "__main__":
    main()
