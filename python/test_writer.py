#!/usr/bin/env python3
"""
Test script for VDB writer with synthetic random vectors.

Usage:
    python test_writer.py --output test.vdb --count 1000 --dim 128
"""

import click
import numpy as np

from vibrato.writer import VDBWriter, read_vdb_header, l2_normalize


@click.command()
@click.option(
    "--output", "-o",
    default="test.vdb",
    help="Output .vdb file path",
)
@click.option(
    "--count", "-n",
    default=1000,
    type=int,
    help="Number of random vectors to generate",
)
@click.option(
    "--dim", "-d",
    default=128,
    type=int,
    help="Vector dimension",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="L2 normalize vectors",
)
def main(output: str, count: int, dim: int, seed: int, normalize: bool):
    """Generate a .vdb file with random vectors for testing."""
    
    click.echo(f"Generating {count} random vectors of dimension {dim}...")
    
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    
    if normalize:
        vectors = l2_normalize(vectors)
        click.echo("Vectors L2 normalized")
    
    # Write to file
    with VDBWriter(output, dimensions=dim) as writer:
        writer.write_batch(vectors)
    
    # Verify
    header = read_vdb_header(output)
    click.echo(f"Created {output}:")
    click.echo(f"  Vectors: {header['count']}")
    click.echo(f"  Dimensions: {header['dimensions']}")
    
    # Check file size
    import os
    file_size = os.path.getsize(output)
    expected_size = 16 + count * dim * 4
    click.echo(f"  File size: {file_size} bytes (expected: {expected_size})")
    
    if file_size == expected_size:
        click.echo("  ✓ File size matches expected")
    else:
        click.echo("  ✗ File size mismatch!", err=True)


if __name__ == "__main__":
    main()
