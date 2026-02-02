#!/usr/bin/env python3
"""
2D HNSW Graph Visualization

This script exports HNSW edges to CSV for visualization.
Use with matplotlib to verify graph structure.

Usage:
    1. Generate edges from Rust: cargo run --example visualize_2d > graph.csv
    2. Run this script: python visualize.py graph.csv
"""

import sys
from pathlib import Path
import click


@click.command()
@click.argument("csv_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output image file")
def main(csv_file: str, output: str):
    """Visualize HNSW graph from CSV edge list."""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        click.echo("matplotlib and numpy required: pip install matplotlib numpy", err=True)
        sys.exit(1)
    
    # Read CSV: expected format is "x1,y1,x2,y2" per line (edge from (x1,y1) to (x2,y2))
    # Or "id,x,y,layer" for nodes
    edges = []
    nodes = {}
    
    with open(csv_file, "r") as f:
        header = f.readline().strip()
        
        if "layer" in header.lower():
            # Node format: id,x,y,layer
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    layer = int(parts[3]) if len(parts) > 3 else 0
                    nodes[node_id] = (x, y, layer)
        else:
            # Edge format: x1,y1,x2,y2
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    x1, y1 = float(parts[0]), float(parts[1])
                    x2, y2 = float(parts[2]), float(parts[3])
                    edges.append(((x1, y1), (x2, y2)))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if edges:
        click.echo(f"Plotting {len(edges)} edges...")
        for (x1, y1), (x2, y2) in edges:
            ax.plot([x1, x2], [y1, y2], "b-", alpha=0.3, linewidth=0.5)
    
    if nodes:
        click.echo(f"Plotting {len(nodes)} nodes...")
        max_layer = max(n[2] for n in nodes.values())
        colors = plt.cm.viridis(np.linspace(0, 1, max_layer + 1))
        
        for node_id, (x, y, layer) in nodes.items():
            size = 20 + layer * 30  # Higher layers = bigger dots
            ax.scatter(x, y, c=[colors[layer]], s=size, alpha=0.7, zorder=10)
    
    ax.set_aspect("equal")
    ax.set_title("HNSW Graph Structure")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        click.echo(f"Saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
