#!/usr/bin/env python3
"""
TensorBoard Log Parser for Ghost Dog Training

Parses binary .tfevents files from TensorBoard logs and outputs
key metrics in readable JSON format.

Usage:
    python parse_logs.py                    # Lists available runs
    python parse_logs.py PPO_3              # Parse specific run
    python parse_logs.py PPO_3 --output metrics.json  # Save to file
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard is not installed!")
    print("Install it with: pip install tensorboard")
    sys.exit(1)


def list_available_runs(log_dir: str) -> List[str]:
    """List all available runs in the TensorBoard log directory."""
    if not os.path.exists(log_dir):
        return []

    runs = []
    for item in os.listdir(log_dir):
        item_path = os.path.join(log_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains .tfevents files
            tfevents_files = [f for f in os.listdir(item_path) if 'tfevents' in f]
            if tfevents_files:
                runs.append(item)

    return sorted(runs)


def parse_tensorboard_logs(log_dir: str, run_name: str) -> Dict:
    """
    Parse TensorBoard logs for a specific run.

    Args:
        log_dir: Base directory containing TensorBoard logs
        run_name: Name of the specific run to parse (e.g., "PPO_3")

    Returns:
        Dictionary containing parsed metrics
    """
    run_path = os.path.join(log_dir, run_name)

    if not os.path.exists(run_path):
        raise ValueError(f"Run '{run_name}' not found in {log_dir}")

    # Create event accumulator
    ea = event_accumulator.EventAccumulator(
        run_path,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 = load all
        }
    )

    print(f"Loading events from {run_path}...", file=sys.stderr)
    ea.Reload()

    # Get all available scalar tags
    available_tags = ea.Tags()['scalars']
    print(f"Found {len(available_tags)} metric tags", file=sys.stderr)

    # Define key metrics to extract (matches what's logged in ghostdog_gym.py)
    key_metrics = [
        'rollout/ep_len_mean',      # Episode length
        'rollout/ep_rew_mean',      # Episode reward
        'time/fps',                  # Training speed
        'time/iterations',           # Number of iterations
        'time/time_elapsed',         # Total training time
        'time/total_timesteps',      # Total timesteps
        'train/approx_kl',           # KL divergence
        'train/clip_fraction',       # Fraction of clipped samples
        'train/clip_range',          # Clipping range
        'train/entropy_loss',        # Entropy loss
        'train/explained_variance',  # Explained variance
        'train/learning_rate',       # Learning rate
        'train/loss',                # Total loss
        'train/n_updates',           # Number of updates
        'train/policy_gradient_loss',# Policy gradient loss
        'train/std',                 # Standard deviation
        'train/value_loss',          # Value loss
    ]

    # Extract data for each metric
    metrics_data = {}
    for tag in key_metrics:
        if tag in available_tags:
            events = ea.Scalars(tag)
            metrics_data[tag] = [
                {
                    'step': event.step,
                    'value': event.value,
                    'wall_time': event.wall_time
                }
                for event in events
            ]
            print(f"  ✓ {tag}: {len(events)} entries", file=sys.stderr)
        else:
            print(f"  ✗ {tag}: not found", file=sys.stderr)

    # Calculate summary statistics
    summary = {}
    for tag, data in metrics_data.items():
        if data:
            values = [entry['value'] for entry in data]
            summary[tag] = {
                'count': len(values),
                'first': values[0],
                'last': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
            }

    return {
        'run_name': run_name,
        'run_path': run_path,
        'total_metrics': len(metrics_data),
        'summary': summary,
        'detailed_metrics': metrics_data
    }


def main():
    parser = argparse.ArgumentParser(
        description='Parse TensorBoard logs for Ghost Dog training analysis'
    )
    parser.add_argument(
        'run_name',
        nargs='?',
        help='Name of the run to parse (e.g., PPO_3, PPO_18). Leave empty to list runs.'
    )
    parser.add_argument(
        '--log-dir',
        default='./controllers/ghostdog_gym/ghostdog_tensorboard',
        help='Path to TensorBoard log directory (default: ./controllers/ghostdog_gym/ghostdog_tensorboard)'
    )
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: print to stdout)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only output summary statistics, not full detailed metrics'
    )

    args = parser.parse_args()

    # List runs if no run name provided
    if not args.run_name:
        print("\nAvailable runs in {}:".format(args.log_dir))
        runs = list_available_runs(args.log_dir)

        if not runs:
            print("  No runs found!")
            print(f"\nMake sure the log directory exists and contains training runs.")
            return

        for run in runs:
            print(f"  - {run}")

        print(f"\nUsage: python {sys.argv[0]} <run_name>")
        print(f"Example: python {sys.argv[0]} {runs[0]}")
        return

    # Parse the specified run
    try:
        data = parse_tensorboard_logs(args.log_dir, args.run_name)

        # Remove detailed metrics if summary-only
        if args.summary_only:
            data.pop('detailed_metrics', None)

        # Output results
        json_output = json.dumps(data, indent=2)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"\nMetrics saved to {args.output}", file=sys.stderr)
        else:
            print(json_output)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
