import itertools
import subprocess
import os
import csv
import json

# Define hyperparameter grid
param_grid = {
    "epochs": [30, 50, 70, 90],
    "lr": [1e-4, 5e-4, 1e-3],
    "batch_size": [64],
    "window_size": [5, 10, 15, 20],
    "hidden_dim": [128],
    "fc_dim": [64],
    "num_layers_lstm": [2],
}

# Generate all combinations
param_combinations = list(itertools.product(*param_grid.values()))
print(f"Total runs to execute: {len(param_combinations)}")

# Prepare logging
results_csv = "tuning/tuning_results.csv"
os.makedirs("tuning", exist_ok=True)

with open(results_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_id", "epochs", "lr", "batch_size", "window_size",
        "hidden_dim", "fc_dim", "num_layers_lstm",
        "ClosedLoop_MSE_avg", "ClosedLoop_MAE_avg"
    ])

# Loop over runs
for i, combo in enumerate(param_combinations):
    param_dict = dict(zip(param_grid.keys(), combo))
    print(f"\nRunning combo {i + 1}/{len(param_combinations)}: {param_dict}")

    # Construct command
    cmd = [
        "python", "run.py",
        "--epochs", str(param_dict["epochs"]),
        "--lr", str(param_dict["lr"]),
        "--batch_size", str(param_dict["batch_size"]),
        "--window_size", str(param_dict["window_size"]),
        "--hidden_dim", str(param_dict["hidden_dim"]),
        "--fc_dim", str(param_dict["fc_dim"]),
        "--num_layers_lstm", str(param_dict["num_layers_lstm"])
    ]

    # Launch run
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Run {i + 1} failed:\n{result.stderr}")
        continue

    # Parse run ID and evaluation metrics from stdout
    try:
        metrics = json.loads(result.stdout.strip().splitlines()[-1])
        run_id = metrics["run_id"]
        mse_avg = metrics["ClosedLoop_MSE_avg"]
        mae_avg = metrics["ClosedLoop_MAE_avg"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Warning: Failed to parse JSON output for run {i + 1}: {e}")
        continue

    if run_id and mse_avg is not None:
        with open(results_csv, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id,
                param_dict["epochs"],
                param_dict["lr"],
                param_dict["batch_size"],
                param_dict["window_size"],
                param_dict["hidden_dim"],
                param_dict["fc_dim"],
                param_dict["num_layers_lstm"],
                mse_avg,
                mae_avg
            ])
    else:
        print(f"Warning: Failed to extract metrics for run {i + 1} (run_id={run_id})")

# After all runs, pick the best
import pandas as pd
df = pd.read_csv(results_csv)
best_row = df.loc[df["ClosedLoop_MSE_avg"].idxmin()]
print("\nBest model based on Closed Loop MSE:")
print(best_row.to_string(index=False))
