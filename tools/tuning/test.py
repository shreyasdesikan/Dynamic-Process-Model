import os
import torch
import numpy as np
import pandas as pd
import csv
from model import BidirectionalLSTMWithMultiHead
from run import clean_batch, window_batch_data, evaluate_model, predict_open_loop
from sklearn.preprocessing import StandardScaler

STATE_COLS = 6
CONTROL_COLS = 7
INPUT_DIM = STATE_COLS + CONTROL_COLS

def load_hparams(run_id, base_path="results"):
    hparam_file = os.path.join(base_path, str(run_id), "hyperparams.txt")
    hparams = {
        "hidden_dim": 128,
        "fc_dim": 64,
        "num_layers_lstm": 2,
        "window_size": 5
    }
    if os.path.exists(hparam_file):
        with open(hparam_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":")
                    key, val = key.strip(), val.strip()
                    if key in hparams:
                        hparams[key] = int(val)
    return hparams

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    test_file = os.path.join(os.path.dirname(__file__), "file_12738.txt")
    out_csv = os.path.join(os.path.dirname(__file__), "tuning", "beat_the_felix_summary.csv")

    # Load and clean test file
    cleaned = clean_batch(test_file)
    scaler = StandardScaler()
    scaler.fit(cleaned)  # Only this file, since all test is done here
    scaled = scaler.transform(cleaned)

    all_dirs = [d for d in os.listdir(results_dir) if d.isdigit()]
    all_dirs.sort()

    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "loop_type", "c_mse", "c_mae", "T_PM_mse", "T_PM_mae",
            "d50_mse", "d50_mae", "d90_mse", "d90_mae",
            "d10_mse", "d10_mae", "T_TM_mse", "T_TM_mae"
        ])

        for run_dir in all_dirs:
            run_id = int(run_dir)
            if run_id is None:
                continue

            hparams = load_hparams(run_id)
            model_path = os.path.join(results_dir, run_dir, f"model_cluster0_run{run_id}.pt")
            if not os.path.exists(model_path):
                print(f"Model not found for run {run_id}, skipping.")
                continue

            model = BidirectionalLSTMWithMultiHead(
                hidden_dim=hparams["hidden_dim"],
                fc_dim=hparams["fc_dim"],
                num_layers=hparams["num_layers_lstm"]
            )
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            # Create windowed data using this model’s window size
            window_size = hparams["window_size"]
            test_X, test_Y = window_batch_data(scaled, window_size)

            # Run evaluation
            closed_metrics = evaluate_model(model, test_X, test_Y, run_id=run_id, cluster_id=0, scaler=scaler)
            open_metrics = predict_open_loop(model, test_X, test_Y, window_size=window_size, input_dim=INPUT_DIM, device="cpu", scaler=scaler, run_id=run_id)

            for loop_type, metrics in [("closed", closed_metrics), ("open", open_metrics)]:
                row = [run_id, loop_type]
                for name in ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']:
                    row.append(metrics[f"{name}_mse"])
                    row.append(metrics[f"{name}_mae"])
                writer.writerow(row)

            print(f"Completed evaluation for run {run_id}")
            
    df = pd.read_csv(out_csv)
    closed_df = df[df["loop_type"] == "closed"]

    # Compute average MSE and MAE across all 6 states
    mse_cols = [col for col in closed_df.columns if col.endswith("_mse")]
    mae_cols = [col for col in closed_df.columns if col.endswith("_mae")]

    closed_df["MSE_avg"] = closed_df[mse_cols].mean(axis=1)
    closed_df["MAE_avg"] = closed_df[mae_cols].mean(axis=1)

    # Select best by lowest MSE_avg, break ties with MAE_avg
    best_row = closed_df.sort_values(["MSE_avg", "MAE_avg"]).iloc[0]

    print("\n🏆 Best model based on Closed Loop evaluation:")
    print(best_row[["run_id", "MSE_avg", "MAE_avg"]].to_string(index=False))

if __name__ == "__main__":
    main()
