import os
import torch
import numpy as np
import pandas as pd
import csv
from model import BidirectionalLSTMWithMultiHead
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

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

def get_run_paths(run_id):
    base_dir = os.path.dirname(__file__)
    return {
        "model": os.path.join(base_dir, f"model_cluster0_run{run_id}.pt"),
        "loss_plot": os.path.join(base_dir, f"loss_plot_run{run_id}.png"),
        "log_file": os.path.join(base_dir, "test_results_log.txt"),
        "hparam_log": os.path.join(base_dir, "hyperparams.txt")
    }

def clean_and_filter(data, z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1, sm_filt=True):
    """Applies stacked Z-score filtering on d10, d50, d90 columns (indices 2, 3, 4)."""
    def apply_zscore_filter(subset, z_thresh):
        cleaned = subset.copy()
        states = cleaned[:, :STATE_COLS]
        for col in [2, 3, 4]:  # d50, d90, d10
            col_data = states[:, col]
            mean = np.mean(col_data)
            std = np.std(col_data)
            z = (col_data - mean) / std
            spike_mask = np.abs(z) > z_thresh
            cleaned[spike_mask, col] = mean
        return cleaned

    # First pass
    first_clean = apply_zscore_filter(data, z_thresh1)
    
    # Second pass
    second_clean = apply_zscore_filter(first_clean, z_thresh2)
    
    if not sm_filt:
        return first_clean, second_clean, second_clean
    
    lowpass_filtered = second_clean.copy()
    b, a = butter(N=2, Wn=cutoff, btype='low', fs=1.0)
    for col in range(6):  # STATE_COLS = 6
        lowpass_filtered[:, col] = filtfilt(b, a, second_clean[:, col])

    return first_clean, second_clean, lowpass_filtered

def clean_batch(file_path, sm_filt=True):
    raw = np.loadtxt(file_path, skiprows=1)
    _, second_zthresh, cleaned = clean_and_filter(raw, z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1, sm_filt=sm_filt)
    return cleaned  # filtered-cleaned data

def window_batch_data(scaled_data, window_size):
    X, Y = [], []
    for t in range(len(scaled_data) - window_size):
        window = scaled_data[t:t + window_size, :]
        target = scaled_data[t + window_size, :STATE_COLS]

        X.append(window.flatten())
        Y.append(target)
        
    return np.array(X), np.array(Y)

def evaluate_model(model, test_X, test_Y, run_id=None, cluster_id=None, scaler=None):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
        scaled_preds = model(test_tensor).cpu().numpy()

        # === Inverse Transform ===
        padded_preds = np.hstack([scaled_preds, np.zeros((scaled_preds.shape[0], CONTROL_COLS))])
        padded_truth = np.hstack([test_Y, np.zeros((test_Y.shape[0], CONTROL_COLS))])

        unscaled_preds = scaler.inverse_transform(padded_preds)[:, :STATE_COLS]
        unscaled_truth = scaler.inverse_transform(padded_truth)[:, :STATE_COLS]

        # === Metrics ===
        mse_unscaled = np.mean((unscaled_preds - unscaled_truth) ** 2, axis=0)
        mae_unscaled = np.mean(np.abs(unscaled_preds - unscaled_truth), axis=0)

        state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

        # === Logging ===
        if run_id is not None:
            paths = get_run_paths(run_id)
            with open(paths["log_file"], "a") as f:
                f.write(f"Run ID: {run_id}, Cluster {cluster_id}, Model - bilstm_multihead - Test MSE and MAE (preprocessed unscaled):\n")
                f.write("Closed Loop:\n")
                for name, mse, mae in zip(state_names, mse_unscaled, mae_unscaled):
                    f.write(f"{name:>5s} | MSE: {mse:.6e}, MAE: {mae:.6e}\n")
                f.write("\n")
    
        return {f"{state}_mse": mse for state, mse in zip(state_names, mse_unscaled)} | \
           {f"{state}_mae": mae for state, mae in zip(state_names, mae_unscaled)}



def predict_open_loop(model, test_X, test_Y, window_size=5, input_dim=13, device='cpu', scaler=None, run_id=None):
    """
    Performs open-loop prediction and reports per-state metrics.
    """

    model.eval()
    with torch.no_grad():
        total_steps = test_X.shape[0]
        windows_per_file = 1000 - window_size
        num_files = total_steps // windows_per_file

        test_X_files = test_X.reshape(num_files, windows_per_file, -1)
        test_Y_files = test_Y.reshape(num_files, windows_per_file, -1)

        mse_list, mae_list = [], []

        for file_idx in range(num_files):
            file_X = test_X_files[file_idx]
            file_Y = test_Y_files[file_idx]

            current_window = file_X[0].copy().reshape(1, -1)
            preds = []

            for step in range(windows_per_file):
                input_tensor = torch.tensor(current_window, dtype=torch.float32).to(device)
                pred = model(input_tensor).cpu().numpy()[0]
                preds.append(pred)

                if step + 1 >= windows_per_file:
                    break

                current_window_matrix = current_window.reshape(window_size, input_dim)
                next_window_matrix = np.zeros_like(current_window_matrix)
                next_window_matrix[:-1] = current_window_matrix[1:]

                next_controls = file_X[step + 1].reshape(window_size, input_dim)[-1, 6:]
                new_row = np.concatenate([pred, next_controls])
                next_window_matrix[-1] = new_row

                current_window = next_window_matrix.reshape(1, -1)

            preds_np = np.array(preds)
            targets_np = file_Y[:len(preds)]

            preds_unscaled = scaler.inverse_transform(
                np.hstack([preds_np, np.zeros((preds_np.shape[0], input_dim - preds_np.shape[1]))])
            )[:, :preds_np.shape[1]]

            truth_unscaled = scaler.inverse_transform(
                np.hstack([targets_np, np.zeros((targets_np.shape[0], input_dim - targets_np.shape[1]))])
            )[:, :targets_np.shape[1]]

            mse = np.mean((preds_unscaled - truth_unscaled) ** 2, axis=0)
            mae = np.mean(np.abs(preds_unscaled - truth_unscaled), axis=0)

            mse_list.append(mse)
            mae_list.append(mae)

        avg_mse_filt = np.mean(mse_list, axis=0)
        avg_mae_filt = np.mean(mae_list, axis=0)

        state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
            
        paths = get_run_paths(run_id)

        with open(paths["log_file"], "a") as f:
            f.write("Open Loop:\n")
            for i, state in enumerate(state_names):
                f.write(f"{state:>5s} | MSE: {avg_mse_filt[i]:.6e}, MAE: {avg_mae_filt[i]:.6e}\n")
            f.write("\n")
    
        return {f"{state}_mse": mse for state, mse in zip(state_names, avg_mse_filt)} | \
           {f"{state}_mae": mae for state, mae in zip(state_names, avg_mae_filt)}
           
def display_results(csv_path="beat_the_felix_summary.csv"):
    """
    Display closed and open loop test results from the summary CSV,
    formatted clearly in separate blocks.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        print("No results found.")
        return

    # Group by loop_type
    closed = df[df["loop_type"] == "closed"]
    open_ = df[df["loop_type"] == "open"]

    if closed.empty or open_.empty:
        print("Expected both closed and open loop results, but one is missing.")
        return

    run_id = int(closed.iloc[0]["run_id"])
    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

    print(f"\n===== Results for Run ID: {run_id} =====\n")

    def print_loop_results(title, data_row):
        print(f"{title} Loop Results:")
        print(f"{'State':>6} | {'MSE':>12} | {'MAE':>12}")
        print("-" * 35)
        for name in state_names:
            mse = data_row[f"{name}_mse"]
            mae = data_row[f"{name}_mae"]
            print(f"{name:>6} | {mse:12.6e} | {mae:12.6e}")
        print()

    print_loop_results("Closed", closed.iloc[0])
    print_loop_results("Open", open_.iloc[0])

def main():
    run_id = 1752708457
    test_file = os.path.join(os.path.dirname(__file__), "file_12738.txt")
    out_csv = os.path.join(os.path.dirname(__file__), "beat_the_felix_summary.csv")

    # Load and clean test file
    cleaned = clean_batch(test_file)
    scaler = StandardScaler()
    scaler.fit(cleaned)  # Only this file, since all test is done here
    scaled = scaler.transform(cleaned)

    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "loop_type", "c_mse", "c_mae", "T_PM_mse", "T_PM_mae",
            "d50_mse", "d50_mae", "d90_mse", "d90_mae",
            "d10_mse", "d10_mae", "T_TM_mse", "T_TM_mae"
        ])

        hparams = load_hparams(run_id)
        
        model_path = os.path.join(os.path.dirname(__file__), f"model_cluster0_run{run_id}.pt")
        if not os.path.exists(model_path):
            print(f"Model not found for run {run_id}, skipping.")
            return

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
    
    display_results(csv_path="beat_the_felix_summary.csv")
        
if __name__ == "__main__":
    main()