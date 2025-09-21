import os
import argparse
import time
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from model import BidirectionalLSTMWithMultiHead, get_loss, get_optimizer

STATE_NAMES = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
STATE_COLS = 6
CONTROL_COLS = 7

class NARXDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)
        # self.X = torch.from_numpy(X).float()
        # self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_run_paths(run_id):
    base_dir = os.path.dirname(__file__)
    root = os.path.join(base_dir, f"results/{run_id}")
    os.makedirs(root, exist_ok=True)
    return {
        "root": root,
        "model": os.path.join(root, f"model_cluster0_run{run_id}.pt"),
        "loss_plot": os.path.join(root, f"loss_plot_run{run_id}.png"),
        "log_file": os.path.join(root, "test_results_log.txt"),
        "hparam_log": os.path.join(root, "hyperparams.txt")
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

def split_data(files, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    np.random.shuffle(files)
    split = int(len(files) * (1 - test_ratio))
    return files[:split], files[split:]

def log_hyperparams(run_id, args, cluster_id):
    paths = get_run_paths(run_id)
    with open(paths["hparam_log"], "a") as f:
        f.write(f"\n===== Run ID: {run_id} | Cluster {cluster_id} =====\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

def plot_losses(train_losses, val_losses, run_id=None, log_hyperparams=True, log_scale=False, model_type="ann"):
    paths = get_run_paths(run_id)
    plt.figure()
    # plt.plot(train_losses, label="Train")
    plt.plot(range(2, len(train_losses)+1), train_losses[1:], label="Train")
    plt.plot(range(2, len(train_losses)+1), val_losses[1:], label="Validation")
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss - {model_type.upper()}")
    plt.legend()
    plt.grid(True)
    if log_hyperparams and run_id is not None:
        plt.savefig(paths["loss_plot"])
    else:
        plt.show()
    plt.close()

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


def log_to_summary_csv(run_id, args, metrics_dict):
    csv_path = os.path.join(os.path.dirname(__file__), "tuning", "results_summary.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "run_id", "epochs", "lr", "batch_size", "window_size", "hidden_dim", "fc_dim", "num_layers_lstm",
            "loop_type", "c_mse", "c_mae", "T_PM_mse", "T_PM_mae",
            "d50_mse", "d50_mae", "d90_mse", "d90_mae",
            "d10_mse", "d10_mae", "T_TM_mse", "T_TM_mae"
        ])
        if not file_exists:
            writer.writeheader()

        for loop_type in ["closed", "open"]:
            writer.writerow({
                "run_id": run_id,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "window_size": args.window_size,
                "hidden_dim": args.hidden_dim,
                "fc_dim": args.fc_dim,
                "num_layers_lstm": args.num_layers_lstm,
                "loop_type": loop_type,
                **metrics_dict[loop_type]
            })


def train_cluster(cluster_id, cluster_path, args, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".txt")]
    train_files, test_files = split_data(all_files, test_ratio=0.1)
    
    train_cleaned = []
    for file in train_files:
        cleaned = clean_batch(file)
        train_cleaned.append(cleaned)
    
    # Step 2: Fit scaler on all cleaned data concatenated
    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_cleaned, axis=0))  # ← fit once across all training data
    
    # Step 3: Apply scaling and windowing per file
    all_train_X, all_train_Y = [], []
    for cleaned in train_cleaned:
        scaled = scaler.transform(cleaned)
        X, Y= window_batch_data(scaled, args.window_size)
        all_train_X.append(X)
        all_train_Y.append(Y)
    train_X = np.concatenate(all_train_X, axis=0)
    train_Y = np.concatenate(all_train_Y, axis=0)
    
    # Same for test data
    test_cleaned= []
    for file in test_files:
        cleaned = clean_batch(file)
        test_cleaned.append(cleaned)
    
    all_test_X, all_test_Y = [], []
    for cleaned in test_cleaned:
        scaled = scaler.transform(cleaned)  # ← use previously fit scaler
        X, Y = window_batch_data(scaled, args.window_size)
        all_test_X.append(X)
        all_test_Y.append(Y)
    test_X = np.concatenate(all_test_X, axis=0)
    test_Y = np.concatenate(all_test_Y, axis=0)

    model = BidirectionalLSTMWithMultiHead(hidden_dim=args.hidden_dim, fc_dim=args.fc_dim, num_layers=args.num_layers_lstm).to(device)
    criterion = get_loss(use_huber=True)
    optimizer = get_optimizer(model, lr=args.lr, use_adamw=True)

    train_losses, val_losses = [], []
    num_epochs = args.epochs
    
    Xtr, Xvl, ytr, yvl = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)
    
    train_loader = DataLoader(NARXDataset(Xtr, ytr), batch_size=args.batch_size)
    val_loader = DataLoader(NARXDataset(Xvl, yvl), batch_size=args.batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()
            val_losses.append(val_loss / len(val_loader))

    log_hyperparams(run_id, args, cluster_id)
    paths = get_run_paths(run_id)
    torch.save(model.state_dict(), paths["model"])

    plot_losses(train_losses, val_losses, run_id=run_id, model_type="bilstm_multihead")

    metrics = {}
    metrics["closed"] = evaluate_model(model, test_X, test_Y, run_id=run_id, cluster_id=cluster_id, scaler=scaler)
    metrics["open"] = predict_open_loop(model, test_X, test_Y, window_size=args.window_size, input_dim=CONTROL_COLS+STATE_COLS, device=device, scaler=scaler, run_id=run_id)
    
    log_to_summary_csv(run_id, args, metrics)
    
    print(json.dumps({
    "run_id": run_id,
    "ClosedLoop_MSE_avg": float(np.mean(list(metrics["closed"].values())[:6])),
    "ClosedLoop_MAE_avg": float(np.mean(list(metrics["closed"].values())[6:]))
    }))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--fc_dim", type=int, default=64)
    parser.add_argument("--num_layers_lstm", type=int, default=2)
    args = parser.parse_args()

    base_path = "../Data/clustered"
    run_id = int(time.time())

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cluster_path = os.path.join(PROJECT_ROOT, "Data", "clustered", "cluster0")
    
    train_cluster(cluster_id=0, cluster_path=cluster_path, args=args, run_id=run_id)

if __name__ == "__main__":
    main()
