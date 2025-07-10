import os
import argparse
import time
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from models.ANN.narx_model import get_model, get_loss, get_optimizer

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

def clean_and_filter(data, z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1, test=False):
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
    
    if test:
        return None, None, second_clean
    
    lowpass_filtered = second_clean.copy()
    b, a = butter(N=2, Wn=cutoff, btype='low', fs=1.0)
    for col in range(6):  # STATE_COLS = 6
        lowpass_filtered[:, col] = filtfilt(b, a, second_clean[:, col])

    return first_clean, second_clean, lowpass_filtered

def load_batch_data(file_path, window_size, scaler=None, test=False):
    raw = np.loadtxt(file_path, skiprows=1)
    _1, _2, cleaned = clean_and_filter(raw, z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1, test=test)
    scaled = scaler.fit_transform(cleaned) if scaler else cleaned

    X, Y = [], []
    for t in range(len(scaled) - window_size):
        window = scaled[t:t + window_size, :]
        target = scaled[t + window_size, :STATE_COLS]
        X.append(window.flatten())
        Y.append(target)

    return np.array(X), np.array(Y)

def create_batches_custom(all_X, all_Y, batch_size):
    total_windows = min([len(x) for x in all_X])
    num_batches = (total_windows + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        batch_X, batch_Y = [], []
        for x, y in zip(all_X, all_Y):
            start = i * batch_size
            end = min(start + batch_size, len(x))
            batch_X.extend(x[start:end])
            batch_Y.extend(y[start:end])
        if batch_X:
            batch_tensor = (
                torch.from_numpy(np.array(batch_X)).float(),
                torch.from_numpy(np.array(batch_Y)).float(),
            )
            batches.append(batch_tensor)

    return batches

def split_data(files, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    np.random.shuffle(files)
    split = int(len(files) * (1 - test_ratio))
    return files[:split], files[split:]

def log_hyperparams(run_id, args, cluster_id):
    path = f"../results/ann/hyperparams.txt"
    with open(path, "a") as f:
        f.write(f"\n===== Run ID: {run_id} | Cluster {cluster_id} =====\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

def plot_losses(train_losses, val_losses, run_id=None, log_hyperparams=False, log_scale=False, model_type="ann"):
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
        plt.savefig(f"../results/ann/loss_plot_run{run_id}.png")
    else:
        plt.show()
    plt.close()

def plot_sample_prediction(model, scaler, test_files, window_size, run_id, cluster_id):
    os.makedirs(f"../results/ann/{run_id}", exist_ok=True)

    # Pick a random test file
    sample_file = random.choice(test_files)
    _, second_zthresh, cleaned_data = clean_and_filter(np.loadtxt(sample_file, skiprows=1), z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1, test=False)

    # Prepare input/output using existing logic
    scaled_data = scaler.fit_transform(cleaned_data)
    X, Y = [], []
    for t in range(len(scaled_data) - window_size):
        window = scaled_data[t:t + window_size, :]
        target = scaled_data[t + window_size, :STATE_COLS]
        X.append(window.flatten())
        Y.append(target)

    X = torch.tensor(np.array(X), dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        Y_pred = model(X).cpu().numpy()

    # Pad and inverse-transform
    def unscale(arr):
        pad = np.zeros((arr.shape[0], CONTROL_COLS))
        padded = np.hstack([arr, pad])
        return scaler.inverse_transform(padded)[:, :STATE_COLS]

    Y_true_unscaled = unscale(np.array(Y))
    Y_pred_unscaled = unscale(Y_pred)

    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

    for i, state in enumerate(state_names):
        plt.figure()
        plt.plot(second_zthresh[:, i], label="Unfiltered")
        # plt.plot(Y_true_unscaled[:, i], label="True")
        plt.plot(Y_pred_unscaled[:, i], label="Predicted", linestyle="--")
        plt.title(f"{state} - Cluster {cluster_id} - Run {run_id}")
        plt.xlabel("Timestep")
        plt.ylabel(state)
        plt.legend()
        plt.grid(True)
        save_path = f"../results/ann/{run_id}/state_{state}_cluster{cluster_id}.png"
        plt.savefig(save_path)
        plt.close()

def evaluate_model(model, test_X, test_Y, run_id=None, cluster_id=None, scaler=None):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
        scaled_preds = model(test_tensor).cpu().numpy()
        scaled_truth = test_Y

        # Scaled MSE
        mse_scaled = np.mean((scaled_preds - scaled_truth) ** 2, axis=0)

        # Unscaled MSE (only on state columns)
        padded_preds = np.hstack([scaled_preds, np.zeros((scaled_preds.shape[0], CONTROL_COLS))])
        padded_truth = np.hstack([scaled_truth, np.zeros((scaled_truth.shape[0], CONTROL_COLS))])
        unscaled_preds = scaler.inverse_transform(padded_preds)[:, :STATE_COLS]
        unscaled_truth = scaler.inverse_transform(padded_truth)[:, :STATE_COLS]
        mse_unscaled = np.mean((unscaled_preds - unscaled_truth) ** 2, axis=0)

        state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

        print("Test MSE per state (scaled):")
        for name, m in zip(state_names, mse_scaled):
            print(f"{name}: {m:.6e}")

        print("\nTest MSE per state (unscaled):")
        for name, m in zip(state_names, mse_unscaled):
            print(f"{name}: {m:.6e}")

        if run_id is not None:
            with open(f"../results/ann/test_results_log.txt", "a") as f:
                f.write(f"Run ID: {run_id}, Cluster {cluster_id} - Test MSE (scaled):\n")
                for name, m in zip(state_names, mse_scaled):
                    f.write(f"{name}: {m:.6e}\n")
                f.write(f"Run ID: {run_id}, Cluster {cluster_id} - Test MSE (unscaled):\n")
                for name, m in zip(state_names, mse_unscaled):
                    f.write(f"{name}: {m:.6e}\n")
                f.write("\n")

def predict_open_loop(model, test_X, test_Y, window_size=5, input_dim=13, device='cpu', scaler=None, run_id=None):
    """
    Performs open-loop prediction and reports per-state metrics.

    Parameters:
    - model: Trained model
    - initial_window: (window_size, 13) numpy array of recent scaled [state+control]
    - windows_per_file: Number of windows to predict in each file
    - window_size: Size of the sliding input window
    - device: Device to run on
    - scaler: Optional scaler to inverse-transform predictions
    - run_id, cluster_id: Optional logging identifiers

    Returns:
    - preds_all: (steps, 6) numpy array of predicted state values
    """
    
    model.eval()
    with torch.no_grad():
        total_steps = test_X.shape[0]
        windows_per_file = 1000 - window_size
        num_files = total_steps // windows_per_file

        # Reshape test_X and test_Y by file
        test_X_files = test_X.reshape(num_files, windows_per_file, -1)
        test_Y_files = test_Y.reshape(num_files, windows_per_file, -1)

        mse_list = []
        mae_list = []

        for file_idx in range(num_files):
            file_X = test_X_files[file_idx]  # (windows_per_file, input_dim * window_size)
            file_Y = test_Y_files[file_idx]  # (windows_per_file, output_dim)
            
            current_window = file_X[0].copy().reshape(1, -1)  # shape: (1, window_size * input_dim)
            preds = []
            
            for step in range(windows_per_file):
                # Predict next state
                input_tensor = torch.tensor(current_window, dtype=torch.float32).to(device)
                pred = model(input_tensor).cpu().numpy()[0]  # shape: (output_dim,)
                preds.append(pred)

                if step + 1 >= windows_per_file:
                    break  # No further window possible

                # Prepare next window:
                current_window_matrix = current_window.reshape(window_size, input_dim)

                # Slide: remove first row, add new row
                next_window_matrix = np.zeros_like(current_window_matrix)
                next_window_matrix[:-1] = current_window_matrix[1:]  # shift

                # Combine predicted state with original control inputs from next timestep
                next_controls = file_X[step + 1].reshape(window_size, input_dim)[-1, 6:]  # last timestep inputs
                new_row = np.concatenate([pred, next_controls])
                next_window_matrix[-1] = new_row

                # Flatten for next iteration
                current_window = next_window_matrix.reshape(1, -1)

            preds_np = np.array(preds)
            targets_np = file_Y[:len(preds)]

            # Inverse transform predictions and targets
            preds_unscaled = scaler.inverse_transform(
                np.hstack([preds_np, np.zeros((preds_np.shape[0], input_dim - preds_np.shape[1]))])
            )[:, :preds_np.shape[1]]

            targets_unscaled = scaler.inverse_transform(
                np.hstack([targets_np, np.zeros((targets_np.shape[0], input_dim - targets_np.shape[1]))])
            )[:, :targets_np.shape[1]]

            mse = np.mean((preds_unscaled - targets_unscaled) ** 2, axis=0)
            mae = np.mean(np.abs(preds_unscaled - targets_unscaled), axis=0)

            mse_list.append(mse)
            mae_list.append(mae)

        avg_mse = np.mean(mse_list, axis=0)
        avg_mae = np.mean(mae_list, axis=0)

        state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

        print(f"\nAverage Test MSE and MAE per State (Open Loop, Run ID: {run_id}):")
        for i, state in enumerate(state_names):
            print(f"{state:>5s} | MSE: {avg_mse[i]:.6e}, MAE: {avg_mae[i]:.6e}")

        with open("../results/ann/test_results_log.txt", "a") as f:
            f.write(f"\nAverage Test MSE and MAE per State (Open Loop, Run ID: {run_id}):\n")
            for i, state in enumerate(state_names):
                f.write(f"{state:>5s} | MSE: {avg_mse[i]:.6e}, MAE: {avg_mae[i]:.6e}\n")
            f.write("\n")

def zscore_filter_trial(sample_file_path, z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1):
    raw = np.loadtxt(sample_file_path, skiprows=1)

    # First filter (always applied)
    def apply_zscore_filter(data, z_thresh):
        cleaned = data.copy()
        for col in [2, 3, 4]:  # d50, d90, d10
            col_data = cleaned[:, col]
            mean = np.mean(col_data)
            std = np.std(col_data)
            z = (col_data - mean) / std
            cleaned[np.abs(z) > z_thresh, col] = mean
        return cleaned

    # First and second pass
    first_clean = apply_zscore_filter(raw.copy(), z_thresh=z_thresh1)
    second_clean = apply_zscore_filter(first_clean.copy(), z_thresh=z_thresh2)
    
    # lowpass_filtered = first_clean.copy()
    # b, a = butter(N=2, Wn=cutoff, btype='low', fs=1.0)
    # for col in range(6):  # STATE_COLS = 6
    #     lowpass_filtered[:, col] = filtfilt(b, a, first_clean[:, col])
    
    lowpass_filtered = second_clean.copy()
    b, a = butter(N=2, Wn=cutoff, btype='low', fs=1.0)
    for col in range(6):  # STATE_COLS = 6
        lowpass_filtered[:, col] = filtfilt(b, a, second_clean[:, col])

    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

    for idx, name in enumerate(state_names): # zip([2, 3, 4], ['d50', 'd90', 'd10'])
        plt.figure(figsize=(10, 4))
        
        plt.subplot(2, 2, 1)
        plt.plot(raw[:, idx], label="Raw", color='red')
        plt.title(f"{name} raw")
        plt.grid()
        
        plt.subplot(2, 2, 2)
        plt.plot(first_clean[:, idx], label="After Z-filter 1", color='blue')
        plt.title(f"{name} after Z-filter 1")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(second_clean[:, idx], label="After Z-filter 2", color='purple')
        plt.title(f"{name} after Z-filter 2")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(lowpass_filtered[:, idx], label=f"After smoothing filter (cutoff={cutoff})", color='green')
        plt.title(f"{name} after smooth filter")
        plt.grid()

        plt.suptitle(f"Z-Score and Smooth Filtering on {name}")
        plt.tight_layout()
        plt.show()


def train_cluster(cluster_id, cluster_path, args, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".txt")]
    train_files, test_files = split_data(all_files, test_ratio=0.1)

    # Load cleaned and scaled train, test data
    scaler = MinMaxScaler()

    all_train_X, all_train_Y = [], []
    for file in train_files:
        X, Y = load_batch_data(file, args.window_size, scaler)
        all_train_X.append(X)
        all_train_Y.append(Y)
    train_X = np.concatenate(all_train_X, axis=0)
    train_Y = np.concatenate(all_train_Y, axis=0)

    all_test_X, all_test_Y = [], []
    for file in test_files:
        X, Y = load_batch_data(file, args.window_size, scaler, test=False)
        all_test_X.append(X)
        all_test_Y.append(Y)
    test_X = np.concatenate(all_test_X, axis=0)
    test_Y = np.concatenate(all_test_Y, axis=0)

    model = get_model(input_size=(args.window_size * (STATE_COLS + CONTROL_COLS)), model_type=args.model, window_size=args.window_size).to(device)
    criterion = get_loss(use_huber=True)
    optimizer = get_optimizer(model, lr=args.lr, use_adamw=True)
    
    if args.test_saved_model is not None:
        model_path = f"../results/ann/model_cluster{cluster_id}_run{args.test_saved_model}.pt"
        if os.path.exists(model_path):
            print(f"🔁 Using saved model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            evaluate_model(model, test_X, test_Y, run_id=args.test_saved_model, cluster_id=cluster_id, scaler=scaler)
            plot_sample_prediction(model, scaler, test_files, args.window_size, run_id=args.test_saved_model, cluster_id=cluster_id)
            return  # Exit early to skip training
        else:
            print(f"❌ Model file not found at {model_path}, proceeding to train a new model.")
    
    if args.open_loop_run is not None:
        model_path = f"../results/ann/model_cluster{cluster_id}_run{args.open_loop_run}.pt"
        if os.path.exists(model_path):
            print(f"🔁 Using saved model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            predict_open_loop(model, test_X, test_Y, window_size=args.window_size, input_dim=CONTROL_COLS+STATE_COLS, device=device, scaler=scaler, run_id=args.open_loop_run)
            return
        else:
            print(f"❌ Model file not found at {model_path}, proceeding to train a new model.")

    train_losses, val_losses = [], []
    num_epochs = args.epochs
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    Xtr, Xvl, ytr, yvl = train_test_split(train_X, train_Y, test_size=0.3, random_state=42)
    
    # train_loader = create_batches_custom(Xtr, ytr, args.batch_size)
    # val_loader = create_batches_custom(Xvl, yvl, args.batch_size)
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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
        
        # Early stopping check
        if args.early_stopping:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                epochs_without_improvement = 0
                best_model_state = model.state_dict()  # Save best model
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}. No improvement for {args.early_stop_patience} epochs.")
                    break

    if args.early_stopping:
        model.load_state_dict(best_model_state)
    
    if args.log_hyperparams:
        log_hyperparams(run_id, args, cluster_id)
    
    if args.save_model:
        torch.save(model.state_dict(), f"../results/ann/model_cluster{cluster_id}_run{run_id}.pt")

    plot_losses(train_losses, val_losses, run_id=run_id, log_hyperparams=args.log_hyperparams,
                log_scale=args.log_scale, model_type=args.model)

    evaluate_model(model, test_X, test_Y, run_id=run_id, cluster_id=cluster_id, scaler=scaler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs="+", type=int, help="List of cluster IDs to train")
    parser.add_argument("--train_all", action="store_true", help="Train all available clusters")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--model", choices=["ann", "lstm", "stacked_lstm", "stacked_lstm_reg", "bilstm_attn", "bilstm_multihead"], default="ann")
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument("--log_hyperparams", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--test_saved_model", type=int, help="Use a saved model by run_id instead of training")
    parser.add_argument("--open_loop_run", type=int, help="Run ID of model to use for open loop prediction")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--test_zscore_filter", action="store_true", help="Run Z-score threshold test and exit")
    args = parser.parse_args()

    base_path = "../Data/clustered"
    run_id = args.test_saved_model if args.test_saved_model is not None else int(time.time())

    if args.train_all:
        clusters = [int(folder.replace("cluster", "")) for folder in os.listdir(base_path) if folder.startswith("cluster")]
    else:
        clusters = args.clusters or []
    
    if args.test_zscore_filter:
        # Pick a random file from a known cluster (adjust path as needed)
        cluster_0_path = os.path.join(base_path, "cluster0")
        all_files = [os.path.join(cluster_0_path, f) for f in os.listdir(cluster_0_path) if f.endswith(".txt")]
        sample_file = random.choice(all_files)

        print(f"\n📊 Running Z-score trial on: {sample_file}")
        zscore_filter_trial(sample_file)
        return

    for cluster_id in clusters:
        cluster_path = os.path.join(base_path, f"cluster{cluster_id}")
        print(f"\n\U0001f527 Training on Cluster {cluster_id}")
        train_cluster(cluster_id, cluster_path, args, run_id)

if __name__ == "__main__":
    main()
