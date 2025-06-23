import os
import argparse
import time
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.ANN.narx_model import get_model, get_loss, get_optimizer

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

def clean_state_spikes(data, z_thresh=3.0):
    """Clean spikes in d50, d90, and d10 columns using z-score thresholding."""
    cleaned = data.copy()
    states = cleaned[:, :STATE_COLS]
    target_indices = [2, 3, 4]  # d50, d90, d10

    for i in target_indices:
        col = states[:, i]
        mean = np.mean(col)
        std = np.std(col)
        z_scores = (col - mean) / std
        spike_mask = np.abs(z_scores) > z_thresh
        mean_val = np.mean(col)
        cleaned[spike_mask, i] = mean_val

    return cleaned

def load_batch_data(file_path, window_size, scaler=None):
    df = np.loadtxt(file_path, skiprows=1)
    scaled = scaler.transform(df) if scaler else df

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
    path = f"../results/hyperparams.txt"
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
        plt.savefig(f"../results/loss_plot_run{run_id}.png")
    else:
        plt.show()
    plt.close()

def plot_sample_prediction(model, scaler, test_files, window_size, run_id, cluster_id):
    os.makedirs(f"../results/{run_id}", exist_ok=True)

    # Pick a random test file
    sample_file = random.choice(test_files)
    raw_data = clean_state_spikes(np.loadtxt(sample_file, skiprows=1), z_thresh=1.0)

    # Prepare input/output using existing logic
    scaled_data = scaler.transform(raw_data)
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
        plt.plot(Y_true_unscaled[:, i], label="True")
        plt.plot(Y_pred_unscaled[:, i], label="Predicted", linestyle="--")
        plt.title(f"{state} - Cluster {cluster_id} - Run {run_id}")
        plt.xlabel("Timestep")
        plt.ylabel(state)
        plt.legend()
        plt.grid(True)
        save_path = f"../results/{run_id}/state_{state}_cluster{cluster_id}.png"
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
            with open(f"../results/test_results_log.txt", "a") as f:
                f.write(f"Run ID: {run_id}, Cluster {cluster_id} - Test MSE (scaled):\n")
                for name, m in zip(state_names, mse_scaled):
                    f.write(f"{name}: {m:.6e}\n")
                f.write(f"Run ID: {run_id}, Cluster {cluster_id} - Test MSE (unscaled):\n")
                for name, m in zip(state_names, mse_unscaled):
                    f.write(f"{name}: {m:.6e}\n")
                f.write("\n")

def train_cluster(cluster_id, cluster_path, args, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".txt")]
    train_files, test_files = split_data(all_files, test_ratio=0.1)

    # Load and clean training data
    all_train_df = []
    for f in train_files:
        raw = np.loadtxt(f, skiprows=1)
        cleaned = clean_state_spikes(raw, z_thresh=1.0)
        # cleaned = raw
        all_train_df.append(cleaned)

    # Fit scaler on cleaned training data
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(all_train_df, axis=0))

    all_train_X, all_train_Y = [], []
    for file in train_files:
        X, Y = load_batch_data(file, args.window_size, scaler)
        all_train_X.append(X)
        all_train_Y.append(Y)

    all_test_X, all_test_Y = [], []
    for file in test_files:
        X, Y = load_batch_data(file, args.window_size, scaler)
        all_test_X.append(X)
        all_test_Y.append(Y)
    test_X = np.concatenate(all_test_X, axis=0)
    test_Y = np.concatenate(all_test_Y, axis=0)

    model = get_model(input_size=(args.window_size * (STATE_COLS + CONTROL_COLS)), model_type=args.model, window_size=args.window_size).to(device)
    criterion = get_loss(use_huber=True)
    optimizer = get_optimizer(model, lr=args.lr, use_adamw=True)
    
    if args.use_saved_model is not None:
        model_path = f"../results/model_cluster{cluster_id}_run{args.use_saved_model}.pt"
        if os.path.exists(model_path):
            print(f"🔁 Using saved model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            evaluate_model(model, test_X, test_Y, run_id=args.use_saved_model, cluster_id=cluster_id, scaler=scaler)
            plot_sample_prediction(model, scaler, test_files, args.window_size, run_id=args.use_saved_model, cluster_id=cluster_id)
            return  # Exit early to skip training
        else:
            print(f"❌ Model file not found at {model_path}, proceeding to train a new model.")

    train_losses, val_losses = [], []
    num_epochs = args.epochs
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_X = np.concatenate(all_train_X, axis=0)
    train_Y = np.concatenate(all_train_Y, axis=0)
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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
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
        torch.save(model.state_dict(), f"../results/model_cluster{cluster_id}_run{run_id}.pt")

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
    parser.add_argument("--model", choices=["ann", "lstm", "stacked_lstm", "bilstm_attn", "bilstm_multihead"], default="ann")
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument("--log_hyperparams", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_saved_model", type=int, help="Use a saved model by run_id instead of training")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Patience for early stopping")
    args = parser.parse_args()

    base_path = "../Data/clustered"
    run_id = args.use_saved_model if args.use_saved_model is not None else int(time.time())

    if args.train_all:
        clusters = [int(folder.replace("cluster", "")) for folder in os.listdir(base_path) if folder.startswith("cluster")]
    else:
        clusters = args.clusters or []

    for cluster_id in clusters:
        cluster_path = os.path.join(base_path, f"cluster{cluster_id}")
        print(f"\n\U0001f527 Training on Cluster {cluster_id}")
        train_cluster(cluster_id, cluster_path, args, run_id)

if __name__ == "__main__":
    main()
