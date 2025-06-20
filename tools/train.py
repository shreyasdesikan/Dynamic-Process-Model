import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.ANN.narx_model import get_model, get_loss, get_optimizer

STATE_COLS = 6
INPUT_COLS = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_batches(cluster_path):
    files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".txt")]
    return files


def load_data(file):
    return np.loadtxt(file, skiprows=1)


def create_dataset(data, window_size):
    # Normalize each column (feature-wise min-max)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data = (data - data_min) / (data_max - data_min + 1e-8)

    X, Y = [], []
    for t in range(window_size, len(data) - 1):
        state_window = data[t - window_size:t, :STATE_COLS].flatten()
        control_input = data[t, STATE_COLS:]
        input_vec = np.concatenate((state_window, control_input))
        target = data[t + 1, :STATE_COLS]
        X.append(input_vec)
        Y.append(target)
    return np.array(X), np.array(Y)


def split_files(files, ratio):
    random.shuffle(files)
    split_point = int(len(files) * ratio)
    return files[:split_point], files[split_point:]


def evaluate_model(model, files, window_size, epochs):
    model.eval()
    criterion = get_loss()
    total_loss, count = 0, 0
    with torch.no_grad():
        for f in files:
            data = load_data(f)
            X, Y = create_dataset(data, window_size)
            for i in range(min(epochs, len(X))):
                x_tensor = torch.tensor(X[i], dtype=torch.float32).to(DEVICE)
                y_tensor = torch.tensor(Y[i], dtype=torch.float32).to(DEVICE)
                pred = model(x_tensor)
                loss = criterion(pred, y_tensor)
                total_loss += loss.item()
                count += 1
    return total_loss / count


def train_cluster(cluster_id, cluster_path, window_size, epochs, lr, steps_per_file, log_scale=False, log_hyperparams=False):
    files = load_batches(cluster_path)
    trainval_files, test_files = split_files(files, 0.9)
    model = get_model().to(DEVICE)
    criterion = get_loss()
    optimizer = get_optimizer(model, lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_files, val_files = split_files(trainval_files, 0.7)
        model.train()
        total_loss, count = 0, 0

        for f in train_files:
            data = load_data(f)
            X, Y = create_dataset(data, window_size)
            if epoch >= len(X):
                continue
            for i in range(min(steps_per_file, len(X))):
                x_tensor = torch.tensor(X[i], dtype=torch.float32).to(DEVICE)
                y_tensor = torch.tensor(Y[i], dtype=torch.float32).to(DEVICE)
                pred = model(x_tensor)
                loss = criterion(pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        val_loss = evaluate_model(model, val_files, window_size, epochs)
        train_losses.append(total_loss / count if count else 0)
        val_losses.append(val_loss)
        print(f"Cluster {cluster_id} | Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Loss (log scale)")
    else:
        plt.ylabel("Loss")
    plt.title(f"Cluster {cluster_id} Loss over Time")
    plt.legend()
    os.makedirs("results", exist_ok=True)
    if log_hyperparams:
        filename = f"results/cluster_{cluster_id}_loss_w{window_size}_e{epochs}_lr{lr}_s{steps_per_file}.png"
    else:
        filename = f"results/cluster_{cluster_id}_loss.png"
    plt.savefig(filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../Data/clustered", help="Path to clustered data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--window_size", type=int, default=5, help="Input window size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--steps_per_file", type=int, default=10, help="Number of windowed steps used per file per epoch")
    parser.add_argument("--clusters", type=int, nargs="*", help="List of cluster indices to train. Leave empty to train all.")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for loss plot")
    parser.add_argument("--log_hyperparams", action="store_true", help="Log hyperparameters and save loss plot with config info")
    args = parser.parse_args()

    all_clusters = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
    selected_clusters = (
        [all_clusters[i] for i in args.clusters if 0 <= i < len(all_clusters)]
        if args.clusters else all_clusters
    )

    for cluster_id, cluster_folder in enumerate(selected_clusters):
        cluster_path = os.path.join(args.data_path, cluster_folder)
        train_cluster(cluster_id, cluster_path, args.window_size, args.epochs, args.lr, args.steps_per_file, args.log_scale, args.log_hyperparams)


if __name__ == "__main__":
    main()
