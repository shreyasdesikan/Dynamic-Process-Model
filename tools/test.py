import os
import random
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tuning.model import BidirectionalLSTMWithMultiHead

STATE_NAMES = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
STATE_COLS = 6
CONTROL_COLS = 7

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

def plot_sample_prediction(model, scaler, test_file, window_size, run_id, cluster_id, sm_filt=True):
    os.makedirs(f"../results/ann/{run_id}", exist_ok=True)

    # Pick a random test file
    _, second_zthresh, filt_data = clean_and_filter(np.loadtxt(test_file, skiprows=1), z_thresh1=1.0, z_thresh2=3.0, cutoff=0.1, sm_filt=sm_filt)

    # Prepare input/output using existing logic
    scaled_data = scaler.fit_transform(filt_data)
    X, Y, Y_unfilt = [], [], []
    for t in range(len(scaled_data) - window_size):
        window = scaled_data[t:t + window_size, :]
        target = scaled_data[t + window_size, :STATE_COLS]
        target_unfilt = second_zthresh[t + window_size, :STATE_COLS]
        X.append(window.flatten())
        Y.append(target)
        Y_unfilt.append(target_unfilt)

    X = torch.tensor(np.array(X), dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        Y_pred = model(X).cpu().numpy()

    # Pad and inverse-transform
    def unscale(arr):
        pad = np.zeros((arr.shape[0], CONTROL_COLS))
        padded = np.hstack([arr, pad])
        return scaler.inverse_transform(padded)[:, :STATE_COLS]

    Y_true_unscaled = unscale(np.array(Y))
    Y_true_unfilt = np.array(Y_unfilt)[:, :STATE_COLS]
    Y_pred_unscaled = unscale(Y_pred)

    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

    for i, state in enumerate(state_names):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # LEFT subplot: Raw vs Predicted
        axes[0].plot(Y_true_unfilt[:, i], label="Unsmoothed Raw data")
        axes[0].plot(Y_pred_unscaled[:, i], label="Predicted", linestyle="--")
        axes[0].set_title(f"{state} (Raw vs Predicted)")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel(state)
        axes[0].legend()
        axes[0].grid(True)

        # Capture y-limits from the left subplot
        y_min, y_max = axes[0].get_ylim()

        # RIGHT subplot: True vs Predicted
        axes[1].plot(Y_true_unscaled[:, i], label="True")
        axes[1].plot(Y_pred_unscaled[:, i], label="Predicted", linestyle="--")
        axes[1].set_title(f"{state} (True vs Predicted)")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel(state)
        axes[1].legend()
        axes[1].grid(True)

        # Apply y-limits to the second plot
        axes[1].set_ylim(y_min, y_max)

        # Save to same location
        save_path = f"../results/ann/{run_id}/state_{state}_cluster{cluster_id}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_path = "../Beat-the-Felix"
    test_file = os.path.join(base_path, "file_12738.txt")
    model_path = os.path.join(base_path, f"model_cluster0_run1752708457.pt")
    
    scaler = StandardScaler()
    model = BidirectionalLSTMWithMultiHead(hidden_dim=128, fc_dim=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    plot_sample_prediction(model, scaler, test_file, window_size=5, run_id=1752708457, cluster_id=0)

if __name__ == "__main__":
    main()