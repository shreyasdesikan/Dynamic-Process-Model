import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import random

# Fix import paths - add parent directories to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools'))

from models.ANN.narx_model import get_model
from train import load_batch_data, clean_state_spikes


CONTROL_COLS = 7
STATE_NAMES = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
STATE_COLS = len(STATE_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU_LOW = 0.1
TAU_HIGH = 0.9
ALPHA = 1 - (TAU_HIGH - TAU_LOW)

class ErrorDataset(Dataset):
    """Dataset for training quantile regressors on prediction errors"""
    def __init__(self, inputs, errors):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.errors = torch.tensor(errors, dtype=torch.float32)
    
    def __len__(self):
        return len(self.errors)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.errors[idx]

class QuantileRegressor(nn.Module):
    """Neural network for quantile regression"""
    def __init__(self, input_size, hidden_sizes=[64, 64], output_size=1):
        super(QuantileRegressor, self).__init__()
        layers = []
        last_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            last_size = hidden_size
        
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def pinball_loss(predictions, targets, tau):
    """Pinball loss for quantile regression"""
    errors = targets - predictions
    loss = torch.where(
        errors >= 0,
        tau * errors,
        (tau - 1) * errors
    )
    return torch.mean(loss)

# ============= TASK 1: Generate Error Dataset and Train Quantile Regressors =============

def generate_error_dataset(model, data_files, window_size, scaler):
    """Generate error dataset from trained model"""
    print("\nTask 1: Generating error dataset...")
    
    model.eval()
    all_inputs = []
    all_errors = {f'state_{i}': [] for i in range(STATE_COLS)}
    
    with torch.no_grad():
        for i, file_path in enumerate(data_files):
            if i % 5 == 0:
                print(f"  Processing file {i+1}/{len(data_files)}...")
                
            X, Y = load_batch_data(file_path, window_size, scaler)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            Y_pred = model(X_tensor).cpu()
            errors = Y_pred - Y_tensor
            
            all_inputs.extend(X)
            for j in range(STATE_COLS):
                all_errors[f'state_{j}'].extend(errors[:, j].numpy())
    
    print(f"Generated {len(all_inputs)} error samples")
    return np.array(all_inputs), {k: np.array(v) for k, v in all_errors.items()}

def train_quantile_regressor(inputs, errors, tau, epochs=80, batch_size=64, lr=1e-3):
    """Train a quantile regressor"""    
    dataset = ErrorDataset(inputs, errors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = inputs.shape[1]
    model = QuantileRegressor(input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_inputs, batch_errors in dataloader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_errors = batch_errors.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs).squeeze()
            loss = pinball_loss(predictions, batch_errors, tau)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    return model

# ============= TASK 2: Conformalize Prediction Intervals =============

def compute_conformity_scores(model, quantile_models, cal_files, window_size, scaler):
    """Compute conformity scores on calibration data"""
    print("\nTask 2: Computing conformity scores...")
    
    conformity_scores = {f'state_{i}': [] for i in range(STATE_COLS)}
    
    model.eval()
    for qr_dict in quantile_models.values():
        for qr in qr_dict.values():
            qr.eval()
    
    with torch.no_grad():
        for file_path in cal_files:
            X, Y = load_batch_data(file_path, window_size, scaler)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            Y_pred = model(X_tensor).cpu()
            
            for i in range(STATE_COLS):
                state_name = f'state_{i}'
                
                # Get quantile predictions (these predict errors, not absolute values)
                q_low = quantile_models[state_name]['low'](X_tensor).cpu().squeeze()
                q_high = quantile_models[state_name]['high'](X_tensor).cpu().squeeze()
                
                # Compute conformity scores
                y_true = Y_tensor[:, i]
                y_pred = Y_pred[:, i]
                
                # E_i = max(q_low - (y_true - y_pred), (y_true - y_pred) - q_high)
                actual_error = y_true - y_pred
                lower_score = q_low - actual_error
                upper_score = actual_error - q_high
                scores = torch.maximum(lower_score, upper_score)
                
                conformity_scores[state_name].extend(scores.numpy())
    
    # Compute correction factors
    correction_factors = {}
    
    for state_name, scores in conformity_scores.items():
        scores = np.array(scores)
        n = len(scores)
        quantile_level = (1 - ALPHA) * (1 + 1/n)
        correction_factors[state_name] = np.quantile(scores, quantile_level)
        print(f"  {state_name}: correction factor = {correction_factors[state_name]:.6f}")
    
    return correction_factors

# ============= TASK 3: Analyze and Visualize Performance =============

def plot_all_states_comparison(model, quantile_models, correction_factors, 
                              test_files, window_size, scaler, save_dir):
    # --------- PART 1: Compute average coverage across all test files ---------
    print("\nComputing average coverage across all test files...")
    state_coverages = {name: {'uncal': [], 'cal': []} for name in STATE_NAMES}

    for test_file in test_files:
        X, Y = load_batch_data(test_file, window_size, scaler)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        for state_idx in range(STATE_COLS):
            state_name = f'state_{state_idx}'
            state_label = STATE_NAMES[state_idx]

            n_points = min(300, len(X))
            indices = np.linspace(0, len(X)-1, n_points, dtype=int)
            X_subset = X_tensor[indices]
            Y_subset = Y[indices, state_idx]

            with torch.no_grad():
                Y_pred_all = model(X_subset).cpu().numpy()
                Y_pred = Y_pred_all[:, state_idx]
                q_low = quantile_models[state_name]['low'](X_subset).cpu().squeeze().numpy()
                q_high = quantile_models[state_name]['high'](X_subset).cpu().squeeze().numpy()

                uncal_lower = Y_pred + q_low
                uncal_upper = Y_pred + q_high
                correction = correction_factors[state_name]
                cal_lower = uncal_lower - correction
                cal_upper = uncal_upper + correction

            def unscale(arr):
                dummy = np.zeros((len(arr), STATE_COLS + CONTROL_COLS))
                dummy[:, state_idx] = arr
                return scaler.inverse_transform(dummy)[:, state_idx]

            Y_true_uns = unscale(Y_subset)
            uncal_lower_uns = unscale(uncal_lower)
            uncal_upper_uns = unscale(uncal_upper)
            cal_lower_uns = unscale(cal_lower)
            cal_upper_uns = unscale(cal_upper)

            uncal_coverage = np.mean((Y_true_uns >= uncal_lower_uns) & (Y_true_uns <= uncal_upper_uns)) * 100
            cal_coverage = np.mean((Y_true_uns >= cal_lower_uns) & (Y_true_uns <= cal_upper_uns)) * 100

            state_coverages[state_label]['uncal'].append(uncal_coverage)
            state_coverages[state_label]['cal'].append(cal_coverage)

    print("\n--- Average Coverage over All Test Files ---")
    for state, vals in state_coverages.items():
        avg_uncal = np.mean(vals['uncal'])
        avg_cal = np.mean(vals['cal'])
        print(f"{state:6}: Uncalibrated {avg_uncal:.2f}% → CQR {avg_cal:.2f}%")

    # --------- PART 2: Plotting for one file only (randomly selected) ---------
    # Pick a random test file
    random.seed(44)
    test_file = random.choice(test_files)
    print(f"Selected test file: {os.path.basename(test_file)}")
    
    # Load the selected file
    X, Y = load_batch_data(test_file, window_size, scaler)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    # Process each state
    for state_idx in range(STATE_COLS):
        state_name = f'state_{state_idx}'
        
        print(f"\nProcessing {STATE_NAMES[state_idx]}...")
        
        # Use 300 points for clarity
        n_points = min(300, len(X))
        indices = np.linspace(0, len(X)-1, n_points, dtype=int)
        
        X_subset = X_tensor[indices]
        Y_subset = Y[indices, state_idx]
        
        with torch.no_grad():
            # Get predictions
            Y_pred_all = model(X_subset).cpu().numpy()
            Y_pred = Y_pred_all[:, state_idx]
            
            # Get quantile predictions
            q_low = quantile_models[state_name]['low'](X_subset).cpu().squeeze().numpy()
            q_high = quantile_models[state_name]['high'](X_subset).cpu().squeeze().numpy()
            
            # Uncalibrated intervals
            uncal_lower = Y_pred + q_low
            uncal_upper = Y_pred + q_high
            
            # Calibrated intervals
            correction = correction_factors[state_name]
            cal_lower = Y_pred + q_low - correction
            cal_upper = Y_pred + q_high + correction
        
        # Unscale for visualization
        def unscale_state(arr, state_idx):
            dummy = np.zeros((len(arr), STATE_COLS + CONTROL_COLS))
            dummy[:, state_idx] = arr
            unscaled = scaler.inverse_transform(dummy)
            return unscaled[:, state_idx]
        
        Y_true_uns = unscale_state(Y_subset, state_idx)
        Y_pred_uns = unscale_state(Y_pred, state_idx)
        uncal_lower_uns = unscale_state(uncal_lower, state_idx)
        uncal_upper_uns = unscale_state(uncal_upper, state_idx)
        cal_lower_uns = unscale_state(cal_lower, state_idx)
        cal_upper_uns = unscale_state(cal_upper, state_idx)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Use time steps for x-axis
        x_plot = np.arange(n_points)
        
        # Plot 1: Mean Prediction Only
        ax1.scatter(x_plot, Y_true_uns, c='black', s=20, alpha=0.6, label='Data')
        ax1.plot(x_plot, Y_pred_uns, 'b-', linewidth=2, label='Mean prediction')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(STATE_NAMES[state_idx])
        ax1.set_title('Mean Prediction Only')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncalibrated QR
        uncal_inside = (Y_true_uns >= uncal_lower_uns) & (Y_true_uns <= uncal_upper_uns)
        uncal_coverage = np.mean(uncal_inside) * 100
        
        ax2.scatter(x_plot[uncal_inside], Y_true_uns[uncal_inside], 
                   c='blue', s=20, alpha=0.6, label='Inside interval')
        ax2.scatter(x_plot[~uncal_inside], Y_true_uns[~uncal_inside], 
                   c='red', s=20, alpha=0.6, marker='x', label='Outside interval')
        ax2.plot(x_plot, Y_pred_uns, 'b-', linewidth=2, label='Mean prediction')
        ax2.fill_between(x_plot, uncal_lower_uns, uncal_upper_uns, 
                        alpha=0.2, color='blue', label='Uncalibrated QR')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel(STATE_NAMES[state_idx])
        ax2.set_title(f'QR - Uncalibrated - Coverage: {uncal_coverage:.2f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CQR (Calibrated)
        cal_inside = (Y_true_uns >= cal_lower_uns) & (Y_true_uns <= cal_upper_uns)
        cal_coverage = np.mean(cal_inside) * 100
        
        ax3.scatter(x_plot[cal_inside], Y_true_uns[cal_inside], 
                   c='green', s=20, alpha=0.6, label='Inside interval')
        ax3.scatter(x_plot[~cal_inside], Y_true_uns[~cal_inside], 
                   c='red', s=20, alpha=0.6, marker='x', label='Outside interval')
        ax3.plot(x_plot, Y_pred_uns, 'g-', linewidth=2, label='Mean prediction')
        ax3.fill_between(x_plot, cal_lower_uns, cal_upper_uns, 
                        alpha=0.2, color='green', label='CQR')
        
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel(STATE_NAMES[state_idx])
        ax3.set_title(f'CQR - Coverage: {cal_coverage:.2f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'{STATE_NAMES[state_idx]} - Comparison of Prediction Methods', fontsize=16, y=1.02)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cqr_comparison_{STATE_NAMES[state_idx]}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {STATE_NAMES[state_idx]}: Uncalibrated {uncal_coverage:.2f}% → CQR {cal_coverage:.2f}%")

def main():    
    window_size = 15
    model_path = "results/ann/model_cluster0_run1751300407.pt"
    data_path = "Data/clustered/cluster0"
    results_dir = "results/ann"
    save_dir = os.path.join(results_dir, f"cqr_results")
    
    # Load data files and split
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".txt")]
    train_files, temp_files = train_test_split(data_files, test_size=0.2, random_state=44)
    cal_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=44)
    
    print(f"Training: {len(train_files)} files")
    print(f"Calibration: {len(cal_files)} files")
    print(f"Test: {len(test_files)} files")
    
    # Load model
    input_size = window_size * (STATE_COLS + CONTROL_COLS)
    model = get_model(input_size=input_size, model_type="bilstm_multihead").to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Initialize and fit scaler
    print("\nFitting scaler on training data...")
    scaler = MinMaxScaler()
    
    for i, file in enumerate(train_files[:10]):
        raw = np.loadtxt(file, skiprows=1)
        cleaned = clean_state_spikes(raw)
        scaler.partial_fit(cleaned)
        if i % 5 == 0:
            print(f"  Fitted on {i+1}/10 files")
    
    # Generate error dataset
    inputs, errors_dict = generate_error_dataset(model, train_files, window_size, scaler)
    
    # Train quantile regressors
    print("\nTraining quantile regressors...")
    quantile_models = {}
    
    for i in range(STATE_COLS):
        state_name = f'state_{i}'
        print(f"\nTraining for {STATE_NAMES[i]}...")
        
        quantile_models[state_name] = {}
        
        print(f"Training τ={TAU_LOW}")
        model_low = train_quantile_regressor(inputs, errors_dict[state_name], tau=TAU_LOW)
        quantile_models[state_name]['low'] = model_low
        
        print(f"Training τ={TAU_HIGH}")
        model_high = train_quantile_regressor(inputs, errors_dict[state_name], tau=TAU_HIGH)
        quantile_models[state_name]['high'] = model_high
    
    # Compute conformity scores
    correction_factors = compute_conformity_scores(model, quantile_models, cal_files, window_size, scaler)
    
    # Create visualizations
    os.makedirs(save_dir, exist_ok=True)
    plot_all_states_comparison(
        model, quantile_models, correction_factors,
        test_files, window_size, scaler, save_dir
    )

    print("\nCQR pipeline completed successfully!")
    print(f"Results saved in: {save_dir}")

if __name__ == "__main__":
    main()