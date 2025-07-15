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

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools'))

from models.ANN.narx_model import get_model
from train import clean_batch, window_batch_data, CONTROL_COLS, STATE_NAMES


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

def unscale_state_values(arr, state_idx, scaler):
    dummy = np.zeros((len(arr), STATE_COLS + CONTROL_COLS))
    dummy[:, state_idx] = arr
    return scaler.inverse_transform(dummy)[:, state_idx]


# ============= TASK 1: Generate Error Dataset and Train Quantile Regressors =============

def generate_error_dataset(model, data_files, window_size, scaler):
    print("\nTask 1: Generating error dataset")
    
    model.eval()
    all_inputs = []
    all_errors = {f'state_{i}': [] for i in range(STATE_COLS)}
    
    with torch.no_grad():
        data_cleaned = []
        for file in data_files:
            cleaned = clean_batch(file)
            data_cleaned.append(cleaned)
        for i, cleaned in enumerate(data_cleaned):
            if i % 5 == 0:
                print(f"  Processing file {i+1}/{len(data_files)}...")
            
            scaled = scaler.transform(cleaned)    
            X, Y = window_batch_data(scaled, window_size)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            Y_pred = model(X_tensor).cpu()
            errors = Y_pred - Y_tensor
            
            all_inputs.extend(X)
            for j in range(STATE_COLS):
                all_errors[f'state_{j}'].extend(errors[:, j].numpy())
    
    print(f"Generated {len(all_inputs)} error samples")
    return np.array(all_inputs), {k: np.array(v) for k, v in all_errors.items()}

def train_quantile_regressor(inputs, errors, tau, epochs=40, batch_size=64, lr=1e-3):
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
    print("\nTask 2: Computing conformity scores")
    
    conformity_scores = {f'state_{i}': [] for i in range(STATE_COLS)}
    
    model.eval()
    for qr_dict in quantile_models.values():
        for qr in qr_dict.values():
            qr.eval()
    
    with torch.no_grad():
        cal_cleaned = []
        for file in cal_files:
            cleaned = clean_batch(file)
            cal_cleaned.append(cleaned)
        for cleaned in cal_cleaned:
            scaled = scaler.transform(cleaned)
            X, Y = window_batch_data(scaled, window_size)
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

def compute_all_predictions_and_coverage(model, quantile_models, correction_factors, test_files, window_size, scaler):
    print("\nComputing predictions and coverage for all test files")

    # Store all computed results
    all_results = []
    
    test_cleaned = []
    for file in test_files:
        cleaned = clean_batch(file)
        test_cleaned.append(cleaned)
    for file_idx, (cleaned, test_file) in enumerate(zip(test_cleaned, test_files)):
        print(f"Processing file {file_idx + 1}/{len(test_files)}: {os.path.basename(test_file)}")
        
        scaled = scaler.transform(cleaned)
        X, Y = window_batch_data(scaled, window_size)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        
        # Store results for this file
        file_results = {
            'filename': test_file,
            'states': {}
        }
        
        for state_idx in range(STATE_COLS):
            state_name = f'state_{state_idx}'
            state_label = STATE_NAMES[state_idx]

            # Sample subset for efficiency
            n_points = min(300, len(X))
            indices = np.linspace(0, len(X)-1, n_points, dtype=int)
            X_subset = X_tensor[indices]
            Y_subset = Y[indices, state_idx]

            with torch.no_grad():
                # Get predictions and quantiles
                Y_pred_all = model(X_subset).cpu().numpy()
                Y_pred = Y_pred_all[:, state_idx]
                q_low = quantile_models[state_name]['low'](X_subset).cpu().squeeze().numpy()
                q_high = quantile_models[state_name]['high'](X_subset).cpu().squeeze().numpy()

                # Compute intervals
                uncal_lower = Y_pred + q_low
                uncal_upper = Y_pred + q_high
                correction = correction_factors[state_name]
                cal_lower = uncal_lower - correction
                cal_upper = uncal_upper + correction

            # Unscale everything
            Y_true_uns = unscale_state_values(Y_subset, state_idx, scaler)
            Y_pred_uns = unscale_state_values(Y_pred, state_idx, scaler)
            uncal_lower_uns = unscale_state_values(uncal_lower, state_idx, scaler)
            uncal_upper_uns = unscale_state_values(uncal_upper, state_idx, scaler)
            cal_lower_uns = unscale_state_values(cal_lower, state_idx, scaler)
            cal_upper_uns = unscale_state_values(cal_upper, state_idx, scaler)

            # Calculate coverage
            uncal_coverage = np.mean((Y_true_uns >= uncal_lower_uns) & (Y_true_uns <= uncal_upper_uns)) * 100
            cal_coverage = np.mean((Y_true_uns >= cal_lower_uns) & (Y_true_uns <= cal_upper_uns)) * 100

            # Store all computed results for the state
            file_results['states'][state_idx] = {
                'state_name': state_name,
                'state_label': state_label,
                'n_points': n_points,
                'Y_true_uns': Y_true_uns,
                'Y_pred_uns': Y_pred_uns,
                'uncal_lower_uns': uncal_lower_uns,
                'uncal_upper_uns': uncal_upper_uns,
                'cal_lower_uns': cal_lower_uns,
                'cal_upper_uns': cal_upper_uns,
                'uncal_coverage': uncal_coverage,
                'cal_coverage': cal_coverage
            }
        
        all_results.append(file_results)

    print("\n--- Average Coverage over All Test Files ---")
    for state_idx in range(STATE_COLS):
        state_label = STATE_NAMES[state_idx]
        uncal_coverages = [result['states'][state_idx]['uncal_coverage'] for result in all_results]
        cal_coverages = [result['states'][state_idx]['cal_coverage'] for result in all_results]
        avg_uncal = np.mean(uncal_coverages)
        avg_cal = np.mean(cal_coverages)
        print(f"{state_label:6}: Uncalibrated {avg_uncal:.2f}% → CQR {avg_cal:.2f}%")

    return all_results

def plot_selected_file_results(selected_results, save_dir):
    filename = selected_results['filename']
    print(f"\nCreating plots for: {os.path.basename(filename)}")
    
    for state_idx in range(STATE_COLS):
        state_data = selected_results['states'][state_idx]
        state_label = state_data['state_label']
        
        print(f"Plotting {state_label}...")
        
        # Extract data
        n_points = state_data['n_points']
        Y_true_uns = state_data['Y_true_uns']
        Y_pred_uns = state_data['Y_pred_uns']
        uncal_lower_uns = state_data['uncal_lower_uns']
        uncal_upper_uns = state_data['uncal_upper_uns']
        cal_lower_uns = state_data['cal_lower_uns']
        cal_upper_uns = state_data['cal_upper_uns']
        uncal_coverage = state_data['uncal_coverage']
        cal_coverage = state_data['cal_coverage']
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        x_plot = np.arange(n_points)
        
        # Plot 1: Mean Prediction Only
        ax1.scatter(x_plot, Y_true_uns, c='black', s=20, alpha=0.6, label='Data')
        ax1.plot(x_plot, Y_pred_uns, 'b-', linewidth=2, label='Mean prediction')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(state_label)
        ax1.set_title('Mean Prediction Only')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Uncalibrated QR
        uncal_inside = (Y_true_uns >= uncal_lower_uns) & (Y_true_uns <= uncal_upper_uns)
        
        ax2.scatter(x_plot[uncal_inside], Y_true_uns[uncal_inside], 
                   c='blue', s=20, alpha=0.6, label='Inside interval')
        ax2.scatter(x_plot[~uncal_inside], Y_true_uns[~uncal_inside], 
                   c='red', s=20, alpha=0.6, marker='x', label='Outside interval')
        ax2.plot(x_plot, Y_pred_uns, 'b-', linewidth=2, label='Mean prediction')
        ax2.fill_between(x_plot, uncal_lower_uns, uncal_upper_uns, 
                        alpha=0.2, color='blue', label='Uncalibrated QR')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel(state_label)
        ax2.set_title(f'QR - Uncalibrated - Coverage: {uncal_coverage:.2f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CQR (Calibrated)
        cal_inside = (Y_true_uns >= cal_lower_uns) & (Y_true_uns <= cal_upper_uns)
        
        ax3.scatter(x_plot[cal_inside], Y_true_uns[cal_inside], 
                   c='green', s=20, alpha=0.6, label='Inside interval')
        ax3.scatter(x_plot[~cal_inside], Y_true_uns[~cal_inside], 
                   c='red', s=20, alpha=0.6, marker='x', label='Outside interval')
        ax3.plot(x_plot, Y_pred_uns, 'g-', linewidth=2, label='Mean prediction')
        ax3.fill_between(x_plot, cal_lower_uns, cal_upper_uns, 
                        alpha=0.2, color='green', label='CQR')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel(state_label)
        ax3.set_title(f'CQR - Coverage: {cal_coverage:.2f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle(f'{state_label} - Comparison of Prediction Methods', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cqr_comparison_{state_label}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{state_label}: Uncalibrated {uncal_coverage:.2f}% → CQR {cal_coverage:.2f}%")

def initialize_scaler(train_files):
    """Initialize and fit scaler on training data"""
    scaler = MinMaxScaler()
    
    all_cleaned_data = []
    for file_path in train_files:
        cleaned = clean_batch(file_path, sm_filt=True)
        all_cleaned_data.append(cleaned)
    
    # Fit scaler on all concatenated data
    concatenated_data = np.concatenate(all_cleaned_data, axis=0)
    scaler.fit(concatenated_data)
    
    print(f"Scaler fitted on {len(concatenated_data)} samples from {len(train_files)} files")
    return scaler

def run_cqr_pipeline(model_path, data_path, results_dir, window_size):
    cluster_name = os.path.basename(data_path.rstrip("/"))
    save_dir = os.path.join(results_dir, "cqr_results", cluster_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nRunning CQR pipeline for {cluster_name}")
    
    # Load data files and split
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".txt")]
    train_files, temp_files = train_test_split(data_files, test_size=0.3, random_state=44)
    cal_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=44)
    
    print(f"Training: {len(train_files)} files")
    print(f"Calibration: {len(cal_files)} files")
    print(f"Test: {len(test_files)} files")
    
    # Load model
    input_size = window_size * (STATE_COLS + CONTROL_COLS)
    model = get_model(input_size=input_size, model_type="stacked_lstm_reg", window_size=window_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Initialize and fit scaler
    scaler = initialize_scaler(train_files)
    
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
    all_results = compute_all_predictions_and_coverage(
        model, quantile_models, correction_factors, test_files, window_size, scaler
    )
    
    random.seed(46)
    selected_results = random.choice(all_results)
    plot_selected_file_results(selected_results, save_dir)
    
    print(f"\nCQR pipeline completed for {cluster_name}")
    print(f"Results saved in: {save_dir}")

def main():
    cluster_configs = [
        ("results/ann/model_cluster0_run1752226124.pt", "Data/clustered/cluster0"),
        ("results/ann/model_cluster1_run1752226909.pt", "Data/clustered/cluster1"),
    ]
    results_dir = "results/ann"
    window_size = 10
    
    for model_path, data_path in cluster_configs:
        run_cqr_pipeline(model_path, data_path, results_dir, window_size)

if __name__ == "__main__":
    main()