import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
import sys


current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools'))


from models.ANN.narx_model import get_model
from train import load_batch_data, clean_state_spikes

STATE_COLS = 6
CONTROL_COLS = 7

# ============= Task 1: Generate Error Datasets and Train Quantile Regressors =============

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
    """
    Pinball loss for quantile regression
    tau: quantile level (e.g., 0.05 for 5th percentile)
    """
    errors = targets - predictions
    return torch.mean(torch.max(tau * errors, (tau - 1) * errors))

def generate_error_dataset(model, data_files, window_size, scaler, device):
    """
    Generate error dataset from trained NARX model
    Returns: inputs, errors for each state variable
    """
    model.eval()
    all_inputs = []
    all_errors = {f'state_{i}': [] for i in range(STATE_COLS)}
    
    with torch.no_grad():
        for file_path in data_files:
            X, Y = load_batch_data(file_path, window_size, scaler)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            # Get predictions
            Y_pred = model(X_tensor).cpu()
            
            # Calculate errors for each state
            errors = Y_pred - Y_tensor
            
            all_inputs.extend(X)
            for i in range(STATE_COLS):
                all_errors[f'state_{i}'].extend(errors[:, i].numpy())
    
    return np.array(all_inputs), {k: np.array(v) for k, v in all_errors.items()}

def train_quantile_regressor(inputs, errors, tau, epochs=100, batch_size=64, lr=1e-3):
    """
    Train a quantile regressor for a specific quantile level
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader
    dataset = ErrorDataset(inputs, errors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = inputs.shape[1]
    model = QuantileRegressor(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_inputs, batch_errors in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_errors = batch_errors.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs).squeeze()
            loss = pinball_loss(predictions, batch_errors, tau)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses

# ============= Task 2: Conformalize Prediction Intervals =============

class ConformizedQuantileRegression:
    """
    Implementation of Conformalized Quantile Regression (CQR)
    """
    def __init__(self, model, quantile_models, scaler, window_size, alpha=0.1):
        self.model = model  # Main NARX model
        self.quantile_models = quantile_models  # Dict of quantile regressors
        self.scaler = scaler
        self.window_size = window_size
        self.alpha = alpha
        self.device = next(model.parameters()).device
        
    def compute_conformity_scores(self, cal_data_files):
        """
        Compute conformity scores on calibration data
        """
        conformity_scores = {f'state_{i}': [] for i in range(STATE_COLS)}
        
        self.model.eval()
        for qr in self.quantile_models.values():
            for m in qr.values():
                m.eval()
        
        with torch.no_grad():
            for file_path in cal_data_files:
                X, Y = load_batch_data(file_path, self.window_size, self.scaler)
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                Y_tensor = torch.tensor(Y, dtype=torch.float32)
                
                # Get predictions from main model
                Y_pred = self.model(X_tensor).cpu()
                
                # Get quantile predictions
                for i in range(STATE_COLS):
                    state_name = f'state_{i}'
                    
                    # Lower quantile prediction
                    q_low = self.quantile_models[state_name]['low'](X_tensor).cpu().squeeze()
                    # Upper quantile prediction  
                    q_high = self.quantile_models[state_name]['high'](X_tensor).cpu().squeeze()
                    
                    # Compute conformity scores
                    y_true = Y_tensor[:, i]
                    y_pred = Y_pred[:, i]
                    
                    # E_i = max(q_low - y_true, y_true - q_high)
                    lower_error = (y_pred + q_low) - y_true
                    upper_error = y_true - (y_pred + q_high)
                    scores = torch.maximum(lower_error, upper_error)
                    
                    conformity_scores[state_name].extend(scores.numpy())
        
        # Compute correction factors (quantiles of conformity scores)
        self.correction_factors = {}
        for state_name, scores in conformity_scores.items():
            scores = np.array(scores)
            # Use (1-alpha)*(1+1/n) quantile as in the paper
            n = len(scores)
            quantile_level = (1 - self.alpha) * (1 + 1/n)
            self.correction_factors[state_name] = np.quantile(scores, quantile_level)
            
        return conformity_scores
    
    def predict_with_intervals(self, X):
        """
        Make predictions with conformalized prediction intervals
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Main prediction
            Y_pred = self.model(X_tensor).cpu().numpy()
            
            # Initialize interval bounds
            lower_bounds = np.zeros_like(Y_pred)
            upper_bounds = np.zeros_like(Y_pred)
            
            # Compute intervals for each state
            for i in range(STATE_COLS):
                state_name = f'state_{i}'
                
                # Get quantile predictions
                q_low = self.quantile_models[state_name]['low'](X_tensor).cpu().squeeze().numpy()
                q_high = self.quantile_models[state_name]['high'](X_tensor).cpu().squeeze().numpy()
                
                # Apply conformalization
                correction = self.correction_factors[state_name]
                lower_bounds[:, i] = Y_pred[:, i] + q_low - correction
                upper_bounds[:, i] = Y_pred[:, i] + q_high + correction
        
        return Y_pred, lower_bounds, upper_bounds

# ============= Task 3: Analyze and Visualize CQR Performance =============

def calculate_coverage(y_true, lower_bounds, upper_bounds):
    """Calculate empirical coverage for each state variable"""
    coverage = {}
    for i in range(STATE_COLS):
        in_interval = (y_true[:, i] >= lower_bounds[:, i]) & (y_true[:, i] <= upper_bounds[:, i])
        coverage[f'state_{i}'] = np.mean(in_interval)
    return coverage

def calculate_interval_widths(lower_bounds, upper_bounds):
    """Calculate interval widths for each state variable"""
    widths = {}
    for i in range(STATE_COLS):
        widths[f'state_{i}'] = upper_bounds[:, i] - lower_bounds[:, i]
    return widths

def visualize_cqr_performance(cqr_model, test_files, window_size, run_id, cluster_id):
    """
    Comprehensive visualization of CQR performance
    """
    os.makedirs(f"../results/ann/cqr_{run_id}", exist_ok=True)
    
    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
    
    # Collect predictions and metrics
    all_y_true = []
    all_y_pred = []
    all_lower = []
    all_upper = []
    
    for file_path in test_files:
        X, Y = load_batch_data(file_path, window_size, cqr_model.scaler)
        y_pred, lower, upper = cqr_model.predict_with_intervals(X)
        
        all_y_true.append(Y)
        all_y_pred.append(y_pred)
        all_lower.append(lower)
        all_upper.append(upper)
    
    # Concatenate all results
    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)
    all_lower = np.vstack(all_lower)
    all_upper = np.vstack(all_upper)
    
    # Unscale for visualization
    def unscale(arr):
        pad = np.zeros((arr.shape[0], CONTROL_COLS))
        padded = np.hstack([arr, pad])
        return cqr_model.scaler.inverse_transform(padded)[:, :STATE_COLS]
    
    y_true_unscaled = unscale(all_y_true)
    y_pred_unscaled = unscale(all_y_pred)
    lower_unscaled = unscale(all_lower)
    upper_unscaled = unscale(all_upper)
    
    # 1. Time series plots with prediction intervals
    n_samples = min(500, len(y_true_unscaled))  # Plot first 500 points for clarity
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, state) in enumerate(zip(axes, state_names)):
        time_steps = np.arange(n_samples)
        
        # Plot true values
        ax.plot(time_steps, y_true_unscaled[:n_samples, i], 'k-', label='True', alpha=0.8)
        
        # Plot predictions
        ax.plot(time_steps, y_pred_unscaled[:n_samples, i], 'b--', label='Predicted', alpha=0.8)
        
        # Plot prediction intervals
        ax.fill_between(time_steps, 
                       lower_unscaled[:n_samples, i], 
                       upper_unscaled[:n_samples, i],
                       alpha=0.3, color='blue', label='90% PI')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(state)
        ax.set_title(f'{state} - CQR Predictions with Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"../results/ann/cqr_{run_id}/time_series_cluster{cluster_id}.png", dpi=300)
    plt.close()
    
    # 2. Coverage analysis
    coverage = calculate_coverage(y_true_unscaled, lower_unscaled, upper_unscaled)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    states = list(coverage.keys())
    coverages = [coverage[s] for s in states]
    
    bars = ax.bar(state_names, coverages)
    ax.axhline(y=0.9, color='r', linestyle='--', label='Target Coverage (90%)')
    
    # Color bars based on coverage
    for bar, cov in zip(bars, coverages):
        if cov < 0.88:
            bar.set_color('red')
        elif cov > 0.92:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(f'Coverage Analysis - Cluster {cluster_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add coverage values on bars
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cov:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"../results/ann/cqr_{run_id}/coverage_analysis_cluster{cluster_id}.png", dpi=300)
    plt.close()
    
    # 3. Interval width analysis
    widths = calculate_interval_widths(lower_unscaled, upper_unscaled)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (ax, state) in enumerate(zip(axes, state_names)):
        ax.hist(widths[f'state_{i}'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Interval Width')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{state} - Interval Width Distribution')
        ax.axvline(np.mean(widths[f'state_{i}']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(widths[f"state_{i}"]):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"../results/ann/cqr_{run_id}/interval_widths_cluster{cluster_id}.png", dpi=300)
    plt.close()
    
    # 4. Reliability diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute coverage at different confidence levels
    confidence_levels = np.arange(0.5, 1.0, 0.05)
    empirical_coverages = []
    
    for conf in confidence_levels:
        # Recompute intervals for this confidence level
        alpha = 1 - conf
        factor = (1 - alpha) * (1 + 1/len(all_y_true))
        
        temp_coverage = []
        for i in range(STATE_COLS):
            # Simple approximation - scale existing intervals
            scale = np.quantile(widths[f'state_{i}'], factor) / np.mean(widths[f'state_{i}'])
            temp_lower = y_pred_unscaled[:, i] - scale * (y_pred_unscaled[:, i] - lower_unscaled[:, i])
            temp_upper = y_pred_unscaled[:, i] + scale * (upper_unscaled[:, i] - y_pred_unscaled[:, i])
            
            in_interval = (y_true_unscaled[:, i] >= temp_lower) & (y_true_unscaled[:, i] <= temp_upper)
            temp_coverage.append(np.mean(in_interval))
        
        empirical_coverages.append(np.mean(temp_coverage))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(confidence_levels, empirical_coverages, 'bo-', label='CQR')
    ax.set_xlabel('Target Coverage')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"../results/ann/cqr_{run_id}/reliability_diagram_cluster{cluster_id}.png", dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== CQR Performance Summary ===")
    print(f"Cluster: {cluster_id}")
    for i, state in enumerate(state_names):
        print(f"\n{state}:")
        print(f"  Coverage: {coverage[f'state_{i}']:.3f}")
        print(f"  Mean Interval Width: {np.mean(widths[f'state_{i}']):.4f}")
        print(f"  Std Interval Width: {np.std(widths[f'state_{i}']):.4f}")

# ============= Main CQR Pipeline =============

def run_cqr_pipeline(cluster_id, cluster_path, model_path, window_size=5, alpha=0.1):
    """
    Complete CQR pipeline for a trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data files
    all_files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".txt")]
    
    # Split into train, calibration, and test
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
    cal_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    print(f"Files - Train: {len(train_files)}, Calibration: {len(cal_files)}, Test: {len(test_files)}")
    
    # Initialize scaler and load model
    scaler = MinMaxScaler()
    
    # Load the trained NARX model
    input_size = window_size * (STATE_COLS + CONTROL_COLS)
    model = get_model(input_size=input_size, model_type="bilstm_multihead").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Fit scaler on train data
    for file in train_files[:5]:  # Use first few files to fit scaler
        raw = np.loadtxt(file, skiprows=1)
        cleaned = clean_state_spikes(raw)
        scaler.partial_fit(cleaned)
    
    print("\n=== Task 1: Generate Error Dataset ===")
    # Generate error dataset
    inputs, errors_dict = generate_error_dataset(model, train_files, window_size, scaler, device)
    print(f"Generated error dataset with {len(inputs)} samples")
    
    print("\n=== Task 1: Train Quantile Regressors ===")
    # Train quantile regressors for each state
    quantile_models = {}
    
    for i in range(STATE_COLS):
        state_name = f'state_{i}'
        print(f"\nTraining quantile regressors for {state_name}")
        
        quantile_models[state_name] = {}
        
        # Train lower quantile (0.05)
        print(f"Training tau=0.05 quantile regressor...")
        model_low, losses_low = train_quantile_regressor(
            inputs, errors_dict[state_name], tau=0.05, epochs=50
        )
        quantile_models[state_name]['low'] = model_low
        
        # Train upper quantile (0.95)
        print(f"Training tau=0.95 quantile regressor...")
        model_high, losses_high = train_quantile_regressor(
            inputs, errors_dict[state_name], tau=0.95, epochs=50
        )
        quantile_models[state_name]['high'] = model_high
    
    print("\n=== Task 2: Conformalize Prediction Intervals ===")
    # Create CQR model
    cqr_model = ConformizedQuantileRegression(
        model, quantile_models, scaler, window_size, alpha=alpha
    )
    
    # Compute conformity scores on calibration data
    conformity_scores = cqr_model.compute_conformity_scores(cal_files)
    
    print("\nCorrection factors:")
    for state_name, factor in cqr_model.correction_factors.items():
        print(f"  {state_name}: {factor:.6f}")
    
    print("\n=== Task 3: Analyze and Visualize Performance ===")
    # Visualize performance on test data
    run_id = model_path.split('run')[1].split('.pt')[0]
    visualize_cqr_performance(cqr_model, test_files, window_size, run_id, cluster_id)
    
    # Save CQR model
    cqr_save_path = f"../results/ann/cqr_model_cluster{cluster_id}_run{run_id}.pt"
    torch.save({
        'narx_model_state': model.state_dict(),
        'quantile_models_state': {
            state: {level: m.state_dict() for level, m in models.items()}
            for state, models in quantile_models.items()
        },
        'correction_factors': cqr_model.correction_factors,
        'scaler': scaler,
        'window_size': window_size,
        'alpha': alpha
    }, cqr_save_path)
    print(f"\nCQR model saved to: {cqr_save_path}")
    
    return cqr_model

# ============= Example Usage =============

if __name__ == "__main__":
    # Configuration
    cluster_id = 0
    cluster_path = "Data/clustered/cluster0"
    model_path = "results/ann/model_cluster0_run1751300407.pt"
    
    # Run the complete CQR pipeline
    cqr_model = run_cqr_pipeline(
        cluster_id=cluster_id,
        cluster_path=cluster_path,
        model_path=model_path,
        window_size=5,
        alpha=0.1  # For 90% prediction intervals
    )
    
    print("\nCQR Pipeline completed successfully!")