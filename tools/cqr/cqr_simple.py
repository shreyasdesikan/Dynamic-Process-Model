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

# Fix import paths - add parent directories to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools'))

from models.ANN.narx_model import get_model
from train import load_batch_data, clean_state_spikes


# Constants
STATE_COLS = 6
CONTROL_COLS = 7
state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

print("="*60)
print("CQR (Conformalized Quantile Regression) Implementation")
print("="*60)

# ============= Helper Classes =============

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

# ============= Core Functions =============

def pinball_loss(predictions, targets, tau):
    """Pinball loss for quantile regression"""
    errors = targets - predictions
    return torch.mean(torch.max(tau * errors, (tau - 1) * errors))

# ============= TASK 1: Generate Error Dataset and Train Quantile Regressors =============

def generate_error_dataset(model, data_files, window_size, scaler, device):
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
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            
            Y_pred = model(X_tensor).cpu()
            errors = Y_pred - Y_tensor
            
            all_inputs.extend(X)
            for j in range(STATE_COLS):
                all_errors[f'state_{j}'].extend(errors[:, j].numpy())
    
    print(f"✓ Generated {len(all_inputs)} error samples")
    return np.array(all_inputs), {k: np.array(v) for k, v in all_errors.items()}

def train_quantile_regressor(inputs, errors, tau, epochs=50, batch_size=64, lr=1e-3):
    """Train a quantile regressor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ErrorDataset(inputs, errors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = inputs.shape[1]
    model = QuantileRegressor(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
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
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    return model

# ============= TASK 2: Conformalize Prediction Intervals =============

def compute_conformity_scores(model, quantile_models, cal_files, window_size, scaler, device):
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
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
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
    alpha = 0.1  # For 90% coverage
    
    for state_name, scores in conformity_scores.items():
        scores = np.array(scores)
        n = len(scores)
        quantile_level = (1 - alpha) * (1 + 1/n)
        correction_factors[state_name] = np.quantile(scores, quantile_level)
        print(f"  {state_name}: correction factor = {correction_factors[state_name]:.6f}")
    
    return correction_factors

# ============= TASK 3: Analyze and Visualize Performance =============

def make_predictions_with_intervals(model, quantile_models, correction_factors, X, scaler, device):
    """Make predictions with conformalized intervals"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        Y_pred = model(X_tensor).cpu().numpy()
        
        lower_bounds = np.zeros_like(Y_pred)
        upper_bounds = np.zeros_like(Y_pred)
        
        for i in range(STATE_COLS):
            state_name = f'state_{i}'
            
            # Get error quantile predictions
            q_low = quantile_models[state_name]['low'](X_tensor).cpu().squeeze().numpy()
            q_high = quantile_models[state_name]['high'](X_tensor).cpu().squeeze().numpy()
            
            # Apply conformalization
            correction = correction_factors[state_name]
            lower_bounds[:, i] = Y_pred[:, i] + q_low - correction
            upper_bounds[:, i] = Y_pred[:, i] + q_high + correction
    
    return Y_pred, lower_bounds, upper_bounds

def visualize_results(Y_true, Y_pred, lower_bounds, upper_bounds, scaler, save_dir):
    """Create comprehensive visualizations"""
    print("\nTask 3: Creating visualizations...")
    
    # Unscale for visualization
    def unscale(arr):
        pad = np.zeros((arr.shape[0], CONTROL_COLS))
        padded = np.hstack([arr, pad])
        return scaler.inverse_transform(padded)[:, :STATE_COLS]
    
    Y_true_uns = unscale(Y_true)
    Y_pred_uns = unscale(Y_pred)
    lower_uns = unscale(lower_bounds)
    upper_uns = unscale(upper_bounds)
    
    # 1. Time series plots
    n_samples = min(500, len(Y_true))
    fig, axes = plt.subplots(6, 1, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, state) in enumerate(zip(axes, state_names)):
        time_steps = np.arange(n_samples)
        
        ax.plot(time_steps, Y_true_uns[:n_samples, i], 'k-', label='True', alpha=0.8)
        ax.plot(time_steps, Y_pred_uns[:n_samples, i], 'b--', label='Predicted', alpha=0.8)
        ax.fill_between(time_steps, 
                       lower_uns[:n_samples, i], 
                       upper_uns[:n_samples, i],
                       alpha=0.3, color='blue', label='90% PI')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(state)
        ax.set_title(f'{state} - Predictions with Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cqr_time_series.png", dpi=300)
    plt.close()
    
    # 2. Coverage analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coverages = []
    for i in range(STATE_COLS):
        coverage = np.mean((Y_true_uns[:, i] >= lower_uns[:, i]) & 
                          (Y_true_uns[:, i] <= upper_uns[:, i]))
        coverages.append(coverage)
    
    bars = ax.bar(state_names, coverages)
    ax.axhline(y=0.9, color='r', linestyle='--', label='Target Coverage (90%)')
    
    for bar, cov in zip(bars, coverages):
        if cov < 0.88:
            bar.set_color('red')
        elif cov > 0.92:
            bar.set_color('orange')
        else:
            bar.set_color('green')
        
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cov:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage Analysis by State Variable')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cqr_coverage.png", dpi=300)
    plt.close()
    
    # 3. Interval widths
    fig, ax = plt.subplots(figsize=(10, 6))
    
    widths = []
    for i in range(STATE_COLS):
        width = np.mean(upper_uns[:, i] - lower_uns[:, i])
        widths.append(width)
    
    bars = ax.bar(state_names, widths)
    ax.set_ylabel('Average Interval Width')
    ax.set_title('Average Prediction Interval Width by State')
    ax.grid(True, alpha=0.3)
    
    for bar, w in zip(bars, widths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{w:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cqr_widths.png", dpi=300)
    plt.close()
    
    print("✓ Visualizations saved")
    
    # Print summary
    print("\n📊 Summary Statistics:")
    print(f"{'State':<10} {'Coverage':<10} {'Avg Width':<10} {'MAE':<10}")
    print("-" * 40)
    for i, state in enumerate(state_names):
        mae = np.mean(np.abs(Y_true_uns[:, i] - Y_pred_uns[:, i]))
        print(f"{state:<10} {coverages[i]:<10.3f} {widths[i]:<10.4f} {mae:<10.4f}")

# ============= Main Pipeline =============

def main():
    """Run complete CQR pipeline"""
        
    model_path = "results/ann/model_cluster0_run1751300407.pt"
    data_path = "Data/clustered/cluster0"
    results_dir = "results/ann"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    window_size = 5
    alpha = 0.1  # For 90% coverage
    
    # Load data files
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".txt")]
    
    # Split data
    train_files, temp_files = train_test_split(data_files, test_size=0.3, random_state=42)
    cal_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    print(f"Training: {len(train_files)} files")
    print(f"Calibration: {len(cal_files)} files")
    print(f"Test: {len(test_files)} files")
    
    # Load model
    input_size = window_size * (STATE_COLS + CONTROL_COLS)
    model = get_model(input_size=input_size, model_type="bilstm_multihead").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Initialize scaler
    scaler = MinMaxScaler()
    for file in train_files[:5]:
        raw = np.loadtxt(file, skiprows=1)
        cleaned = clean_state_spikes(raw)
        scaler.partial_fit(cleaned)
    
    # TASK 1: Generate error dataset and train quantile regressors
    inputs, errors_dict = generate_error_dataset(model, train_files, window_size, scaler, device)
    
    print("\nTraining quantile regressors...")
    quantile_models = {}
    
    for i in range(STATE_COLS):
        state_name = f'state_{i}'
        print(f"\n  Training for {state_names[i]}...")
        
        quantile_models[state_name] = {}
        
        # Lower quantile (0.05)
        print(f"    Training τ=0.05...")
        model_low = train_quantile_regressor(inputs, errors_dict[state_name], tau=0.05)
        quantile_models[state_name]['low'] = model_low
        
        # Upper quantile (0.95)
        print(f"    Training τ=0.95...")
        model_high = train_quantile_regressor(inputs, errors_dict[state_name], tau=0.95)
        quantile_models[state_name]['high'] = model_high
    
    # TASK 2: Conformalize prediction intervals
    correction_factors = compute_conformity_scores(
        model, quantile_models, cal_files, window_size, scaler, device
    )
    
    # TASK 3: Evaluate on test data
    print("\nEvaluating on test data...")
    
    # Collect test predictions
    all_Y_true = []
    all_Y_pred = []
    all_lower = []
    all_upper = []
    
    for file_path in test_files[:10]:  # Use subset for speed
        X, Y = load_batch_data(file_path, window_size, scaler)
        Y_pred, lower, upper = make_predictions_with_intervals(
            model, quantile_models, correction_factors, X, scaler, device
        )
        
        all_Y_true.append(Y)
        all_Y_pred.append(Y_pred)
        all_lower.append(lower)
        all_upper.append(upper)
    
    # Concatenate results
    Y_true_all = np.vstack(all_Y_true)
    Y_pred_all = np.vstack(all_Y_pred)
    lower_all = np.vstack(all_lower)
    upper_all = np.vstack(all_upper)
    
    # Create visualizations
    save_dir = os.path.join(results_dir, f"cqr_results")
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_results(Y_true_all, Y_pred_all, lower_all, upper_all, scaler, save_dir)
    
    # Save CQR model
    save_path = os.path.join(save_dir, "cqr_model.pt")
    torch.save({
        'narx_model_state': model.state_dict(),
        'quantile_models_state': {
            state: {level: m.state_dict() for level, m in models.items()}
            for state, models in quantile_models.items()
        },
        'correction_factors': correction_factors,
        'window_size': window_size,
        'alpha': alpha
    }, save_path)
    
    print(f"\nCQR model saved to: {save_path}")
    print("\nCQR pipeline completed successfully!")
    print(f"Results saved in: {save_dir}")

if __name__ == "__main__":
    main()