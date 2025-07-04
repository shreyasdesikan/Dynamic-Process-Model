#!/usr/bin/env python3
"""
Run CQR (Conformalized Quantile Regression) on a trained NARX model
Usage: python run_cqr.py --cluster 0 --model_path ../results/ann/model_cluster0_run1751300407.pt
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cqr_implementation import run_cqr_pipeline, ConformizedQuantileRegression
from train import load_batch_data, clean_state_spikes

def load_cqr_model(cqr_model_path):
    """Load a saved CQR model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cqr_model_path, map_location=device)
    
    # Reconstruct the model
    from models.ANN.narx_model import get_model
    from cqr_implementation import QuantileRegressor
    
    window_size = checkpoint['window_size']
    input_size = window_size * 13  # 6 states + 7 controls
    
    # Load NARX model
    narx_model = get_model(input_size=input_size, model_type="ann").to(device)
    narx_model.load_state_dict(checkpoint['narx_model_state'])
    
    # Load quantile models
    quantile_models = {}
    for state_name, state_models in checkpoint['quantile_models_state'].items():
        quantile_models[state_name] = {}
        for level, state_dict in state_models.items():
            qr_model = QuantileRegressor(input_size).to(device)
            qr_model.load_state_dict(state_dict)
            quantile_models[state_name][level] = qr_model
    
    # Create CQR model
    cqr_model = ConformizedQuantileRegression(
        narx_model, 
        quantile_models, 
        checkpoint['scaler'],
        checkpoint['window_size'],
        checkpoint['alpha']
    )
    cqr_model.correction_factors = checkpoint['correction_factors']
    
    return cqr_model

def predict_single_file(cqr_model, file_path, output_path=None):
    """Make predictions with uncertainty for a single file"""
    # Load and preprocess data
    X, Y = load_batch_data(file_path, cqr_model.window_size, cqr_model.scaler)
    
    # Make predictions
    y_pred, lower, upper = cqr_model.predict_with_intervals(X)
    
    # Unscale predictions
    def unscale(arr):
        pad = np.zeros((arr.shape[0], 7))  # 7 control columns
        padded = np.hstack([arr, pad])
        return cqr_model.scaler.inverse_transform(padded)[:, :6]
    
    y_true_unscaled = unscale(Y)
    y_pred_unscaled = unscale(y_pred)
    lower_unscaled = unscale(lower)
    upper_unscaled = unscale(upper)
    
    # Calculate metrics
    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
    
    print(f"\nPrediction results for: {os.path.basename(file_path)}")
    print("-" * 60)
    
    for i, state in enumerate(state_names):
        mae = np.mean(np.abs(y_true_unscaled[:, i] - y_pred_unscaled[:, i]))
        coverage = np.mean((y_true_unscaled[:, i] >= lower_unscaled[:, i]) & 
                          (y_true_unscaled[:, i] <= upper_unscaled[:, i]))
        avg_width = np.mean(upper_unscaled[:, i] - lower_unscaled[:, i])
        
        print(f"{state}:")
        print(f"  MAE: {mae:.6f}")
        print(f"  Coverage: {coverage:.3f}")
        print(f"  Avg Interval Width: {avg_width:.6f}")
    
    # Save results if requested
    if output_path:
        results = {
            'file': file_path,
            'timestamp': datetime.now().isoformat(),
            'y_true': y_true_unscaled,
            'y_pred': y_pred_unscaled,
            'lower_bounds': lower_unscaled,
            'upper_bounds': upper_unscaled,
            'state_names': state_names
        }
        np.savez(output_path, **results)
        print(f"\nResults saved to: {output_path}")
    
    return y_pred_unscaled, lower_unscaled, upper_unscaled

def main():
    parser = argparse.ArgumentParser(description='Run CQR on trained NARX model')
    parser.add_argument('--cluster', type=int, required=True, help='Cluster ID')
    parser.add_argument('--model_path', type=str, help='Path to trained NARX model')
    parser.add_argument('--cqr_model_path', type=str, help='Path to saved CQR model')
    parser.add_argument('--train_cqr', action='store_true', help='Train new CQR model')
    parser.add_argument('--predict_file', type=str, help='Make predictions for a specific file')
    parser.add_argument('--output', type=str, help='Output path for predictions')
    parser.add_argument('--window_size', type=int, default=5, help='Window size')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage rate (default: 0.1 for 90% intervals)')
    
    args = parser.parse_args()
    
    cluster_path = f"../Data/clustered/cluster{args.cluster}"
    
    if args.train_cqr:
        if not args.model_path:
            print("Error: --model_path required when training CQR")
            sys.exit(1)
            
        print(f"Training CQR for model: {args.model_path}")
        cqr_model = run_cqr_pipeline(
            cluster_id=args.cluster,
            cluster_path=cluster_path,
            model_path=args.model_path,
            window_size=args.window_size,
            alpha=args.alpha
        )
        
    elif args.cqr_model_path:
        print(f"Loading CQR model from: {args.cqr_model_path}")
        cqr_model = load_cqr_model(args.cqr_model_path)
        
        if args.predict_file:
            predict_single_file(cqr_model, args.predict_file, args.output)
    
    else:
        print("Error: Either --train_cqr or --cqr_model_path must be specified")
        sys.exit(1)

if __name__ == "__main__":
    main()