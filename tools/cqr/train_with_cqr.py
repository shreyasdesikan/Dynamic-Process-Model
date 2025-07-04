"""
Extended training script that includes CQR functionality
This can be used as a drop-in replacement for train.py with added CQR capabilities
"""

import argparse
import os
import time
import torch
from pathlib import Path

# Import all original functions from train.py
from train import *
from cqr_implementation import run_cqr_pipeline

def train_cluster_with_cqr(cluster_id, cluster_path, args, run_id):
    """
    Extended version of train_cluster that includes CQR training
    """
    # First, train the regular model using the original function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if we should use a saved model
    if args.use_saved_model is not None:
        model_path = f"../results/ann/model_cluster{cluster_id}_run{args.use_saved_model}.pt"
        if os.path.exists(model_path):
            print(f"🔁 Using saved model from {model_path}")
            
            # Load model for evaluation
            all_files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".txt")]
            train_files, test_files = split_data(all_files, test_ratio=0.1)
            
            scaler = MinMaxScaler()
            all_test_X, all_test_Y = [], []
            for file in test_files:
                X, Y = load_batch_data(file, args.window_size, scaler)
                all_test_X.append(X)
                all_test_Y.append(Y)
            test_X = np.concatenate(all_test_X, axis=0)
            test_Y = np.concatenate(all_test_Y, axis=0)
            
            model = get_model(input_size=(args.window_size * (STATE_COLS + CONTROL_COLS)), 
                            model_type=args.model, window_size=args.window_size).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            evaluate_model(model, test_X, test_Y, run_id=args.use_saved_model, 
                          cluster_id=cluster_id, scaler=scaler)
            plot_sample_prediction(model, scaler, test_files, args.window_size, 
                                 run_id=args.use_saved_model, cluster_id=cluster_id)
            
            # Now run CQR if requested
            if args.train_cqr:
                print(f"\n🔬 Training CQR for cluster {cluster_id}")
                cqr_model = run_cqr_pipeline(
                    cluster_id=cluster_id,
                    cluster_path=cluster_path,
                    model_path=model_path,
                    window_size=args.window_size,
                    alpha=args.cqr_alpha
                )
            return
    
    # Otherwise, train a new model
    print(f"\n🔧 Training new model for Cluster {cluster_id}")
    
    # Original training code
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
        X, Y = load_batch_data(file, args.window_size, scaler)
        all_test_X.append(X)
        all_test_Y.append(Y)
    test_X = np.concatenate(all_test_X, axis=0)
    test_Y = np.concatenate(all_test_Y, axis=0)

    model = get_model(input_size=(args.window_size * (STATE_COLS + CONTROL_COLS)), 
                     model_type=args.model, window_size=args.window_size).to(device)
    criterion = get_loss(use_huber=True)
    optimizer = get_optimizer(model, lr=args.lr, use_adamw=True)
    
    train_losses, val_losses = [], []
    num_epochs = args.epochs
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
        
        # Early stopping check
        if args.early_stopping:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop_patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}. No improvement for {args.early_stop_patience} epochs.")
                    break

    if args.early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    model_path = None
    if args.save_model:
        model_path = f"../results/ann/model_cluster{cluster_id}_run{run_id}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
    
    if args.log_hyperparams:
        log_hyperparams(run_id, args, cluster_id)

    plot_losses(train_losses, val_losses, run_id=run_id, log_hyperparams=args.log_hyperparams,
                log_scale=args.log_scale, model_type=args.model)

    evaluate_model(model, test_X, test_Y, run_id=run_id, cluster_id=cluster_id, scaler=scaler)
    plot_sample_prediction(model, scaler, test_files, args.window_size, run_id=run_id, cluster_id=cluster_id)
    
    # Train CQR if requested
    if args.train_cqr and model_path:
        print(f"\n🔬 Training CQR for cluster {cluster_id}")
        cqr_model = run_cqr_pipeline(
            cluster_id=cluster_id,
            cluster_path=cluster_path,
            model_path=model_path,
            window_size=args.window_size,
            alpha=args.cqr_alpha
        )
        
        # Additional CQR evaluation
        if args.evaluate_cqr:
            evaluate_cqr_on_test(cqr_model, test_files, run_id, cluster_id)

def evaluate_cqr_on_test(cqr_model, test_files, run_id, cluster_id):
    """
    Evaluate CQR model on test data and save results
    """
    print(f"\n📊 Evaluating CQR on test data...")
    
    all_coverages = []
    all_widths = []
    state_names = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
    
    for file_path in test_files[:5]:  # Evaluate on subset
        X, Y = load_batch_data(file_path, cqr_model.window_size, cqr_model.scaler)
        y_pred, lower, upper = cqr_model.predict_with_intervals(X)
        
        # Calculate coverage and width for each state
        for i in range(STATE_COLS):
            coverage = np.mean((Y[:, i] >= lower[:, i]) & (Y[:, i] <= upper[:, i]))
            width = np.mean(upper[:, i] - lower[:, i])
            all_coverages.append(coverage)
            all_widths.append(width)
    
    # Save results
    results_path = f"../results/ann/cqr_evaluation_cluster{cluster_id}_run{run_id}.txt"
    with open(results_path, 'w') as f:
        f.write(f"CQR Evaluation Results\n")
        f.write(f"Cluster: {cluster_id}, Run: {run_id}\n")
        f.write(f"Average Coverage: {np.mean(all_coverages):.3f}\n")
        f.write(f"Average Width: {np.mean(all_widths):.4f}\n")
        f.write(f"Coverage Std: {np.std(all_coverages):.3f}\n")
        f.write(f"Width Std: {np.std(all_widths):.4f}\n")
    
    print(f"✅ CQR evaluation saved to {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Train NARX models with optional CQR')
    
    # Original arguments
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
    parser.add_argument("--test_zscore_threshold", action="store_true", help="Run Z-score threshold test and exit")
    
    # New CQR arguments
    parser.add_argument("--train_cqr", action="store_true", help="Train CQR after model training")
    parser.add_argument("--cqr_alpha", type=float, default=0.1, help="Miscoverage rate for CQR (default: 0.1 for 90% intervals)")
    parser.add_argument("--evaluate_cqr", action="store_true", help="Evaluate CQR on test data")
    parser.add_argument("--cqr_only", action="store_true", help="Only train CQR on existing models")
    
    args = parser.parse_args()

    base_path = "../Data/clustered"
    run_id = args.use_saved_model if args.use_saved_model is not None else int(time.time())
    
    # Create results directory
    Path("../results/ann").mkdir(parents=True, exist_ok=True)

    if args.train_all:
        clusters = [int(folder.replace("cluster", "")) for folder in os.listdir(base_path) if folder.startswith("cluster")]
    else:
        clusters = args.clusters or []
    
    if args.test_zscore_threshold:
        cluster_0_path = os.path.join(base_path, "cluster0")
        all_files = [os.path.join(cluster_0_path, f) for f in os.listdir(cluster_0_path) if f.endswith(".txt")]
        sample_file = random.choice(all_files)
        print(f"\n📊 Running Z-score trial on: {sample_file}")
        zscore_threshold_trial(sample_file)
        return
    
    # CQR only mode
    if args.cqr_only:
        for cluster_id in clusters:
            cluster_path = os.path.join(base_path, f"cluster{cluster_id}")
            model_path = f"../results/ann/model_cluster{cluster_id}_run{run_id}.pt"
            
            if os.path.exists(model_path):
                print(f"\n🔬 Training CQR for existing model in Cluster {cluster_id}")
                cqr_model = run_cqr_pipeline(
                    cluster_id=cluster_id,
                    cluster_path=cluster_path,
                    model_path=model_path,
                    window_size=args.window_size,
                    alpha=args.cqr_alpha
                )
            else:
                print(f"⚠️ Model not found: {model_path}")
        return

    # Regular training (with optional CQR)
    for cluster_id in clusters:
        cluster_path = os.path.join(base_path, f"cluster{cluster_id}")
        print(f"\n{'='*60}")
        print(f"Processing Cluster {cluster_id}")
        print(f"{'='*60}")
        train_cluster_with_cqr(cluster_id, cluster_path, args, run_id)

if __name__ == "__main__":
    main()