import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def compare_model_logs(log1_path, log2_path, output_dir="comparison_plots"):
    """
    Compare two model training logs and generate visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the log files
    log1 = pd.read_csv(log1_path)
    log2 = pd.read_csv(log2_path)
    
    print(f"Loaded log files:")
    print(f"  - {log1_path}: {len(log1)} rows")
    print(f"  - {log2_path}: {len(log2)} rows")
    
    # Get all metrics (excluding epoch and lr)
    metrics = [col for col in log1.columns if col not in ['epoch', 'lr']]
    
    # Create individual plots for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        plt.plot(log1['epoch'], log1[metric], 'b-', linewidth=2, label='Model 1 (Original)')
        plt.plot(log2['epoch'], log2[metric], 'r-', linewidth=2, label='Model 2 (Improved)')
        
        # Add titles and labels
        plt.title(f'Comparison of {metric.title()}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        
        # Format y-label based on metric type
        if 'loss' in metric:
            plt.ylabel('Loss', fontsize=14)
        elif 'iou' in metric or 'dice' in metric:
            plt.ylabel('Score', fontsize=14)
        else:
            plt.ylabel(metric.title(), fontsize=14)
            
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Highlight best points for model 2
        if 'loss' in metric:
            best_epoch = log2['epoch'][log2[metric].idxmin()]
            best_value = log2[metric].min()
            plt.scatter(best_epoch, best_value, color='red', s=100, zorder=5)
            plt.annotate(f'Best: {best_value:.4f}', 
                         (best_epoch, best_value), 
                         xytext=(10, -20), 
                         textcoords='offset points',
                         fontsize=12)
        else:  # For metrics where higher is better
            best_epoch = log2['epoch'][log2[metric].idxmax()]
            best_value = log2[metric].max()
            plt.scatter(best_epoch, best_value, color='red', s=100, zorder=5)
            plt.annotate(f'Best: {best_value:.4f}', 
                         (best_epoch, best_value), 
                         xytext=(10, 10), 
                         textcoords='offset points',
                         fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=150)
        plt.close()
        print(f"Created plot for {metric}")
    
    # Create combined plots for train vs validation
    pairs = [
        ('loss', 'val_loss'),
        ('iou', 'val_iou')
    ]
    
    for train_metric, val_metric in pairs:
        if train_metric in metrics and val_metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # Training metrics
            plt.plot(log1['epoch'], log1[train_metric], 'b-', linewidth=2, label='Model 1 - Training')
            plt.plot(log2['epoch'], log2[train_metric], 'r-', linewidth=2, label='Model 2 - Training')
            
            # Validation metrics
            plt.plot(log1['epoch'], log1[val_metric], 'b--', linewidth=2, label='Model 1 - Validation')
            plt.plot(log2['epoch'], log2[val_metric], 'r--', linewidth=2, label='Model 2 - Validation')
            
            metric_name = train_metric.replace('_', ' ').title()
            plt.title(f'Training vs Validation {metric_name}', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            
            if 'loss' in train_metric:
                plt.ylabel('Loss', fontsize=14)
            else:
                plt.ylabel('Score', fontsize=14)
                
            plt.legend(fontsize=12)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{train_metric}_vs_{val_metric}.png'), dpi=150)
            plt.close()
            print(f"Created combined plot for {train_metric} vs {val_metric}")
    
    # Create metrics improvement summary
    summary_data = {
        'Metric': [],
        'Model 1 (Best)': [],
        'Model 2 (Best)': [],
        'Improvement': [],
        'Relative Improvement (%)': []
    }
    
    for metric in metrics:
        summary_data['Metric'].append(metric)
        
        if 'loss' in metric:  # For loss metrics, lower is better
            best1 = log1[metric].min()
            best2 = log2[metric].min()
            improvement = best1 - best2
            rel_improvement = (improvement / best1) * 100 if best1 != 0 else np.inf
        else:  # For IOU/Dice metrics, higher is better
            best1 = log1[metric].max()
            best2 = log2[metric].max()
            improvement = best2 - best1
            rel_improvement = (improvement / best1) * 100 if best1 != 0 else np.inf
        
        summary_data['Model 1 (Best)'].append(f"{best1:.4f}")
        summary_data['Model 2 (Best)'].append(f"{best2:.4f}")
        summary_data['Improvement'].append(f"{improvement:.4f}")
        summary_data['Relative Improvement (%)'].append(f"{rel_improvement:.2f}%")
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'improvement_summary.csv'), index=False)
    print(f"Saved improvement summary to {os.path.join(output_dir, 'improvement_summary.csv')}")
    
    # Learning rate comparison
    plt.figure(figsize=(12, 6))
    plt.plot(log1['epoch'], log1['lr'], 'b-', linewidth=2, label='Model 1 (Original)')
    plt.plot(log2['epoch'], log2['lr'], 'r-', linewidth=2, label='Model 2 (Improved)')
    
    plt.title('Learning Rate Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.yscale('log')  # Use log scale for learning rate visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'), dpi=150)
    plt.close()
    print(f"Created learning rate comparison plot")
    
    print(f"\nAll comparison plots and summary have been saved to '{output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two model training logs')
    parser.add_argument('--log1', type=str, default='log.csv', help='Path to the first log file (original model)')
    parser.add_argument('--log2', type=str, default='log_up.csv', help='Path to the second log file (improved model)')
    parser.add_argument('--output', type=str, default='comparison_plots', help='Directory to save comparison plots')
    
    args = parser.parse_args()
    
    # Generate comparison plots
    compare_model_logs(args.log1, args.log2, args.output)
