#!/usr/bin/env python3
"""
Enhanced Model Comparison Script with File Saving

Usage: 
    python compare_models.py --dd tdlm_20250811_123506 --ar tdlm_20250811_140810
    python compare_models.py --dd tdlm_20250811_123506 --ar tdlm_20250811_140810 --save-results
"""

import sys
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append('.')

from src.utils import load_config, set_random_seeds
from src.data import create_dataloaders
from src.model import TinyDiffusionTransformer
from src.diffusion import DiscreteDiffusion
from src.evaluation import compare_ar_vs_diffusion

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Autoregressive vs Diffusion models with automatic validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--dd', '--diffusion', dest='diffusion_exp', required=True,
                       help='Diffusion experiment directory name or path')
    parser.add_argument('--ar', '--autoregressive', dest='ar_exp', required=True,
                       help='Autoregressive experiment directory name or path')
    
    # Optional arguments
    parser.add_argument('--experiments-dir', default='experiments',
                       help='Base experiments directory (default: experiments)')
    parser.add_argument('--max-batches', type=int, default=50,
                       help='Maximum batches to evaluate (default: 50)')
    
    # NEW: Output options
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed results to JSON file')
    parser.add_argument('--output-dir', default='comparison_results',
                       help='Directory to save results (default: comparison_results)')
    parser.add_argument('--output-name', 
                       help='Custom output filename (default: auto-generated)')
    
    return parser.parse_args()

def convert_to_serializable(obj):
    """
    Convert PyTorch tensors and other non-serializable objects to JSON-serializable format.
    """
    import torch
    
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # Handle dataclass or other objects with attributes
        return {key: convert_to_serializable(value) for key, value in obj.__dict__.items()}
    else:
        return obj

def save_comparison_results(
    results: dict,
    diffusion_path: Path,
    ar_path: Path,
    args,
    output_dir: Path
):
    """
    Save detailed comparison results to JSON file.
    
    Args:
        results: Results from compare_ar_vs_diffusion
        diffusion_path: Path to diffusion experiment
        ar_path: Path to AR experiment  
        args: Command line arguments
        output_dir: Output directory
    """
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if args.output_name:
        filename = args.output_name
        if not filename.endswith('.json'):
            filename += '.json'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        diff_name = diffusion_path.name[:15]  # Truncate for filename
        ar_name = ar_path.name[:15]
        filename = f"comparison_{diff_name}_vs_{ar_name}_{timestamp}.json"
    
    output_file = output_dir / filename
    
    # Prepare detailed results
    ar_results = results["autoregressive"]
    diff_results = results["diffusion"]
    
    # Convert all values to JSON-serializable format
    detailed_results = {
        "comparison_info": {
            "timestamp": datetime.now().isoformat(),
            "diffusion_experiment": str(diffusion_path),
            "ar_experiment": str(ar_path),
            "max_batches_evaluated": args.max_batches,
            "comparison_method": "enhanced_with_fair_metrics"
        },
        
        "autoregressive_results": {
            "model_name": ar_results.model_name,
            "traditional_metrics": {
                "loss": float(ar_results.loss) if ar_results.loss is not None else None,
                "perplexity": float(ar_results.perplexity) if ar_results.perplexity is not None else None,
                "num_tokens": int(ar_results.num_tokens) if ar_results.num_tokens is not None else None
            },
            "fair_comparison_metrics": {
                "downstream_accuracy": convert_to_serializable(ar_results.downstream_accuracy),
                "relative_likelihood_gap": float(ar_results.relative_likelihood_gap) if ar_results.relative_likelihood_gap is not None else None
            },
            "evaluation_time": float(ar_results.evaluation_time) if ar_results.evaluation_time is not None else None
        },
        
        "diffusion_results": {
            "model_name": diff_results.model_name,
            "traditional_metrics": {
                "loss": float(diff_results.loss) if diff_results.loss is not None else None,
                "perplexity": float(diff_results.perplexity) if diff_results.perplexity is not None else None,
                "num_tokens": int(diff_results.num_tokens) if diff_results.num_tokens is not None else None
            },
            "fair_comparison_metrics": {
                "downstream_accuracy": convert_to_serializable(diff_results.downstream_accuracy),
                "relative_likelihood_gap": float(diff_results.relative_likelihood_gap) if diff_results.relative_likelihood_gap is not None else None
            },
            "evaluation_time": float(diff_results.evaluation_time) if diff_results.evaluation_time is not None else None
        },
        
        "comparison_summary": {
            "traditional_winner": results.get("traditional_winner", "Unknown"),
            "traditional_warning": "AR computes exact likelihood, Diffusion computes upper bound",
            "recommendation": "Focus on downstream task performance for reliable comparison"
        }
    }
    
    # Add empirical winner if downstream results available
    if (ar_results.downstream_accuracy and diff_results.downstream_accuracy):
        ar_wins = sum(1 for task in ar_results.downstream_accuracy 
                     if ar_results.downstream_accuracy[task] > diff_results.downstream_accuracy.get(task, 0))
        diff_wins = len(ar_results.downstream_accuracy) - ar_wins
        
        detailed_results["comparison_summary"]["empirical_analysis"] = {
            "empirical_winner": "Diffusion" if diff_wins > ar_wins else "Autoregressive",
            "downstream_task_wins": {
                "autoregressive": ar_wins,
                "diffusion": diff_wins
            },
            "task_details": {}
        }
        
        # Add task-by-task results
        for task in ar_results.downstream_accuracy:
            ar_acc = float(ar_results.downstream_accuracy[task])
            diff_acc = float(diff_results.downstream_accuracy.get(task, 0.0))
            winner = "diffusion" if diff_acc > ar_acc else "autoregressive"
            
            detailed_results["comparison_summary"]["empirical_analysis"]["task_details"][task] = {
                "ar_accuracy": ar_acc,
                "diffusion_accuracy": diff_acc,
                "winner": winner
            }
    
    # Convert everything to JSON-serializable format
    detailed_results = convert_to_serializable(detailed_results)
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"ERROR saving results to {output_file}: {e}")
        return None

def create_summary_report(results: dict, diffusion_path: Path, ar_path: Path, output_dir: Path):
    """Create a human-readable summary report."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f"comparison_summary_{timestamp}.txt"
    
    ar_results = results["autoregressive"]
    diff_results = results["diffusion"]
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MODEL COMPARISON SUMMARY REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXPERIMENT DETAILS:\n")
            f.write(f"  Diffusion Experiment:      {diffusion_path.name}\n")
            f.write(f"  Autoregressive Experiment: {ar_path.name}\n\n")
            
            f.write("TRADITIONAL METRICS (with caveats):\n")
            f.write("  WARNING: AR computes exact likelihood, Diffusion computes upper bound\n")
            f.write(f"  AR Loss:       {float(ar_results.loss):.4f}\n")
            f.write(f"  AR Perplexity: {float(ar_results.perplexity):.2f}\n")
            f.write(f"  Diffusion Loss:       {float(diff_results.loss):.4f}\n")
            f.write(f"  Diffusion Perplexity: {float(diff_results.perplexity):.2f}\n")
            f.write(f"  Traditional Winner: {results.get('traditional_winner', 'Unknown')}\n\n")
            
            if ar_results.downstream_accuracy and diff_results.downstream_accuracy:
                f.write("DOWNSTREAM TASK PERFORMANCE (Recommended Metric):\n")
                ar_wins = 0
                diff_wins = 0
                
                for task in ar_results.downstream_accuracy:
                    ar_acc = float(ar_results.downstream_accuracy[task])
                    diff_acc = float(diff_results.downstream_accuracy.get(task, 0.0))
                    
                    if diff_acc > ar_acc:
                        winner = "Diffusion"
                        diff_wins += 1
                    else:
                        winner = "Autoregressive"
                        ar_wins += 1
                    
                    f.write(f"  {task}:\n")
                    f.write(f"    AR Accuracy:       {ar_acc:.1%}\n")
                    f.write(f"    Diffusion Accuracy: {diff_acc:.1%}\n")
                    f.write(f"    Winner: {winner}\n\n")
                
                empirical_winner = "Diffusion" if diff_wins > ar_wins else "Autoregressive"
                f.write(f"EMPIRICAL WINNER: {empirical_winner}\n")
                f.write(f"  Task Wins - Diffusion: {diff_wins}, Autoregressive: {ar_wins}\n\n")
            
            f.write("RECOMMENDATION:\n")
            f.write("  Focus on downstream task performance for more reliable model comparison.\n")
            f.write("  Traditional validation loss comparison has theoretical limitations.\n")
            f.write("="*70 + "\n")
        
        print(f"Summary report saved to: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"ERROR creating summary report: {e}")
        return None

# [Keep all the existing utility functions from the previous version]
def resolve_experiment_path(exp_name: str, experiments_dir: str) -> Path:
    """Resolve experiment path from name or full path."""
    exp_path = Path(exp_name)
    
    if exp_path.exists():
        return exp_path.resolve()
    
    experiments_base = Path(experiments_dir)
    full_path = experiments_base / exp_name
    
    if full_path.exists():
        return full_path.resolve()
    
    current_dir_path = Path('.') / exp_name
    if current_dir_path.exists():
        return current_dir_path.resolve()
    
    raise FileNotFoundError(f"Could not find experiment directory: {exp_name}")

def validate_training_mode(experiment_dir: Path, expected_mode: str) -> bool:
    """Validate that experiment has the expected training mode."""
    results_file = experiment_dir / "results" / "final_results.json"
    
    if not results_file.exists():
        print(f"WARNING: Could not find {results_file}")
        return False
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        actual_mode = results["config"]["model"]["training_mode"]
        
        if actual_mode == expected_mode:
            print(f"✓ {experiment_dir.name}: Confirmed {actual_mode} mode")
            return True
        else:
            print(f"✗ {experiment_dir.name}: Expected {expected_mode}, found {actual_mode}")
            return False
            
    except (KeyError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not read training_mode from {results_file}: {e}")
        return False

def find_best_checkpoint(experiment_dir: Path) -> Path:
    """Find the best model checkpoint in the experiment directory."""
    checkpoints_dir = experiment_dir / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    best_checkpoint = checkpoints_dir / "best_model.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    
    pt_files = list(checkpoints_dir.glob("*.pt"))
    if pt_files:
        checkpoint = pt_files[0]
        print(f"Warning: Using {checkpoint.name} (best_model.pt not found)")
        return checkpoint
    
    raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")

def get_config_from_experiment(experiment_dir: Path) -> Path:
    """Get config file path from experiment directory."""
    config_file = experiment_dir / "configs" / "config.yaml"
    
    if config_file.exists():
        return config_file
    
    default_config = Path("config/example_config.yaml")
    print(f"Warning: Config not found in {config_file}, using {default_config}")
    return default_config

def load_model_from_experiment(experiment_dir: Path, training_mode: str, device):
    """Load model from experiment directory."""
    config_path = get_config_from_experiment(experiment_dir)
    config = load_config(str(config_path))
    config.model.training_mode = training_mode
    
    model = TinyDiffusionTransformer(config).to(device)
    
    checkpoint_path = find_best_checkpoint(experiment_dir)
    print(f"Loading {training_mode} model from: {checkpoint_path}")
    
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def main():
    args = parse_arguments()
    
    print("="*70)
    print("ENHANCED MODEL COMPARISON WITH FILE SAVING")
    print("="*70)
    
    # Resolve paths and validate
    try:
        diffusion_path = resolve_experiment_path(args.diffusion_exp, args.experiments_dir)
        ar_path = resolve_experiment_path(args.ar_exp, args.experiments_dir)
        
        print(f"\nDiffusion experiment: {diffusion_path}")
        print(f"Autoregressive experiment: {ar_path}")
        
        if not (validate_training_mode(diffusion_path, "diffusion") and 
                validate_training_mode(ar_path, "autoregressive")):
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Setup and run comparison
    set_random_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        ar_model, ar_config = load_model_from_experiment(ar_path, "autoregressive", device)
        diffusion_model, diff_config = load_model_from_experiment(diffusion_path, "diffusion", device)
        
        _, _, test_loader = create_dataloaders(ar_config)
        diffusion = DiscreteDiffusion(diff_config)
        
        print(f"\nRunning comparison with {args.max_batches} batches...")
        print("="*70)
        
        results = compare_ar_vs_diffusion(
            ar_model=ar_model,
            diffusion_model=diffusion_model,
            diffusion=diffusion,
            test_dataloader=test_loader,
            max_batches=args.max_batches
        )
        
        # Save results if requested
        if args.save_results:
            print(f"\nSaving results...")
            output_dir = Path(args.output_dir)
            
            try:
                json_file = save_comparison_results(results, diffusion_path, ar_path, args, output_dir)
                txt_file = create_summary_report(results, diffusion_path, ar_path, output_dir)
                
                if json_file and txt_file:
                    print(f"All results saved successfully to: {output_dir}")
                else:
                    print(f"Some files may not have been saved correctly")
                    
            except Exception as e:
                print(f"ERROR saving results: {e}")
                print(f"Comparison completed but results not saved")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())