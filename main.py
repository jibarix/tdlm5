#!/usr/bin/env python3
"""
TDLM Main Training Script

Simple entry point for training discrete diffusion language models.
Usage: python main.py --config config/example_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Add src to path for imports
sys.path.append('.')

from src.utils import (
    load_config, setup_logging, set_random_seeds,
    create_experiment_dir, generate_experiment_id
)
from src.data import create_dataloaders
from src.model import TinyDiffusionTransformer
from src.diffusion import DiscreteDiffusion
from src.training import DiffusionTrainer
# FIXED Issue 8: Removed unused import
# from src.evaluation import compare_ar_vs_diffusion


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train TDLM discrete diffusion language model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--experiment-id', type=str, help='Custom experiment ID (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Output directory for experiments')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation (requires --resume)')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Generate experiment ID and create directory
    experiment_id = args.experiment_id or generate_experiment_id()
    experiment_dir = create_experiment_dir(args.output_dir, experiment_id)
    
    # Setup logging
    logger = setup_logging(config, experiment_id, experiment_dir / 'logs')
    logger.info("=" * 60)
    logger.info(f"TDLM Training Started - Experiment: {experiment_id}")
    logger.info("=" * 60)
    
    # Set random seeds for reproducibility
    seed = getattr(config.reproducibility, 'seed', 42)
    set_random_seeds(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {device_name} ({memory_gb:.1f}GB)")
    else:
        logger.info("Using CPU (CUDA not available)")
    
    # Save config to experiment directory
    config.save(experiment_dir / 'configs' / 'config.yaml')
    
    # FIXED Issues 3, 7: Enhanced evaluation configuration
    enhanced_eval_config = getattr(config, 'enhanced_evaluation', None)
    if enhanced_eval_config is None:
        # Default enhanced evaluation settings
        include_downstream = True
        max_eval_batches = None
        logger.info("Using default enhanced evaluation settings")
    else:
        include_downstream = getattr(enhanced_eval_config, 'include_downstream', True)
        max_eval_batches = getattr(enhanced_eval_config, 'max_eval_batches', None)
        logger.info(f"Enhanced evaluation config: include_downstream={include_downstream}, max_batches={max_eval_batches}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    logger.info(f"Data ready: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    logger.info("Creating model...")
    model = TinyDiffusionTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Create diffusion process (if needed)
    diffusion = None
    if config.model.training_mode == "diffusion":
        diffusion = DiscreteDiffusion(config)
        logger.info("Discrete diffusion process created")
    
    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    
    # FIXED Issue 6: Eval-only mode with configuration control
    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume checkpoint")
        
        logger.info("Running evaluation only...")
        from src.evaluation import evaluate_single_model
        
        # FIXED Issues 2, 5: Pass enhanced evaluation parameters
        results = evaluate_single_model(
            model=model,
            diffusion=diffusion,
            test_dataloader=test_loader,
            model_name=f"{config.model.training_mode.title()} Model",
            max_batches=max_eval_batches,
            include_downstream=include_downstream
        )
        
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {results.model_name}")
        logger.info(f"Loss: {results.loss:.4f}")
        logger.info(f"Perplexity: {results.perplexity:.2f}")
        logger.info(f"Tokens evaluated: {results.num_tokens:,}")
        logger.info(f"Evaluation time: {results.evaluation_time:.2f}s")
        
        # FIXED Issue 4: Log enhanced results
        if results.downstream_accuracy:
            logger.info("Downstream Task Performance:")
            for task, accuracy in results.downstream_accuracy.items():
                logger.info(f"  {task}: {accuracy:.1%}")
        
        if results.relative_likelihood_gap:
            logger.info(f"Relative Likelihood Gap: {results.relative_likelihood_gap:.3f}")
        
        if results.comparison_warnings:
            logger.info("Comparison Warnings:")
            for key, warning in results.comparison_warnings.items():
                logger.info(f"  {key}: {warning}")
        
        return
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        experiment_dir=experiment_dir
    )
    
    # Resume training state if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        training_results = trainer.train()
        training_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Final loss: {training_results['final_loss']:.4f}")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Total steps: {training_results['total_steps']:,}")
        logger.info(f"Training time: {training_time:.2f}s ({training_time/3600:.2f}h)")
        
        # Run final evaluation
        logger.info("Running final evaluation...")
        from src.evaluation import evaluate_single_model
        
        # FIXED Issues 2, 5, 10: Pass enhanced evaluation parameters with performance control
        final_results = evaluate_single_model(
            model=model,
            diffusion=diffusion,
            test_dataloader=test_loader,
            model_name=f"Final {config.model.training_mode.title()} Model",
            max_batches=max_eval_batches,
            include_downstream=include_downstream
        )
        
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Test Loss: {final_results.loss:.4f}")
        logger.info(f"Test Perplexity: {final_results.perplexity:.2f}")
        logger.info(f"Tokens evaluated: {final_results.num_tokens:,}")
        
        # FIXED Issue 4: Log enhanced results
        if final_results.downstream_accuracy:
            logger.info("Downstream Task Performance:")
            for task, accuracy in final_results.downstream_accuracy.items():
                logger.info(f"  {task}: {accuracy:.1%}")
        
        if final_results.relative_likelihood_gap:
            logger.info(f"Relative Likelihood Gap: {final_results.relative_likelihood_gap:.3f}")
        
        if final_results.comparison_warnings:
            logger.info("Comparison Warnings:")
            for key, warning in final_results.comparison_warnings.items():
                logger.info(f"  {key}: {warning}")
        
        # FIXED Issues 1, 9: Save enhanced results with proper serialization
        def convert_to_serializable(obj):
            """Convert numpy/torch types to Python native types for JSON serialization."""
            if obj is None:
                return None
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            if hasattr(obj, 'item'):  # numpy/torch scalar
                return obj.item()
            if hasattr(obj, 'tolist'):  # numpy/torch array
                return obj.tolist()
            return obj
        
        results_summary = {
            'experiment_id': experiment_id,
            'config': config.to_dict(),
            'training_results': training_results,
            'final_evaluation': {
                # Traditional metrics
                'loss': final_results.loss,
                'perplexity': final_results.perplexity,
                'num_tokens': final_results.num_tokens,
                'evaluation_time': final_results.evaluation_time,
                # FIXED Issue 1: Enhanced metrics now included
                'downstream_accuracy': convert_to_serializable(final_results.downstream_accuracy),
                'relative_likelihood_gap': convert_to_serializable(final_results.relative_likelihood_gap),
                'comparison_warnings': convert_to_serializable(final_results.comparison_warnings)
            },
            'training_time_hours': training_time / 3600,
            'device': str(device),
            # FIXED Issue 7: Save enhanced evaluation configuration
            'enhanced_evaluation_config': {
                'include_downstream': include_downstream,
                'max_eval_batches': max_eval_batches
            }
        }
        
        import json
        with open(experiment_dir / 'results' / 'final_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {experiment_dir / 'results' / 'final_results.json'}")
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT COMPLETE: {experiment_id}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info(f"Experiment directory: {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()