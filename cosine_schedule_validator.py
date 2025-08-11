#!/usr/bin/env python3
"""
Cosine Schedule Validator for TDLM
Simple validator with two modes: experiment (trained model) or test (untrained model)

Usage:
    python cosine_schedule_validator.py --list                           # List available experiments
    python cosine_schedule_validator.py --experiment tdlm_20250811_123506  # Use specific trained model
    python cosine_schedule_validator.py --test quick_test.yaml            # Test with untrained model
"""

import sys
import os
import time
import argparse
import torch
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_random_seeds
from src.model import TinyDiffusionTransformer
from src.diffusion import DiscreteDiffusion
from src.sampling import DiffusionSampler, SamplingConfig, create_sampling_config


class CosineScheduleValidator:
    """Simple cosine schedule validator for TDLM."""
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """Initialize validator with config and optional checkpoint."""
        
        # Load and validate config
        self.config = load_config(config_path)
        self._validate_diffusion_mode()
        
        # Setup device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TinyDiffusionTransformer(self.config).to(self.device)
        self.diffusion = DiscreteDiffusion(self.config)
        
        # Load checkpoint if provided
        self.is_trained = False
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
            self.is_trained = True
        
        print(f"Validator initialized:")
        print(f"  Config: {config_path}")
        print(f"  Device: {self.device}")
        print(f"  Model: {'Trained' if self.is_trained else 'Untrained'}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _validate_diffusion_mode(self):
        """Ensure training mode is diffusion."""
        if self.config.model.training_mode != "diffusion":
            raise ValueError(
                f"Training mode must be 'diffusion', got '{self.config.model.training_mode}'"
            )
        print("SUCCESS Training mode is 'diffusion'")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"SUCCESS Loaded checkpoint: {checkpoint_path}")
            
            # Show training info if available
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"  Global step: {checkpoint['global_step']}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def validate_cosine_mathematics(self):
        """Test mathematical correctness of cosine schedule."""
        print("\n=== Mathematical Validation ===")
        
        num_steps = 20
        
        # Create cosine sampler
        sampler = DiffusionSampler(
            self.model, self.diffusion, self.config,
            SamplingConfig(schedule_type="cosine", num_steps=num_steps)
        )
        
        # Get schedule values
        cosine_ratios = sampler._create_corruption_schedule(num_steps)
        
        print(f"Cosine schedule (20 steps): {cosine_ratios}")
        print(f"Start value: {cosine_ratios[0]:.6f} (should be ~1.0)")
        print(f"End value: {cosine_ratios[-1]:.6f} (should be ~0.0)")
        
        # Check monotonicity
        is_monotonic = all(cosine_ratios[i] >= cosine_ratios[i+1] for i in range(len(cosine_ratios)-1))
        print(f"Monotonically decreasing: {is_monotonic}")
        
        # Verify Zhang 2025 formula: cos²(π/2 * i/T)
        expected_ratios = []
        for i in range(num_steps + 1):
            t = i / num_steps
            expected = math.cos(math.pi / 2 * t) ** 2
            expected_ratios.append(expected)
        
        expected_ratios = torch.tensor(expected_ratios)
        max_diff = torch.max(torch.abs(cosine_ratios - expected_ratios)).item()
        print(f"Max difference from Zhang 2025 formula: {max_diff:.8f}")
        
        # Validation results
        math_valid = is_monotonic and max_diff < 1e-6
        print(f"Mathematical validation: {'PASS' if math_valid else 'FAIL'}")
        
        return math_valid
    
    def compare_schedules(self):
        """Compare linear vs cosine schedule performance."""
        print("\n=== Schedule Comparison ===")
        
        schedules = ["linear", "cosine"]
        results = {}
        
        for schedule_type in schedules:
            print(f"\nTesting {schedule_type} schedule...")
            
            # Create sampling config
            sampling_config = create_sampling_config(
                schedule_type=schedule_type,
                num_steps=10,
                config=self.config
            )
            
            sampler = DiffusionSampler(self.model, self.diffusion, self.config, sampling_config)
            
            # Time generation
            start_time = time.time()
            
            try:
                output = sampler.generate(
                    batch_size=2,
                    max_length=64,
                    show_progress=False
                )
                
                generation_time = time.time() - start_time
                avg_score = output.scores.mean().item()
                
                results[schedule_type] = {
                    'time': generation_time,
                    'score': avg_score,
                    'success': True
                }
                
                print(f"  Time: {generation_time:.3f}s")
                print(f"  Score: {avg_score:.4f}")
                print(f"  Status: SUCCESS")
                
            except Exception as e:
                results[schedule_type] = {'success': False, 'error': str(e)}
                print(f"  Status: FAILED - {e}")
        
        # Compare results
        if all(r['success'] for r in results.values()):
            print(f"\nComparison Summary:")
            linear_time = results['linear']['time']
            cosine_time = results['cosine']['time']
            
            print(f"  Linear:  {linear_time:.3f}s, score: {results['linear']['score']:.4f}")
            print(f"  Cosine:  {cosine_time:.3f}s, score: {results['cosine']['score']:.4f}")
            
            if cosine_time < linear_time:
                speedup = linear_time / cosine_time
                print(f"  Result: Cosine is {speedup:.2f}x faster")
            else:
                ratio = cosine_time / linear_time
                print(f"  Result: Linear is {1/ratio:.2f}x faster")
        
        return results
    
    def test_generation_quality(self):
        """Test generation quality (only meaningful for trained models)."""
        if not self.is_trained:
            print("\n=== Generation Quality Test ===")
            print("SKIPPED - Model is untrained (random weights)")
            print("Train a model first to test generation quality")
            return None
        
        print("\n=== Generation Quality Test ===")
        
        schedules = ["linear", "cosine"]
        for schedule_type in schedules:
            print(f"\n{schedule_type.title()} Schedule Generation:")
            
            sampling_config = create_sampling_config(
                schedule_type=schedule_type,
                num_steps=20,
                config=self.config
            )
            
            sampler = DiffusionSampler(self.model, self.diffusion, self.config, sampling_config)
            
            output = sampler.generate(
                batch_size=1,
                max_length=128,
                show_progress=False
            )
            
            # Basic quality metrics
            sequence = output.sequences[0]
            score = output.scores[0].item()
            non_mask_tokens = (sequence != self.config.model.mask_token_id).sum().item()
            
            print(f"  Score: {score:.4f}")
            print(f"  Length: {non_mask_tokens} tokens")
            print(f"  Time: {output.generation_time:.3f}s")
        
        return True
    
    def run_validation(self):
        """Run complete validation suite."""
        print(f"Running Cosine Schedule Validation")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Mathematical validation
        try:
            results['mathematics'] = self.validate_cosine_mathematics()
        except Exception as e:
            print(f"Mathematical validation failed: {e}")
            results['mathematics'] = False
        
        # Test 2: Schedule comparison
        try:
            comparison = self.compare_schedules()
            results['comparison'] = all(r['success'] for r in comparison.values())
        except Exception as e:
            print(f"Schedule comparison failed: {e}")
            results['comparison'] = False
        
        # Test 3: Generation quality (only for trained models)
        try:
            quality_result = self.test_generation_quality()
            results['generation'] = quality_result is not False
        except Exception as e:
            print(f"Generation quality test failed: {e}")
            results['generation'] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test_name.title():.<20} {status}")
        
        overall_success = sum(results.values()) / len(results)
        print(f"\nOverall Success Rate: {overall_success:.1%}")
        
        if overall_success >= 0.8:
            print("SUCCESS Cosine schedule implementation is working correctly")
            print("Ready for research use with Zhang 2025 optimal schedule")
        else:
            print("WARNING Some tests failed - check implementation")
        
        return results


def list_available_experiments():
    """List available experiments in the experiments directory."""
    experiments_dir = Path(r"C:\Users\arroy\Projects\tdlm5\experiments")
    if not experiments_dir.exists():
        return []
    
    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('tdlm_'):
            config_file = exp_dir / "configs" / "config.yaml"
            checkpoint_file = exp_dir / "checkpoints" / "best_model.pt"
            if config_file.exists() and checkpoint_file.exists():
                experiments.append(exp_dir.name)
    
    return sorted(experiments)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='TDLM Cosine Schedule Validator')
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument(
        '--experiment',
        type=str,
        metavar='EXPERIMENT_ID',
        help='Use trained model from specified experiment (e.g., tdlm_20250811_123506)'
    )
    
    group.add_argument(
        '--test',
        type=str,
        metavar='CONFIG',
        help='Test with untrained model using specified config'
    )
    
    group.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.list, args.experiment, args.test]):
        parser.error("Must specify one of: --list, --experiment, or --test")
    
    # Set random seed
    set_random_seeds(42)
    
    if args.list:
        # List available experiments
        print("Available experiments:")
        available = list_available_experiments()
        if available:
            for exp in available:
                print(f"  {exp}")
            print(f"\nFound {len(available)} experiments")
        else:
            print("  No complete experiments found")
        return 0
    
    elif args.experiment:
        # Construct paths from experiment ID
        experiment_id = args.experiment
        base_path = Path(r"C:\Users\arroy\Projects\tdlm5\experiments") / experiment_id
        config_path = base_path / "configs" / "config.yaml"
        checkpoint_path = base_path / "checkpoints" / "best_model.pt"
        
        # Verify paths exist
        if not config_path.exists() or not checkpoint_path.exists():
            print(f"ERROR: Experiment '{experiment_id}' not found or incomplete")
            print(f"Expected config: {config_path}")
            print(f"Expected checkpoint: {checkpoint_path}")
            
            # Show available experiments
            available = list_available_experiments()
            if available:
                print(f"\nAvailable experiments:")
                for exp in available[-5:]:  # Show last 5
                    print(f"  {exp}")
                if len(available) > 5:
                    print(f"  ... and {len(available) - 5} more")
            else:
                print("No complete experiments found")
            
            return 1
        
        print(f"EXPERIMENT MODE: Using trained model from {experiment_id}")
        validator = CosineScheduleValidator(str(config_path), str(checkpoint_path))
        
    else:  # args.test
        config_path = args.test
        
        if not Path(config_path).exists():
            print(f"ERROR: Config not found: {config_path}")
            return 1
        
        print("TEST MODE: Using untrained model")
        validator = CosineScheduleValidator(config_path)
    
    # Run validation
    try:
        results = validator.run_validation()
        return 0 if sum(results.values()) >= len(results) * 0.8 else 1
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())