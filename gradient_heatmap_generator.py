#!/usr/bin/env python3
"""
TDLM Gradient Heatmap Generator

Simple tool to generate gradient flow analysis heatmaps from wandb run data.
Uses wandb API to fetch metrics data directly from wandb servers.
Only works with diffusion training runs (validates config.yaml automatically).

Requirements:
    pip install wandb seaborn matplotlib pyyaml

Usage:
    python gradient_heatmap_generator.py run-20250811_123519-fh5nsj6z

Generates 4 heatmaps:
1. Corruption Level vs Time - gradient norms across corruption levels over training
2. Gradient Ratios Over Time - balance ratios evolution during training  
3. Gradient Health Matrix - stability and balance indicators
4. Gradient Summary Dashboard - comprehensive overview

Saves to: research/heatmaps/run-id/
"""

import sys
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Optional, Dict, Any, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GradientHeatmapGenerator:
    """Simple gradient heatmap generator for TDLM wandb runs."""
    
    def __init__(self, run_id: str, wandb_base_path: str = r"C:\Users\arroy\Projects\tdlm5\wandb"):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required. Install with: pip install wandb")
        
        self.run_id = run_id
        self.wandb_base_path = Path(wandb_base_path)
        self.run_path = self.wandb_base_path / run_id
        self.output_path = Path("research/heatmaps") / run_id
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Gradient Heatmap Generator")
        print(f"Run ID: {run_id}")
        print(f"Wandb path: {self.run_path}")
        print(f"Output path: {self.output_path}")
        
        # Initialize wandb API
        self.api = wandb.Api()
        
        # Optional override for entity/project
        self.override_wandb_info = None
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load wandb metadata (optional, for local validation only)."""
        metadata_path = self.run_path / "files" / "wandb-metadata.json"
        
        if not metadata_path.exists():
            print(f"WARNING: Local metadata not found: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"SUCCESS Loaded local metadata: {metadata.get('program', 'unknown')} - {metadata.get('startedAt', 'unknown')}")
            return metadata
        except Exception as e:
            print(f"WARNING: Failed to load metadata: {e}")
            return None
    
    def validate_diffusion_run(self) -> bool:
        """Validate that this is a diffusion training run by checking config.yaml."""
        config_path = self.run_path / "files" / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Navigate the nested config structure: model.value.training_mode
            training_mode = config.get('model', {}).get('value', {}).get('training_mode', 'unknown')
            
            print(f"Training mode detected: {training_mode}")
            
            if training_mode != 'diffusion':
                print(f"ERROR: This run uses '{training_mode}' mode, not 'diffusion'")
                print(f"Gradient flow metrics are only available for diffusion training runs.")
                print(f"Please use a diffusion training run for gradient heatmap analysis.")
                return False
            
            print(f"SUCCESS Confirmed diffusion training run")
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to parse config.yaml: {e}")
    
    def load_metrics_data(self) -> pd.DataFrame:
        """Load metrics data from wandb using API."""
        # Get project and entity info
        project_name, entity_name = self._get_wandb_info()
        
        # Extract the actual wandb run ID (part after last dash)
        wandb_run_id = self.run_id.split('-')[-1]  # e.g., "7mb5daw1" from "run-20250811_193243-7mb5daw1"
        
        # Construct full run path for API: entity/project/run_id
        full_run_path = f"{entity_name}/{project_name}/{wandb_run_id}"
        
        print(f"Fetching data from wandb: {full_run_path}")
        
        try:
            # Get run from wandb
            run = self.api.run(full_run_path)
            
            # Get metrics history
            history = run.history()
            
            if history.empty:
                raise ValueError("No metrics history found in wandb run")
            
            print(f"SUCCESS Loaded metrics from wandb: ({len(history)} rows, {len(history.columns)} columns)")
            
            # Show available gradient columns for debugging
            gradient_cols = [col for col in history.columns if 'gradient_norm' in col]
            if gradient_cols:
                print(f"Found gradient metrics: {gradient_cols}")
            else:
                print("WARNING: No gradient metrics found in this run")
            
            return history
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data from wandb: {e}")
    
    def _get_wandb_info(self) -> Tuple[str, str]:
        """Extract project name and entity from config or metadata."""
        project_name = None
        entity_name = None
        
        # Use override if provided
        if self.override_wandb_info:
            override_project, override_entity = self.override_wandb_info
            if override_project:
                project_name = override_project
            if override_entity:
                entity_name = override_entity
        
        # Try to get from config if not overridden
        if not project_name or not entity_name:
            try:
                config_path = self.run_path / "files" / "config.yaml"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Look for project/entity in monitoring section
                    monitoring = config.get('monitoring', {}).get('value', {}).get('wandb', {})
                    if not project_name:
                        project_name = monitoring.get('project')
                    if not entity_name:
                        entity_name = monitoring.get('entity')
            except Exception:
                pass
        
        # Try to get from metadata if still missing
        if not project_name or not entity_name:
            try:
                metadata_path = self.run_path / "files" / "wandb-metadata.json" 
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if not project_name:
                        project_name = metadata.get('project')
                    if not entity_name:
                        entity_name = metadata.get('entity')
            except Exception:
                pass
        
        # Fallback defaults based on the example
        if not project_name:
            project_name = "tdlm-quick-test"
            print(f"WARNING: Using default project name: {project_name}")
        
        if not entity_name:
            entity_name = "jibarix-none"
            print(f"WARNING: Using default entity: {entity_name}")
        
        print(f"Using wandb entity/project: {entity_name}/{project_name}")
        return project_name, entity_name

    
    def extract_gradient_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean gradient-related metrics."""
        # Gradient columns we're looking for
        gradient_cols = [
            'research/gradient_norm_low_corruption',
            'research/gradient_norm_medium_corruption', 
            'research/gradient_norm_high_corruption',
            'research/gradient_norm_ratio_high_to_low',
            'research/gradient_norm_total',
            'train/step'
        ]
        
        # Find available gradient columns
        available_cols = [col for col in gradient_cols if col in df.columns]
        
        if not available_cols:
            print(f"\nERROR: No gradient norm metrics found in this run.")
            print(f"Available columns: {list(df.columns)}")
            print(f"\nThis could mean:")
            print(f"  1. This run was created before gradient tracking was implemented")
            print(f"  2. This run failed before gradient metrics were logged")
            print(f"  3. This run was too short (gradient metrics start after step 0)")
            print(f"\nPlease use a diffusion training run that contains gradient norm metrics.")
            print(f"Look for runs with columns like: research/gradient_norm_low_corruption")
            raise ValueError("No gradient norm metrics found - cannot generate heatmaps")
        
        # Extract gradient data and remove rows with all NaN gradient values
        gradient_df = df[available_cols].copy()
        
        # Remove rows where all gradient metrics are NaN
        gradient_metrics_only = [col for col in available_cols if col.startswith('research/gradient_norm')]
        gradient_df = gradient_df.dropna(subset=gradient_metrics_only, how='all')
        
        if len(gradient_df) == 0:
            print(f"\nERROR: All gradient metric values are NaN or missing.")
            print(f"Found gradient columns: {gradient_metrics_only}")
            print(f"But all values are empty/NaN in this run.")
            print(f"\nThis suggests the run didn't complete enough training steps to log gradient metrics.")
            print(f"Please use a run that trained for more steps.")
            raise ValueError("All gradient metrics are NaN - cannot generate heatmaps")
        
        print(f"SUCCESS Extracted gradient data: {len(gradient_df)} time points")
        print(f"Available metrics: {[col.split('/')[-1] for col in available_cols]}")
        
        return gradient_df
    
    def create_corruption_level_vs_time_heatmap(self, df: pd.DataFrame) -> str:
        """Heatmap 1: Corruption Level vs Time"""
        # Prepare data matrix
        time_steps = df['train/step'].dropna().values
        
        # Create matrix: rows = corruption levels, columns = time steps
        corruption_data = []
        corruption_labels = ['Low (0-30%)', 'Medium (30-70%)', 'High (70-100%)']
        corruption_cols = [
            'research/gradient_norm_low_corruption',
            'research/gradient_norm_medium_corruption',
            'research/gradient_norm_high_corruption'
        ]
        
        for col in corruption_cols:
            if col in df.columns:
                values = df[col].fillna(0).values
                corruption_data.append(values)
            else:
                corruption_data.append(np.zeros(len(time_steps)))
        
        # Create heatmap
        plt.figure(figsize=(15, 6))
        
        # Subsample for readability if too many time points
        if len(time_steps) > 50:
            step_size = len(time_steps) // 50
            time_steps = time_steps[::step_size]
            corruption_data = [data[::step_size] for data in corruption_data]
        
        heatmap_data = np.array(corruption_data)
        
        # Create time step labels
        time_labels = [f"{int(step/1000)}k" if step >= 1000 else str(int(step)) for step in time_steps[::max(1, len(time_steps)//10)]]
        
        sns.heatmap(heatmap_data, 
                   yticklabels=corruption_labels,
                   xticklabels=time_labels,
                   annot=False,
                   cmap='viridis',
                   cbar_kws={'label': 'Gradient Norm'})
        
        plt.title('Gradient Norms Across Corruption Levels Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Corruption Level', fontsize=12)
        plt.tight_layout()
        
        filename = f"{self.run_id}_corruption_level_vs_time.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SUCCESS Generated: {filename}")
        return str(filepath)
    
    def create_gradient_ratios_heatmap(self, df: pd.DataFrame) -> str:
        """Heatmap 2: Gradient Ratios Over Time"""
        time_steps = df['train/step'].dropna().values
        
        # Calculate various ratios
        ratios_data = []
        ratio_labels = []
        
        # High/Low ratio (if available)
        if 'research/gradient_norm_ratio_high_to_low' in df.columns:
            ratios_data.append(df['research/gradient_norm_ratio_high_to_low'].fillna(1.0).values)
            ratio_labels.append('High/Low Ratio')
        
        # Calculate Medium/Low ratio
        if all(col in df.columns for col in ['research/gradient_norm_medium_corruption', 'research/gradient_norm_low_corruption']):
            medium_low_ratio = (df['research/gradient_norm_medium_corruption'] / 
                              (df['research/gradient_norm_low_corruption'] + 1e-8)).fillna(1.0).values
            ratios_data.append(medium_low_ratio)
            ratio_labels.append('Medium/Low Ratio')
        
        # Calculate High/Medium ratio  
        if all(col in df.columns for col in ['research/gradient_norm_high_corruption', 'research/gradient_norm_medium_corruption']):
            high_medium_ratio = (df['research/gradient_norm_high_corruption'] / 
                               (df['research/gradient_norm_medium_corruption'] + 1e-8)).fillna(1.0).values
            ratios_data.append(high_medium_ratio)
            ratio_labels.append('High/Medium Ratio')
        
        if not ratios_data:
            # Fallback: create dummy data
            ratios_data = [np.ones(len(time_steps))]
            ratio_labels = ['No Ratio Data']
        
        # Create heatmap
        plt.figure(figsize=(15, 6))
        
        # Subsample for readability
        if len(time_steps) > 50:
            step_size = len(time_steps) // 50
            time_steps = time_steps[::step_size]
            ratios_data = [data[::step_size] for data in ratios_data]
        
        heatmap_data = np.array(ratios_data)
        
        # Create time step labels
        time_labels = [f"{int(step/1000)}k" if step >= 1000 else str(int(step)) for step in time_steps[::max(1, len(time_steps)//10)]]
        
        sns.heatmap(heatmap_data,
                   yticklabels=ratio_labels,
                   xticklabels=time_labels, 
                   annot=False,
                   cmap='RdBu_r',
                   center=1.0,  # Center colormap at 1.0 (balanced ratio)
                   cbar_kws={'label': 'Gradient Ratio'})
        
        plt.title('Gradient Balance Ratios Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Gradient Ratios', fontsize=12)
        plt.tight_layout()
        
        filename = f"{self.run_id}_gradient_ratios_over_time.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SUCCESS Generated: {filename}")
        return str(filepath)
    
    def create_gradient_health_matrix(self, df: pd.DataFrame) -> str:
        """Heatmap 3: Gradient Health Matrix"""
        # Create health indicators matrix
        health_metrics = {}
        
        corruption_cols = [
            'research/gradient_norm_low_corruption',
            'research/gradient_norm_medium_corruption', 
            'research/gradient_norm_high_corruption'
        ]
        
        # Calculate health indicators for each corruption level
        for col in corruption_cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    level_name = col.split('_')[-1].title()  # Extract 'corruption' -> 'Corruption'
                    
                    health_metrics[f'{level_name} Mean'] = [data.mean()]
                    health_metrics[f'{level_name} Std'] = [data.std()]
                    health_metrics[f'{level_name} Stability'] = [1.0 / (1.0 + data.std())]  # Higher = more stable
                    health_metrics[f'{level_name} Range'] = [data.max() - data.min()]
        
        # Add ratio stability
        if 'research/gradient_norm_ratio_high_to_low' in df.columns:
            ratio_data = df['research/gradient_norm_ratio_high_to_low'].dropna()
            if len(ratio_data) > 0:
                health_metrics['Ratio Balance'] = [1.0 / (1.0 + abs(ratio_data.mean() - 1.0))]  # Closer to 1.0 = better
                health_metrics['Ratio Stability'] = [1.0 / (1.0 + ratio_data.std())]
        
        # Convert to DataFrame for heatmap
        health_df = pd.DataFrame(health_metrics).T
        health_df.columns = ['Value']
        
        # Create heatmap
        plt.figure(figsize=(8, 10))
        
        sns.heatmap(health_df.values, 
                   yticklabels=health_df.index,
                   xticklabels=['Health Score'],
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   cbar_kws={'label': 'Health Score'})
        
        plt.title('Gradient Health Matrix\n(Higher Values = Better Health)', fontsize=14, fontweight='bold')
        plt.ylabel('Health Indicators', fontsize=12)
        plt.tight_layout()
        
        filename = f"{self.run_id}_gradient_health_matrix.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SUCCESS Generated: {filename}")
        return str(filepath)
    
    def create_gradient_summary_dashboard(self, df: pd.DataFrame) -> str:
        """Heatmap 4: Gradient Summary Dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        corruption_cols = [
            'research/gradient_norm_low_corruption',
            'research/gradient_norm_medium_corruption', 
            'research/gradient_norm_high_corruption'
        ]
        
        # Panel 1: Average gradient norms
        avg_norms = []
        corruption_labels = ['Low', 'Medium', 'High']
        
        for col in corruption_cols:
            if col in df.columns:
                avg_norms.append(df[col].mean())
            else:
                avg_norms.append(0)
        
        ax1.bar(corruption_labels, avg_norms, color=['lightblue', 'orange', 'lightcoral'])
        ax1.set_title('Average Gradient Norms by Corruption Level', fontweight='bold')
        ax1.set_ylabel('Gradient Norm')
        
        # Panel 2: Gradient trends over time
        time_steps = df['train/step'].dropna()
        for i, col in enumerate(corruption_cols):
            if col in df.columns:
                ax2.plot(time_steps, df[col], label=corruption_labels[i], alpha=0.7)
        
        ax2.set_title('Gradient Norms Over Training', fontweight='bold')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Gradient Norm')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Gradient ratio evolution
        if 'research/gradient_norm_ratio_high_to_low' in df.columns:
            ratio_data = df['research/gradient_norm_ratio_high_to_low'].dropna()
            ax3.plot(time_steps, ratio_data, color='purple', alpha=0.7)
            ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Balance')
            ax3.fill_between(time_steps, 0.5, 1.5, alpha=0.2, color='green', label='Healthy Range')
        
        ax3.set_title('High/Low Gradient Ratio', fontweight='bold')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Stability matrix
        stability_data = []
        stability_labels = []
        
        for i, col in enumerate(corruption_cols):
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 10:  # Need enough points for rolling std
                    rolling_std = data.rolling(window=min(10, len(data)//2)).std()
                    stability = 1.0 / (1.0 + rolling_std.mean())  # Higher = more stable
                    stability_data.append([stability])
                    stability_labels.append(f'{corruption_labels[i]} Stability')
        
        if stability_data:
            stability_matrix = np.array(stability_data)
            sns.heatmap(stability_matrix, 
                       yticklabels=stability_labels,
                       xticklabels=['Stability Score'],
                       annot=True, fmt='.3f',
                       cmap='RdYlGn', ax=ax4,
                       cbar_kws={'label': 'Stability'})
        
        ax4.set_title('Training Stability Scores', fontweight='bold')
        
        plt.suptitle(f'Gradient Flow Analysis Dashboard - {self.run_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{self.run_id}_gradient_summary_dashboard.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SUCCESS Generated: {filename}")
        return str(filepath)
    
    def save_processed_data(self, df: pd.DataFrame) -> str:
        """Save the processed gradient data as CSV."""
        filename = f"{self.run_id}_gradient_metrics.csv"
        filepath = self.output_path / filename
        df.to_csv(filepath, index=False)
        
        print(f"SUCCESS Saved data: {filename} ({len(df)} rows)")
        return str(filepath)
    
    def generate_all_heatmaps(self) -> None:
        """Generate all gradient heatmaps and save data."""
        try:
            # First validate this is a diffusion run
            print(f"\nValidating run type...")
            if not self.validate_diffusion_run():
                return  # Exit early if not a diffusion run
            
            # Load data from wandb API
            print(f"\nFetching data from wandb...")
            df = self.load_metrics_data()
            gradient_df = self.extract_gradient_data(df)
            
            print(f"\nGenerating heatmaps...")
            
            # Generate all 4 heatmaps
            self.create_corruption_level_vs_time_heatmap(gradient_df)
            self.create_gradient_ratios_heatmap(gradient_df)
            self.create_gradient_health_matrix(gradient_df)
            self.create_gradient_summary_dashboard(gradient_df)
            
            # Save processed data
            self.save_processed_data(gradient_df)
            
            print(f"\nSUCCESS All heatmaps generated!")
            print(f"Output directory: {self.output_path}")
            print(f"Files created:")
            for file in self.output_path.glob("*"):
                print(f"  {file.name}")
            
        except ValueError as e:
            # Handle gradient data extraction errors gracefully
            print(f"\nCannot generate heatmaps: {e}")
            return
        except Exception as e:
            print(f"ERROR: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate gradient flow heatmaps from TDLM wandb runs')
    parser.add_argument('run_id', help='Wandb run ID (e.g., run-20250811_123519-fh5nsj6z)')
    parser.add_argument('--wandb-path', default=r"C:\Users\arroy\Projects\tdlm5\wandb", 
                       help='Base path to wandb directory (for config validation)')
    parser.add_argument('--entity', help='Wandb entity name (overrides auto-detection)')
    parser.add_argument('--project', help='Wandb project name (overrides auto-detection)')
    
    args = parser.parse_args()
    
    # Check wandb availability first
    if not WANDB_AVAILABLE:
        print("ERROR: wandb package is required but not installed")
        print("Install with: pip install wandb")
        return 1
    
    # Validate run ID format
    if not args.run_id.startswith('run-'):
        print(f"ERROR: Run ID should start with 'run-', got: {args.run_id}")
        return 1
    
    try:
        generator = GradientHeatmapGenerator(args.run_id, args.wandb_path)
        
        # Override entity/project if specified
        if args.entity or args.project:
            generator.override_wandb_info = (args.project, args.entity)
        
        generator.generate_all_heatmaps()
        return 0
        
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print(f"Install missing packages with: pip install wandb")
        return 1
    except FileNotFoundError as e:
        print(f"FILE NOT FOUND: {e}")
        print(f"Make sure the run ID exists and wandb path is correct.")
        return 1
    except ValueError as e:
        if "gradient norm metrics" in str(e).lower():
            # Don't show confusing "VALIDATION ERROR" for missing gradient metrics
            print(f"\nRun found but cannot generate heatmaps.")
            print(f"Use a diffusion run that has logged gradient metrics.")
        else:
            print(f"VALIDATION ERROR: {e}")
            print(f"Check that the run exists in wandb and you have access to it.")
        return 1
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())