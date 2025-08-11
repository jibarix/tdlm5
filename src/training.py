"""
Enhanced Training module for TDLM with Critical Metrics.

Maintains core training functionality and adds:
- Mask Prediction Accuracy (fundamental diffusion metric)
- Corruption-Level Performance (research validation across masking ratios)
- Loss Weight Effectiveness (Austin et al. 2021 theoretical validation)

Configurable metrics frequencies:
- Core metrics: every logging_steps
- Research metrics: every detailed_metrics_steps  
- Advanced analysis: every attention_analysis_steps

PRESERVED: All original gradient accumulation, validation, checkpointing logic.
"""

import os
import time
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Install with: pip install wandb")

from .model import TinyDiffusionTransformer
from .diffusion import DiscreteDiffusion
from .data import DataCollatorOutput
from .utils import TDLMConfig, get_environment_info, save_environment_info


class DiffusionTrainer:
    """
    Enhanced trainer class for TDLM models with critical metrics.
    
    Maintains all core functionality, adds research-validated metrics.
    """
    
    def __init__(
        self,
        model: TinyDiffusionTransformer,
        diffusion: Optional[DiscreteDiffusion],
        config: TDLMConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        device: torch.device,
        experiment_dir: Union[str, Path],
        tokenizer=None  # Optional for compatibility
    ):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.tokenizer = tokenizer
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_mode = config.model.training_mode
        
        # Training configuration
        self.gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        self.gradient_clip = getattr(config.training, 'gradient_clip', 1.0)
        self.eval_every_n_steps = getattr(config.training, 'eval_every_n_steps', 1000)
        self.save_every_n_steps = getattr(config.training, 'save_every_n_steps', 1000)
        self.save_every_n_epochs = getattr(config.training, 'save_every_n_epochs', 5)
        self.logging_steps = getattr(config.training, 'logging_steps', 100)
        
        # ADDED: Metrics configuration
        self.detailed_metrics_steps = getattr(config.training, 'detailed_metrics_steps', 500)
        self.attention_analysis_steps = getattr(config.training, 'attention_analysis_steps', 2000)
        
        # Scheduler configuration
        self.warmup_start_factor = getattr(config.training, 'warmup_start_factor', 1e-6)
        self.warmup_end_factor = getattr(config.training, 'warmup_end_factor', 1.0)
        
        # Checkpoint configuration
        self.checkpoint_map_location = getattr(config.training, 'checkpoint_map_location', 'cpu')
        
        # Numerical stability
        self.perplexity_cap = getattr(config.training, 'perplexity_cap', 10.0)
        
        # ADDED: Metrics tracking storage
        self.metrics_history = {
            'mask_accuracy': [],
            'corruption_performance': [],
            'loss_weight_effectiveness': []
        }
        
        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create checkpoint directory
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb if enabled
        self.use_wandb = False
        if WANDB_AVAILABLE and hasattr(config, 'monitoring') and hasattr(config.monitoring, 'wandb'):
            wandb_config = config.monitoring.wandb
            if getattr(wandb_config, 'enabled', False):
                experiment_name = self.experiment_dir.name
                
                # Get environment info for wandb
                env_info = get_environment_info()
                
                # Merge config with environment info for wandb
                wandb_config_dict = config.to_dict()
                wandb_config_dict.update({
                    'environment': env_info,
                    'experiment_name': experiment_name
                })
                
                wandb.init(
                    project=getattr(wandb_config, 'project', 'tdlm'),
                    entity=getattr(wandb_config, 'entity', None),
                    name=experiment_name,
                    config=wandb_config_dict,
                    tags=getattr(wandb_config, 'tags', []),
                )
                
                # Save environment info to experiment directory
                save_environment_info(self.experiment_dir, experiment_name)
                
                self.use_wandb = True
                logging.info("Weights & Biases logging enabled")
                logging.info(f"Experiment: {experiment_name}")
                logging.info(f"Git commit: {env_info.get('git_commit', 'unknown')}")
                logging.info(f"Git branch: {env_info.get('git_branch', 'unknown')}")
        
        logging.info(f"DiffusionTrainer initialized for {self.training_mode} mode")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"Validation every {self.eval_every_n_steps} steps (no end-of-epoch validation)")
        logging.info(f"Checkpoint saving: every {self.save_every_n_steps} steps" + 
                    (f", every {self.save_every_n_epochs} epochs" if self.save_every_n_epochs > 0 else ", no epoch-based saving"))
        logging.info(f"LR warmup: {self.warmup_start_factor:.1e} → {self.warmup_end_factor:.1f} over warmup period")
        
        # ADDED: Log metrics configuration
        logging.info(f"Metrics configuration:")
        logging.info(f"  Core metrics every {self.logging_steps} steps")
        logging.info(f"  Research metrics every {self.detailed_metrics_steps} steps")
        logging.info(f"  Attention analysis every {self.attention_analysis_steps} steps")
        
        # Log basic model info to wandb
        if self.use_wandb:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            wandb.log({
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/hidden_size': config.model.hidden_size,
                'model/num_layers': config.model.num_layers,
                'model/num_heads': config.model.num_heads,
            })

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay separation."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': float(self.config.training.weight_decay)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return AdamW(
            param_groups,
            lr=float(self.config.training.learning_rate),
            betas=(
                float(getattr(self.config.training, 'beta1', 0.9)),
                float(getattr(self.config.training, 'beta2', 0.95))
            ),
            eps=float(getattr(self.config.training, 'eps', 1e-8))
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Calculate total steps
        num_epochs = self.config.training.num_epochs
        steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        total_steps = num_epochs * steps_per_epoch
        
        # Get config values
        warmup_ratio = float(getattr(self.config.training, 'warmup_ratio', 0.01))
        base_lr = float(self.config.training.learning_rate)
        min_lr = float(getattr(self.config.training, 'min_learning_rate', base_lr * 0.1))
        warmup_steps = int(warmup_ratio * total_steps)

        # Create warmup + cosine schedule
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=self.warmup_start_factor, 
            end_factor=self.warmup_end_factor, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        logging.info(f"Total training steps: {total_steps} (warmup: {warmup_steps})")
        logging.info(f"Warmup schedule: {self.warmup_start_factor:.1e} → {self.warmup_end_factor:.1f} → {min_lr:.1e}")
        return scheduler
    
    # ADDED: Core metrics computation methods
    def _compute_mask_prediction_accuracy(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask_positions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute mask prediction accuracy - the fundamental diffusion metric.
        
        This is the most critical metric for discrete diffusion models.
        Without this, you can't tell if your model is actually learning.
        """
        with torch.no_grad():
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Create valid mask (masked positions AND not ignore tokens)
            valid_mask = mask_positions & (labels != -100)
            
            if valid_mask.any():
                # Compute accuracy only on valid masked positions
                correct = (predictions == labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
                
                # Additional useful metrics
                total_masked = mask_positions.sum().item()
                total_valid = valid_mask.sum().item()
                
                return {
                    'mask_accuracy': accuracy.item(),
                    'total_masked_tokens': total_masked,
                    'total_valid_tokens': total_valid,
                    'valid_ratio': total_valid / max(total_masked, 1)
                }
            else:
                return {
                    'mask_accuracy': 0.0,
                    'total_masked_tokens': 0,
                    'total_valid_tokens': 0,
                    'valid_ratio': 0.0
                }
    
    def _compute_corruption_level_performance(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask_positions: torch.Tensor, 
        mask_ratios: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute performance across different corruption levels.
        
        Critical for research validation - shows model handles full spectrum.
        Validates the model works across all masking ratios (0-30%, 30-70%, 70-100%).
        """
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            batch_size = mask_ratios.shape[0]
            
            # Define corruption level buckets
            low_corruption = []    # 0-30% masked
            medium_corruption = [] # 30-70% masked  
            high_corruption = []   # 70-100% masked
            
            for i in range(batch_size):
                ratio = mask_ratios[i].item()
                
                # Get valid positions for this sequence
                seq_mask_positions = mask_positions[i]
                seq_valid_mask = seq_mask_positions & (labels[i] != -100)
                
                if seq_valid_mask.any():
                    # Compute accuracy for this sequence
                    seq_correct = (predictions[i] == labels[i]) & seq_valid_mask
                    seq_accuracy = seq_correct.sum().float() / seq_valid_mask.sum().float()
                    
                    # Categorize by corruption level
                    if ratio <= 0.3:
                        low_corruption.append(seq_accuracy.item())
                    elif ratio <= 0.7:
                        medium_corruption.append(seq_accuracy.item())
                    else:
                        high_corruption.append(seq_accuracy.item())
            
            # Compute averages for each corruption level
            results = {}
            
            if low_corruption:
                results['low_corruption_accuracy'] = sum(low_corruption) / len(low_corruption)
                results['low_corruption_count'] = len(low_corruption)
            else:
                results['low_corruption_accuracy'] = 0.0
                results['low_corruption_count'] = 0
                
            if medium_corruption:
                results['medium_corruption_accuracy'] = sum(medium_corruption) / len(medium_corruption)
                results['medium_corruption_count'] = len(medium_corruption)
            else:
                results['medium_corruption_accuracy'] = 0.0
                results['medium_corruption_count'] = 0
                
            if high_corruption:
                results['high_corruption_accuracy'] = sum(high_corruption) / len(high_corruption)
                results['high_corruption_count'] = len(high_corruption)
            else:
                results['high_corruption_accuracy'] = 0.0
                results['high_corruption_count'] = 0
            
            return results
    
    def _compute_loss_weight_effectiveness(
        self, 
        loss_weights: torch.Tensor, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask_positions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate Austin et al. (2021) theoretical formulation effectiveness.
        
        Research validation - ensures theoretical correctness.
        Checks if loss weights correlate with prediction difficulty.
        """
        with torch.no_grad():
            # Get confidence scores (max probability after softmax)
            probs = F.softmax(logits, dim=-1)
            confidence_scores = torch.max(probs, dim=-1)[0]
            
            # Create valid mask
            valid_mask = mask_positions & (labels != -100)
            
            if valid_mask.any():
                # Get valid loss weights and confidence scores
                valid_weights = loss_weights[valid_mask]
                valid_confidence = confidence_scores[valid_mask]
                
                # Compute prediction difficulty (inverse of confidence)
                prediction_difficulty = 1.0 - valid_confidence
                
                # Compute correlation between weights and difficulty
                if len(valid_weights) > 1:
                    # Compute Pearson correlation coefficient
                    weight_mean = valid_weights.mean()
                    difficulty_mean = prediction_difficulty.mean()
                    
                    numerator = ((valid_weights - weight_mean) * (prediction_difficulty - difficulty_mean)).sum()
                    weight_std = ((valid_weights - weight_mean) ** 2).sum().sqrt()
                    difficulty_std = ((prediction_difficulty - difficulty_mean) ** 2).sum().sqrt()
                    
                    denominator = weight_std * difficulty_std
                    
                    if denominator > 1e-8:
                        correlation = (numerator / denominator).item()
                    else:
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                return {
                    'weight_difficulty_correlation': correlation,
                    'avg_loss_weight': valid_weights.mean().item(),
                    'avg_prediction_difficulty': prediction_difficulty.mean().item(),
                    'weight_std': valid_weights.std().item() if len(valid_weights) > 1 else 0.0,
                    'difficulty_std': prediction_difficulty.std().item() if len(valid_weights) > 1 else 0.0
                }
            else:
                return {
                    'weight_difficulty_correlation': 0.0,
                    'avg_loss_weight': 0.0,
                    'avg_prediction_difficulty': 0.0,
                    'weight_std': 0.0,
                    'difficulty_std': 0.0
                }
    
    def _compute_attention_analysis(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Advanced attention pattern analysis (less frequent).
        
        Analyzes bidirectional attention health for diffusion models.
        """
        with torch.no_grad():
            # This would require accessing attention weights from the model
            # For now, we'll compute some proxy metrics from hidden states
            
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            # Compute attention entropy proxy (variance in hidden states)
            if attention_mask is not None:
                # Mask out padding positions
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                valid_positions = attention_mask.sum(dim=1, keepdim=True)
                
                # Compute variance across sequence dimension
                mean_hidden = masked_hidden.sum(dim=1, keepdim=True) / valid_positions.unsqueeze(-1)
                variance = ((masked_hidden - mean_hidden) ** 2).sum(dim=1) / valid_positions.unsqueeze(-1)
                avg_variance = variance.mean()
            else:
                variance = hidden_states.var(dim=1)
                avg_variance = variance.mean()
            
            return {
                'attention_entropy_proxy': avg_variance.item(),
                'hidden_state_norm': hidden_states.norm(dim=-1).mean().item(),
                'sequence_diversity': hidden_states.std(dim=1).mean().item()
            }
    
    def train(self) -> Dict[str, float]:
        """Main training loop with metrics integration."""
        logging.info("Starting training...")
        start_time = time.time()
        training_time = 0.0
        
        self.model.train()
        epoch_metrics = {}
        
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                epoch_metrics = self._train_epoch()
                
                # Configurable epoch-based checkpoint saving
                if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                    self._save_checkpoint()
                    logging.info(f"Saved epoch-based checkpoint at epoch {epoch + 1}")
                
                logging.info(f"Epoch {epoch + 1} completed - Train Loss: {epoch_metrics['avg_loss']:.4f}")
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
        finally:
            training_time = time.time() - start_time
            logging.info(f"Training completed in {training_time:.2f} seconds")
            
            # Log final summary to wandb
            if self.use_wandb:
                summary_metrics = {
                    'summary/training_time_hours': training_time / 3600,
                    'summary/total_steps': self.global_step,
                    'summary/final_loss': epoch_metrics.get('avg_loss', float('inf')),
                    'summary/best_val_loss': self.best_val_loss,
                }
                wandb.log(summary_metrics)
                wandb.finish()
        
        return {
            'final_loss': epoch_metrics.get('avg_loss', float('inf')),
            'best_val_loss': self.best_val_loss,
            'total_steps': self.global_step,
            'training_time': training_time
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with metrics collection."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Get current learning rate for this batch
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Training step with metrics collection - MODIFIED to collect metrics data
            loss, metrics_data = self._training_step_with_metrics(batch, batch_idx)
            total_loss += loss
            num_batches += 1
            
            # Update learning rate (only after actual optimizer steps)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scheduler.step()
            
            # OPTIMIZED: Consolidated step-based logging to ensure synchronized wandb graphs
            should_log_core = (self.global_step % self.logging_steps == 0 and self.global_step > 0)
            should_validate = (self.global_step % self.eval_every_n_steps == 0 and self.global_step > 0)
            
            if should_log_core or should_validate:
                # Initialize consolidated metrics dictionary for this step
                step_metrics = {
                    'train/step': self.global_step,
                    'train/epoch': self.current_epoch,
                }
                step_log_info = []
                
                # Add core training metrics if logging step
                if should_log_core:
                    # Calculate training speed
                    batch_time = time.time() - batch_start_time
                    num_tokens = batch.input_ids.numel()
                    tokens_per_second = num_tokens / batch_time if batch_time > 0 else 0
                    
                    # Core training metrics
                    step_metrics.update({
                        'train/loss': loss,
                        'train/learning_rate': current_lr,
                        'train/tokens_per_second': tokens_per_second,
                    })
                    
                    step_log_info.append(f"Step {self.global_step:6d} | Loss: {loss:.4f} | LR: {current_lr:.2e} | {tokens_per_second:.0f} tokens/s")
                    
                    # Add diffusion-specific metrics if available
                    if metrics_data and self.training_mode == "diffusion":
                        if 'mask_accuracy' in metrics_data:
                            step_metrics.update({
                                'metrics/mask_accuracy': metrics_data['mask_accuracy']['mask_accuracy'],
                                'metrics/total_masked_tokens': metrics_data['mask_accuracy']['total_masked_tokens'],
                                'metrics/valid_ratio': metrics_data['mask_accuracy']['valid_ratio']
                            })
                            step_log_info.append(f"  Mask Accuracy: {metrics_data['mask_accuracy']['mask_accuracy']:.3f}")
                        
                        # Add detailed research metrics if this step aligns
                        if (self.global_step % self.detailed_metrics_steps == 0 and 
                            'corruption_performance' in metrics_data):
                            
                            corruption_metrics = metrics_data['corruption_performance']
                            step_metrics.update({
                                'research/low_corruption_accuracy': corruption_metrics['low_corruption_accuracy'],
                                'research/medium_corruption_accuracy': corruption_metrics['medium_corruption_accuracy'],
                                'research/high_corruption_accuracy': corruption_metrics['high_corruption_accuracy'],
                                'research/low_corruption_count': corruption_metrics['low_corruption_count'],
                                'research/medium_corruption_count': corruption_metrics['medium_corruption_count'],
                                'research/high_corruption_count': corruption_metrics['high_corruption_count']
                            })
                            
                            cp = corruption_metrics
                            step_log_info.append(f"  Corruption Performance - Low: {cp['low_corruption_accuracy']:.3f}, "
                                              f"Medium: {cp['medium_corruption_accuracy']:.3f}, "
                                              f"High: {cp['high_corruption_accuracy']:.3f}")
                        
                        # Add loss weight effectiveness if this step aligns
                        if (self.global_step % self.detailed_metrics_steps == 0 and 
                            'loss_weight_effectiveness' in metrics_data):
                            
                            weight_metrics = metrics_data['loss_weight_effectiveness']
                            step_metrics.update({
                                'research/weight_difficulty_correlation': weight_metrics['weight_difficulty_correlation'],
                                'research/avg_loss_weight': weight_metrics['avg_loss_weight'],
                                'research/avg_prediction_difficulty': weight_metrics['avg_prediction_difficulty'],
                                'research/weight_std': weight_metrics['weight_std']
                            })
                            
                            lwe = weight_metrics
                            step_log_info.append(f"  Loss Weight Effectiveness - Correlation: {lwe['weight_difficulty_correlation']:.3f}")
                        
                        # Add advanced attention analysis if this step aligns
                        if (self.global_step % self.attention_analysis_steps == 0 and 
                            'attention_analysis' in metrics_data):
                            
                            attention_metrics = metrics_data['attention_analysis']
                            step_metrics.update({
                                'advanced/attention_entropy_proxy': attention_metrics['attention_entropy_proxy'],
                                'advanced/hidden_state_norm': attention_metrics['hidden_state_norm'],
                                'advanced/sequence_diversity': attention_metrics['sequence_diversity']
                            })
                            
                            step_log_info.append(f"  Attention Analysis - Entropy: {attention_metrics['attention_entropy_proxy']:.3f}")
                
                # Add validation metrics if validation step
                if should_validate:
                    val_loss = self._validate()
                    step_metrics.update({
                        'val/loss': val_loss,
                        'val/perplexity': math.exp(min(val_loss, self.perplexity_cap)),
                    })
                    step_log_info.append(f"  Validation Loss: {val_loss:.4f}")
                    
                    # Save best model based on step validation
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                        step_log_info.append(f"  New best validation loss: {val_loss:.4f}")
                    
                    self.model.train()
                
                # Log all consolidated info to console
                for info_line in step_log_info:
                    logging.info(info_line)
                
                # SINGLE wandb.log() call with ALL metrics for this step
                if self.use_wandb:
                    wandb.log(step_metrics)
                
                if should_log_core:
                    batch_start_time = time.time()
            
            # Save checkpoint (only check after actual optimizer steps)
            if self.global_step % self.save_every_n_steps == 0 and self.global_step > 0:
                self._save_checkpoint()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{current_lr:.2e}'})
        
        progress_bar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'avg_loss': avg_loss}
    
    def _training_step_with_metrics(self, batch: DataCollatorOutput, batch_idx: int) -> Tuple[float, Optional[Dict]]:
        """Enhanced training step with metrics collection."""
        # Handle dictionary input for compatibility
        if isinstance(batch, dict):
            if 'labels' not in batch:
                labels = batch['input_ids'].clone()
                if 'attention_mask' in batch and batch['attention_mask'] is not None:
                    labels[batch['attention_mask'] == 0] = -100
                batch['labels'] = labels
            batch = DataCollatorOutput(**batch)
        
        # Compute loss and collect metrics data based on training mode
        if self.training_mode == "diffusion":
            loss, metrics_data = self._diffusion_training_step_with_metrics(batch)
        else:
            loss = self._autoregressive_training_step(batch)
            metrics_data = None  # No diffusion metrics for AR mode
        
        # Scale loss by gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        # Gradient step - Use batch_idx instead of global_step
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1  # Only increment after actual optimizer step
        
        return loss.item() * self.gradient_accumulation_steps, metrics_data
    
    def _diffusion_training_step_with_metrics(self, batch: DataCollatorOutput) -> Tuple[torch.Tensor, Dict]:
        """Enhanced diffusion training step with metrics collection."""
        input_ids = batch.input_ids
        labels = batch.labels
        attention_mask = batch.attention_mask
        
        # Apply forward diffusion process
        corrupted_ids, mask_positions, mask_ratios = self.diffusion.forward_process_with_ratios(
            input_ids, 
            attention_mask=attention_mask
        )
        
        # Forward pass
        outputs = self.model(input_ids=corrupted_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs['logits']
        hidden_states = outputs.get('hidden_states')
        
        # Compute time-dependent loss weights
        loss_weights = self.diffusion.compute_loss_weights(mask_positions, mask_ratios)
        
        # Compute loss only on masked positions with proper weighting
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_mask_positions = mask_positions.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Create loss mask
        loss_mask = flat_mask_positions & (flat_labels != -100)
        
        if loss_mask.any():
            losses = F.cross_entropy(flat_logits[loss_mask], flat_labels[loss_mask], reduction='none')
            weighted_losses = losses * flat_weights[loss_mask]
            loss = weighted_losses.mean()
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # ADDED: Collect metrics data for monitoring
        metrics_data = {}
        
        # Only compute metrics during actual optimizer steps and if we have valid data
        if loss_mask.any():
            # Core metric: Mask prediction accuracy
            metrics_data['mask_accuracy'] = self._compute_mask_prediction_accuracy(
                logits, labels, mask_positions
            )
            
            # Research metrics (collected every step but logged less frequently)
            metrics_data['corruption_performance'] = self._compute_corruption_level_performance(
                logits, labels, mask_positions, mask_ratios
            )
            
            metrics_data['loss_weight_effectiveness'] = self._compute_loss_weight_effectiveness(
                loss_weights, logits, labels, mask_positions
            )
            
            # Advanced analysis (if hidden states available)
            if hidden_states is not None:
                metrics_data['attention_analysis'] = self._compute_attention_analysis(
                    hidden_states, attention_mask
                )
        
        return loss, metrics_data
    
    def _training_step(self, batch: DataCollatorOutput, batch_idx: int) -> float:
        """Original simple training step with correct gradient accumulation - PRESERVED for compatibility."""
        # Handle dictionary input for compatibility
        if isinstance(batch, dict):
            if 'labels' not in batch:
                labels = batch['input_ids'].clone()
                if 'attention_mask' in batch and batch['attention_mask'] is not None:
                    labels[batch['attention_mask'] == 0] = -100
                batch['labels'] = labels
            batch = DataCollatorOutput(**batch)
        
        # Compute loss based on training mode
        if self.training_mode == "diffusion":
            loss = self._diffusion_training_step(batch)
        else:
            loss = self._autoregressive_training_step(batch)
        
        # Scale loss by gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        # Gradient step - Use batch_idx instead of global_step
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1  # Only increment after actual optimizer step
        
        return loss.item() * self.gradient_accumulation_steps
    
    def _diffusion_training_step(self, batch: DataCollatorOutput) -> torch.Tensor:
        """Original simple diffusion training step - PRESERVED for compatibility."""
        input_ids = batch.input_ids
        labels = batch.labels
        attention_mask = batch.attention_mask
        
        # Apply forward diffusion process
        corrupted_ids, mask_positions, mask_ratios = self.diffusion.forward_process_with_ratios(
            input_ids, 
            attention_mask=attention_mask
        )
        
        # Forward pass
        outputs = self.model(input_ids=corrupted_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs['logits']
        
        # Compute time-dependent loss weights
        loss_weights = self.diffusion.compute_loss_weights(mask_positions, mask_ratios)
        
        # Compute loss only on masked positions with proper weighting
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_mask_positions = mask_positions.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Create loss mask
        loss_mask = flat_mask_positions & (flat_labels != -100)
        
        if loss_mask.any():
            losses = F.cross_entropy(flat_logits[loss_mask], flat_labels[loss_mask], reduction='none')
            weighted_losses = losses * flat_weights[loss_mask]
            loss = weighted_losses.mean()
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        return loss
    
    def _autoregressive_training_step(self, batch: DataCollatorOutput) -> torch.Tensor:
        """Simple autoregressive training step - PRESERVED unchanged."""
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            return_dict=True
        )
        
        return outputs['loss']
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Simple validation - PRESERVED unchanged."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.val_dataloader:
            batch = self._move_batch_to_device(batch)
            
            if self.training_mode == "diffusion":
                batch_loss, batch_tokens = self._validate_diffusion_batch(batch)
            else:
                batch_loss, batch_tokens = self._validate_autoregressive_batch(batch)
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        return avg_loss
    
    @torch.no_grad()
    def _validate_diffusion_batch(self, batch: DataCollatorOutput) -> Tuple[float, int]:
        """Validate single diffusion batch - PRESERVED unchanged."""
        corrupted_ids, mask_positions, mask_ratios = self.diffusion.forward_process_with_ratios(
            batch.input_ids, attention_mask=batch.attention_mask
        )
        outputs = self.model(input_ids=corrupted_ids, attention_mask=batch.attention_mask, return_dict=True)
        
        # Same loss computation as training
        loss_weights = self.diffusion.compute_loss_weights(mask_positions, mask_ratios)
        
        vocab_size = outputs['logits'].size(-1)
        flat_logits = outputs['logits'].view(-1, vocab_size)
        flat_labels = batch.labels.view(-1)
        flat_mask_positions = mask_positions.view(-1)
        flat_weights = loss_weights.view(-1)
        
        loss_mask = flat_mask_positions & (flat_labels != -100)
        
        if loss_mask.any():
            losses = F.cross_entropy(flat_logits[loss_mask], flat_labels[loss_mask], reduction='none')
            weighted_losses = losses * flat_weights[loss_mask]
            loss = weighted_losses.mean().item()
            num_tokens = loss_mask.sum().item()
        else:
            loss = 0.0
            num_tokens = 1
        
        return loss, num_tokens
    
    @torch.no_grad()
    def _validate_autoregressive_batch(self, batch: DataCollatorOutput) -> Tuple[float, int]:
        """Validate single autoregressive batch - PRESERVED unchanged."""
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            return_dict=True
        )
        loss = outputs['loss'].item()
        num_tokens = (batch.labels != -100).sum().item()
        
        return loss, num_tokens
    
    def _move_batch_to_device(self, batch: DataCollatorOutput) -> DataCollatorOutput:
        """Move batch tensors to device - PRESERVED unchanged."""
        return DataCollatorOutput(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device) if batch.attention_mask is not None else None,
            labels=batch.labels.to(self.device) if batch.labels is not None else None,
            original_lengths=batch.original_lengths.to(self.device) if batch.original_lengths is not None else None
        )
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint - PRESERVED unchanged."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'training_mode': self.training_mode
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model to: {best_path}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        logging.info(f"Saved checkpoint to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint - PRESERVED unchanged."""
        checkpoint_path = Path(checkpoint_path)
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.checkpoint_map_location, weights_only=False)
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logging.info(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")


def setup_training_environment(config: TDLMConfig, experiment_dir: Path) -> Dict[str, any]:
    """Setup training environment - PRESERVED unchanged."""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Using GPU: {device_name} ({memory_gb:.1f}GB)")
    else:
        device = torch.device('cpu')
        device_name = "CPU"
        memory_gb = 0
        logging.warning("CUDA not available, using CPU")
    
    # Setup random seeds
    seed = getattr(config.reproducibility, 'seed', 42)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return {
        'device': device,
        'device_name': device_name,
        'memory_gb': memory_gb,
        'seed': seed
    }


# Export main classes - PRESERVED unchanged
__all__ = ['DiffusionTrainer', 'setup_training_environment']