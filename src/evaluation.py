"""
Simplified Evaluation module for TDLM.

Focused only on the core research question:
"Does diffusion training achieve better perplexity than autoregressive training?"

Removed over-engineering: generation quality metrics, downstream tasks, 
complex reporting, statistical testing, external dependencies.
"""

import math
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import TinyDiffusionTransformer
from .diffusion import DiscreteDiffusion
from .data import DataCollatorOutput
from .utils import TDLMConfig


@dataclass
class EvaluationResults:
    """Simple container for evaluation results."""
    model_name: str
    loss: float
    perplexity: float
    num_tokens: int
    evaluation_time: float


class PerplexityEvaluator:
    """
    Simple evaluator for computing perplexity on language modeling tasks.
    
    Handles both diffusion and autoregressive models.
    """
    
    def __init__(self):
        pass
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: TinyDiffusionTransformer,
        dataloader: torch.utils.data.DataLoader,
        diffusion: Optional[DiscreteDiffusion] = None,
        max_batches: Optional[int] = None,
        model_name: str = "model"
    ) -> EvaluationResults:
        """
        Evaluate model and return perplexity.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            diffusion: DiscreteDiffusion for diffusion models (None for AR)
            max_batches: Maximum number of batches to evaluate
            model_name: Name for the model
            
        Returns:
            EvaluationResults with loss and perplexity
        """
        start_time = time.time()
        model.eval()
        device = next(model.parameters()).device
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        # Determine number of batches to process
        max_batches = max_batches or len(dataloader)
        max_batches = min(max_batches, len(dataloader))
        
        progress_bar = tqdm(dataloader, desc=f"Evaluating {model_name}", total=max_batches)
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= max_batches:
                break
                
            # Move batch to device
            batch = self._move_batch_to_device(batch, device)
            
            # Compute loss based on model type
            batch_loss, batch_tokens = self._compute_batch_loss(model, batch, diffusion)
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1
            
            # Update progress
            current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
            progress_bar.set_postfix({
                'loss': f"{total_loss / total_tokens:.4f}",
                'ppl': f"{current_ppl:.2f}"
            })
        
        progress_bar.close()
        
        # Compute final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        
        evaluation_time = time.time() - start_time
        
        logging.info(f"{model_name} Evaluation Results:")
        logging.info(f"  Loss: {avg_loss:.4f}")
        logging.info(f"  Perplexity: {perplexity:.2f}")
        logging.info(f"  Tokens: {total_tokens:,}")
        logging.info(f"  Time: {evaluation_time:.2f}s")
        
        return EvaluationResults(
            model_name=model_name,
            loss=avg_loss,
            perplexity=perplexity,
            num_tokens=total_tokens,
            evaluation_time=evaluation_time
        )
    
    def _compute_batch_loss(
        self,
        model: TinyDiffusionTransformer,
        batch: DataCollatorOutput,
        diffusion: Optional[DiscreteDiffusion]
    ) -> Tuple[float, int]:
        """Compute loss for a single batch."""
        if model.training_mode == "autoregressive":
            return self._compute_ar_loss(model, batch)
        else:
            if diffusion is None:
                raise ValueError("Diffusion process required for diffusion model evaluation")
            return self._compute_diffusion_loss(model, batch, diffusion)
    
    def _compute_ar_loss(
        self,
        model: TinyDiffusionTransformer,
        batch: DataCollatorOutput
    ) -> Tuple[float, int]:
        """Compute autoregressive loss."""
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            return_dict=True
        )
        loss = outputs['loss']
        
        # Count valid tokens (exclude padding)
        valid_tokens = (batch.labels != -100).sum().item()
        
        return loss.item(), valid_tokens
    
    def _compute_diffusion_loss(
        self,
        model: TinyDiffusionTransformer,
        batch: DataCollatorOutput,
        diffusion: DiscreteDiffusion
    ) -> Tuple[float, int]:
        """Compute diffusion loss with proper time-dependent weighting."""
        input_ids = batch.input_ids
        labels = batch.labels
        attention_mask = batch.attention_mask
        
        # Apply forward diffusion
        corrupted_ids, mask_positions, mask_ratios = diffusion.forward_process_with_ratios(
            input_ids,
            attention_mask=attention_mask  
        )
        
        # Forward pass
        outputs = model(corrupted_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs['logits']
        
        # Compute time-dependent loss weights (Austin et al. 2021)
        loss_weights = diffusion.compute_loss_weights(mask_positions, mask_ratios)
        
        # Compute loss only on masked positions with proper weighting
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_mask_positions = mask_positions.view(-1)
        flat_weights = loss_weights.view(-1)
        
        # Create loss mask: masked positions AND not ignore tokens
        loss_mask = flat_mask_positions & (flat_labels != -100)
        
        if loss_mask.any():
            # Compute cross-entropy loss for valid masked positions
            losses = F.cross_entropy(
                flat_logits[loss_mask],
                flat_labels[loss_mask],
                reduction='none'
            )
            
            # Apply time-dependent weights
            weighted_losses = losses * flat_weights[loss_mask]
            loss = weighted_losses.mean()
            num_tokens = loss_mask.sum().item()
        else:
            loss = 0.0
            num_tokens = 1  # Avoid division by zero
        
        return loss, num_tokens
    
    def _move_batch_to_device(self, batch: DataCollatorOutput, device: torch.device) -> DataCollatorOutput:
        """Move batch tensors to device."""
        return DataCollatorOutput(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device) if batch.attention_mask is not None else None,
            labels=batch.labels.to(device) if batch.labels is not None else None,
            original_lengths=batch.original_lengths.to(device) if batch.original_lengths is not None else None
        )


def compare_ar_vs_diffusion(
    ar_model: TinyDiffusionTransformer,
    diffusion_model: TinyDiffusionTransformer,
    diffusion: DiscreteDiffusion,
    test_dataloader: torch.utils.data.DataLoader,
    max_batches: Optional[int] = None
) -> Dict[str, EvaluationResults]:
    """
    Simple AR vs Diffusion comparison.
    
    The core function for validating "Diffusion Beats Autoregressive" claims.
    
    Args:
        ar_model: Autoregressive model
        diffusion_model: Diffusion model  
        diffusion: Diffusion process
        test_dataloader: Test data
        max_batches: Maximum batches to evaluate (for speed)
        
    Returns:
        Dictionary with comparison results
    """
    logging.info("=== AR vs Diffusion Comparison ===")
    
    evaluator = PerplexityEvaluator()
    
    # Evaluate autoregressive model
    logging.info("Evaluating Autoregressive Model...")
    ar_results = evaluator.evaluate_model(
        model=ar_model,
        dataloader=test_dataloader,
        diffusion=None,  # No diffusion for AR
        max_batches=max_batches,
        model_name="Autoregressive"
    )
    
    # Evaluate diffusion model
    logging.info("Evaluating Diffusion Model...")
    diffusion_results = evaluator.evaluate_model(
        model=diffusion_model,
        dataloader=test_dataloader,
        diffusion=diffusion,
        max_batches=max_batches,
        model_name="Diffusion"
    )
    
    # Compare results
    logging.info("\n=== COMPARISON RESULTS ===")
    logging.info(f"Autoregressive - Loss: {ar_results.loss:.4f}, Perplexity: {ar_results.perplexity:.2f}")
    logging.info(f"Diffusion      - Loss: {diffusion_results.loss:.4f}, Perplexity: {diffusion_results.perplexity:.2f}")
    
    # Determine winner
    if diffusion_results.perplexity < ar_results.perplexity:
        winner = "Diffusion"
        improvement = (ar_results.perplexity - diffusion_results.perplexity) / ar_results.perplexity * 100
        logging.info(f"\n🎉 DIFFUSION WINS! {improvement:.1f}% better perplexity")
    else:
        winner = "Autoregressive"
        improvement = (diffusion_results.perplexity - ar_results.perplexity) / diffusion_results.perplexity * 100
        logging.info(f"\n🏆 AUTOREGRESSIVE WINS! {improvement:.1f}% better perplexity")
    
    logging.info("=" * 40)
    
    return {
        "autoregressive": ar_results,
        "diffusion": diffusion_results,
        "winner": winner
    }


def evaluate_single_model(
    model: TinyDiffusionTransformer,
    diffusion: Optional[DiscreteDiffusion],
    test_dataloader: torch.utils.data.DataLoader,
    model_name: str = "model",
    max_batches: Optional[int] = None
) -> EvaluationResults:
    """
    Evaluate a single model.
    
    Convenience function for evaluating one model.
    """
    evaluator = PerplexityEvaluator()
    return evaluator.evaluate_model(
        model=model,
        dataloader=test_dataloader,
        diffusion=diffusion,
        max_batches=max_batches,
        model_name=model_name
    )


# For backward compatibility with existing code
def run_ar_vs_diffusion_comparison(
    ar_model: TinyDiffusionTransformer,
    diffusion_model: TinyDiffusionTransformer,
    diffusion: DiscreteDiffusion,
    config: TDLMConfig,
    test_dataloader: torch.utils.data.DataLoader,
    tokenizer,  # Unused but kept for compatibility
    output_dir   # Unused but kept for compatibility
) -> Dict[str, EvaluationResults]:
    """
    Backward compatibility wrapper for the comparison function.
    """
    return compare_ar_vs_diffusion(
        ar_model=ar_model,
        diffusion_model=diffusion_model,
        diffusion=diffusion,
        test_dataloader=test_dataloader,
        max_batches=100  # Reasonable default for quick evaluation
    )


# Export main functions
__all__ = [
    'EvaluationResults',
    'PerplexityEvaluator',
    'compare_ar_vs_diffusion',
    'evaluate_single_model', 
    'run_ar_vs_diffusion_comparison'
]