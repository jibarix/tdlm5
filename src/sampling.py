"""
Revised Sampling module for TDLM discrete diffusion models.

Focused on core functionality with theoretically optimal corruption schedules:
- Basic text generation from trained diffusion models
- Zhang (2025) optimal cosine schedule for inference
- Simple iterative denoising with confidence-based remasking
- Clean interface without over-engineering

Implements latest research findings for optimal generation quality.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from tqdm import tqdm

from .model import TinyDiffusionTransformer
from .diffusion import DiscreteDiffusion
from .utils import TDLMConfig


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    num_steps: int = 20
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    confidence_threshold: float = 0.8
    max_length: Optional[int] = None
    seed: Optional[int] = None
    
    # Theoretically optimal corruption schedule (Zhang 2025)
    schedule_type: Literal["linear", "cosine"] = "cosine"


@dataclass 
class GenerationOutput:
    """Output from generation process."""
    sequences: torch.Tensor
    scores: Optional[torch.Tensor] = None
    num_steps_taken: Optional[int] = None
    generation_time: Optional[float] = None


class DiffusionSampler:
    """
    Revised sampler for TDLM discrete diffusion models.
    
    Handles basic text generation through iterative denoising with
    theoretically optimal corruption schedules (Zhang 2025).
    """
    
    def __init__(
        self,
        model: TinyDiffusionTransformer,
        diffusion: DiscreteDiffusion,
        config: TDLMConfig,
        sampling_config: Optional[SamplingConfig] = None
    ):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        
        # MODIFIED: Create sampling_config with fallback to main config
        if sampling_config is None:
            sampling_config = SamplingConfig()
        
        # ADDED: Override sampling_config with main config values if available
        # This allows main config to provide defaults for sampling parameters
        if hasattr(config, 'diffusion'):
            # Read confidence_threshold from main config if available
            if hasattr(config.diffusion, 'confidence_threshold'):
                sampling_config.confidence_threshold = config.diffusion.confidence_threshold
        
        self.sampling_config = sampling_config
        
        # Set model to eval mode
        self.model.eval()
        
        # Cache commonly used values
        self.mask_token_id = config.model.mask_token_id
        self.vocab_size = config.model.vocab_size
        self.device = next(model.parameters()).device
        
        print(f"DiffusionSampler initialized:")
        print(f"  Steps: {self.sampling_config.num_steps}")
        print(f"  Temperature: {self.sampling_config.temperature}")
        print(f"  Confidence threshold: {self.sampling_config.confidence_threshold}")
        print(f"  Schedule type: {self.sampling_config.schedule_type} (Zhang 2025 optimal: cosine)")
    
    def _create_corruption_schedule(self, num_steps: int) -> torch.Tensor:
        """
        Create corruption schedule for inference.
        
        Implements Zhang (2025) theoretically optimal cosine schedule by default.
        The cosine schedule ensures each denoising step is equally difficult,
        providing the most efficient path from noise to clean data.
        
        Args:
            num_steps: Number of denoising steps
            
        Returns:
            Tensor of corruption ratios from 1.0 (fully masked) to 0.0 (clean)
        """
        if self.sampling_config.schedule_type == "cosine":
            # Zhang (2025) optimal: Cosine discretization grid
            # Formula: t(i) = cos²(π/2 * i/T) - starts at 1.0, ends at 0.0
            steps = torch.linspace(0, 1, num_steps + 1)
            cosine_ratios = torch.cos(math.pi / 2 * steps) ** 2
            return cosine_ratios
            
        elif self.sampling_config.schedule_type == "linear":
            # Simple linear schedule (for comparison/ablation)
            return torch.linspace(1.0, 0.0, num_steps + 1)
            
        else:
            raise ValueError(f"Unknown schedule type: {self.sampling_config.schedule_type}")
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_length: Optional[int] = None,
        prompt: Optional[torch.Tensor] = None,
        show_progress: bool = True
    ) -> GenerationOutput:
        """
        Generate text sequences using discrete diffusion with optimal scheduling.
        
        Args:
            batch_size: Number of sequences to generate
            max_length: Maximum sequence length
            prompt: Optional prompt to condition on [batch_size, prompt_len]
            show_progress: Whether to show progress bar
            
        Returns:
            GenerationOutput with generated sequences
        """
        import time
        start_time = time.time()
        
        # Set random seed if specified
        if self.sampling_config.seed is not None:
            torch.manual_seed(self.sampling_config.seed)
            
        # Determine sequence length
        if max_length is None:
            max_length = self.sampling_config.max_length or self.config.model.max_seq_length
        
        # Handle prompting
        if prompt is not None:
            if prompt.shape[0] != batch_size:
                raise ValueError(f"Prompt batch size {prompt.shape[0]} != {batch_size}")
            prompt_length = prompt.shape[1]
            if prompt_length >= max_length:
                raise ValueError(f"Prompt length {prompt_length} >= max_length {max_length}")
            generation_length = max_length - prompt_length
        else:
            prompt_length = 0
            generation_length = max_length
        
        # Initialize with fully masked sequence
        if prompt is not None:
            # Start with prompt + masked tokens
            initial_state = torch.full((batch_size, generation_length), self.mask_token_id, device=self.device)
            current_ids = torch.cat([prompt, initial_state], dim=1)
        else:
            # Start with fully masked sequence
            current_ids = torch.full((batch_size, max_length), self.mask_token_id, device=self.device)
        
        # Create theoretically optimal corruption schedule
        mask_ratios = self._create_corruption_schedule(self.sampling_config.num_steps)
        
        # Log schedule information
        if show_progress:
            schedule_info = f"Using {self.sampling_config.schedule_type} schedule"
            if self.sampling_config.schedule_type == "cosine":
                schedule_info += " (Zhang 2025 optimal)"
            print(schedule_info)
        
        # Sampling loop with progress bar
        progress_desc = f"Generating ({self.sampling_config.schedule_type})"
        progress_bar = tqdm(range(self.sampling_config.num_steps), desc=progress_desc, disable=not show_progress)
        
        for step in progress_bar:
            current_mask_ratio = mask_ratios[step].item()
            target_mask_ratio = mask_ratios[step + 1].item()
            
            # Forward pass through model
            outputs = self.model(current_ids, return_dict=True)
            logits = outputs['logits']
            
            # Apply sampling controls
            if self.sampling_config.temperature != 1.0:
                logits = logits / self.sampling_config.temperature
            
            if self.sampling_config.top_k is not None and self.sampling_config.top_k > 0:
                logits = self._apply_top_k(logits, self.sampling_config.top_k)
            
            if self.sampling_config.top_p is not None and self.sampling_config.top_p < 1.0:
                logits = self._apply_top_p(logits, self.sampling_config.top_p)
            
            # Sample new tokens for masked positions
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, max_length)
            
            # Update masked positions with sampled tokens
            currently_masked = (current_ids == self.mask_token_id)
            current_ids = torch.where(currently_masked, sampled_tokens, current_ids)
            
            # Explicitly preserve prompt tokens
            if prompt is not None:
                current_ids[:, :prompt_length] = prompt
            
            # Apply remasking for next step (except last step)
            if step < self.sampling_config.num_steps - 1:
                current_ids = self._apply_remasking(
                    current_ids, logits, target_mask_ratio, prompt_length
                )
            
            # Update progress
            num_masked = (current_ids == self.mask_token_id).sum().item()
            progress_bar.set_postfix({'masked_tokens': num_masked})
        
        progress_bar.close()
        
        # Final cleanup - replace any remaining masks
        final_masked = (current_ids == self.mask_token_id)
        if final_masked.any():
            outputs = self.model(current_ids, return_dict=True)
            logits = outputs['logits']
            
            if self.sampling_config.temperature != 1.0:
                logits = logits / self.sampling_config.temperature
            
            probs = F.softmax(logits, dim=-1)
            final_samples = torch.multinomial(probs.view(-1, self.vocab_size), 1).view_as(current_ids)
            current_ids = torch.where(final_masked, final_samples, current_ids)
            
            # Preserve prompt one final time
            if prompt is not None:
                current_ids[:, :prompt_length] = prompt
        
        generation_time = time.time() - start_time
        
        # Compute sequence scores (average log probability)
        with torch.no_grad():
            outputs = self.model(current_ids, return_dict=True)
            logits = outputs['logits']
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probabilities of generated tokens
            token_log_probs = log_probs.gather(2, current_ids.unsqueeze(-1)).squeeze(-1)
            
            # Mask out prompt tokens from scoring if present
            if prompt is not None:
                score_mask = torch.ones_like(current_ids, dtype=torch.bool)
                score_mask[:, :prompt_length] = False
                token_log_probs = token_log_probs.masked_fill(~score_mask, 0.0)
                sequence_scores = token_log_probs.sum(dim=1) / score_mask.sum(dim=1).float()
            else:
                sequence_scores = token_log_probs.mean(dim=1)
        
        return GenerationOutput(
            sequences=current_ids,
            scores=sequence_scores,
            num_steps_taken=self.sampling_config.num_steps,
            generation_time=generation_time
        )
    
    def _apply_remasking(
        self,
        current_ids: torch.Tensor,
        logits: torch.Tensor,
        target_mask_ratio: float,
        prompt_length: int = 0
    ) -> torch.Tensor:
        """
        Apply confidence-based remasking with target ratio from optimal schedule.
        
        Remasks the lowest confidence predictions to achieve target mask ratio.
        The target_mask_ratio comes from the theoretically optimal schedule.
        
        Args:
            current_ids: Current token sequences
            logits: Model logits for confidence scoring
            target_mask_ratio: Target masking ratio from optimal schedule
            prompt_length: Length of prompt to preserve
            
        Returns:
            Remasked token sequences
        """
        batch_size, seq_len = current_ids.shape
        
        # Calculate target number of masked tokens
        target_masked = int(target_mask_ratio * seq_len)
        
        if target_masked <= 0:
            return current_ids
        
        # Get confidence scores (max probability)
        probs = F.softmax(logits, dim=-1)
        confidence_scores = torch.max(probs, dim=-1)[0]
        
        # Don't remask prompt tokens
        if prompt_length > 0:
            confidence_scores[:, :prompt_length] = 1.0  # High confidence = don't remask
        
        # Don't remask positions that are already masked
        currently_masked = (current_ids == self.mask_token_id)
        confidence_scores = confidence_scores.masked_fill(currently_masked, 1.0)
        
        # Select lowest confidence tokens to remask
        for i in range(batch_size):
            if target_masked < seq_len:
                # Get indices of lowest confidence tokens
                _, indices = torch.topk(confidence_scores[i], k=target_masked, largest=False)
                
                # Create remask positions
                remask_positions = torch.zeros(seq_len, dtype=torch.bool, device=current_ids.device)
                remask_positions[indices] = True
                
                # Apply remasking
                current_ids[i] = torch.where(remask_positions, self.mask_token_id, current_ids[i])
        
        return current_ids
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = top_k_values[..., -1:]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float('-inf'))


# Simple utility function - MODIFIED to read from main config
def create_sampling_config(
    schedule_type: Literal["linear", "cosine"] = "cosine",
    config: Optional[TDLMConfig] = None,
    **kwargs
) -> SamplingConfig:
    """
    Create sampling configuration with theoretically optimal defaults.
    
    Args:
        schedule_type: "cosine" (Zhang 2025 optimal) or "linear" (comparison)
        config: Optional main config to read defaults from
        **kwargs: Additional sampling configuration parameters
        
    Returns:
        SamplingConfig with specified parameters
    """
    config_dict = {"schedule_type": schedule_type}
    
    # ADDED: Read defaults from main config if provided
    if config is not None and hasattr(config, 'diffusion'):
        # Override with main config values if they exist
        if hasattr(config.diffusion, 'confidence_threshold'):
            config_dict['confidence_threshold'] = config.diffusion.confidence_threshold
    
    # Override with explicit kwargs
    config_dict.update(kwargs)
    return SamplingConfig(**config_dict)


# Export main classes and functions
__all__ = [
    'DiffusionSampler',
    'SamplingConfig', 
    'GenerationOutput',
    'create_sampling_config'
]