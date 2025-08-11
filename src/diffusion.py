"""
Simplified Discrete Diffusion for TDLM Training.

Focused only on what's needed for training:
- Forward process (corruption via masking)
- Loss weight computation (Austin et al. 2021 formulation)
- Single masking schedule

Generation/sampling is handled by sampling.py, so removed all reverse process complexity.
"""

import torch
from typing import Tuple, Optional, Dict
from .utils import TDLMConfig



class DiscreteDiffusion:
    """
    Simplified discrete diffusion class for training only.
    
    Handles:
    - Forward process: applies masking corruption 
    - Loss weight computation: time-dependent weighting (Austin et al. 2021)
    
    Does NOT handle:
    - Reverse process (handled by sampling.py)
    - Multi-step generation (handled by sampling.py)
    - Multiple strategies (simplified to essentials)
    """
    
    def __init__(self, config: TDLMConfig):
        self.config = config
        self.mask_token_id = getattr(config.model, 'mask_token_id', 100)
        self.vocab_size = getattr(config.model, 'vocab_size', 32000)
        
        # Masking schedule parameters
        self.min_mask_ratio = getattr(config.diffusion, 'min_mask_ratio', 0.0)
        self.max_mask_ratio = getattr(config.diffusion, 'max_mask_ratio', 1.0)
        
        # Single ratio per sequence (based on research findings)
        self.single_ratio_per_sequence = getattr(config.diffusion, 'single_ratio_per_sequence', True)
        
        # ADDED: Max loss weight for training stability (configurable)
        self.max_loss_weight = getattr(config.diffusion, 'max_loss_weight', 5.0)
        
        print(f"DiscreteDiffusion initialized:")
        print(f"  Mask token ID: {self.mask_token_id}")
        print(f"  Mask ratio range: [{self.min_mask_ratio}, {self.max_mask_ratio}]")
        print(f"  Single ratio per sequence: {self.single_ratio_per_sequence}")
        print(f"  Max loss weight: {self.max_loss_weight}")  # ADDED log
    
    def forward_process_with_ratios(
        self, 
        input_ids: torch.Tensor,
        mask_ratios: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process and return mask ratios.
        
        This is the main method used during training.
        
        Args:
            input_ids: Clean token sequences [batch_size, seq_len]
            mask_ratios: Optional predefined mask ratios [batch_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (corrupted_ids, mask_positions, mask_ratios):
            - corrupted_ids: Sequence with some tokens replaced by [MASK]
            - mask_positions: Boolean tensor indicating masked positions
            - mask_ratios: Actual mask ratios used [batch_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Sample mask ratios if not provided
        if mask_ratios is None:
            if self.single_ratio_per_sequence:
                # Single ratio for all sequences in batch (Nie et al. 2025)
                single_ratio = torch.rand(1) * (self.max_mask_ratio - self.min_mask_ratio) + self.min_mask_ratio
                mask_ratios = single_ratio.expand(batch_size).to(device)
            else:
                # Different ratio per sequence
                mask_ratios = torch.rand(batch_size) * (self.max_mask_ratio - self.min_mask_ratio) + self.min_mask_ratio
                mask_ratios = mask_ratios.to(device)
        
        # Apply masking
        corrupted_ids = input_ids.clone()
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            ratio = mask_ratios[i].item()
            
            # Get valid (non-padding) positions for this sequence
            if attention_mask is not None:
                valid_positions = attention_mask[i].bool()
                valid_indices = torch.where(valid_positions)[0]
                num_valid = len(valid_indices)
            else:
                # Fallback to all positions if no attention mask provided
                valid_indices = torch.arange(seq_len, device=device)
                num_valid = seq_len
            
            # Calculate number of tokens to mask based on valid content length
            num_mask = int(ratio * num_valid)
            
            if num_mask > 0 and num_valid > 0:
                # Randomly select positions to mask from valid positions only
                if num_mask >= num_valid:
                    # Mask all valid positions if ratio is very high
                    indices = valid_indices
                else:
                    # Randomly sample from valid positions
                    perm_indices = torch.randperm(num_valid, device=device)[:num_mask]
                    indices = valid_indices[perm_indices]
                
                mask_positions[i, indices] = True
                corrupted_ids[i, indices] = self.mask_token_id
        
        return corrupted_ids, mask_positions, mask_ratios
    
    def compute_loss_weights(
        self, 
        mask_positions: torch.Tensor,
        mask_ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correct time-dependent loss weights for masked positions.
        
        Implements the theoretically grounded formulation from Austin et al. (2021):
        weight = α_t' / (1 - α_t)
        
        This is CRITICAL for proper diffusion training and fair comparison with AR models.
        
        Args:
            mask_positions: Boolean mask of masked positions [batch_size, seq_len]
            mask_ratios: Mask ratios used for each sequence [batch_size]
            
        Returns:
            Loss weights [batch_size, seq_len] with correct time-dependent weighting
        """
        batch_size, seq_len = mask_positions.shape
        device = mask_positions.device
        
        # Initialize weights tensor
        weights = torch.zeros_like(mask_positions, dtype=torch.float)
        
        # Compute time-dependent weights for each sequence
        for i in range(batch_size):
            ratio = mask_ratios[i].item()
            
            # Convert mask ratio to α_t (probability of keeping a token)
            alpha_t = 1.0 - ratio
            
            # Prevent division by zero and numerical instability
            alpha_t = max(1e-8, min(1.0 - 1e-8, alpha_t))
            
            # Compute the correct time-dependent weight: α_t' / (1 - α_t)
            # For absorbing diffusion, α_t' = α_t (see Austin et al. 2021)
            alpha_t_prime = alpha_t
            weight = alpha_t_prime / (1.0 - alpha_t + 1e-8)

            # CRITICAL: Clip extreme weights for training stability
            # CHANGED: Use configurable max_loss_weight instead of hardcoded 5.0
            weight = min(weight, self.max_loss_weight)
            
            # Apply weight to all positions in this sequence
            weights[i] = weight
        
        # Only apply weights to actually masked positions
        weights = weights * mask_positions.float()
        
        return weights.detach()
    
    def forward_process(
        self, 
        input_ids: torch.Tensor,
        mask_ratios: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple forward process without returning ratios.
        Convenience method for compatibility.
        """
        corrupted_ids, mask_positions, _ = self.forward_process_with_ratios(
            input_ids, mask_ratios, attention_mask
        )
        return corrupted_ids, mask_positions
    
    def get_config_summary(self) -> Dict[str, str]:
        """Get summary of diffusion configuration."""
        return {
            "mask_token_id": str(self.mask_token_id),
            "mask_ratio_range": f"[{self.min_mask_ratio}, {self.max_mask_ratio}]",
            "single_ratio_per_sequence": str(self.single_ratio_per_sequence),
            "vocab_size": str(self.vocab_size)
        }


# Export main class only
__all__ = ['DiscreteDiffusion']