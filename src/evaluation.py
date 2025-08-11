"""
Updated Evaluation module for TDLM with fair comparison metrics.

Based on "Diffusion Language Models are Super Data Learners" critique:
- Added downstream task evaluation (primary recommendation)
- Added relative likelihood analysis (secondary metric)  
- Enhanced comparison function with proper warnings
- Kept existing validation loss with caveats
"""

import math
import time
import logging
from typing import Dict, Optional, Tuple, List
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
    """Enhanced results container with fair comparison metrics."""
    model_name: str
    loss: float
    perplexity: float
    num_tokens: int
    evaluation_time: float
    
    # NEW: Fair comparison metrics from paper
    downstream_accuracy: Optional[Dict[str, float]] = None
    relative_likelihood_gap: Optional[float] = None
    comparison_warnings: Optional[Dict[str, str]] = None


# NEW: Sample downstream tasks for fair comparison
SAMPLE_HELLASWAG = [
    {
        "context": "A person is trying to bounce a ball. The ball",
        "choices": [
            "bounces high into the air and comes back down",
            "turns into a cat and runs away", 
            "starts singing a song loudly",
            "becomes completely invisible"
        ],
        "correct": 0
    },
    {
        "context": "Someone is cooking pasta. They",
        "choices": [
            "put the pasta in boiling water",
            "put the pasta in the freezer",
            "throw the pasta at the wall",
            "eat the pasta raw"
        ],
        "correct": 0
    }
    # Add more examples as needed
]

SAMPLE_MMLU = [
    {
        "context": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "correct": 2
    },
    {
        "context": "Which of the following is a prime number?",
        "choices": ["4", "6", "7", "8"],
        "correct": 2
    }
    # Add more examples as needed
]


class PerplexityEvaluator:
    """
    Enhanced evaluator with fair comparison metrics.
    
    Handles both diffusion and autoregressive models with proper comparison methodology.
    """
    
    def __init__(self):
        self.downstream_tasks = {
            "hellaswag_sample": SAMPLE_HELLASWAG,
            "mmlu_sample": SAMPLE_MMLU
        }
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: TinyDiffusionTransformer,
        dataloader: torch.utils.data.DataLoader,
        diffusion: Optional[DiscreteDiffusion] = None,
        max_batches: Optional[int] = None,
        model_name: str = "model",
        include_downstream: bool = True
    ) -> EvaluationResults:
        """
        Enhanced evaluation with fair comparison metrics.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            diffusion: DiscreteDiffusion for diffusion models (None for AR)
            max_batches: Maximum number of batches to evaluate
            model_name: Name for the model
            include_downstream: Whether to run downstream task evaluation
            
        Returns:
            EvaluationResults with traditional + fair comparison metrics
        """
        start_time = time.time()
        model.eval()
        device = next(model.parameters()).device
        
        # 1. Traditional metrics (validation loss/perplexity)
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        max_batches = max_batches or len(dataloader)
        max_batches = min(max_batches, len(dataloader))
        
        progress_bar = tqdm(dataloader, desc=f"Evaluating {model_name}", total=max_batches)
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= max_batches:
                break
                
            batch = self._move_batch_to_device(batch, device)
            batch_loss, batch_tokens = self._compute_batch_loss(model, batch, diffusion)
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1
            
            current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
            progress_bar.set_postfix({
                'loss': f"{total_loss / total_tokens:.4f}",
                'ppl': f"{current_ppl:.2f}"
            })
        
        progress_bar.close()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 10))
        
        # 2. NEW: Fair comparison metrics (if requested)
        downstream_accuracy = None
        relative_likelihood_gap = None
        
        if include_downstream:
            logging.info(f"Running downstream task evaluation for {model_name}...")
            downstream_accuracy = self._evaluate_downstream_tasks(model, device)
            relative_likelihood_gap = self._compute_relative_likelihood_gap(model, device)
        
        # 3. Add comparison warnings
        comparison_warnings = {
            "validation_loss": f"AR models compute exact likelihood, diffusion models compute upper bound",
            "recommendation": "Use downstream_accuracy for more reliable comparison",
            "implementation_status": "Uses correct time-dependent loss weighting (Austin et al. 2021)"
        }
        
        evaluation_time = time.time() - start_time
        
        logging.info(f"{model_name} Evaluation Results:")
        logging.info(f"  Loss: {avg_loss:.4f}")
        logging.info(f"  Perplexity: {perplexity:.2f}")
        if downstream_accuracy:
            for task, acc in downstream_accuracy.items():
                logging.info(f"  {task}: {acc:.1%}")
        logging.info(f"  Tokens: {total_tokens:,}")
        logging.info(f"  Time: {evaluation_time:.2f}s")
        
        return EvaluationResults(
            model_name=model_name,
            loss=avg_loss,
            perplexity=perplexity,
            num_tokens=total_tokens,
            evaluation_time=evaluation_time,
            downstream_accuracy=downstream_accuracy,
            relative_likelihood_gap=relative_likelihood_gap,
            comparison_warnings=comparison_warnings
        )
    
    def _evaluate_downstream_tasks(self, model, device) -> Dict[str, float]:
        """
        NEW: Evaluate on downstream tasks (paper's primary recommendation).
        
        This provides the most empirical comparison between AR and diffusion models.
        """
        accuracies = {}
        
        for task_name, examples in self.downstream_tasks.items():
            correct = 0
            total = 0
            
            for example in examples:
                predicted_choice = self._predict_multiple_choice(model, example, device)
                if predicted_choice == example["correct"]:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            accuracies[task_name] = accuracy
        
        return accuracies
    
    def _predict_multiple_choice(self, model, example, device) -> int:
        """Predict which choice is most likely given the context."""
        choice_scores = []
        
        for choice in example["choices"]:
            # Create full text: context + choice
            full_text = example["context"] + " " + choice
            
            # Tokenize (simplified - you may need to adjust for your tokenizer)
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                inputs = tokenizer(
                    full_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(device)
                
                # Get model likelihood
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_dict=True
                )
                
                # Compute average log likelihood
                logits = outputs['logits']
                vocab_size = logits.size(-1)
                
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
                shift_labels = inputs['input_ids'][..., 1:].contiguous().view(-1)
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Average log likelihood as score
                score = token_log_probs.mean().item()
                choice_scores.append(score)
                
            except Exception as e:
                logging.warning(f"Error in choice prediction: {e}")
                choice_scores.append(float('-inf'))
        
        # Return index of highest scoring choice
        return torch.argmax(torch.tensor(choice_scores)).item()
    
    def _compute_relative_likelihood_gap(self, model, device) -> float:
        """
        NEW: Compute relative likelihood gap (ΔLL) between correct and incorrect choices.
        
        The paper shows this metric continues improving even when validation loss increases.
        """
        total_gap = 0.0
        num_examples = 0
        
        for task_name, examples in self.downstream_tasks.items():
            for example in examples:
                choice_likelihoods = []
                
                # Get likelihood for each choice
                for choice in example["choices"]:
                    full_text = example["context"] + " " + choice
                    likelihood = self._compute_text_likelihood(model, full_text, device)
                    choice_likelihoods.append(likelihood)
                
                # Compute gap: LL_correct - mean(LL_incorrect)
                correct_idx = example["correct"]
                correct_ll = choice_likelihoods[correct_idx]
                incorrect_lls = [ll for i, ll in enumerate(choice_likelihoods) if i != correct_idx]
                
                if incorrect_lls:
                    gap = correct_ll - sum(incorrect_lls) / len(incorrect_lls)
                    total_gap += gap
                    num_examples += 1
        
        return total_gap / num_examples if num_examples > 0 else 0.0
    
    def _compute_text_likelihood(self, model, text, device) -> float:
        """Compute likelihood of text under the model."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256
            ).to(device)
            
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            
            logits = outputs['logits']
            vocab_size = logits.size(-1)
            
            shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = inputs['input_ids'][..., 1:].contiguous().view(-1)
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            return token_log_probs.mean().item()
            
        except Exception as e:
            logging.warning(f"Error computing text likelihood: {e}")
            return float('-inf')
    
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
        
        corrupted_ids, mask_positions, mask_ratios = diffusion.forward_process_with_ratios(
            input_ids,
            attention_mask=attention_mask  
        )
        
        outputs = model(corrupted_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs['logits']
        
        # Compute time-dependent loss weights (Austin et al. 2021)
        loss_weights = diffusion.compute_loss_weights(mask_positions, mask_ratios)
        
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_mask_positions = mask_positions.view(-1)
        flat_weights = loss_weights.view(-1)
        
        loss_mask = flat_mask_positions & (flat_labels != -100)
        
        if loss_mask.any():
            losses = F.cross_entropy(
                flat_logits[loss_mask],
                flat_labels[loss_mask],
                reduction='none'
            )
            weighted_losses = losses * flat_weights[loss_mask]
            loss = weighted_losses.mean()
            num_tokens = loss_mask.sum().item()
        else:
            loss = 0.0
            num_tokens = 1
        
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
    UPDATED: Enhanced AR vs Diffusion comparison with fair metrics.
    
    Based on "Diffusion Language Models are Super Data Learners" recommendations:
    - Primary metric: Downstream task performance
    - Secondary metric: Relative likelihood analysis
    - Traditional metrics: Validation loss (with caveats)
    
    IMPORTANT: AR computes exact likelihood, Diffusion computes upper bound.
    Downstream task performance provides more reliable comparison.
    """
    logging.info("=== ENHANCED AR vs DIFFUSION COMPARISON ===")
    logging.info("Based on 'Diffusion Language Models are Super Data Learners'")
    logging.info("WARNING: Validation loss comparison has theoretical limitations")
    logging.info("RECOMMENDATION: Focus on downstream task performance")
    
    evaluator = PerplexityEvaluator()
    
    # Evaluate autoregressive model
    logging.info("Evaluating Autoregressive Model...")
    ar_results = evaluator.evaluate_model(
        model=ar_model,
        dataloader=test_dataloader,
        diffusion=None,
        max_batches=max_batches,
        model_name="Autoregressive",
        include_downstream=True
    )
    
    # Evaluate diffusion model
    logging.info("Evaluating Diffusion Model...")
    diffusion_results = evaluator.evaluate_model(
        model=diffusion_model,
        dataloader=test_dataloader,
        diffusion=diffusion,
        max_batches=max_batches,
        model_name="Diffusion",
        include_downstream=True
    )
    
    # ENHANCED COMPARISON with proper warnings
    logging.info("\n=== ENHANCED COMPARISON RESULTS ===")
    
    # Traditional metrics (with caveats)
    logging.info("TRADITIONAL METRICS (Validation Loss/Perplexity):")
    logging.info("CAVEAT: AR=exact likelihood, Diffusion=upper bound - not directly comparable")
    logging.info(f"Autoregressive - Loss: {ar_results.loss:.4f}, Perplexity: {ar_results.perplexity:.2f}")
    logging.info(f"Diffusion      - Loss: {diffusion_results.loss:.4f}, Perplexity: {diffusion_results.perplexity:.2f}")
    
    # NEW: Fair comparison metrics
    logging.info("\nFAIR COMPARISON METRICS (Recommended):")
    
    if ar_results.downstream_accuracy and diffusion_results.downstream_accuracy:
        logging.info("Downstream Task Performance:")
        ar_wins = 0
        diff_wins = 0
        
        for task in ar_results.downstream_accuracy:
            ar_acc = ar_results.downstream_accuracy[task]
            diff_acc = diffusion_results.downstream_accuracy.get(task, 0.0)
            
            winner = "Diffusion" if diff_acc > ar_acc else "Autoregressive"
            if diff_acc > ar_acc:
                diff_wins += 1
            else:
                ar_wins += 1
                
            logging.info(f"  {task}: AR={ar_acc:.1%}, Diffusion={diff_acc:.1%} -> {winner}")
        
        empirical_winner = "Diffusion" if diff_wins > ar_wins else "Autoregressive"
        logging.info(f"EMPIRICAL WINNER (Downstream Tasks): {empirical_winner}")
    
    if ar_results.relative_likelihood_gap and diffusion_results.relative_likelihood_gap:
        logging.info(f"Relative Likelihood Gap: AR={ar_results.relative_likelihood_gap:.3f}, Diffusion={diffusion_results.relative_likelihood_gap:.3f}")
    
    # Traditional winner determination (with warning)
    if diffusion_results.perplexity < ar_results.perplexity:
        traditional_winner = "Diffusion"
        improvement = (ar_results.perplexity - diffusion_results.perplexity) / ar_results.perplexity * 100
        logging.info(f"\nTRADITIONAL WINNER: DIFFUSION ({improvement:.1f}% better perplexity)")
    else:
        traditional_winner = "Autoregressive"  
        improvement = (diffusion_results.perplexity - ar_results.perplexity) / diffusion_results.perplexity * 100
        logging.info(f"\nTRADITIONAL WINNER: AUTOREGRESSIVE ({improvement:.1f}% better perplexity)")
    
    logging.info("WARNING: Traditional comparison has theoretical limitations")
    logging.info("RECOMMENDATION: Use downstream task performance for more reliable comparison")
    logging.info("=" * 60)
    
    return {
        "autoregressive": ar_results,
        "diffusion": diffusion_results,
        "traditional_winner": traditional_winner,
        "comparison_methodology": "enhanced_with_fair_metrics"
    }


def evaluate_single_model(
    model: TinyDiffusionTransformer,
    diffusion: Optional[DiscreteDiffusion],
    test_dataloader: torch.utils.data.DataLoader,
    model_name: str = "model",
    max_batches: Optional[int] = None
) -> EvaluationResults:
    """
    UPDATED: Enhanced single model evaluation with fair metrics.
    """
    evaluator = PerplexityEvaluator()
    return evaluator.evaluate_model(
        model=model,
        dataloader=test_dataloader,
        diffusion=diffusion,
        max_batches=max_batches,
        model_name=model_name,
        include_downstream=True
    )


# For backward compatibility
def run_ar_vs_diffusion_comparison(
    ar_model: TinyDiffusionTransformer,
    diffusion_model: TinyDiffusionTransformer,
    diffusion: DiscreteDiffusion,
    config: TDLMConfig,
    test_dataloader: torch.utils.data.DataLoader,
    tokenizer,
    output_dir
) -> Dict[str, EvaluationResults]:
    """Backward compatibility wrapper."""
    return compare_ar_vs_diffusion(
        ar_model=ar_model,
        diffusion_model=diffusion_model,
        diffusion=diffusion,
        test_dataloader=test_dataloader,
        max_batches=100
    )


# Export main functions
__all__ = [
    'EvaluationResults',
    'PerplexityEvaluator',
    'compare_ar_vs_diffusion',
    'evaluate_single_model', 
    'run_ar_vs_diffusion_comparison'
]