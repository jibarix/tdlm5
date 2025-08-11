# TDLM Training Metrics: Developer Guide

A comprehensive guide to understanding the metrics used in TDLM training and how they validate the two core components of discrete diffusion models.

## Overview

TDLM implements a three-tier metrics system that validates both theoretical correctness and practical performance across the two core components of text diffusion models. Each metric serves a specific purpose in monitoring model health and research compliance.

**Three-Tier Collection System:**
- **Tier 1 (Essential)**: Core metrics - fundamental performance indicators
- **Tier 2 (Research)**: Research validation - theoretical compliance verification including gradient flow balance
- **Tier 3 (Advanced)**: Deep analysis - architectural health monitoring

## Core Component 1: Forward Process Metrics (Data → Noise)

The forward process systematically corrupts clean text into noise. These metrics validate the corruption mechanism and scheduling effectiveness.

### Subcomponent 1B: Corruption Process Validation

**Corruption Performance Metrics**
```python
research/low_corruption_accuracy     # 0-30% masking performance
research/medium_corruption_accuracy  # 30-70% masking performance  
research/high_corruption_accuracy    # 70-100% masking performance
research/low_corruption_count        # Number of sequences in each bucket
research/medium_corruption_count
research/high_corruption_count
```

**What These Measure:**
- Model's ability to handle different levels of corruption
- Validates that variable masking ratios (LLaDA 2025 finding) work correctly
- Ensures model doesn't collapse at extreme corruption levels

**Impact on Model:**
- **Low corruption (0-30%)**: Tests infilling capability - model should achieve high accuracy
- **Medium corruption (30-70%)**: Tests true bidirectional reasoning - critical performance range
- **High corruption (70-100%)**: Tests generation from minimal context - should show gradual degradation

**Warning Signs:**
- High corruption accuracy > medium corruption accuracy (unnatural pattern)
- Any corruption level showing 0% accuracy (training failure)
- Large gaps between corruption levels (poor generalization)

### Subcomponent 1C: Corruption Schedule Validation

**Mask Coverage Metrics**
```python
metrics/total_masked_tokens    # Total tokens corrupted per batch
metrics/valid_ratio           # Ratio of valid to total masked tokens
```

**What These Measure:**
- Validates corruption process is working correctly
- Ensures proper masking distribution across sequences
- Confirms no degenerate masking patterns

**Impact on Model:**
- **total_masked_tokens**: Should vary based on batch mask ratios - not constant
- **valid_ratio**: Should be close to 1.0 - low values indicate padding issues

## Core Component 2: Reverse Process Metrics (Noise → Data)

The reverse process learns to denoise corrupted text. These metrics validate the denoising network and objective function.

### Subcomponent 2A: Denoising Network Performance

**Core Denoising Metric**
```python
metrics/mask_accuracy    # Percentage of masked tokens predicted correctly
```

**What This Measures:**
- **The fundamental diffusion metric** - percentage of masked tokens correctly predicted
- Direct measure of the model's core capability
- Primary indicator of training progress

**Impact on Model:**
- **Above 60%**: Good performance, model is learning effectively
- **40-60%**: Moderate performance, needs more training or hyperparameter adjustment
- **Below 40%**: Poor performance, indicates training issues or model capacity problems

**This is the single most important metric for discrete diffusion models.**

### Subcomponent 2A: Denoising Network Health

**Attention Pattern Analysis**
```python
advanced/attention_entropy_proxy    # Variance in hidden states (attention health proxy)
advanced/hidden_state_norm         # Average magnitude of hidden representations
advanced/sequence_diversity        # Variation across sequence positions
```

**What These Measure:**
- **attention_entropy_proxy**: Health of bidirectional attention patterns
- **hidden_state_norm**: Representation magnitude stability
- **sequence_diversity**: Model's ability to differentiate sequence positions

**Impact on Model:**
- **attention_entropy_proxy**: Should be stable, not decreasing (indicates attention collapse)
- **hidden_state_norm**: Should be stable around 1-10 range (proper normalization)
- **sequence_diversity**: Should remain consistent (model maintains positional awareness)

**Warning Signs:**
- Steadily decreasing attention entropy (attention pattern collapse)
- Exploding or vanishing hidden state norms (training instability)
- Decreasing sequence diversity (model losing positional sensitivity)

### Subcomponent 2B: Objective Function Validation

**Austin et al. (2021) Theoretical Compliance**
```python
research/weight_difficulty_correlation    # Correlation between loss weights and prediction difficulty
research/avg_loss_weight                 # Average time-dependent weight value
research/avg_prediction_difficulty       # Average model uncertainty on masked tokens
research/weight_std                      # Standard deviation of loss weights
```

**What These Measure:**
- **weight_difficulty_correlation**: Validates that Austin et al. time-dependent weights correlate with actual prediction difficulty
- **avg_loss_weight**: Ensures loss weights are in reasonable range (not too extreme)
- **avg_prediction_difficulty**: Model's uncertainty level on denoising task
- **weight_std**: Variation in loss weighting across different corruption levels

**Impact on Model:**
- **weight_difficulty_correlation**: Should be positive (0.1-0.5) - validates theoretical formulation
- **avg_loss_weight**: Should be stable 1-3 range - extreme values indicate numerical issues
- **avg_prediction_difficulty**: Should decrease over training - model becoming more confident
- **weight_std**: Should be moderate - too high indicates unstable weighting

**Gradient Flow Balance Validation**
```python
research/gradient_norm_low_corruption     # Gradient magnitude for 0-30% masking
research/gradient_norm_medium_corruption  # Gradient magnitude for 30-70% masking  
research/gradient_norm_high_corruption    # Gradient magnitude for 70-100% masking
research/gradient_norm_ratio_high_to_low  # Ratio of high to low corruption gradients
research/gradient_norm_total             # Overall gradient norm across all corruption levels
```

**What These Measure:**
- **Gradient balance across corruption spectrum**: Validates that loss weighting creates balanced learning signals
- **Training stability per corruption level**: Ensures no corruption level dominates or vanishes during learning
- **LLaDA variable masking validation**: Confirms that different mask ratios per sequence don't create gradient imbalances
- **Austin et al. weight effectiveness**: Verifies that time-dependent weights actually balance gradient magnitudes

**Impact on Model:**
- **gradient_norm_low_corruption**: Should be moderate (0.1-2.0) - too high indicates overpowering easy examples
- **gradient_norm_medium_corruption**: Should be highest (0.5-3.0) - this is the core learning signal
- **gradient_norm_high_corruption**: Should be lower but not vanishing (0.1-1.5) - model still learning from minimal context
- **gradient_norm_ratio_high_to_low**: Should be 0.2-2.0 - balanced learning across corruption spectrum
- **gradient_norm_total**: Should be stable during training - not exploding or vanishing

**Warning Signs:**
- **Vanishing high corruption gradients** (< 0.01): Model not learning from heavily corrupted examples
- **Exploding low corruption gradients** (> 5.0): Model overfitting to easy infilling tasks
- **Extreme ratios** (< 0.1 or > 10.0): Severely imbalanced learning across corruption levels
- **Oscillating gradient norms**: Indicates numerical instability in loss weighting

**Critical for Research Validity:**
This validates that the implementation correctly follows Austin et al. (2021) theoretical framework. Without proper loss weighting, comparisons with autoregressive models are fundamentally unfair. **Gradient norm tracking per corruption level is essential evidence that variable masking ratios (LLaDA 2025) create balanced learning signals rather than gradient imbalances.**

## System-Level Performance Metrics

These metrics monitor overall training health and efficiency, independent of the diffusion-specific components.

### Training Efficiency
```python
train/loss                 # Current training loss value
train/learning_rate        # Current learning rate from scheduler
train/tokens_per_second    # Training throughput
```

**What These Measure:**
- **train/loss**: Direct optimization target - should decrease over training
- **train/learning_rate**: Learning rate schedule execution
- **train/tokens_per_second**: Hardware utilization and training speed

### Validation Performance
```python
val/loss          # Validation loss (generalization measure)
val/perplexity    # Exponential of validation loss (interpretability)
```

**What These Measure:**
- **val/loss**: Model's generalization capability
- **val/perplexity**: Human-interpretable measure of model uncertainty

**Impact on Model:**
- **val/loss**: Should track training loss but remain stable - large gaps indicate overfitting
- **val/perplexity**: Lower is better - values above 100 indicate poor performance

### Training Summary
```python
summary/training_time_hours    # Total training duration
summary/total_steps           # Number of optimization steps completed
summary/final_loss           # Final training loss achieved
summary/best_val_loss        # Best validation loss during training
```

## Metrics Interpretation Framework

### Healthy Training Indicators

**Core Health (Check every 100 steps):**
- Mask accuracy steadily increasing (target: 60%+)
- Training loss decreasing smoothly
- Validation loss tracking training loss

**Research Compliance (Check every 500 steps):**
- Positive weight-difficulty correlation (0.1-0.5)
- Balanced performance across corruption levels
- Stable loss weight values (1-3 range)
- Balanced gradient norms across corruption levels (0.2-2.0 ratio range)

**Advanced Health (Check every 2000 steps):**
- Stable attention entropy patterns
- Consistent hidden state norms
- Maintained sequence diversity

### Warning Signs and Debugging

**Training Collapse Indicators:**
- Mask accuracy stagnating below 40%
- Negative weight-difficulty correlation
- Vanishing or exploding hidden state norms
- Extreme gradient norm ratios (< 0.1 or > 10.0)
- Vanishing gradients on high corruption (< 0.01)

**Overfitting Indicators:**
- Large gap between training and validation loss
- Improving training metrics but degrading validation metrics

**Implementation Issues:**
- Zero values for corruption performance (masking failure)
- Extreme loss weight values (numerical instability)
- Constant tokens_per_second (data loading bottleneck)

## Practical Usage

### Development Phase
Focus on Tier 1 metrics for rapid iteration:
```python
# Monitor these every 100 steps
metrics/mask_accuracy    # Is the model learning?
train/loss              # Is optimization working?
val/loss                # Is the model generalizing?
```

### Research Validation
Include Tier 2 metrics for publication:
```python
# Validate theoretical compliance every 500 steps
research/weight_difficulty_correlation    # Austin et al. compliance
research/corruption_performance          # Full spectrum validation
research/gradient_norm_ratio_high_to_low  # Balanced learning validation
research/gradient_norm_medium_corruption  # Core learning signal strength
```

### Deep Analysis
Use Tier 3 metrics for architectural insights:
```python
# Analyze model internals every 2000 steps
advanced/attention_entropy_proxy    # Attention health
advanced/sequence_diversity        # Positional awareness
```

## Configuration Example

Align all metrics frequencies for clean wandb graphs:
```yaml
training:
  logging_steps: 100              # Core metrics frequency
  detailed_metrics_steps: 500     # Research metrics frequency  
  attention_analysis_steps: 2000  # Advanced metrics frequency
  eval_every_n_steps: 100        # Sync validation with core metrics
```

## Key Takeaways

1. **Mask accuracy is the fundamental metric** - without this, nothing else matters
2. **Corruption performance validates LLaDA 2025 findings** - essential for competitive performance
3. **Loss weight effectiveness validates Austin et al. 2021 theory** - critical for fair comparisons
4. **Gradient flow balance confirms theoretical weight effectiveness** - validates that loss weighting actually works
5. **Attention analysis prevents architectural collapse** - early warning system
6. **All metrics should align temporally** - enables clean experimental tracking

This metrics system ensures both practical performance and theoretical correctness, providing comprehensive validation of discrete diffusion language model training.