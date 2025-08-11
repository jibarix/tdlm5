# Tiny Discrete Diffusion Language Model (TDLM)

A research implementation of discrete diffusion for language modeling, optimized for data-constrained settings and small model scales. TDLM demonstrates that diffusion models can outperform autoregressive models when data is scarce.

## Why Discrete Diffusion?

Recent research shows diffusion models are "super data learners" that excel when **data, not compute, is the bottleneck**:

- **Superior Data Efficiency**: Extracts more signal from repeated data through bidirectional modeling
- **Implicit Data Augmentation**: Trains on diverse token orderings vs. fixed left-to-right factorization  
- **Continued Improvement**: Benefits from 100+ epochs while AR models saturate at ~15 epochs
- **Theoretical Foundation**: Implements correct time-weighted ELBO from Austin et al. (2021)

## Quick Start

### Installation
```bash
git clone <repository-url>
cd tdlm
pip install torch transformers datasets wandb tqdm pyyaml
```

### Training
```bash
# Train diffusion model (recommended for data-constrained settings)
python main.py --config config/example_config.yaml

# Train autoregressive model for comparison  
python main.py --config config/example_config.yaml
# (Change training_mode to "autoregressive" in config)

# Resume from checkpoint
python main.py --config config/example_config.yaml --resume experiments/exp_id/checkpoints/best_model.pt

# Evaluation only
python main.py --config config/example_config.yaml --resume path/to/checkpoint.pt --eval-only
```

### Generation
```python
from src.sampling import DiffusionSampler, create_sampling_config
from src.model import TinyDiffusionTransformer
from src.diffusion import DiscreteDiffusion
from src.utils import load_config

# Load trained model
config = load_config('config/example_config.yaml')
model = TinyDiffusionTransformer(config)
diffusion = DiscreteDiffusion(config)

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate with optimal schedule
sampler = DiffusionSampler(model, diffusion, config)
output = sampler.generate(
    batch_size=4,
    max_length=256,
    show_progress=True
)
```

## Key Features

### Research-Aligned Implementation
- **Theoretically Optimal**: Zhang (2025) cosine schedule for inference
- **Latest Training**: Variable masking ratios (LLaDA 2025 finding)  
- **Correct Objective**: Proper time-dependent loss weighting (critical for fair AR comparison)
- **Advanced Sampling**: Confidence-based remasking (Sahoo et al. 2024)

### Training Metrics
- **Fundamental Metrics**: Mask prediction accuracy as primary performance indicator
- **Corruption-Level Analysis**: Performance breakdown across low/medium/high corruption ratios
- **Theoretical Validation**: Loss weight effectiveness tracking (Austin et al. 2021)
- **Attention Monitoring**: Bidirectional attention pattern health analysis
- **Multi-Tier Tracking**: Essential, important, and advanced metrics organized for research

### Modern Architecture
- **7M parameter Transformer** optimized for RTX 3070 Ti (8GB VRAM)
- **RoPE + SwiGLU + RMSNorm**: State-of-the-art architectural components
- **Bidirectional Attention**: Full context modeling for diffusion training
- **Unified Codebase**: Supports both diffusion and autoregressive modes

### Production Ready
- **Simple Interface**: `python main.py --config file.yaml`
- **Weights & Biases Integration**: Comprehensive experiment tracking with metrics
- **Robust Data Pipeline**: WikiText-2 with attention-mask-aware corruption
- **Flexible Configuration**: YAML-based setup with research-backed defaults
- **Evaluation Suite**: Fair AR vs. diffusion comparison framework

## Architecture Overview

```
Input Text → Tokenization → Forward Diffusion → Transformer → Enhanced Loss Computation
    ↓              ↓             ↓                ↓            ↓
"Hello world"  [1234, 5678]  [MASK, 5678]    Predictions   Weighted CE Loss
                                  ↓                ↓            ↓
                            Bidirectional     Time-dependent  Research Metrics
                            Attention         Reweighting     (Accuracy, Corruption, etc.)
```

### Core Components
1. **Input Representation**: Direct BPE tokenization (discrete)
2. **Forward Process**: Random masking with uniform ratio sampling  
3. **Reverse Process**: Transformer-based denoising with confidence remasking
4. **Objective Function**: Theoretically correct weighted cross-entropy
5. **Corruption Schedule**: Optimal cosine discretization for inference
6. **Monitoring**: Research-validated metrics across three tiers

## Metrics System

### Tier 1 (Essential) - Every 100 steps
- **Mask Prediction Accuracy**: Fundamental diffusion performance metric
- **Corruption Performance**: Low (0-30%), Medium (30-70%), High (70-100%) masking accuracy
- **Loss Weight Effectiveness**: Theoretical validation of Austin et al. (2021) formulation

### Tier 2 (Important) - Every 500 steps  
- **Attention Entropy**: Bidirectional attention distribution health
- **Prediction Confidence**: Model uncertainty quantification
- **Self-Attention Ratio**: Position bias detection

### Tier 3 (Advanced) - Less frequent
- **Schedule Optimality**: Zhang (2025) empirical verification
- **Token Frequency Bias**: Vocabulary coverage analysis
- **Denoising Convergence**: Generation efficiency metrics

## Expected Results

Based on Prabhudesai et al. (2025) findings and monitoring:

- **Data-Constrained**: Diffusion outperforms AR after critical compute threshold
- **Multi-Epoch Training**: Diffusion improves for 500+ epochs vs. AR's 15 epochs  
- **Validation Loss**: Lower perplexity when trained beyond Chinchilla-optimal point
- **Downstream Tasks**: Better generalization across NLP benchmarks
- **Tracking**: Comprehensive metrics validate theoretical predictions

## Configuration

Key settings in `config/example_config.yaml`:

```yaml
model:
  training_mode: "diffusion"  # or "autoregressive"
  hidden_size: 128
  num_layers: 3
  max_seq_length: 256

diffusion:
  single_ratio_per_sequence: false  # LLaDA critical finding
  min_mask_ratio: 0.0
  max_mask_ratio: 1.0

sampling:
  schedule_type: "cosine"  # Zhang (2025) optimal
  num_steps: 20
  confidence_threshold: 0.8

training:
  batch_size: 16
  learning_rate: 2e-4
  num_epochs: 100
  # Metrics configuration
  logging_steps: 100                    # Tier 1 metrics
  detailed_metrics_steps: 500           # Tier 2 metrics  
  generation_eval_steps: 2000           # Generation quality
```

## Testing and Validation

For comprehensive testing of all components, see the **[Comprehensive Test Guide](tdlm_test_guide_v0.3.md)** which covers:

- **Component Testing**: Utils, model, diffusion, data pipeline validation
- **Training**: Research-validated metrics verification
- **Integration Testing**: Full pipeline validation with monitoring
- **Research Validation**: LLaDA, Zhang, and Austin et al. implementation verification

### Quick Test
```bash
# Run quick validation of training system
python main.py --config config/quick_test.yaml

# Test metrics computation
python -c "from src.training import DiffusionTrainer; print('Metrics system ready')"
```

## File Structure

```
tdlm/
├── main.py                    # Training entry point
├── src/
│   ├── utils.py              # Configuration and utilities
│   ├── model.py              # Transformer architecture
│   ├── diffusion.py          # Discrete diffusion process
│   ├── training.py           # Training with research metrics
│   ├── sampling.py           # Generation with optimal scheduling
│   ├── evaluation.py         # AR vs diffusion comparison
│   └── data.py               # WikiText-2 data pipeline
├── config/
│   ├── example_config.yaml   # Research-validated configuration
│   └── quick_test.yaml       # Fast testing configuration
├── docs/
│   ├── why_discrete_diffusion_v0.2.md # Technical design rationale
│   ├── TDM_DEV_GUIDE_v0.7.md # Comprehensive developer guide
│   └── tdlm_test_guide_v0.3.md # Complete testing guide
└── experiments/              # Auto-generated experiment outputs
    └── tdlm_YYYYMMDD_HHMMSS/
        ├── checkpoints/
        ├── logs/
        ├── configs/
        └── results/
```

## Research Foundation

This implementation builds on cutting-edge research with validation:

- **Austin et al. (2021)**: Foundational discrete diffusion framework with proper loss weighting
- **Prabhudesai et al. (2025)**: "Diffusion Beats Autoregressive in Data-Constrained Settings"  
- **Nie et al. (2025)**: LLaDA variable masking strategy and competitive performance validation
- **Sahoo et al. (2024)**: Confidence-based remasking and metrics
- **Zhang (2025)**: Proof of cosine schedule optimality for inference

### Validation Features
- **Theoretical Compliance**: All metrics validate research paper requirements
- **Performance Tracking**: Multi-tier system captures essential to advanced metrics
- **Research Reproducibility**: Monitoring ensures proper implementation

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite the foundational papers:

```bibtex
@article{prabhudesai2025diffusion,
  title={Diffusion Beats Autoregressive in Data-Constrained Settings},
  author={Prabhudesai, Mihir and Wu, Mengning and Zadeh, Amir and Fragkiadaki, Katerina and Pathak, Deepak},
  journal={arXiv preprint arXiv:2507.15857},
  year={2025}
}

@article{austin2021structured,
  title={Structured denoising diffusion models in discrete state-spaces},
  author={Austin, Jacob and Johnson, Daniel D and Ho, Jonathan and Tarlow, Daniel and van den Berg, Rianne},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}

@article{nie2025llada,
  title={Large Language Diffusion Models},
  author={Nie, Shuyang and Zhu, Feiyu and You, Zhengyan and Zhang, Xiaoyuan and Ou, Jitao and Hu, Jingzheng and Zhou, Jian and Lin, Yingkai and Wen, Jian-Rong and Li, Cheng},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## Contributing

Contributions welcome! Please ensure implementations follow the latest research and maintain theoretical correctness of the diffusion objective. All enhancements should include appropriate metrics validation and testing as outlined in the test guide.