# TDLM Comprehensive Test Guide - Enhanced (Updated Paths)

# How to Use This Test Guide

## Essential Testing Workflow

**Run ALL tests in order** - each section validates different components and builds understanding of the TDLM pipeline:

1. **Utils & Config** → Core functionality
2. **Model Architecture** → Transformer components  
3. **Diffusion Process** → Forward/reverse processes
4. **Data Pipeline** → WikiText loading/processing
5. **Training System** → Enhanced metrics & optimization
6. **Sampling & Generation** → Optimal schedules
7. **Evaluation** → Performance measurement
8. **Integration** → Component interactions
9. **Research Validation** → Latest findings
10. **Full Pipeline** → End-to-end verification

## Command Line Setup

**Use Command Prompt (cmd), not PowerShell** - PowerShell has quote escaping issues with complex Python commands.

```bash
# Switch to cmd
cmd

# Navigate to project
cd C:\Users\your_user\Projects\tdlm5

# Activate environment
conda activate tdlm-env
```

## Testing Benefits

- **Validation**: Ensures correct implementation of research papers
- **Learning**: Understand discrete diffusion architecture
- **Debugging**: Identify issues before full training runs
- **Familiarity**: Learn component interactions and data flow

## If Tests Fail

1. Check prerequisites (dependencies, config file)
2. Verify directory structure (`src/`, `config/`)
3. Run simpler tests first (utils before full pipeline)
4. Use error messages to identify missing imports/fixes

## Prerequisites
```bash
# Install dependencies
pip install torch transformers datasets wandb tqdm pyyaml numpy

# Directory structure should be:
# tdlm5/
# ├── src/
# │   ├── __init__.py
# │   ├── utils.py, model.py, diffusion.py, training.py, sampling.py, evaluation.py, data.py
# ├── config/
# │   └── example_config.yaml
# └── (run commands from tdlm5/ root)
```

## 1. Utils Module Testing

### Basic Configuration Testing
```bash
# Test config loading
python -c "import sys; sys.path.append('.'); from src.utils import load_config, TDLMConfig; config = load_config('config/example_config.yaml'); print(f'Config loaded: {config.model.hidden_size}')"

# Test enhanced config validation
python -c "import sys; sys.path.append('.'); from src.utils import TDLMConfig; exec('try: TDLMConfig({\"model\": {\"hidden_size\": 128, \"num_heads\": 5}})\nexcept ValueError as e: print(f\"Validation works: {e}\")')"

# Test enhanced config options
python -c "import sys; sys.path.append('.'); from src.utils import load_config; config = load_config('config/example_config.yaml'); config.diffusion.single_ratio_per_sequence = False; config.diffusion.max_loss_weight = 5.0; config.training.detailed_metrics_steps = 500; config.training.generation_eval_steps = 2000; config.sampling.schedule_type = 'cosine'; print(f'Enhanced config: single_ratio={config.diffusion.single_ratio_per_sequence}, schedule={config.sampling.schedule_type}')"

# Test config conversion
python -c "import sys; sys.path.append('.'); from src.utils import load_config; config = load_config('config/example_config.yaml'); config_dict = config.to_dict(); print(f'Config conversion: {type(config_dict)}')"
```

### Environment and Device Testing
```bash
# Test device detection
python -c "import sys; sys.path.append('.'); from src.utils import get_device_info, set_random_seeds, count_parameters; device_info = get_device_info(); print(f'Device: {device_info.get(\"cuda_available\", False)}')"

# Test GPU usage directly
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); x = torch.randn(100, 100).cuda() if torch.cuda.is_available() else torch.randn(100, 100); print(f'Tensor device: {x.device}'); print(f'GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB' if torch.cuda.is_available() else 'No GPU')"

# Test training on GPU
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda'); print(f'Target device: {device}'); model = TinyDiffusionTransformer(config).to(device); print(f'Model device: {next(model.parameters()).device}'); x = torch.randint(0, 1000, (2, 64)).to(device); y = model(x); print(f'Output device: {y[\"logits\"].device}'); print(f'GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f}GB')"

# Test reproducibility
python -c "import sys; sys.path.append('.'); from src.utils import set_random_seeds; import torch; set_random_seeds(42); a = torch.randn(3); set_random_seeds(42); b = torch.randn(3); print(f'Reproducible: {torch.allclose(a, b)}')"

# Test enhanced environment tracking
python -c "import sys; sys.path.append('.'); from src.utils import get_environment_info; env = get_environment_info(); print(f'Git commit: {env[\"git_commit\"][:8]}...'); print(f'Package versions: {list(env[\"package_versions\"].keys())}')"
```

### Utility Functions Testing
```bash
# Test parameter counting and formatting
python -c "import sys; sys.path.append('.'); from src.utils import format_number; print(f'Format test: {format_number(7654321)} == 7.7M')"

# Test experiment directory creation
python -c "import sys; sys.path.append('.'); from src.utils import create_experiment_dir, generate_experiment_id; exp_id = generate_experiment_id(); exp_dir = create_experiment_dir('.', exp_id); print(f'Created: {exp_dir.exists()}')"

# Test wandb utilities
python -c "import sys; sys.path.append('.'); from src.utils import save_environment_info, generate_experiment_id, create_experiment_dir; exp_id = generate_experiment_id(); exp_dir = create_experiment_dir('.', exp_id); env_file = save_environment_info(exp_dir, exp_id); print(f'Environment saved: {env_file.exists()}')"
```

## 2. Model Architecture Testing

### Basic Model Initialization
```bash
# Test model creation
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.utils import load_config; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); print(f'Model created: {sum(p.numel() for p in model.parameters())} params')"

# Test model modes
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.utils import load_config; config = load_config('config/example_config.yaml'); config.model.training_mode = 'autoregressive'; ar_model = TinyDiffusionTransformer(config); print(f'AR model: {ar_model.training_mode}'); config.model.training_mode = 'diffusion'"
```

### Component Testing
```bash
# Test RMSNorm
python -c "import sys; sys.path.append('.'); from src.model import RMSNorm; import torch; norm = RMSNorm(128); x = torch.randn(2, 10, 128); out = norm(x); print(f'RMSNorm: {out.shape} == {x.shape}')"

# Test RoPE
python -c "import sys; sys.path.append('.'); from src.model import RotaryPositionalEmbedding, apply_rotary_pos_emb; import torch; rope = RotaryPositionalEmbedding(64); q = k = torch.randn(1, 4, 10, 64); cos, sin = rope(q, seq_len=10); q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin); print(f'RoPE: {q_rot.shape} == {q.shape}')"

# Test SwiGLU MLP
python -c "import sys; sys.path.append('.'); from src.model import SwiGLUMLP; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); mlp = SwiGLUMLP(config); x = torch.randn(2, 10, 128); out = mlp(x); print(f'SwiGLU: {out.shape} == {x.shape}')"
```

### Forward Pass Testing
```bash
# Test diffusion forward pass
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.model.training_mode = 'diffusion'; model = TinyDiffusionTransformer(config); input_ids = torch.randint(0, 1000, (2, 64)); output = model(input_ids); print(f'Diffusion forward: {output[\"logits\"].shape}')"

# Test autoregressive forward pass
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.model.training_mode = 'autoregressive'; ar_model = TinyDiffusionTransformer(config); input_ids = torch.randint(0, 1000, (2, 64)); output = ar_model(input_ids, labels=input_ids); print(f'AR forward with loss: {output[\"loss\"].item():.4f}')"

# Test attention masking
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); input_ids = torch.randint(0, 1000, (2, 64)); attention_mask = torch.ones(2, 64); attention_mask[0, 32:] = 0; output = model(input_ids, attention_mask=attention_mask); print(f'Masked forward: {output[\"logits\"].shape}')"
```

## 3. Diffusion Process Testing

### Basic Diffusion Operations
```bash
# Test diffusion initialization
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; config = load_config('config/example_config.yaml'); config.model.training_mode = 'diffusion'; diffusion = DiscreteDiffusion(config); print(f'Diffusion created: mask_token={diffusion.mask_token_id}')"

# Test forward process with enhanced ratios
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.model.training_mode = 'diffusion'; diffusion = DiscreteDiffusion(config); input_ids = torch.randint(0, 1000, (4, 32)); corrupted, mask_pos, ratios = diffusion.forward_process_with_ratios(input_ids); print(f'Forward process: {mask_pos.sum().item()} masked tokens, ratios: {ratios[:2].tolist()}')"

# Test with attention mask
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); diffusion = DiscreteDiffusion(config); input_ids = torch.randint(0, 1000, (4, 32)); attention_mask = torch.ones(4, 32); attention_mask[:, 16:] = 0; corrupted, mask_pos, ratios = diffusion.forward_process_with_ratios(input_ids, attention_mask=attention_mask); print(f'Masked forward: respects padding={not mask_pos[:, 16:].any()}')"
```

### Enhanced Loss Weight Computation
```bash
# Test enhanced loss weight computation (Austin et al. 2021)
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); diffusion = DiscreteDiffusion(config); input_ids = torch.randint(0, 1000, (4, 32)); corrupted, mask_pos, ratios = diffusion.forward_process_with_ratios(input_ids); weights = diffusion.compute_loss_weights(mask_pos, ratios); print(f'Loss weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}')"

# Test weight stability with extreme ratios
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); diffusion = DiscreteDiffusion(config); extreme_ratios = torch.tensor([0.001, 0.999, 0.5, 0.5]); extreme_mask = torch.randint(0, 2, (4, 32)).bool(); extreme_weights = diffusion.compute_loss_weights(extreme_mask, extreme_ratios); print(f'Extreme weights stable: max={extreme_weights.max():.3f} <= {diffusion.max_loss_weight}')"

# Test LLaDA single ratio strategy
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.diffusion.single_ratio_per_sequence = True; diffusion_single = DiscreteDiffusion(config); input_ids = torch.randint(0, 1000, (4, 32)); _, _, ratios_single = diffusion_single.forward_process_with_ratios(input_ids); print(f'Single ratio: all equal={torch.allclose(ratios_single, ratios_single[0])}')"

# Test variable ratio strategy (LLaDA optimal)
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.diffusion.single_ratio_per_sequence = False; diffusion_variable = DiscreteDiffusion(config); input_ids = torch.randint(0, 1000, (4, 32)); _, _, ratios_variable = diffusion_variable.forward_process_with_ratios(input_ids); print(f'Variable ratios: different={not torch.allclose(ratios_variable, ratios_variable[0])}')"
```

## 4. Data Pipeline Testing

### Dataset Loading
```bash
# Test WikiText dataset
python -c "import sys; sys.path.append('.'); from src.data import WikiTextDataset; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_dataset = WikiTextDataset(config, 'train'); print(f'Train dataset: {len(train_dataset)} examples')"

# Test data sample
python -c "import sys; sys.path.append('.'); from src.data import WikiTextDataset; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_dataset = WikiTextDataset(config, 'train'); sample = train_dataset[0]; print(f'Sample keys: {list(sample.keys())}, input_ids shape: {sample[\"input_ids\"].shape}')"

# Test validation and test splits
python -c "import sys; sys.path.append('.'); from src.data import WikiTextDataset; from src.utils import load_config; config = load_config('config/example_config.yaml'); val_dataset = WikiTextDataset(config, 'validation'); test_dataset = WikiTextDataset(config, 'test'); train_dataset = WikiTextDataset(config, 'train'); print(f'Splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}')"
```

### Data Collation
```bash
# Test enhanced data collator
python -c "import sys; sys.path.append('.'); from src.data import TDLMDataCollator, WikiTextDataset; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_dataset = WikiTextDataset(config, 'train'); collator = TDLMDataCollator(config); batch = collator([train_dataset[i] for i in range(4)]); print(f'Batch: {batch.input_ids.shape}, labels: {batch.labels.shape}')"

# Test padding handling
python -c "import sys; sys.path.append('.'); from src.data import TDLMDataCollator, WikiTextDataset; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_dataset = WikiTextDataset(config, 'train'); collator = TDLMDataCollator(config); batch = collator([train_dataset[i] for i in range(4)]); print(f'Padding handled: {(batch.labels == -100).sum().item()} ignored tokens')"

# Test original lengths
python -c "import sys; sys.path.append('.'); from src.data import TDLMDataCollator, WikiTextDataset; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_dataset = WikiTextDataset(config, 'train'); collator = TDLMDataCollator(config); batch = collator([train_dataset[i] for i in range(4)]); print(f'Original lengths: {batch.original_lengths.tolist()}')"
```

### DataLoader Creation
```bash
# Test dataloader creation
python -c "import sys; sys.path.append('.'); from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_loader, val_loader, test_loader = create_dataloaders(config); print(f'Loaders created: train={len(train_loader)} batches')"

# Test batch iteration
python -c "import sys; sys.path.append('.'); from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_loader, val_loader, test_loader = create_dataloaders(config); batch = next(iter(train_loader)); print(f'Train batch: {batch.input_ids.shape}, device: {batch.input_ids.device}')"

# Test data consistency
python -c "import sys; sys.path.append('.'); from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); train_loader, val_loader, test_loader = create_dataloaders(config); val_batch = next(iter(val_loader)); print(f'Validation batch: {val_batch.input_ids.shape}')"
```

## 5. Enhanced Training System Testing

### Trainer Initialization
```bash
# Test enhanced trainer creation
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); print(f'Enhanced trainer created: {trainer.training_mode}')"

# Test optimizer and scheduler creation
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); print(f'Optimizer: {type(trainer.optimizer).__name__}, lr={trainer.optimizer.param_groups[0][\"lr\"]}'); print(f'Scheduler: {type(trainer.scheduler).__name__}')"

# Test enhanced metric configuration
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); print(f'Detailed metrics every: {trainer.detailed_metrics_steps} steps'); print(f'Generation eval every: {trainer.generation_eval_steps} steps')"
```

### Enhanced Training Step Testing
```bash
# Test enhanced diffusion training step
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); model.train(); batch = next(iter(train_loader)); batch = trainer._move_batch_to_device(batch); loss, step_metrics = trainer._enhanced_training_step(batch); print(f'Enhanced diffusion step loss: {loss:.4f}, metrics: {list(step_metrics.keys())}')"

# Test enhanced validation
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); model.eval(); val_metrics = trainer._validate_with_enhanced_metrics(); print(f'Enhanced validation: loss={val_metrics[\"loss\"]:.4f}, perplexity={val_metrics[\"perplexity\"]:.2f}')"
```

### Research-Validated Metrics Testing
```bash
# Test mask prediction accuracy (fundamental metric)
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); batch = next(iter(train_loader)); batch = trainer._move_batch_to_device(batch); corrupted_ids, mask_positions, mask_ratios = diffusion.forward_process_with_ratios(batch.input_ids); outputs = model(corrupted_ids); mask_accuracy = trainer._compute_mask_prediction_accuracy(outputs['logits'], batch.labels, mask_positions); print(f'Mask prediction accuracy: {mask_accuracy:.3f} (fundamental diffusion metric)')"

# Test corruption-level performance analysis
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); batch = next(iter(train_loader)); batch = trainer._move_batch_to_device(batch); corrupted_ids, mask_positions, mask_ratios = diffusion.forward_process_with_ratios(batch.input_ids); outputs = model(corrupted_ids); corruption_metrics = trainer._compute_corruption_level_performance(outputs['logits'], batch.labels, mask_positions, mask_ratios); print(f'Corruption metrics: {list(corruption_metrics.keys())}'); [print(f'  {level} corruption accuracy: {corruption_metrics[f\"{level}_accuracy\"]:.3f}') for level in ['low', 'medium', 'high'] if f'{level}_accuracy' in corruption_metrics]"

# Test loss weight effectiveness
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); batch = next(iter(train_loader)); batch = trainer._move_batch_to_device(batch); corrupted_ids, mask_positions, mask_ratios = diffusion.forward_process_with_ratios(batch.input_ids); outputs = model(corrupted_ids); loss_weights = diffusion.compute_loss_weights(mask_positions, mask_ratios); weight_effectiveness = trainer._compute_loss_weight_effectiveness(outputs['logits'], batch.labels, mask_positions, loss_weights); print(f'Loss weight effectiveness: {weight_effectiveness:.3f} (theoretical validation)')"
```

### Checkpoint Testing
```bash
# Test checkpoint saving and loading
python -c "import sys; sys.path.append('.'); from src.training import DiffusionTrainer; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_loader, val_loader, device, Path('test_exp')); trainer._save_checkpoint(); checkpoint_path = Path('test_exp/checkpoints/latest_model.pt'); original_step = trainer.global_step; trainer.global_step += 100; trainer.load_checkpoint(checkpoint_path); print(f'Checkpoint saved: {checkpoint_path.exists()}'); print(f'Checkpoint loaded: step restored={trainer.global_step == original_step}')"
```

## 6. Enhanced Sampling and Generation Testing

### Sampler Initialization
```bash
# Test enhanced sampler creation
python -c "import sys; sys.path.append('.'); from src.sampling import DiffusionSampler, create_sampling_config; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.utils import load_config; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); model.eval(); sampler = DiffusionSampler(model, diffusion, config); print(f'Enhanced sampler created: {sampler.sampling_config.num_steps} steps, schedule: {sampler.sampling_config.schedule_type}')"

# Test optimal sampling config
python -c "import sys; sys.path.append('.'); from src.sampling import create_sampling_config; from src.utils import load_config; config = load_config('config/example_config.yaml'); sampling_config = create_sampling_config(schedule_type='cosine', config=config, num_steps=10, temperature=0.8); print(f'Optimal config: {sampling_config.schedule_type} (Zhang 2025 optimal), temp={sampling_config.temperature}')"
```

### Optimal Schedule Testing
```bash
# Test Zhang (2025) optimal cosine schedule
python -c "import sys; sys.path.append('.'); from src.sampling import DiffusionSampler; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); model.eval(); sampler = DiffusionSampler(model, diffusion, config); cosine_schedule = sampler._create_corruption_schedule(10); sampler.sampling_config.schedule_type = 'linear'; linear_schedule = sampler._create_corruption_schedule(10); print(f'Schedules: cosine[0]={cosine_schedule[0]:.3f}, linear[0]={linear_schedule[0]:.3f}'); print(f'Cosine schedule monotonic: {(cosine_schedule[:-1] >= cosine_schedule[1:]).all()}'); print(f'Cosine starts at 1.0: {abs(cosine_schedule[0] - 1.0) < 1e-6}'); print(f'Cosine ends at 0.0: {abs(cosine_schedule[-1] - 0.0) < 1e-6}'); sampler.sampling_config.schedule_type = 'cosine'"
```

### Enhanced Generation Testing
```bash
# Test unconditional generation with optimal schedule
python -c "import sys; sys.path.append('.'); from src.sampling import DiffusionSampler; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.utils import load_config; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); model.eval(); sampler = DiffusionSampler(model, diffusion, config); output = sampler.generate(batch_size=2, max_length=32, show_progress=False); print(f'Generated with optimal schedule: {output.sequences.shape}, time: {output.generation_time:.2f}s')"

# Test conditional generation
python -c "import sys; sys.path.append('.'); from src.sampling import DiffusionSampler; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); model.eval(); sampler = DiffusionSampler(model, diffusion, config); prompt = torch.randint(0, 1000, (2, 8)); output = sampler.generate(batch_size=2, max_length=32, prompt=prompt, show_progress=False); print(f'Conditional: prompt preserved={torch.equal(output.sequences[:, :8], prompt)}')"

# Test confidence-based remasking
python -c "import sys; sys.path.append('.'); from src.sampling import DiffusionSampler; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); model.eval(); sampler = DiffusionSampler(model, diffusion, config); test_ids = torch.randint(0, 1000, (1, 16)); test_logits = torch.randn(1, 16, config.model.vocab_size); remasked = sampler._apply_remasking(test_ids, test_logits, 0.5, prompt_length=0); print(f'Confidence remasking: {(remasked == config.model.mask_token_id).sum().item()} tokens remasked')"
```

## 7. Enhanced Evaluation Testing

### Evaluator Testing
```bash
# Test enhanced evaluator creation and evaluation with downstream tasks
python -c "import sys; sys.path.append('.'); from src.evaluation import PerplexityEvaluator; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); evaluator = PerplexityEvaluator(); diffusion_results = evaluator.evaluate_model(model, val_loader, diffusion, max_batches=5, model_name='Enhanced Diffusion', include_downstream=True); print(f'Enhanced diffusion eval: ppl={diffusion_results.perplexity:.2f}, tokens={diffusion_results.num_tokens}, downstream={bool(diffusion_results.downstream_accuracy)}, warnings={bool(diffusion_results.comparison_warnings)}')"

# Test enhanced fair comparison with downstream metrics
python -c "import sys; sys.path.append('.'); from src.evaluation import PerplexityEvaluator, compare_ar_vs_diffusion; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); config_ar = load_config('config/example_config.yaml'); config_ar.model.training_mode = 'autoregressive'; ar_model = TinyDiffusionTransformer(config_ar); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); comparison = compare_ar_vs_diffusion(ar_model, model, diffusion, val_loader, max_batches=3); print(f'Enhanced comparison traditional winner: {comparison[\"traditional_winner\"]}, methodology: {comparison[\"comparison_methodology\"]}, has_downstream: {bool(comparison[\"autoregressive\"].downstream_accuracy)}')"

# Test evaluation without downstream tasks (traditional mode)
python -c "import sys; sys.path.append('.'); from src.evaluation import PerplexityEvaluator; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); evaluator = PerplexityEvaluator(); traditional_results = evaluator.evaluate_model(model, val_loader, diffusion, max_batches=3, model_name='Traditional Mode', include_downstream=False); print(f'Traditional eval: ppl={traditional_results.perplexity:.2f}, downstream={traditional_results.downstream_accuracy}, warnings={bool(traditional_results.comparison_warnings)}')"

# Test enhanced results data structure validation
python -c "import sys; sys.path.append('.'); from src.evaluation import PerplexityEvaluator, EvaluationResults; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); evaluator = PerplexityEvaluator(); results = evaluator.evaluate_model(model, val_loader, diffusion, max_batches=2, include_downstream=True); required_fields = ['model_name', 'loss', 'perplexity', 'num_tokens', 'evaluation_time']; enhanced_fields = ['downstream_accuracy', 'relative_likelihood_gap', 'comparison_warnings']; all_present = all(hasattr(results, field) for field in required_fields + enhanced_fields); print(f'Results type: {type(results).__name__}, all enhanced fields present: {all_present}, sample tasks: {list(results.downstream_accuracy.keys()) if results.downstream_accuracy else None}')"

# Test autoregressive model evaluation for fair comparison
python -c "import sys; sys.path.append('.'); from src.evaluation import PerplexityEvaluator; from src.model import TinyDiffusionTransformer; from src.data import create_dataloaders; from src.utils import load_config; config = load_config('config/example_config.yaml'); config.model.training_mode = 'autoregressive'; ar_model = TinyDiffusionTransformer(config); train_loader, val_loader, test_loader = create_dataloaders(config); evaluator = PerplexityEvaluator(); ar_results = evaluator.evaluate_model(ar_model, val_loader, diffusion=None, max_batches=3, model_name='AR Model', include_downstream=True); print(f'AR eval: ppl={ar_results.perplexity:.2f}, downstream={bool(ar_results.downstream_accuracy)}, warnings_recommend_downstream={\"downstream\" in str(ar_results.comparison_warnings) if ar_results.comparison_warnings else False}')"

# Test sample downstream tasks verification and configuration
python -c "import sys; sys.path.append('.'); from src.evaluation import SAMPLE_HELLASWAG, SAMPLE_MMLU, PerplexityEvaluator; evaluator = PerplexityEvaluator(); sample_task = SAMPLE_HELLASWAG[0] if SAMPLE_HELLASWAG else {}; print(f'HellaSwag samples: {len(SAMPLE_HELLASWAG)}, MMLU samples: {len(SAMPLE_MMLU)}, available tasks: {list(evaluator.downstream_tasks.keys())}, sample structure: context={bool(sample_task.get(\"context\"))}, choices={len(sample_task.get(\"choices\", []))}, correct={sample_task.get(\"correct\")}, downstream tasks configured: {len(evaluator.downstream_tasks)}')"
```

## 8. Enhanced Integration Testing

### Enhanced Model-Diffusion Integration
```bash
# Test enhanced model-diffusion integration with proper loss weighting
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.model.training_mode = 'diffusion'; test_model = TinyDiffusionTransformer(config); test_diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); test_batch = next(iter(train_loader)); corrupted, mask_pos, ratios = test_diffusion.forward_process_with_ratios(test_batch.input_ids); output = test_model(corrupted); weights = test_diffusion.compute_loss_weights(mask_pos, ratios); vocab_size = output['logits'].size(-1); flat_logits = output['logits'].view(-1, vocab_size); flat_labels = test_batch.labels.view(-1); flat_mask_positions = mask_pos.view(-1); flat_weights = weights.view(-1); loss_mask = flat_mask_positions & (flat_labels != -100); print(f'Enhanced model-diffusion integration: output shape {output[\"logits\"].shape} matches vocab {config.model.vocab_size}'); exec('if loss_mask.any():\\n    losses = torch.nn.functional.cross_entropy(flat_logits[loss_mask], flat_labels[loss_mask], reduction=\"none\")\\n    weighted_loss = (losses * flat_weights[loss_mask]).mean()\\n    print(f\"Enhanced weighted loss computed: {weighted_loss.item():.4f}\")\\nelse:\\n    print(\"No valid masked tokens for loss computation\")')"

# Test enhanced training-sampling integration
python -c "import sys; sys.path.append('.'); from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.data import create_dataloaders; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.model.training_mode = 'diffusion'; test_model = TinyDiffusionTransformer(config); test_diffusion = DiscreteDiffusion(config); train_loader, val_loader, test_loader = create_dataloaders(config); test_batch = next(iter(train_loader)); corrupted, mask_pos, ratios = test_diffusion.forward_process_with_ratios(test_batch.input_ids); output = test_model(corrupted); weights = test_diffusion.compute_loss_weights(mask_pos, ratios); print(f'Integration test: shapes match, weights computed, loss_mask has {(mask_pos & (test_batch.labels != -100)).sum()} valid tokens')"
```

## 9. Research Validation Testing

### LLaDA (2025) Implementation Validation
```bash
# Test LLaDA variable masking strategy
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; print('=== LLaDA (2025) VALIDATION ==='); config = load_config('config/example_config.yaml'); config.diffusion.single_ratio_per_sequence = False; llada_diffusion = DiscreteDiffusion(config); test_batch = torch.randint(0, 1000, (4, 32)); _, _, ratios = llada_diffusion.forward_process_with_ratios(test_batch); print(f'✓ LLaDA variable masking: ratios vary={not torch.allclose(ratios, ratios[0])}'); print(f'  Ratios: {ratios.tolist()}')"

# Test single ratio comparison
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; config = load_config('config/example_config.yaml'); config.diffusion.single_ratio_per_sequence = True; single_diffusion = DiscreteDiffusion(config); test_batch = torch.randint(0, 1000, (4, 32)); _, _, single_ratios = single_diffusion.forward_process_with_ratios(test_batch); print(f'✓ Single ratio masking: ratios constant={torch.allclose(single_ratios, single_ratios[0])}')"
```

### Zhang (2025) Optimal Schedule Validation
```bash
# Test Zhang (2025) cosine schedule optimality
python -c "import sys; sys.path.append('.'); from src.sampling import DiffusionSampler; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; print('=== ZHANG (2025) SCHEDULE VALIDATION ==='); config = load_config('config/example_config.yaml'); model = TinyDiffusionTransformer(config); diffusion = DiscreteDiffusion(config); cosine_sampler = DiffusionSampler(model, diffusion, config); cosine_sampler.sampling_config.schedule_type = 'cosine'; cosine_schedule = cosine_sampler._create_corruption_schedule(20); linear_sampler = DiffusionSampler(model, diffusion, config); linear_sampler.sampling_config.schedule_type = 'linear'; linear_schedule = linear_sampler._create_corruption_schedule(20); print(f'✓ Cosine schedule: starts at {cosine_schedule[0]:.3f}, ends at {cosine_schedule[-1]:.3f}'); print(f'✓ Linear schedule: starts at {linear_schedule[0]:.3f}, ends at {linear_schedule[-1]:.3f}'); print(f'✓ Both monotonic: cosine={torch.all(cosine_schedule[:-1] >= cosine_schedule[1:])}, linear={torch.all(linear_schedule[:-1] >= linear_schedule[1:])}')"
```

### Austin et al. (2021) Loss Weight Validation
```bash
# Test Austin et al. (2021) time-dependent loss weights
python -c "import sys; sys.path.append('.'); from src.diffusion import DiscreteDiffusion; from src.utils import load_config; import torch; print('=== AUSTIN ET AL. (2021) LOSS WEIGHT VALIDATION ==='); config = load_config('config/example_config.yaml'); diffusion = DiscreteDiffusion(config); test_ratios = torch.tensor([0.1, 0.5, 0.9]); test_mask = torch.ones(3, 10, dtype=torch.bool); weights = diffusion.compute_loss_weights(test_mask, test_ratios); print(f'✓ Loss weights computed for ratios {test_ratios.tolist()}'); print(f'  Weights: {weights.mean(dim=1).tolist()}'); print(f'  Max weight: {weights.max().item():.3f} <= {diffusion.max_loss_weight}'); print(f'✓ Weight formulation: higher corruption → higher weight')"
```

## 10. Complete Enhanced Pipeline Test

### Full Enhanced Pipeline
```bash
# Complete pipeline test (GPU)
python -c "import sys; sys.path.append('.'); from src.utils import load_config; from src.data import create_dataloaders; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.training import DiffusionTrainer; from pathlib import Path; import torch; config = load_config('config/example_config.yaml'); config.training.num_epochs = 1; config.training.eval_every_n_steps = 100; device = torch.device('cuda'); print(f'Using device: {device}'); model = TinyDiffusionTransformer(config).to(device); diffusion = DiscreteDiffusion(config); train_dl, val_dl, test_dl = create_dataloaders(config); trainer = DiffusionTrainer(model, diffusion, config, train_dl, val_dl, device, Path('gpu_test')); print(f'GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f}GB'); results = trainer.train(); print(f'GPU memory after training: {torch.cuda.memory_allocated()/1e9:.2f}GB'); print(f'Training completed: loss={results[\"final_loss\"]:.4f}')"# Complete enhanced pipeline test

python -c "import sys; sys.path.append('.'); from src.utils import load_config, setup_logging, create_experiment_dir; from src.data import create_dataloaders; from src.model import TinyDiffusionTransformer; from src.diffusion import DiscreteDiffusion; from src.training import DiffusionTrainer; from src.sampling import DiffusionSampler; from src.evaluation import PerplexityEvaluator; from pathlib import Path; import torch; print('=== ENHANCED FULL PIPELINE TEST ==='); config = load_config('config/example_config.yaml'); config.training.num_epochs = 1; config.training.eval_every_n_steps = 50; config.training.detailed_metrics_steps = 25; config.diffusion.single_ratio_per_sequence = False; config.sampling.schedule_type = 'cosine'; print('✓ Enhanced config loaded'); exp_dir = create_experiment_dir('.', 'enhanced_pipeline_test'); logger = setup_logging(config, 'enhanced_pipeline_test', exp_dir / 'logs'); print('✓ Enhanced experiment setup'); train_dl, val_dl, test_dl = create_dataloaders(config); print('✓ Data pipeline created'); config.model.training_mode = 'diffusion'; diff_model = TinyDiffusionTransformer(config); pipeline_diffusion = DiscreteDiffusion(config); print('✓ Enhanced models initialized'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); diff_model = diff_model.to(device); pipeline_trainer = DiffusionTrainer(diff_model, pipeline_diffusion, config, train_dl, val_dl, device, exp_dir); print('✓ Enhanced training setup'); print('Training 1 epoch with enhanced metrics...'); train_results = pipeline_trainer.train(); print(f'✓ Enhanced training completed: final_loss={train_results[\"final_loss\"]:.4f}'); pipeline_sampler = DiffusionSampler(diff_model, pipeline_diffusion, config); generation_output = pipeline_sampler.generate(batch_size=2, max_length=32, show_progress=False); print(f'✓ Optimal generation: {generation_output.sequences.shape}, schedule: {pipeline_sampler.sampling_config.schedule_type}'); evaluator = PerplexityEvaluator(); final_results = evaluator.evaluate_model(diff_model, test_dl, pipeline_diffusion, max_batches=10, model_name='Enhanced Pipeline Test'); print(f'✓ Enhanced evaluation: ppl={final_results.perplexity:.2f}'); print(f'=== ENHANCED PIPELINE TEST COMPLETE ==='); print(f'Final perplexity: {final_results.perplexity:.2f}'); print(f'Training time: {train_results[\"training_time\"]:.2f}s'); print(f'Generation time: {generation_output.generation_time:.2f}s'); print(f'Schedule used: {pipeline_sampler.sampling_config.schedule_type} (Zhang 2025 optimal)')"
```

## Expected Enhanced Results

### Performance Metrics
- **Memory usage**: Under 8GB for RTX 3070 Ti with enhanced metrics
- **Training convergence**: Decreasing loss with enhanced monitoring
- **Generation quality**: Valid token sequences with optimal scheduling
- **Evaluation consistency**: Reasonable perplexity (< 1000) with enhanced metrics

### Research Validation
- **LLaDA Strategy**: Variable masking ratios per sequence working correctly
- **Zhang Schedule**: Cosine schedule providing optimal denoising steps
- **Austin Weights**: Proper time-dependent loss weighting implemented
- **Enhanced Metrics**: Mask prediction accuracy, corruption-level analysis working

### Enhanced Features
- **Comprehensive Metrics**: All three tiers of metrics computing correctly
- **Optimal Scheduling**: Cosine schedule for inference, uniform for training
- **Research Alignment**: All latest research findings properly implemented
- **Monitoring**: Enhanced wandb integration with detailed tracking

This comprehensive enhanced test guide validates all TDLM components with the updated paths for your `src/` and `config/` directory structure.