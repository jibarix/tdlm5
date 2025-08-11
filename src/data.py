"""
Simplified Data processing module for TDLM.
Removed over-engineering, kept only what's actually needed.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from datasets import load_dataset
from transformers import AutoTokenizer

from .utils import TDLMConfig, format_number


@dataclass
class DataCollatorOutput:
    """Output structure for data collator."""
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    original_lengths: Optional[torch.Tensor] = None


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for TDLM training."""
    
    def __init__(self, config: TDLMConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.max_length = config.model.max_seq_length
        
        # Load tokenizer directly
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and process dataset
        self._load_and_process_dataset()
        
        logging.info(f"Loaded {split} split: {len(self)} examples")
    
    def _load_and_process_dataset(self):
        """Load and preprocess WikiText-2."""
        # Load dataset
        dataset_name = "wikitext-2-raw-v1"
        self.raw_dataset = load_dataset("wikitext", dataset_name, split=self.split)
        
        # Filter valid texts (non-empty, non-header)
        def is_valid(example):
            text = example['text'].strip()
            return len(text) > 0 and not text.startswith('=')
        
        filtered_dataset = self.raw_dataset.filter(is_valid)
        
        # Tokenize
        def tokenize_batch(examples):
            tokenized = self.tokenizer(
                examples['text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return tokenized
        
        self.processed_dataset = filtered_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=filtered_dataset.column_names
        )
        
        self.processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    def __len__(self) -> int:
        return len(self.processed_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_dataset[idx]


class TDLMDataCollator:
    """Data collator for TDLM training."""
    
    def __init__(self, config: TDLMConfig):
        self.config = config
        self.training_mode = config.model.training_mode
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> DataCollatorOutput:
        """Collate a batch of examples."""
        # Stack tensors
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
        
        # Calculate original lengths
        original_lengths = attention_mask.sum(dim=1)
        
        # Create labels based on training mode
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return DataCollatorOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            original_lengths=original_lengths
        )


def create_dataloaders(
    config: TDLMConfig,
    num_workers: Optional[int] = None,
    distributed: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    if num_workers is None:
        num_workers = getattr(config.data, 'num_workers', 4)
    
    # Create datasets
    train_dataset = WikiTextDataset(config, split="train")
    val_dataset = WikiTextDataset(config, split="validation") 
    test_dataset = WikiTextDataset(config, split="test")
    
    # Create collator
    collator = TDLMDataCollator(config)
    
    # Batch size from training config
    batch_size = config.training.batch_size
    
    # Distributed samplers if needed
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    
    # Create dataloaders
    dataloader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collator,
        'num_workers': num_workers,
        'pin_memory': True
    }
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler, 
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )
    
    logging.info(f"Dataset Statistics:")
    logging.info(f"  Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    logging.info(f"  Validation: {len(val_dataset)} examples, {len(val_loader)} batches") 
    logging.info(f"  Test: {len(test_dataset)} examples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# Export main functions and classes
__all__ = [
    'WikiTextDataset',
    'TDLMDataCollator', 
    'DataCollatorOutput',
    'create_dataloaders'
]