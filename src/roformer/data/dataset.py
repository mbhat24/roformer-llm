"""
Dataset classes for RoFormer LLM training
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional
from ..data.tokenizer import WordTokenizer


class TextDataset(Dataset):
    """Text dataset for language modeling"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: WordTokenizer,
        max_length: int = 256,
        add_bos: bool = True,
        add_eos: bool = True
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: WordTokenizer instance
            max_length: Maximum sequence length
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            add_bos=self.add_bos,
            add_eos=self.add_eos
        )
        
        # For language modeling, input = tokens[:-1], target = tokens[1:]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return input_ids, target_ids


def collate_fn(batch):
    """Collate function to pad sequences"""
    input_ids, target_ids = zip(*batch)
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    padded_target_ids = torch.zeros(len(target_ids), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
        padded_input_ids[i, :len(inp)] = inp
        padded_target_ids[i, :len(tgt)] = tgt
    
    # Create attention mask
    attention_mask = (padded_input_ids != 0).long()
    
    return padded_input_ids, padded_target_ids, attention_mask
