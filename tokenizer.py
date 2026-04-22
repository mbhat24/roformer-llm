"""
Simple character-level tokenizer for demonstration
For production use, you'd want to use a proper tokenizer like HuggingFace's
"""

import torch
from collections import Counter


class SimpleTokenizer:
    """
    Simple character-level tokenizer for demonstration purposes
    """
    
    def __init__(self, texts=None):
        """
        Args:
            texts: List of texts to build vocabulary from
        """
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        
        if texts is not None:
            self.build_vocab(texts)
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Count character frequencies
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        self.char_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Add characters sorted by frequency
        for char, _ in char_counter.most_common():
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
        
        # Create reverse mapping
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def encode(self, text, max_length=None, add_eos=True):
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_eos: Whether to add EOS token
        
        Returns:
            Token IDs tensor
        """
        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.char_to_id[self.unk_token]))
        
        if add_eos:
            tokens.append(self.char_to_id[self.eos_token])
        
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.char_to_id[self.pad_token]] * (max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, token_ids, skip_special=True):
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token IDs tensor or list
            skip_special: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        special_tokens = {self.char_to_id[t] for t in [self.pad_token, self.unk_token, self.eos_token]}
        
        text = ""
        for token_id in token_ids:
            if skip_special and token_id in special_tokens:
                continue
            if token_id in self.id_to_char:
                text += self.id_to_char[token_id]
        
        return text
    
    def get_vocab_size(self):
        """Return vocabulary size"""
        return self.vocab_size
    
    def get_pad_token_id(self):
        """Return pad token ID"""
        return self.char_to_id[self.pad_token]
    
    def get_eos_token_id(self):
        """Return EOS token ID"""
        return self.char_to_id[self.eos_token]
