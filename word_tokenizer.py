"""
Word-level tokenizer for better text generation
"""

import torch
from collections import Counter
from typing import List, Optional


class WordTokenizer:
    """
    Word-level tokenizer with special tokens
    """
    
    def __init__(self, texts: Optional[List[str]] = None, vocab_size: int = 10000):
        """
        Args:
            texts: List of texts to build vocabulary from
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        
        if texts is not None:
            self.build_vocab(texts)
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word frequencies
        word_counter = Counter()
        for text in texts:
            words = text.lower().split()
            word_counter.update(words)
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.eos_token, self.bos_token]
        self.word_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Add most common words
        for word, _ in word_counter.most_common(self.vocab_size - len(special_tokens)):
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
        
        # Create reverse mapping
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        print(f"Built vocabulary with {len(self.word_to_id)} words")
        print(f"Top 10 words: {list(self.word_to_id.keys())[4:14]}")
    
    def encode(self, text: str, max_length: Optional[int] = None, add_bos: bool = True, add_eos: bool = True) -> torch.Tensor:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        
        Returns:
            Token IDs tensor
        """
        words = text.lower().split()
        tokens = []
        
        if add_bos:
            tokens.append(self.word_to_id[self.bos_token])
        
        for word in words:
            tokens.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))
        
        if add_eos:
            tokens.append(self.word_to_id[self.eos_token])
        
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.word_to_id[self.pad_token]] * (max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor, skip_special: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token IDs tensor or list
            skip_special: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        special_tokens = {self.word_to_id[t] for t in [self.pad_token, self.unk_token, self.eos_token, self.bos_token]}
        
        words = []
        for token_id in token_ids:
            if skip_special and token_id in special_tokens:
                continue
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self.word_to_id)
    
    def get_pad_token_id(self) -> int:
        """Return pad token ID"""
        return self.word_to_id[self.pad_token]
    
    def get_eos_token_id(self) -> int:
        """Return EOS token ID"""
        return self.word_to_id[self.eos_token]
    
    def get_bos_token_id(self) -> int:
        """Return BOS token ID"""
        return self.word_to_id[self.bos_token]
