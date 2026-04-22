"""
Data preprocessing pipeline for RoFormer LLM
"""

import os
import re
from typing import List, Optional
from pathlib import Path


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        min_length: int = 10,
        max_length: int = 10000
    ):
        """
        Initialize preprocessor
        
        Args:
            lowercase: Convert to lowercase
            remove_special_chars: Remove special characters
            remove_numbers: Remove numbers
            min_length: Minimum text length (words)
            max_length: Maximum text length (words)
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean text
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z\s\.,!?;:]', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 256,
        overlap: int = 0
    ) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text
            chunk_size: Target words per chunk
            overlap: Word overlap between chunks
        
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= self.min_length:
                chunks.append(chunk)
        
        return chunks
    
    def filter_chunks(self, chunks: List[str]) -> List[str]:
        """
        Filter chunks by length
        
        Args:
            chunks: List of text chunks
        
        Returns:
            Filtered chunks
        """
        filtered = []
        for chunk in chunks:
            word_count = len(chunk.split())
            if self.min_length <= word_count <= self.max_length:
                filtered.append(chunk)
        
        return filtered
    
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        chunk_size: int = 256
    ) -> List[str]:
        """
        Process a file
        
        Args:
            input_path: Input file path
            output_path: Output file path (optional)
            chunk_size: Chunk size
        
        Returns:
            List of processed chunks
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean
        text = self.clean_text(text)
        
        # Split into chunks
        chunks = self.split_into_chunks(text, chunk_size)
        
        # Filter
        chunks = self.filter_chunks(chunks)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(chunk + '\n')
        
        return chunks


class DataPipeline:
    """Complete data processing pipeline"""
    
    def __init__(self, config):
        """
        Initialize data pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_special_chars=True,
            min_length=10,
            max_length=config.data.max_length
        )
    
    def process_raw_data(self, raw_data_path: str, output_path: str):
        """
        Process raw data file
        
        Args:
            raw_data_path: Path to raw data
            output_path: Path to save processed data
        """
        print(f"Processing raw data: {raw_data_path}")
        chunks = self.preprocessor.process_file(
            raw_data_path,
            output_path,
            chunk_size=self.config.data.max_length
        )
        print(f"Processed {len(chunks)} chunks")
        print(f"Saved to: {output_path}")
        
        return chunks
    
    def validate_data(self, data_path: str) -> bool:
        """
        Validate processed data
        
        Args:
            data_path: Path to processed data
        
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(data_path):
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return False
        
        # Check each line
        for line in lines:
            if len(line.strip().split()) < 5:
                return False
        
        return True
