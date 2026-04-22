"""
Custom data loading for RoFormer LLM
Supports user-provided text files, URLs, or direct text input
"""

import os
import requests
from typing import List, Optional
from pathlib import Path


class CustomDataLoader:
    """Load custom training data from various sources"""
    
    @staticmethod
    def from_file(file_path: str) -> List[str]:
        """
        Load data from a text file
        
        Args:
            file_path: Path to text file
        
        Returns:
            List of text chunks
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks (by paragraphs or lines)
        chunks = CustomDataLoader._split_text(text)
        print(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    
    @staticmethod
    def from_url(url: str) -> List[str]:
        """
        Load data from a URL
        
        Args:
            url: URL to fetch text from
        
        Returns:
            List of text chunks
        """
        print(f"Fetching data from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        
        chunks = CustomDataLoader._split_text(text)
        print(f"Loaded {len(chunks)} chunks from URL")
        return chunks
    
    @staticmethod
    def from_text(text: str, chunk_size: int = 256) -> List[str]:
        """
        Load data from direct text input
        
        Args:
            text: Raw text string
            chunk_size: Words per chunk
        
        Returns:
            List of text chunks
        """
        chunks = CustomDataLoader._split_text(text, chunk_size)
        print(f"Created {len(chunks)} chunks from provided text")
        return chunks
    
    @staticmethod
    def from_directory(directory: str, pattern: str = "*.txt") -> List[str]:
        """
        Load data from all text files in a directory
        
        Args:
            directory: Path to directory
            pattern: File pattern to match
        
        Returns:
            List of text chunks
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_chunks = []
        for file_path in directory.glob(pattern):
            chunks = CustomDataLoader.from_file(str(file_path))
            all_chunks.extend(chunks)
        
        print(f"Loaded {len(all_chunks)} total chunks from {len(list(directory.glob(pattern)))} files")
        return all_chunks
    
    @staticmethod
    def _split_text(text: str, chunk_size: int = 256) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Text to split
            chunk_size: Target words per chunk
        
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            words = paragraph.split()
            
            # If paragraph is small enough, keep it
            if len(words) <= chunk_size:
                if len(words) > 10:  # Minimum 10 words
                    chunks.append(' '.join(words))
            else:
                # Split large paragraphs
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if len(chunk.split()) > 10:
                        chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def save_chunks(chunks: List[str], output_path: str):
        """
        Save chunks to file
        
        Args:
            chunks: List of text chunks
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + '\n')
        
        print(f"Saved {len(chunks)} chunks to {output_path}")


def interactive_data_loader():
    """Interactive data loading for users"""
    print("="*60)
    print("CUSTOM DATA LOADER")
    print("="*60)
    print("\nChoose data source:")
    print("1. Load from file")
    print("2. Load from URL")
    print("3. Paste text directly")
    print("4. Load from directory")
    print("5. Use built-in Shakespeare dataset")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    chunks = []
    
    if choice == '1':
        file_path = input("Enter file path: ").strip()
        chunks = CustomDataLoader.from_file(file_path)
    
    elif choice == '2':
        url = input("Enter URL: ").strip()
        chunks = CustomDataLoader.from_url(url)
    
    elif choice == '3':
        print("\nPaste your text (press Ctrl+D or Ctrl+Z to finish):")
        text = ""
        try:
            while True:
                line = input()
                text += line + "\n"
        except EOFError:
            pass
        
        chunk_size = int(input("Enter chunk size (words, default 256): ") or "256")
        chunks = CustomDataLoader.from_text(text, chunk_size)
    
    elif choice == '4':
        directory = input("Enter directory path: ").strip()
        pattern = input("Enter file pattern (default *.txt): ") or "*.txt"
        chunks = CustomDataLoader.from_directory(directory, pattern)
    
    elif choice == '5':
        # Use existing Shakespeare data
        if os.path.exists('data/training_chunks.txt'):
            with open('data/training_chunks.txt', 'r', encoding='utf-8') as f:
                chunks = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(chunks)} chunks from Shakespeare dataset")
        else:
            print("Shakespeare dataset not found. Please run download_data.py first")
            return None
    
    else:
        print("Invalid choice")
        return None
    
    if chunks:
        save = input(f"\nSave chunks to data/custom_chunks.txt? (y/n): ").strip().lower()
        if save == 'y':
            CustomDataLoader.save_chunks(chunks, 'data/custom_chunks.txt')
        
        return chunks
    
    return None


if __name__ == "__main__":
    chunks = interactive_data_loader()
    if chunks:
        print(f"\nReady to train with {len(chunks)} chunks")
