"""
Download and prepare real text dataset for training
"""

import requests
import os
from typing import List


def download_shakespeare() -> str:
    """Download Shakespeare's complete works from Project Gutenberg"""
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    
    print("Downloading Shakespeare's complete works...")
    response = requests.get(url)
    text = response.text
    
    # Remove header and footer (Project Gutenberg license text)
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx]
    
    # Save to file
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/shakespeare.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Downloaded {len(text)} characters")
    print(f"Saved to data/raw/shakespeare.txt")
    
    return text


def prepare_training_data(text: str, chunk_size: 512) -> List[str]:
    """
    Split text into chunks for training
    
    Args:
        text: Full text
        chunk_size: Number of words per chunk
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:  # Only keep chunks with at least 50 words
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} training chunks")
    print(f"Average chunk size: {sum(len(c.split()) for c in chunks) / len(chunks):.1f} words")
    
    return chunks


def main():
    """Main function to download and prepare data"""
    # Download Shakespeare
    text = download_shakespeare()
    
    # Prepare chunks
    chunks = prepare_training_data(text, chunk_size=256)
    
    # Save chunks
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/training_chunks.txt', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')
    
    print(f"\nTraining data ready!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to data/processed/training_chunks.txt")


if __name__ == "__main__":
    main()
