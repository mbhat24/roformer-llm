"""
Standard LLM benchmarks for RoFormer
Includes perplexity, generation quality, and performance metrics
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Dict
from ..model.llm import RoFormerLLM, RoFormerLLMConfig
from ..data.tokenizer import WordTokenizer


class LLMBenchmarks:
    """Standard LLM evaluation benchmarks"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, texts: List[str], max_length: int = 256) -> float:
        """
        Compute perplexity on a set of texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
        
        Returns:
            Perplexity score (lower is better)
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_pad_token_id(), reduction='sum')
        
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer.encode(text, max_length=max_length, add_bos=True, add_eos=True)
                if len(tokens) < 10:
                    continue
                
                input_ids = tokens[:-1].unsqueeze(0).to(self.device)
                target_ids = tokens[1:].unsqueeze(0).to(self.device)
                
                logits = self.model(input_ids)
                
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += target_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def benchmark_generation_speed(self, prompts: List[str], max_new_tokens: int = 50) -> Dict:
        """
        Benchmark text generation speed
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Number of tokens to generate
        
        Returns:
            Dictionary with speed metrics
        """
        self.model.eval()
        
        total_time = 0
        total_tokens = 0
        
        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = self.tokenizer.encode(prompt, max_length=100, add_bos=False, add_eos=False)
                prompt_ids = prompt_ids.unsqueeze(0).to(self.device)
                
                start_time = time.time()
                generated = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    do_sample=True
                )
                inference_time = time.time() - start_time
                
                total_time += inference_time
                total_tokens += (generated.shape[1] - prompt_ids.shape[1])
        
        avg_time = total_time / len(prompts)
        avg_tokens = total_tokens / len(prompts)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'avg_tokens_generated': avg_tokens,
            'tokens_per_second': tokens_per_second,
            'total_prompts': len(prompts)
        }
    
    def benchmark_memory_usage(self) -> Dict:
        """
        Benchmark memory usage during inference
        
        Returns:
            Dictionary with memory metrics
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_mem = process.memory_info().rss / 1024 / 1024
        
        # Memory during inference
        prompt = "The attention mechanism allows models to focus on relevant parts of the input."
        prompt_ids = self.tokenizer.encode(prompt, max_length=100, add_bos=False, add_eos=False)
        prompt_ids = prompt_ids.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(prompt_ids, max_new_tokens=50, temperature=0.8)
        
        peak_mem = process.memory_info().rss / 1024 / 1024
        
        return {
            'baseline_memory_mb': baseline_mem,
            'peak_memory_mb': peak_mem,
            'memory_increase_mb': peak_mem - baseline_mem
        }
    
    def evaluate_coherence(self, prompts: List[str], max_new_tokens: int = 30) -> Dict:
        """
        Evaluate generation coherence (simple heuristic)
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Number of tokens to generate
        
        Returns:
            Dictionary with coherence metrics
        """
        self.model.eval()
        
        coherence_scores = []
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = self.tokenizer.encode(prompt, max_length=100, add_bos=False, add_eos=False)
                prompt_ids = prompt_ids.unsqueeze(0).to(self.device)
                
                generated = self.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    do_sample=True
                )
                
                generated_text = self.tokenizer.decode(generated[0].cpu().numpy())
                generated_texts.append(generated_text)
                
                # Simple coherence: check for repeated words
                words = generated_text.split()
                unique_words = len(set(words))
                repetition_ratio = 1 - (unique_words / len(words)) if len(words) > 0 else 0
                coherence_scores.append(1 - repetition_ratio)  # Higher is better
        
        return {
            'avg_coherence': np.mean(coherence_scores),
            'generated_texts': generated_texts,
            'coherence_scores': coherence_scores
        }
    
    def run_full_benchmark(self, test_texts: List[str], prompts: List[str]) -> Dict:
        """
        Run all benchmarks
        
        Args:
            test_texts: Texts for perplexity evaluation
            prompts: Prompts for generation benchmarks
        
        Returns:
            Dictionary with all benchmark results
        """
        print("="*60)
        print("RUNNING FULL BENCHMARK SUITE")
        print("="*60)
        
        results = {}
        
        # Perplexity
        print("\n1. Computing Perplexity...")
        perplexity = self.compute_perplexity(test_texts)
        results['perplexity'] = perplexity
        print(f"   Perplexity: {perplexity:.2f}")
        
        # Generation speed
        print("\n2. Benchmarking Generation Speed...")
        speed_results = self.benchmark_generation_speed(prompts)
        results['generation_speed'] = speed_results
        print(f"   Avg inference time: {speed_results['avg_inference_time_ms']:.1f}ms")
        print(f"   Tokens per second: {speed_results['tokens_per_second']:.1f}")
        
        # Memory usage
        print("\n3. Benchmarking Memory Usage...")
        memory_results = self.benchmark_memory_usage()
        results['memory_usage'] = memory_results
        print(f"   Baseline memory: {memory_results['baseline_memory_mb']:.1f} MB")
        print(f"   Peak memory: {memory_results['peak_memory_mb']:.1f} MB")
        print(f"   Memory increase: {memory_results['memory_increase_mb']:.1f} MB")
        
        # Coherence
        print("\n4. Evaluating Generation Coherence...")
        coherence_results = self.evaluate_coherence(prompts)
        results['coherence'] = coherence_results
        print(f"   Avg coherence score: {coherence_results['avg_coherence']:.3f}")
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        
        return results


def load_trained_model(checkpoint_path: str = 'roformer_shakespeare_fast.pt'):
    """Load trained model for benchmarking"""
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config_dict = checkpoint['config']
    config = RoFormerLLMConfig.from_dict(config_dict)
    
    model = RoFormerLLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_position_embeddings=config.max_position_embeddings,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint['tokenizer']
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    return model, tokenizer, device


def main():
    """Run benchmarks on trained model"""
    # Load model
    model, tokenizer, device = load_trained_model()
    
    # Create benchmark suite
    benchmarks = LLMBenchmarks(model, tokenizer, device)
    
    # Test data (use held-out Shakespeare chunks)
    with open('data/training_chunks.txt', 'r', encoding='utf-8') as f:
        all_chunks = [line.strip() for line in f if line.strip()]
    
    # Use last 100 chunks for testing
    test_texts = all_chunks[-100:]
    
    # Prompts for generation
    prompts = [
        "To be, or not to be",
        "What's in a name",
        "All the world's a stage",
        "Shall I compare thee",
        "The course of true love"
    ]
    
    # Run benchmarks
    results = benchmarks.run_full_benchmark(test_texts, prompts)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Generation Speed: {results['generation_speed']['tokens_per_second']:.1f} tokens/sec")
    print(f"Memory Usage: {results['memory_usage']['memory_increase_mb']:.1f} MB")
    print(f"Coherence Score: {results['coherence']['avg_coherence']:.3f}")
    print("="*60)
    
    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
