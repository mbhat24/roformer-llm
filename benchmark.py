"""
Performance benchmark for RoFormer LLM
Shows inference speed, memory usage, and generation quality
"""

import torch
import time
import psutil
import os
from model import RoFormerLLM, RoFormerLLMConfig
from tokenizer import SimpleTokenizer


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_model(checkpoint_path='roformer_llm.pt'):
    """Load trained model from checkpoint"""
    print("="*60)
    print("Loading RoFormer LLM")
    print("="*60)
    
    start_time = time.time()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    load_time = time.time() - start_time
    
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
    model.eval()
    
    tokenizer = checkpoint['tokenizer']
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model loaded in {load_time:.2f}s")
    print(f"\nModel Configuration:")
    print(f"  - Embedding dimension: {config.embed_dim}")
    print(f"  - Attention heads: {config.num_heads}")
    print(f"  - Transformer layers: {config.num_layers}")
    print(f"  - Feed-forward dimension: {config.ff_dim}")
    print(f"  - Max sequence length: {config.max_position_embeddings}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"\nModel Size:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    return model, tokenizer


def benchmark_inference(model, tokenizer, device='cpu'):
    """Benchmark inference performance"""
    print("\n" + "="*60)
    print("Inference Performance Benchmark")
    print("="*60)
    
    model = model.to(device)
    
    prompts = [
        "The attention mechanism",
        "Machine learning",
        "Rotary position embeddings",
        "Deep learning",
        "Transformers"
    ]
    
    temperatures = [0.5, 0.8, 1.0]
    
    results = []
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        total_time = 0
        total_tokens = 0
        
        for prompt in prompts:
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt, max_length=50, add_eos=False)
            prompt_tokens = len(prompt_ids)
            prompt_ids = prompt_ids.unsqueeze(0).to(device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=30,
                    temperature=temp,
                    do_sample=True
                )
            inference_time = time.time() - start_time
            
            # Count generated tokens
            generated_tokens = generated.shape[1] - prompt_tokens
            total_tokens += generated_tokens
            total_time += inference_time
            
            # Calculate tokens per second
            tokens_per_sec = generated_tokens / inference_time if inference_time > 0 else 0
            
            results.append({
                'prompt': prompt,
                'temperature': temp,
                'inference_time': inference_time,
                'generated_tokens': generated_tokens,
                'tokens_per_sec': tokens_per_sec
            })
            
            print(f"  '{prompt}': {inference_time*1000:.1f}ms, {generated_tokens} tokens, {tokens_per_sec:.1f} tokens/sec")
        
        avg_time = total_time / len(prompts)
        avg_tokens = total_tokens / len(prompts)
        avg_tps = avg_tokens / avg_time if avg_time > 0 else 0
        
        print(f"\n  Average: {avg_time*1000:.1f}ms, {avg_tokens:.1f} tokens, {avg_tps:.1f} tokens/sec")
    
    return results


def benchmark_memory(model, tokenizer, device='cpu'):
    """Benchmark memory usage"""
    print("\n" + "="*60)
    print("Memory Usage Benchmark")
    print("="*60)
    
    model = model.to(device)
    
    # Baseline memory
    baseline_memory = get_memory_usage()
    print(f"\nBaseline memory: {baseline_memory:.1f} MB")
    
    # Memory after loading model
    model_memory = get_memory_usage()
    print(f"Model memory: {model_memory:.1f} MB")
    print(f"Model overhead: {model_memory - baseline_memory:.1f} MB")
    
    # Memory during inference
    prompt = "The attention mechanism allows models to focus on relevant parts."
    prompt_ids = tokenizer.encode(prompt, max_length=100, add_eos=False).unsqueeze(0).to(device)
    
    pre_inference_memory = get_memory_usage()
    
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8)
    
    post_inference_memory = get_memory_usage()
    
    print(f"\nPre-inference memory: {pre_inference_memory:.1f} MB")
    print(f"Post-inference memory: {post_inference_memory:.1f} MB")
    print(f"Inference memory increase: {post_inference_memory - pre_inference_memory:.1f} MB")


def interactive_demo(model, tokenizer, device='cpu'):
    """Interactive generation demo"""
    print("\n" + "="*60)
    print("Interactive Generation Demo")
    print("="*60)
    print("\nType a prompt to generate text (or 'quit' to exit)")
    print("You can also specify temperature like: 'prompt|0.7'\n")
    
    model = model.to(device)
    
    while True:
        user_input = input("Prompt> ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        # Parse temperature
        if '|' in user_input:
            parts = user_input.split('|')
            prompt = parts[0].strip()
            try:
                temperature = float(parts[1].strip())
            except:
                temperature = 0.8
        else:
            prompt = user_input
            temperature = 0.8
        
        # Encode
        prompt_ids = tokenizer.encode(prompt, max_length=100, add_eos=False)
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
        
        # Generate with timing
        print(f"\nGenerating (temperature={temperature})...")
        start_time = time.time()
        
        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=True
            )
        
        inference_time = time.time() - start_time
        
        # Decode
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        
        print(f"\nGenerated ({inference_time*1000:.1f}ms):")
        print(f"{generated_text}")
        print("-" * 60)


def main():
    """Main benchmark function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    model, tokenizer = load_model()
    
    # Run benchmarks
    benchmark_inference(model, tokenizer, device)
    benchmark_memory(model, tokenizer, device)
    
    # Interactive demo
    interactive_demo(model, tokenizer, device)
    
    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)


if __name__ == "__main__":
    main()
