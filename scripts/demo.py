"""
Demo script for RoFormer LLM
Shows how to use the trained model for text generation
"""

import torch
from model import RoFormerLLM, RoFormerLLMConfig
from tokenizer import SimpleTokenizer


def load_model(checkpoint_path='roformer_llm.pt'):
    """Load trained model from checkpoint"""
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
    model.eval()
    
    tokenizer = checkpoint['tokenizer']
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """Generate text from a prompt"""
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, max_length=100, add_eos=False)
    prompt_ids = prompt_ids.unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].cpu().numpy())
    
    return generated_text


def interactive_demo():
    """Interactive text generation demo"""
    print("Loading model...")
    model, tokenizer = load_model()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\n" + "="*50)
    print("RoFormer LLM Interactive Demo")
    print("="*50)
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("Enter prompt: ")
        
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=100)
        print(f"\nGenerated: {generated}")
        print("\n" + "-"*50 + "\n")


def batch_generation_demo():
    """Demo with multiple prompts"""
    print("Loading model...")
    model, tokenizer = load_model()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    prompts = [
        "The attention mechanism",
        "Machine learning",
        "Rotary position embeddings",
        "Deep learning",
        "Transformers"
    ]
    
    print("\n" + "="*50)
    print("Batch Generation Demo")
    print("="*50 + "\n")
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=0.7)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_generation_demo()
    else:
        interactive_demo()
