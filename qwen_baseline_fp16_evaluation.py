"""
Baseline FP16 Evaluation for Qwen 2.5 0.5B Model
Computes:
- Perplexity on WikiText-2 test set
- Generates responses for all 96 prompts
- Records logits (top-50 for first 10 tokens)
- Stores baseline metrics for comparison
"""

import torch
import json
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    devNumber = torch.cuda.current_device()
    devName = torch.cuda.get_device_name(devNumber)
    print(f"GPU: {devName} (Device {devNumber})")

# Load Qwen 0.5B model in FP16
print("\n" + "="*80)
print("LOADING QWEN 2.5 0.5B MODEL (FP16)")
print("="*80)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)
model.eval()
print(f"✓ Model loaded on {model.device} in {model.dtype}")

# =============================================================================
# PERPLEXITY COMPUTATION ON WIKITEXT-2
# =============================================================================
def compute_perplexity(model, tokenizer, test_data, max_samples=None):
    """Compute perplexity on WikiText-2 test set"""
    print(f"\n{'='*80}")
    print("COMPUTING PERPLEXITY ON WIKITEXT-2")
    print(f"{'='*80}")
    
    total_loss = 0.0
    total_tokens = 0
    
    samples = test_data if max_samples is None else test_data[:max_samples]
    
    with torch.no_grad():
        for item in tqdm(samples, desc="Computing PPL"):
            text = item['text']
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs.input_ids.to(device)
            
            if input_ids.size(1) < 2:
                continue
            
            # Forward pass with use_cache=False to avoid DynamicCache issues
            outputs = model(input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss
            
            # Accumulate
            num_tokens = input_ids.size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"✓ Perplexity: {perplexity:.4f}")
    return perplexity

# =============================================================================
# PROMPT SUITE GENERATION (96 PROMPTS)
# =============================================================================
def generate_prompt_responses(model, tokenizer, prompts, record_logits=True):
    """Generate responses for all 96 prompts and record logits"""
    print(f"\n{'='*80}")
    print("GENERATING RESPONSES FOR 96 PROMPTS")
    print(f"{'='*80}")
    
    results = []
    total_time = 0.0
    success_count = 0
    
    with torch.no_grad():
        for idx, prompt_item in enumerate(tqdm(prompts, desc="Generating")):
            prompt_text = prompt_item['prompt']
            category = prompt_item['category']
            
            try:
                # Tokenize
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                
                # Generate
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    output_scores=record_logits,
                    return_dict_in_generate=True
                )
                gen_time = time.time() - start_time
                total_time += gen_time
                
                # Decode response
                generated_ids = outputs.sequences[0]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract logits for first 10 tokens (top 50 per token)
                logits_record = []
                if record_logits and hasattr(outputs, 'scores'):
                    for token_idx, scores in enumerate(outputs.scores[:10]):
                        # Get top 50 logits
                        top_logits, top_indices = torch.topk(scores[0], k=min(50, scores.size(-1)))
                        logits_record.append({
                            'token_position': token_idx,
                            'top_50_logits': top_logits.cpu().tolist(),
                            'top_50_token_ids': top_indices.cpu().tolist()
                        })
                
                results.append({
                    'prompt_id': idx,
                    'category': category,
                    'prompt': prompt_text,
                    'response': generated_text,
                    'generation_time': gen_time,
                    'logits': logits_record,
                    'success': True
                })
                success_count += 1
                
            except Exception as e:
                print(f"\n✗ Error on prompt {idx}: {str(e)}")
                results.append({
                    'prompt_id': idx,
                    'category': category,
                    'prompt': prompt_text,
                    'response': None,
                    'generation_time': 0.0,
                    'logits': [],
                    'success': False,
                    'error': str(e)
                })
    
    avg_time = total_time / len(prompts) if prompts else 0.0
    print(f"✓ Successful generations: {success_count}/{len(prompts)}")
    print(f"✓ Average generation time: {avg_time:.4f}s")
    
    return results, avg_time, success_count

# =============================================================================
# MAIN EVALUATION
# =============================================================================
print("\n" + "="*80)
print("BASELINE FP16 EVALUATION - QWEN 2.5 0.5B")
print("="*80)

# Load datasets
print("\nLoading datasets...")
with open('wikitext2_test.json', 'r') as f:
    wikitext2_test = json.load(f)
print(f"✓ WikiText-2 test: {len(wikitext2_test)} samples")

with open('prompt_suite_96.json', 'r') as f:
    prompt_suite_dict = json.load(f)

# Convert dictionary format to list format for processing
prompt_suite = []
for category, prompts in prompt_suite_dict.items():
    for prompt_text in prompts:
        prompt_suite.append({'category': category, 'prompt': prompt_text})
        
print(f"✓ Prompt suite: {len(prompt_suite)} prompts across {len(prompt_suite_dict)} categories")

# 1. Compute perplexity
perplexity = compute_perplexity(model, tokenizer, wikitext2_test, max_samples=500)

# 2. Generate prompt responses
prompt_results, avg_gen_time, success_count = generate_prompt_responses(
    model, tokenizer, prompt_suite, record_logits=True
)

# 3. Compute category-wise statistics
category_stats = {}
for result in prompt_results:
    cat = result['category']
    if cat not in category_stats:
        category_stats[cat] = {'total': 0, 'success': 0, 'avg_time': []}
    category_stats[cat]['total'] += 1
    if result['success']:
        category_stats[cat]['success'] += 1
        category_stats[cat]['avg_time'].append(result['generation_time'])

for cat in category_stats:
    times = category_stats[cat]['avg_time']
    category_stats[cat]['avg_time'] = np.mean(times) if times else 0.0
    category_stats[cat]['success_rate'] = category_stats[cat]['success'] / category_stats[cat]['total']

# 4. Save results
results = {
    'model_name': model_name,
    'model_type': 'baseline_fp16',
    'dtype': 'torch.float16',
    'device': str(device),
    'timestamp': datetime.now().isoformat(),
    'metrics': {
        'perplexity': float(perplexity),
        'num_prompts': len(prompt_suite),
        'successful_generations': success_count,
        'success_rate': success_count / len(prompt_suite),
        'avg_generation_time': float(avg_gen_time),
        'category_stats': {k: {
            'total': v['total'],
            'success': v['success'],
            'success_rate': float(v['success_rate']),
            'avg_time': float(v['avg_time'])
        } for k, v in category_stats.items()}
    },
    'prompt_results': prompt_results
}

output_file = 'qwen_baseline_fp16_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"Model: {model_name}")
print(f"Precision: FP16")
print(f"Device: {device}")
print(f"\nRESULTS:")
print(f"  Perplexity (WikiText-2): {perplexity:.4f}")
print(f"  Successful prompts: {success_count}/{len(prompt_suite)}")
print(f"  Success rate: {100*success_count/len(prompt_suite):.1f}%")
print(f"  Avg generation time: {avg_gen_time:.4f}s")
print(f"\nResults saved to: {output_file}")
print("="*80)
