"""
INT8 Quantization for Qwen 2.5 0.5B Model
Using LLM.int8 (BitsAndBytes) - Weight-only INT8 quantization
Minimal degradation, acts as practical baseline
"""

import torch
import json
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime
import scipy.special

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    devNumber = torch.cuda.current_device()
    devName = torch.cuda.get_device_name(devNumber)
    print(f"GPU: {devName} (Device {devNumber})")

# =============================================================================
# LOAD BASELINE RESULTS FOR COMPARISON
# =============================================================================
print("\n" + "="*80)
print("LOADING BASELINE FP16 RESULTS")
print("="*80)

with open('qwen_baseline_fp16_results.json', 'r') as f:
    baseline_results = json.load(f)

baseline_ppl = baseline_results['metrics']['perplexity']
baseline_prompts = baseline_results['prompt_results']
print(f"Baseline PPL: {baseline_ppl:.4f}")
print(f"Baseline prompts: {len(baseline_prompts)}")

# =============================================================================
# LOAD CALIBRATION DATASET
# =============================================================================
print("\n" + "="*80)
print("LOADING CALIBRATION DATASET")
print("="*80)

with open('calibration_dataset.json', 'r') as f:
    calibration_data = json.load(f)
print(f"Calibration samples: {len(calibration_data)}")

# =============================================================================
# INT8 QUANTIZATION WITH BITSANDBYTES
# =============================================================================
print("\n" + "="*80)
print("APPLYING INT8 QUANTIZATION (LLM.int8)")
print("="*80)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Configure INT8 quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Outlier threshold
    llm_int8_has_fp16_weight=False
)

print("Loading model with INT8 quantization...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)
model_int8.eval()
print(f"✓ INT8 Model loaded on {model_int8.device}")

# Check model size
def get_model_size(model):
    """Estimate model size in MB"""
    total_params = sum(p.numel() for p in model.parameters())
    # INT8 = 1 byte per parameter
    size_mb = total_params / (1024 * 1024)
    return size_mb, total_params

int8_size_mb, int8_params = get_model_size(model_int8)
print(f"INT8 Model size: ~{int8_size_mb:.2f} MB ({int8_params:,} parameters)")

# =============================================================================
# PERPLEXITY COMPUTATION
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
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs.input_ids.to(device)
            
            if input_ids.size(1) < 2:
                continue
            
            outputs = model(input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss
            
            num_tokens = input_ids.size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"✓ Perplexity: {perplexity:.4f}")
    return perplexity

# =============================================================================
# KL DIVERGENCE COMPUTATION
# =============================================================================
def compute_kl_divergence(baseline_logits, quantized_logits):
    """
    Compute KL divergence between baseline and quantized model outputs
    KL(P_baseline || P_quantized)
    """
    # Convert logits to probabilities using softmax
    baseline_probs = scipy.special.softmax(baseline_logits, axis=-1)
    quantized_probs = scipy.special.softmax(quantized_logits, axis=-1)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    baseline_probs = np.clip(baseline_probs, epsilon, 1.0)
    quantized_probs = np.clip(quantized_probs, epsilon, 1.0)
    
    # KL divergence: sum(P * log(P/Q))
    kl_div = np.sum(baseline_probs * np.log(baseline_probs / quantized_probs))
    
    return kl_div

# =============================================================================
# PROMPT SUITE GENERATION WITH KL DIVERGENCE
# =============================================================================
def generate_prompt_responses_with_kl(model, tokenizer, prompts, baseline_prompts, record_logits=True):
    """Generate responses and compute KL divergence against baseline"""
    print(f"\n{'='*80}")
    print("GENERATING RESPONSES FOR 96 PROMPTS")
    print(f"{'='*80}")
    
    results = []
    total_time = 0.0
    success_count = 0
    kl_divergences = []
    
    with torch.no_grad():
        for idx, prompt_item in enumerate(tqdm(prompts, desc="Generating")):
            prompt_text = prompt_item['prompt']
            category = prompt_item['category']
            
            try:
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                
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
                
                generated_ids = outputs.sequences[0]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract logits and compute KL divergence
                logits_record = []
                token_kl_divs = []
                
                if record_logits and hasattr(outputs, 'scores'):
                    baseline_prompt_logits = baseline_prompts[idx]['logits']
                    
                    for token_idx, scores in enumerate(outputs.scores[:10]):
                        top_logits, top_indices = torch.topk(scores[0], k=min(50, scores.size(-1)))
                        logits_record.append({
                            'token_position': token_idx,
                            'top_50_logits': top_logits.cpu().tolist(),
                            'top_50_token_ids': top_indices.cpu().tolist()
                        })
                        
                        # Compute KL divergence for this token
                        if token_idx < len(baseline_prompt_logits):
                            baseline_token_logits = np.array(baseline_prompt_logits[token_idx]['top_50_logits'])
                            quantized_token_logits = top_logits.cpu().numpy()
                            
                            token_kl = compute_kl_divergence(baseline_token_logits, quantized_token_logits)
                            token_kl_divs.append(token_kl)
                
                avg_kl = np.mean(token_kl_divs) if token_kl_divs else 0.0
                kl_divergences.append(avg_kl)
                
                results.append({
                    'prompt_id': idx,
                    'category': category,
                    'prompt': prompt_text,
                    'response': generated_text,
                    'generation_time': gen_time,
                    'logits': logits_record,
                    'kl_divergence': float(avg_kl),
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
                    'kl_divergence': None,
                    'success': False,
                    'error': str(e)
                })
    
    avg_time = total_time / len(prompts) if prompts else 0.0
    avg_kl = np.mean([kl for kl in kl_divergences if kl is not None])
    
    print(f"✓ Successful generations: {success_count}/{len(prompts)}")
    print(f"✓ Average generation time: {avg_time:.4f}s")
    print(f"✓ Average KL divergence: {avg_kl:.4f}")
    
    return results, avg_time, success_count, avg_kl

# =============================================================================
# MAIN EVALUATION
# =============================================================================
print("\n" + "="*80)
print("INT8 QUANTIZED MODEL EVALUATION")
print("="*80)

# Load test dataset
with open('wikitext2_test.json', 'r') as f:
    wikitext2_test = json.load(f)

with open('prompt_suite_96.json', 'r') as f:
    prompt_suite_dict = json.load(f)

# Convert to list format (use all 96 prompts - 8 per category)
prompt_suite = []
for category, prompts in prompt_suite_dict.items():
    for prompt_text in prompts:  # All prompts per category
        prompt_suite.append({'category': category, 'prompt': prompt_text})

# 1. Compute perplexity (500 samples as in baseline)
int8_ppl = compute_perplexity(model_int8, tokenizer, wikitext2_test, max_samples=500)

# 2. Calculate perplexity degradation
ppl_degradation = ((int8_ppl - baseline_ppl) / baseline_ppl) * 100

print(f"\n{'='*80}")
print(f"PERPLEXITY COMPARISON")
print(f"{'='*80}")
print(f"Baseline FP16 PPL: {baseline_ppl:.4f}")
print(f"INT8 PPL: {int8_ppl:.4f}")
print(f"ΔPPLᵧ%: {ppl_degradation:+.2f}%")

# 3. Generate prompt responses with KL divergence
prompt_results, avg_gen_time, success_count, avg_kl = generate_prompt_responses_with_kl(
    model_int8, tokenizer, prompt_suite, baseline_prompts, record_logits=True
)

# 4. Compute category-wise statistics
category_stats = {}
for result in prompt_results:
    cat = result['category']
    if cat not in category_stats:
        category_stats[cat] = {
            'total': 0, 
            'success': 0, 
            'avg_time': [], 
            'kl_divergences': []
        }
    category_stats[cat]['total'] += 1
    if result['success']:
        category_stats[cat]['success'] += 1
        category_stats[cat]['avg_time'].append(result['generation_time'])
        if result['kl_divergence'] is not None:
            category_stats[cat]['kl_divergences'].append(result['kl_divergence'])

for cat in category_stats:
    times = category_stats[cat]['avg_time']
    kls = category_stats[cat]['kl_divergences']
    category_stats[cat]['avg_time'] = np.mean(times) if times else 0.0
    category_stats[cat]['avg_kl'] = np.mean(kls) if kls else 0.0
    category_stats[cat]['success_rate'] = category_stats[cat]['success'] / category_stats[cat]['total']

# 5. Save results
results = {
    'model_name': model_name,
    'model_type': 'int8_quantized',
    'quantization_method': 'LLM.int8 (BitsAndBytes)',
    'device': str(device),
    'timestamp': datetime.now().isoformat(),
    'model_size_mb': float(int8_size_mb),
    'baseline_comparison': {
        'baseline_ppl': float(baseline_ppl),
        'quantized_ppl': float(int8_ppl),
        'ppl_degradation_percent': float(ppl_degradation)
    },
    'metrics': {
        'perplexity': float(int8_ppl),
        'num_prompts': len(prompt_suite),
        'successful_generations': success_count,
        'success_rate': success_count / len(prompt_suite),
        'avg_generation_time': float(avg_gen_time),
        'avg_kl_divergence': float(avg_kl),
        'category_stats': {k: {
            'total': v['total'],
            'success': v['success'],
            'success_rate': float(v['success_rate']),
            'avg_time': float(v['avg_time']),
            'avg_kl': float(v['avg_kl'])
        } for k, v in category_stats.items()}
    },
    'prompt_results': prompt_results
}

output_file = 'qwen_int8_quantized_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("INT8 QUANTIZATION COMPLETE")
print("="*80)
print(f"Model: {model_name}")
print(f"Quantization: INT8 (LLM.int8)")
print(f"Device: {device}")
print(f"\nQUANTIZATION METRICS:")
print(f"  Model size: ~{int8_size_mb:.2f} MB")
print(f"  Baseline PPL: {baseline_ppl:.4f}")
print(f"  INT8 PPL: {int8_ppl:.4f}")
print(f"  ΔPPLᵧ%: {ppl_degradation:+.2f}%")
print(f"\nGENERATION METRICS:")
print(f"  Successful prompts: {success_count}/{len(prompt_suite)}")
print(f"  Success rate: {100*success_count/len(prompt_suite):.1f}%")
print(f"  Avg generation time: {avg_gen_time:.4f}s")
print(f"  Avg KL divergence: {avg_kl:.4f}")
print(f"\nResults saved to: {output_file}")
print("="*80)
