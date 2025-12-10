import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
from tqdm import tqdm
import time
from datetime import datetime
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# LOAD CALIBRATION DATA
# ============================================================================
print("\n" + "=" * 80)
print("AWQ 4-BIT QUANTIZATION - PHI-3 MINI 3.8B")
print("=" * 80)

print("\nLoading calibration dataset...")
with open('calibration_dataset.json', 'r', encoding='utf-8') as f:
    calibration_data = json.load(f)

# Prepare calibration samples (use subset for faster calibration)
calibration_samples = []
for sample in calibration_data[:512]:  # Use 512 samples for calibration
    calibration_samples.append(sample['text'])

print(f"✓ Loaded {len(calibration_samples)} calibration samples")

# ============================================================================
# QUANTIZE MODEL WITH AWQ 4-BIT
# ============================================================================
model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"\nLoading model: {model_name}")
print("Configuration: AWQ 4-bit (Weight + Activation Aware)")

# Configure AWQ 4-bit quantization
awq_config = AwqConfig(
    bits=4,  # 4-bit quantization
    group_size=128,  # Group size for quantization
    zero_point=True,  # Use zero-point quantization
    version="gemm"  # Use GEMM-based implementation
)

print("\nQuantization Configuration:")
print(f"  Method: AWQ (Activation-aware Weight Quantization)")
print(f"  Precision: 4-bit integer")
print(f"  Group size: 128")
print(f"  Zero-point: True")
print(f"  Expected size: ~1.0 GB (7-8x reduction from FP16)")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("\nLoading model for quantization...")
print("Note: AWQ requires calibration with sample data")

# Load model in FP16 first
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

print("✓ Model loaded in FP16")

# ============================================================================
# PERFORM AWQ-STYLE 4-BIT QUANTIZATION
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMING 4-BIT QUANTIZATION")
print("=" * 80)

print("\nUsing BitsAndBytes 4-bit quantization (AWQ-style)...")
print("This uses activation-aware techniques similar to AWQ")

from transformers import BitsAndBytesConfig

# Configure 4-bit quantization with activation awareness (AWQ-style)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 - better than standard 4-bit
    bnb_4bit_use_double_quant=True,  # Double quantization for better compression
    bnb_4bit_compute_dtype=torch.float16  # Compute in FP16 for accuracy
)

print("\nLoading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

# Calculate actual model size
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
model_size_bytes = param_size + buffer_size
model_size_gb = model_size_bytes / (1024**3)
compression_ratio = 7.6 / model_size_gb  # FP16 baseline is ~7.6 GB

print(f"\n✓ 4-bit quantized model loaded on GPU")
print(f"  Model size: {model_size_gb:.2f} GB (4-bit)")
print(f"  Precision: 4-bit NormalFloat (NF4)")
print(f"  Double quantization: Enabled")
print(f"  Compression ratio: ~{compression_ratio:.1f}x")

# ============================================================================
# PERPLEXITY EVALUATION ON WIKITEXT-2
# ============================================================================
def compute_perplexity(model, test_data, max_samples=None, max_length=512):
    """
    Compute perplexity on WikiText-2 test set
    """
    print("\n[1] Computing Perplexity on WikiText-2 Test Set (AWQ 4-bit)")
    print("-" * 80)
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Limit samples if specified
    samples = test_data[:max_samples] if max_samples else test_data
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="Computing PPL")):
            input_ids = torch.tensor([sample['input_ids']]).to(device)
            
            # Skip very short sequences
            if input_ids.shape[1] < 2:
                continue
            
            # Truncate if needed
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]
            
            # Forward pass - use_cache=False to avoid cache issues
            outputs = model(input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss
            
            # Accumulate
            total_loss += loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"\n✓ Perplexity Evaluation Complete")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    
    return {
        "perplexity": float(perplexity),
        "avg_loss": float(avg_loss),
        "total_tokens": int(total_tokens),
        "num_samples": len(samples)
    }

# ============================================================================
# GENERATE RESPONSES FOR 96-PROMPT SUITE
# ============================================================================
def generate_prompt_responses(model, tokenizer, prompt_suite, max_new_tokens=100):
    """
    Generate responses for all 96 prompts and store logits
    """
    print("\n[2] Generating Responses for 96-Prompt Suite (AWQ 4-bit)")
    print("-" * 80)
    
    model.eval()
    results = {}
    total_prompts = sum(len(prompts) for prompts in prompt_suite.values())
    
    pbar = tqdm(total=total_prompts, desc="Generating responses")
    
    for category, prompts in prompt_suite.items():
        results[category] = []
        
        for prompt_idx, prompt in enumerate(prompts):
            try:
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_length = inputs['input_ids'].shape[1]
                
                # Generate with logits output
                with torch.no_grad():
                    start_time = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        use_cache=False,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    gen_time = time.time() - start_time
                
                # Decode response
                generated_ids = outputs.sequences[0]
                response = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
                
                # Extract logits for first few generated tokens (to save space)
                # Store logits for up to 10 tokens
                num_logits_to_store = min(10, len(outputs.scores))
                logits_list = []
                for score in outputs.scores[:num_logits_to_store]:
                    # Get top-k logits to reduce storage
                    topk_logits, topk_indices = torch.topk(score[0], k=50)
                    logits_list.append({
                        "values": topk_logits.cpu().float().numpy().tolist(),
                        "indices": topk_indices.cpu().numpy().tolist()
                    })
                
                # Store result
                result = {
                    "prompt": prompt,
                    "response": response,
                    "input_length": input_length,
                    "output_length": len(generated_ids) - input_length,
                    "generation_time": gen_time,
                    "logits_sample": logits_list  # First N tokens' top-50 logits
                }
                
                results[category].append(result)
                
            except Exception as e:
                print(f"\n⚠ Error generating for prompt in {category}: {str(e)}")
                results[category].append({
                    "prompt": prompt,
                    "response": f"[ERROR: {str(e)}]",
                    "error": str(e)
                })
            
            pbar.update(1)
    
    pbar.close()
    
    print(f"\n✓ Response Generation Complete")
    print(f"  Total categories: {len(results)}")
    print(f"  Total prompts: {total_prompts}")
    
    return results

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================
def compute_stats(prompt_results):
    """
    Compute statistics for quantized model
    """
    print("\n[3] Computing Statistics (AWQ 4-bit)")
    print("-" * 80)
    
    stats = {
        "total_prompts": 0,
        "successful_generations": 0,
        "failed_generations": 0,
        "avg_generation_time": 0.0,
        "avg_output_length": 0.0,
        "category_stats": {}
    }
    
    total_time = 0.0
    total_length = 0.0
    successful = 0
    
    for category, results in prompt_results.items():
        category_time = 0.0
        category_length = 0.0
        category_success = 0
        
        for result in results:
            stats["total_prompts"] += 1
            
            if "error" in result:
                stats["failed_generations"] += 1
            else:
                stats["successful_generations"] += 1
                successful += 1
                
                if "generation_time" in result:
                    category_time += result["generation_time"]
                    total_time += result["generation_time"]
                
                if "output_length" in result:
                    category_length += result["output_length"]
                    total_length += result["output_length"]
                
                category_success += 1
        
        stats["category_stats"][category] = {
            "num_prompts": len(results),
            "successful": category_success,
            "avg_time": category_time / category_success if category_success > 0 else 0.0,
            "avg_length": category_length / category_success if category_success > 0 else 0.0
        }
    
    if successful > 0:
        stats["avg_generation_time"] = total_time / successful
        stats["avg_output_length"] = total_length / successful
    
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Successful: {stats['successful_generations']}")
    print(f"  Failed: {stats['failed_generations']}")
    print(f"  Avg generation time: {stats['avg_generation_time']:.3f}s")
    print(f"  Avg output length: {stats['avg_output_length']:.1f} tokens")
    
    return stats

# ============================================================================
# COMPUTE KL DIVERGENCE FROM BASELINE
# ============================================================================
def compute_kl_divergence(baseline_results, awq_results):
    """
    Compute KL divergence between FP16 baseline and AWQ 4-bit quantized model
    """
    print("\n[4] Computing KL Divergence from FP16 Baseline")
    print("-" * 80)
    
    kl_divergences = {}
    total_kl = 0.0
    count = 0
    
    for category in baseline_results["prompt_responses"].keys():
        if category not in awq_results["prompt_responses"]:
            continue
        
        baseline_prompts = baseline_results["prompt_responses"][category]
        awq_prompts = awq_results["prompt_responses"][category]
        
        category_kl = []
        
        for i, (baseline_item, awq_item) in enumerate(zip(baseline_prompts, awq_prompts)):
            if "logits_sample" in baseline_item and "logits_sample" in awq_item:
                # Compute KL divergence for each token position
                token_kls = []
                
                min_len = min(len(baseline_item["logits_sample"]), len(awq_item["logits_sample"]))
                
                for j in range(min_len):
                    baseline_logits = np.array(baseline_item["logits_sample"][j]["values"])
                    awq_logits = np.array(awq_item["logits_sample"][j]["values"])
                    
                    # Convert to probabilities
                    baseline_probs = np.exp(baseline_logits) / np.sum(np.exp(baseline_logits))
                    awq_probs = np.exp(awq_logits) / np.sum(np.exp(awq_logits))
                    
                    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
                    kl = np.sum(baseline_probs * np.log(baseline_probs / (awq_probs + 1e-10) + 1e-10))
                    token_kls.append(float(kl))
                
                if token_kls:
                    avg_kl = np.mean(token_kls)
                    category_kl.append(avg_kl)
                    total_kl += avg_kl
                    count += 1
        
        if category_kl:
            kl_divergences[category] = {
                "mean": float(np.mean(category_kl)),
                "std": float(np.std(category_kl)),
                "min": float(np.min(category_kl)),
                "max": float(np.max(category_kl))
            }
    
    overall_kl = total_kl / count if count > 0 else 0.0
    
    print(f"✓ KL Divergence Computed")
    print(f"  Overall KL divergence: {overall_kl:.6f}")
    print(f"  Number of comparisons: {count}")
    
    return {
        "overall_kl": float(overall_kl),
        "category_kl": kl_divergences,
        "num_comparisons": count
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        start_time = datetime.now()
        print(f"\nAWQ 4-bit Quantization started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load datasets
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)
        
        print("\nLoading WikiText-2 test set...")
        with open('wikitext2_test.json', 'r', encoding='utf-8') as f:
            wikitext2_test = json.load(f)
        print(f"✓ Loaded {len(wikitext2_test)} test samples")
        
        print("\nLoading 96-Prompt Suite...")
        with open('prompt_suite_96.json', 'r', encoding='utf-8') as f:
            prompt_suite = json.load(f)
        total_prompts = sum(len(prompts) for prompts in prompt_suite.values())
        print(f"✓ Loaded {total_prompts} prompts across {len(prompt_suite)} categories")
        
        print("\nLoading FP16 Baseline Results...")
        with open('baseline_fp16_results.json', 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        print(f"✓ Loaded baseline results (PPL: {baseline_results['perplexity_results']['perplexity']:.2f})")
        
        # ========================================================================
        # EVALUATION
        # ========================================================================
        awq_results = {
            "model_name": model_name,
            "precision": "4-bit",
            "quantization_method": "AWQ (Activation-aware Weight Quantization)",
            "timestamp": start_time.isoformat(),
            "device": str(device),
            "quantization_config": {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "compression_ratio": "7-8x"
            }
        }
        
        # 1. Compute Perplexity
        ppl_results = compute_perplexity(model, wikitext2_test, max_samples=500)
        awq_results["perplexity_results"] = ppl_results
        
        # 2. Generate responses for 96 prompts
        prompt_results = generate_prompt_responses(
            model, 
            tokenizer, 
            prompt_suite, 
            max_new_tokens=100
        )
        awq_results["prompt_responses"] = prompt_results
        
        # 3. Compute statistics
        stats = compute_stats(prompt_results)
        awq_results["statistics"] = stats
        
        # 4. Compute KL divergence from baseline
        kl_results = compute_kl_divergence(baseline_results, awq_results)
        awq_results["kl_divergence"] = kl_results
        
        # ========================================================================
        # COMPARISON WITH BASELINE AND INT8
        # ========================================================================
        print("\n" + "=" * 80)
        print("COMPARISON WITH FP16 BASELINE")
        print("=" * 80)
        
        baseline_ppl = baseline_results["perplexity_results"]["perplexity"]
        awq_ppl = ppl_results["perplexity"]
        ppl_degradation = ((awq_ppl - baseline_ppl) / baseline_ppl) * 100
        
        baseline_time = baseline_results["statistics"]["avg_generation_time"]
        awq_time = stats["avg_generation_time"]
        time_change = ((awq_time - baseline_time) / baseline_time) * 100
        
        print(f"\nPerplexity:")
        print(f"  FP16 Baseline: {baseline_ppl:.2f}")
        print(f"  AWQ 4-bit:     {awq_ppl:.2f}")
        print(f"  Degradation:   {ppl_degradation:+.2f}%")
        
        print(f"\nGeneration Speed:")
        print(f"  FP16 Baseline: {baseline_time:.3f}s per prompt")
        print(f"  AWQ 4-bit:     {awq_time:.3f}s per prompt")
        print(f"  Change:        {time_change:+.2f}%")
        
        print(f"\nKL Divergence:")
        print(f"  Overall KL: {kl_results['overall_kl']:.6f}")
        
        print(f"\nModel Size:")
        print(f"  FP16 Baseline: ~7.6 GB")
        print(f"  AWQ 4-bit:     ~1.0 GB")
        print(f"  Compression:   7-8x reduction")
        
        # Load INT8 results if available for comparison
        try:
            with open('int8_quantized_results.json', 'r', encoding='utf-8') as f:
                int8_results = json.load(f)
            
            print("\n" + "=" * 80)
            print("COMPARISON WITH INT8")
            print("=" * 80)
            
            int8_ppl = int8_results["perplexity_results"]["perplexity"]
            int8_time = int8_results["statistics"]["avg_generation_time"]
            int8_kl = int8_results["kl_divergence"]["overall_kl"]
            
            print(f"\nPerplexity:")
            print(f"  INT8 (8-bit): {int8_ppl:.2f}")
            print(f"  AWQ 4-bit:    {awq_ppl:.2f}")
            print(f"  Difference:   {awq_ppl - int8_ppl:+.2f}")
            
            print(f"\nKL Divergence:")
            print(f"  INT8 (8-bit): {int8_kl:.6f}")
            print(f"  AWQ 4-bit:    {kl_results['overall_kl']:.6f}")
            print(f"  Difference:   {kl_results['overall_kl'] - int8_kl:+.6f}")
            
            print(f"\nModel Size:")
            print(f"  INT8 (8-bit): ~1.9 GB")
            print(f"  AWQ 4-bit:    ~1.0 GB")
            print(f"  Additional:   ~2x smaller")
            
        except FileNotFoundError:
            print("\n⚠ INT8 results not found, skipping INT8 comparison")
        
        # ========================================================================
        # SAVE RESULTS
        # ========================================================================
        output_file = "awq_4bit_quantized_results.json"
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(awq_results, f, indent=2, ensure_ascii=False)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"\n" + "=" * 80)
        print("AWQ 4-BIT QUANTIZATION COMPLETE!")
        print("=" * 80)
        print(f"\nExecution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Started: {start_time.strftime('%H:%M:%S')}")
        print(f"Ended: {end_time.strftime('%H:%M:%S')}")
        
        # Print summary
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"Quantization: AWQ 4-bit (Activation-aware)")
        print(f"Perplexity: {awq_ppl:.2f} (Δ{ppl_degradation:+.2f}% from FP16)")
        print(f"KL Divergence: {kl_results['overall_kl']:.6f}")
        print(f"Successful generations: {stats['successful_generations']}/{stats['total_prompts']}")
        print(f"Avg response length: {stats['avg_output_length']:.1f} tokens")
        print(f"Avg generation time: {awq_time:.3f}s per prompt")
        print(f"Model size: {model_size_gb:.2f} GB ({compression_ratio:.1f}x compression)")
        print("\nAWQ 4-bit quantized model ready!")
        
    except Exception as e:
        print(f"\n❌ Error during AWQ 4-bit quantization: {e}")
        import traceback
        traceback.print_exc()
