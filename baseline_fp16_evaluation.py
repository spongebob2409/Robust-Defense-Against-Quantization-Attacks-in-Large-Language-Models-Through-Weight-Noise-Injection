import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from datetime import datetime

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# LOAD MODEL AND TOKENIZER (FP16 Baseline)
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE FP16 EVALUATION - PHI-3 MINI 3.8B")
print("=" * 80)

model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"\nLoading model: {model_name}")
print("Configuration: FP16 (Float16) - Baseline")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation='eager'
)

print(f"✓ Model loaded on {device}")
print(f"  Model size: ~7.6 GB (FP16)")
print(f"  Precision: float16")

# ============================================================================
# PERPLEXITY EVALUATION ON WIKITEXT-2
# ============================================================================
def compute_perplexity(model, test_data, max_samples=None, max_length=512):
    """
    Compute perplexity on WikiText-2 test set
    """
    print("\n[1] Computing Perplexity on WikiText-2 Test Set")
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
    print("\n[2] Generating Responses for 96-Prompt Suite")
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
# COMPUTE BASELINE STATISTICS
# ============================================================================
def compute_baseline_stats(prompt_results):
    """
    Compute statistics for baseline model
    """
    print("\n[3] Computing Baseline Statistics")
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
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        start_time = datetime.now()
        print(f"\nEvaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        
        # ========================================================================
        # EVALUATION
        # ========================================================================
        baseline_results = {
            "model_name": model_name,
            "precision": "float16",
            "timestamp": start_time.isoformat(),
            "device": str(device)
        }
        
        # 1. Compute Perplexity
        ppl_results = compute_perplexity(model, wikitext2_test, max_samples=500)
        baseline_results["perplexity_results"] = ppl_results
        
        # 2. Generate responses for 96 prompts
        prompt_results = generate_prompt_responses(
            model, 
            tokenizer, 
            prompt_suite, 
            max_new_tokens=100
        )
        baseline_results["prompt_responses"] = prompt_results
        
        # 3. Compute statistics
        stats = compute_baseline_stats(prompt_results)
        baseline_results["statistics"] = stats
        
        # ========================================================================
        # SAVE RESULTS
        # ========================================================================
        output_file = "baseline_fp16_results.json"
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=2, ensure_ascii=False)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"\n" + "=" * 80)
        print("BASELINE FP16 EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"\nExecution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Started: {start_time.strftime('%H:%M:%S')}")
        print(f"Ended: {end_time.strftime('%H:%M:%S')}")
        
        # Print summary
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"Perplexity: {ppl_results['perplexity']:.2f}")
        print(f"Successful generations: {stats['successful_generations']}/{stats['total_prompts']}")
        print(f"Avg response length: {stats['avg_output_length']:.1f} tokens")
        print(f"Avg generation time: {stats['avg_generation_time']:.3f}s per prompt")
        print("\nBaseline metrics ready for quantization comparison!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
