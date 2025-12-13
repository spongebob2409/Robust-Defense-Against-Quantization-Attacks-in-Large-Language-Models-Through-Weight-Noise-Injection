import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import time
from datetime import datetime

# ============================================================
# Device info
# ============================================================
print("Checking device...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================
# Load Mistral-7B INT8 (BitsAndBytes)
# ============================================================
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model with INT8 quantization (BitsAndBytes)...")

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    use_cache=True
)

# Configure for generation
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = True
model.eval()

# ============================================================
# Model size info (dynamically calculated)
# ============================================================
def get_model_size(model):
    """Calculate model size from loaded model"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate actual memory (considering quantization)
    total_memory = 0
    for param in model.parameters():
        # INT8 quantized parameters use 1 byte per element
        if hasattr(param, 'quant_state') or param.dtype == torch.int8:
            total_memory += param.numel() * 1  # 1 byte for int8
        else:
            total_memory += param.numel() * param.element_size()
    
    return {
        "total_params": total_params,
        "total_params_billions": total_params / 1e9,
        "memory_bytes": total_memory,
        "memory_gb": total_memory / (1024**3),
        "quantization": "INT8"
    }

model_info = get_model_size(model)

print("\nModel Info:")
print(f"  Parameters: {model_info['total_params_billions']:.2f}B")
print(f"  Memory: {model_info['memory_gb']:.2f} GB")
print(f"  Quantization: INT8 (BitsAndBytes)")
print(f"  Device: {model.device}")

# ============================================================
# Perplexity computation
# ============================================================
def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on dataset"""
    model.eval()
    total_loss, total_tokens = 0, 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )

            # Move to model device
            input_ids = enc.input_ids.to(model.device)
            if input_ids.shape[1] < 2:
                continue

            try:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.shape[1]
                total_tokens += input_ids.shape[1]
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                continue

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss)), float(avg_loss)

# ============================================================
# Generation with logits collection
# ============================================================
def generate_responses(model, tokenizer, prompts, max_new_tokens=100):
    """Generate responses and collect logits"""
    model.eval()
    results = []

    with torch.no_grad():
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to model device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]

                start = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True
                )
                gen_time = time.time() - start

                gen_ids = outputs.sequences[0]
                response_ids = gen_ids[input_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

                # Collect logits (top-k to save space)
                logits_list = []
                if hasattr(outputs, 'scores') and outputs.scores:
                    for score in outputs.scores[:10]:  # Store first 10 tokens
                        top_k = 100
                        top_values, top_indices = torch.topk(score[0], k=min(top_k, score.shape[-1]))
                        logits_list.append({
                            'values': top_values.cpu().numpy().tolist(),
                            'indices': top_indices.cpu().numpy().tolist()
                        })

                results.append({
                    "prompt_id": idx,
                    "prompt": prompt,
                    "response": response_text,
                    "output_length": len(response_ids),
                    "generation_time": gen_time,
                    "tokens_per_second": len(response_ids) / gen_time if gen_time > 0 else 0,
                    "logits": logits_list
                })

                if idx < 3:
                    print(f"\n--- Sample {idx+1} ---")
                    print(f"Prompt: {prompt[:100]}")
                    print(f"Response: {response_text[:200]}")

            except Exception as e:
                print(f"\nError on prompt {idx}: {str(e)}")
                results.append({
                    "prompt_id": idx,
                    "prompt": prompt,
                    "error": str(e)
                })

    return results

# ============================================================
# Load datasets
# ============================================================
print("\nLoading WikiText-2...")
with open("wikitext2_test.json", "r", encoding="utf-8") as f:
    wikitext = json.load(f)
texts = [x["text"] for x in wikitext[:1000]]
print(f"Loaded {len(texts)} WikiText-2 samples")

print("\nLoading prompt suite...")
with open("prompt_suite_96.json", "r", encoding="utf-8") as f:
    prompt_suite = json.load(f)

prompts, categories = [], []
for cat, plist in prompt_suite.items():
    for p in plist:
        prompts.append(p)
        categories.append(cat)
print(f"Loaded {len(prompts)} prompts across {len(prompt_suite)} categories")

# ============================================================
# Run evaluation
# ============================================================
print("\n" + "="*80)
print("STEP 1: Computing Perplexity on WikiText-2 (INT8 Model)")
print("="*80)

ppl_start = time.time()
ppl, loss = compute_perplexity(model, tokenizer, texts)
ppl_time = time.time() - ppl_start

print(f"\nPerplexity: {ppl:.4f}")
print(f"Avg Loss: {loss:.4f}")
print(f"Computation time: {ppl_time:.2f}s")

print("\n" + "="*80)
print("STEP 2: Generating Responses for 96 Prompts (INT8 Model)")
print("="*80)

gen_start = time.time()
gen_results = generate_responses(model, tokenizer, prompts, max_new_tokens=100)
total_gen_time = time.time() - gen_start

print(f"\nGeneration completed in {total_gen_time:.2f}s")
print(f"Average time per prompt: {total_gen_time / len(prompts):.2f}s")

# ============================================================
# Compile metrics
# ============================================================
print("\n" + "="*80)
print("STEP 3: Compiling INT8 Quantization Metrics")
print("="*80)

success = [r for r in gen_results if "response" in r and "error" not in r]
failed = [r for r in gen_results if "error" in r]

int8_metrics = {
    "model_info": {
        "model_name": model_name,
        "model_type": "Mistral-7B-Instruct-v0.2",
        "precision": "INT8",
        "quantization_method": "LLM.int8 (BitsAndBytes)",
        "parameters": model_info,
        "device": str(model.device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    },
    "quantization_config": {
        "method": "BitsAndBytes INT8",
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "weight_only": True
    },
    "evaluation_info": {
        "timestamp": datetime.now().isoformat(),
        "wikitext_samples": len(texts),
        "num_prompts": len(prompts),
        "max_new_tokens": 100,
        "temperature": 0.7
    },
    "perplexity_metrics": {
        "perplexity": ppl,
        "average_loss": loss,
        "computation_time_seconds": ppl_time,
        "dataset": "WikiText-2",
        "num_samples": len(texts)
    },
    "generation_metrics": {
        "total_prompts": len(prompts),
        "successful_generations": len(success),
        "failed_generations": len(failed),
        "total_time_seconds": total_gen_time,
        "average_time_per_prompt": total_gen_time / len(prompts),
        "average_tokens_per_second": np.mean([r["tokens_per_second"] for r in success])
    },
    "prompt_responses": []
}

# Add category information
for i, result in enumerate(gen_results):
    result['category'] = categories[i]
    int8_metrics['prompt_responses'].append(result)

# Category-wise statistics
category_stats = {}
for category in prompt_suite.keys():
    cat_results = [r for r in gen_results if r.get('category') == category and 'error' not in r]
    category_stats[category] = {
        "num_prompts": len([r for r in gen_results if r.get('category') == category]),
        "avg_response_length": np.mean([r["output_length"] for r in cat_results]) if cat_results else 0,
        "avg_generation_time": np.mean([r["generation_time"] for r in cat_results]) if cat_results else 0,
        "success_rate": len(cat_results) / len([r for r in gen_results if r.get('category') == category]) if cat_results else 0
    }

int8_metrics['category_statistics'] = category_stats

# Save results
output_file = "int8_quantized_mistral7b_results.json"
print(f"\nSaving results to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(int8_metrics, f, indent=2, ensure_ascii=False)

print(f"Results saved successfully!")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("INT8 QUANTIZATION EVALUATION SUMMARY")
print("="*80)
print(f"\nModel: {model_name}")
print(f"Parameters: {model_info['total_params_billions']:.2f}B ({model_info['total_params']:,})")
print(f"Quantization: INT8 (BitsAndBytes)")
print(f"Memory Usage: {model_info['memory_gb']:.2f} GB")
print(f"\nPerplexity (WikiText-2): {ppl:.4f}")
print(f"Total Prompts: {len(prompts)}")
print(f"Successful Generations: {len(success)}")
print(f"Failed Generations: {len(failed)}")
print(f"Average Generation Time: {int8_metrics['generation_metrics']['average_time_per_prompt']:.2f}s")
print(f"Average Tokens/Second: {int8_metrics['generation_metrics']['average_tokens_per_second']:.2f}")

print("\nCategory Statistics:")
for category, stats in category_stats.items():
    print(f"  {category}:")
    print(f"    Prompts: {stats['num_prompts']}")
    print(f"    Avg Response Length: {stats['avg_response_length']:.1f} tokens")
    print(f"    Success Rate: {stats['success_rate']*100:.1f}%")

print(f"\n✅ Mistral-7B INT8 quantization evaluation completed!")
print(f"Results saved to: {output_file}")
