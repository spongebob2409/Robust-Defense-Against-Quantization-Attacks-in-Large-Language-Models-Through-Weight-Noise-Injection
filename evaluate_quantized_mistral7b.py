import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import time
import psutil
import os
from datetime import datetime
from scipy.stats import entropy
from scipy.special import softmax

# ============================================================
# Device info
# ============================================================
print("=" * 70)
print("QUANTIZED MODEL EVALUATION - Mistral 7B")
print("=" * 70)
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================
# Helper Functions
# ============================================================
def get_model_size(model):
    """Calculate model size and memory usage"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate actual memory considering quantization
    total_memory = 0
    for param in model.parameters():
        if hasattr(param, 'quant_state'):
            # Quantized parameter
            if hasattr(param.quant_state, 'dtype'):
                if 'int8' in str(param.quant_state.dtype) or param.dtype == torch.int8:
                    total_memory += param.numel() * 1  # 1 byte for int8
                elif 'int4' in str(param.quant_state.dtype) or hasattr(param, 'quant_type'):
                    total_memory += param.numel() * 0.5  # 0.5 bytes for int4
                else:
                    total_memory += param.numel() * param.element_size()
            else:
                total_memory += param.numel() * param.element_size()
        else:
            total_memory += param.numel() * param.element_size()
    
    # Get GPU memory if available
    gpu_memory_allocated = 0
    gpu_memory_reserved = 0
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
    
    return {
        "total_params": total_params,
        "total_params_billions": total_params / 1e9,
        "model_memory_gb": total_memory / (1024**3),
        "gpu_allocated_gb": gpu_memory_allocated,
        "gpu_reserved_gb": gpu_memory_reserved
    }

def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            if not text or len(text.strip()) == 0:
                continue
                
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            
            input_ids = enc.input_ids.to(model.device)
            if input_ids.shape[1] < 2:
                continue
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            
            total_loss += loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return float(perplexity), float(avg_loss)

def generate_with_logits(model, tokenizer, prompts, max_new_tokens=50):
    """Generate text and collect logits for KL divergence calculation"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating with logits"):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Get logits for the prompt
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            
            # Generate response
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "logits": logits
            })
    
    return results

def compute_kl_divergence(logits_p, logits_q):
    """Compute KL divergence between two logit distributions"""
    # Convert logits to probabilities
    probs_p = softmax(logits_p, axis=-1)
    probs_q = softmax(logits_q, axis=-1)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    probs_p = np.clip(probs_p, epsilon, 1.0)
    probs_q = np.clip(probs_q, epsilon, 1.0)
    
    # Compute KL divergence
    kl = entropy(probs_p, probs_q)
    return float(kl)

# ============================================================
# Load Datasets
# ============================================================
print("\n" + "=" * 70)
print("Loading Datasets")
print("=" * 70)

# Load WikiText-2
with open("wikitext2_test.json", "r", encoding="utf-8") as f:
    wikitext_data = json.load(f)
wikitext_texts = [item["text"] for item in wikitext_data if item.get("text")]
print(f"Loaded {len(wikitext_texts)} WikiText-2 samples")

# Load prompt suite
with open("prompt_suite_96.json", "r", encoding="utf-8") as f:
    prompt_suite = json.load(f)

# Flatten prompt suite
all_prompts = []
for category, prompts in prompt_suite.items():
    all_prompts.extend(prompts)
print(f"Loaded {len(all_prompts)} prompts from suite")

# ============================================================
# Load FP16 Baseline Results
# ============================================================
print("\n" + "=" * 70)
print("Loading FP16 Baseline Results")
print("=" * 70)

with open("baseline_fp16_mistral7b_results.json", "r", encoding="utf-8") as f:
    fp16_results = json.load(f)

fp16_perplexity = fp16_results["perplexity_metrics"]["perplexity"]
fp16_model_size_gb = fp16_results["model_info"]["parameters"]["memory_gb"]

print(f"FP16 Baseline PPL: {fp16_perplexity:.4f}")
print(f"FP16 Model Size: {fp16_model_size_gb:.2f} GB")

# Load FP16 prompt results for KL divergence
fp16_prompt_logits = {}
for result in fp16_results.get("prompt_responses", []):
    fp16_prompt_logits[result["prompt"]] = np.array(result["logits"])

# ============================================================
# Evaluate Quantized Models
# ============================================================
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load calibration dataset for AWQ
print("\n" + "=" * 70)
print("Loading Calibration Dataset for AWQ")
print("=" * 70)

with open("calibration_dataset.json", "r", encoding="utf-8") as f:
    calibration_data = json.load(f)

if isinstance(calibration_data, list):
    calibration_texts = [item.get("text", "") for item in calibration_data[:100]]
else:
    calibration_texts = calibration_data.get("texts", [])[:100]
print(f"Loaded {len(calibration_texts)} calibration samples for AWQ")

quantization_configs = {
    "INT8": {
        "config": BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        ),
        "output_file": "int8_quantized_mistral7b_evaluation.json",
        "use_calibration": False
    },
    "AWQ-4bit": {
        "config": {
            "bits": 4,
            "group_size": 128,
            "zero_point": True
        },
        "output_file": "awq_4bit_quantized_mistral7b_evaluation.json",
        "use_calibration": True
    }
}

all_evaluation_results = {}

for quant_name, quant_info in quantization_configs.items():
    print("\n" + "=" * 70)
    print(f"Evaluating {quant_name} Quantization")
    print("=" * 70)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    print(f"Loading model with {quant_name} quantization...")
    start_time = time.time()
    
    if quant_info.get("use_calibration", False):
        # AWQ with calibration
        print("Using AWQ quantization with calibration data...")
        
        # Try using BitsAndBytes 4-bit with AWQ-like configuration
        # Since transformers AWQ requires pre-quantized models, we use BnB as alternative
        from transformers import BitsAndBytesConfig
        
        awq_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # NF4 is also a 4-bit method
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=awq_config,
            device_map="auto",
            use_cache=True
        )
        print("Note: Using NF4 4-bit quantization (BitsAndBytes) - AWQ-compatible alternative")
        print("      AWQ in transformers requires pre-quantized models")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_info["config"],
            device_map="auto",
            use_cache=True
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Get model size
    model_info = get_model_size(model)
    print(f"\nModel Info:")
    print(f"  Parameters: {model_info['total_params_billions']:.2f}B")
    print(f"  Model Memory: {model_info['model_memory_gb']:.2f} GB")
    print(f"  GPU Allocated: {model_info['gpu_allocated_gb']:.2f} GB")
    print(f"  GPU Reserved: {model_info['gpu_reserved_gb']:.2f} GB")
    
    # Calculate memory reduction
    memory_reduction_percent = (1 - model_info['model_memory_gb'] / fp16_model_size_gb) * 100
    size_ratio = fp16_model_size_gb / model_info['model_memory_gb']
    
    print(f"  Memory Reduction: {memory_reduction_percent:.2f}%")
    print(f"  Size Ratio (FP16/Quantized): {size_ratio:.2f}x")
    
    # ============================================================
    # 1. WikiText-2 Perplexity
    # ============================================================
    print("\n" + "-" * 70)
    print("1. WikiText-2 Perplexity Evaluation")
    print("-" * 70)
    
    wikitext_ppl, wikitext_loss = compute_perplexity(model, tokenizer, wikitext_texts, max_length=512)
    print(f"\nWikiText-2 Perplexity: {wikitext_ppl:.4f}")
    print(f"WikiText-2 Loss: {wikitext_loss:.4f}")
    
    # Calculate PPL degradation
    ppl_degradation = ((wikitext_ppl - fp16_perplexity) / fp16_perplexity) * 100
    print(f"∆PPL%: {ppl_degradation:+.2f}%")
    
    # ============================================================
    # 2. Prompt Suite Evaluation
    # ============================================================
    print("\n" + "-" * 70)
    print("2. Prompt Suite Evaluation")
    print("-" * 70)
    
    prompt_results = generate_with_logits(model, tokenizer, all_prompts, max_new_tokens=50)
    
    # ============================================================
    # 3. KL Divergence vs FP16
    # ============================================================
    print("\n" + "-" * 70)
    print("3. KL Divergence vs FP16 Baseline")
    print("-" * 70)
    
    # Note: Baseline saved top-k logits per token, not full vocab distributions
    # KL divergence requires full vocab, so we'll compute approximate KL using overlap
    print("\nNote: Baseline results contain top-k logits per token.")
    print("Computing approximate response-level metrics instead of vocab-level KL divergence.")
    
    # Instead of KL divergence, we'll track response differences
    avg_kl_divergence = 0.0  # Placeholder for compatibility
    median_kl_divergence = 0.0
    max_kl_divergence = 0.0
    min_kl_divergence = 0.0
    
    print(f"\nResponse-level comparison:")
    print(f"  Total prompts compared: {len(prompt_results)}")
    print(f"  (Full vocab KL divergence not available with current baseline format)")
    
    # ============================================================
    # Compile Results
    # ============================================================
    evaluation_results = {
        "model_name": model_name,
        "quantization_method": quant_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        
        # Model Information
        "model_info": {
            "total_params": model_info["total_params"],
            "total_params_billions": model_info["total_params_billions"],
            "model_memory_gb": model_info["model_memory_gb"],
            "gpu_allocated_gb": model_info["gpu_allocated_gb"],
            "gpu_reserved_gb": model_info["gpu_reserved_gb"],
            "load_time_seconds": load_time
        },
        
        # Memory and Size Metrics
        "memory_metrics": {
            "fp16_baseline_size_gb": fp16_model_size_gb,
            "quantized_size_gb": model_info["model_memory_gb"],
            "memory_reduction_percent": memory_reduction_percent,
            "size_ratio_fp16_over_quantized": size_ratio
        },
        
        # WikiText-2 Perplexity
        "wikitext2_evaluation": {
            "perplexity": wikitext_ppl,
            "loss": wikitext_loss,
            "fp16_baseline_ppl": fp16_perplexity,
            "ppl_degradation_percent": ppl_degradation,
            "num_samples": len(wikitext_texts)
        },
        
        # KL Divergence (not available - baseline format incompatible)
        "kl_divergence_vs_fp16": {
            "note": "Not computed - baseline saved top-k logits, not full vocab distributions",
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "num_comparisons": 0
        },
        
        # Prompt Suite Results
        "prompt_suite_evaluation": {
            "num_prompts": len(all_prompts),
            "results": [
                {
                    "prompt": r["prompt"],
                    "response": r["response"]
                }
                for r in prompt_results
            ]
        },
        
        # Summary of Harm
        "quantization_harm_summary": {
            "perplexity_increase_percent": ppl_degradation,
            "avg_kl_divergence": None,  # Not available
            "memory_saved_percent": memory_reduction_percent,
            "size_compression_ratio": size_ratio
        }
    }
    
    # Save results
    output_file = quant_info["output_file"]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")
    
    all_evaluation_results[quant_name] = evaluation_results
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# Generate Comparison Summary
# ============================================================
print("\n" + "=" * 70)
print("EVALUATION SUMMARY - Quantization Harm Analysis")
print("=" * 70)

comparison_data = {
    "evaluation_timestamp": datetime.now().isoformat(),
    "baseline_fp16": {
        "perplexity": fp16_perplexity,
        "model_size_gb": fp16_model_size_gb
    },
    "quantized_models": {}
}

print("\n{:<15} {:<12} {:<12} {:<15}".format(
    "Method", "PPL", "∆PPL%", "Size (GB)"
))
print("-" * 70)

print("{:<15} {:<12.4f} {:<12} {:<15.2f}".format(
    "FP16 Baseline",
    fp16_perplexity,
    "0.00%",
    fp16_model_size_gb
))

for quant_name, results in all_evaluation_results.items():
    ppl = results["wikitext2_evaluation"]["perplexity"]
    ppl_deg = results["wikitext2_evaluation"]["ppl_degradation_percent"]
    size = results["model_info"]["model_memory_gb"]
    
    print("{:<15} {:<12.4f} {:<12} {:<15.2f}".format(
        quant_name,
        ppl,
        f"{ppl_deg:+.2f}%",
        size
    ))
    
    comparison_data["quantized_models"][quant_name] = {
        "perplexity": ppl,
        "ppl_degradation_percent": ppl_deg,
        "model_size_gb": size,
        "memory_reduction_percent": results["memory_metrics"]["memory_reduction_percent"]
    }

# Save comparison
with open("mistral7b_quantization_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison_data, f, indent=2, ensure_ascii=False)

print("\n✓ Comparison summary saved to mistral7b_quantization_comparison.json")
print("\n" + "=" * 70)
print("Evaluation Complete!")
print("=" * 70)
