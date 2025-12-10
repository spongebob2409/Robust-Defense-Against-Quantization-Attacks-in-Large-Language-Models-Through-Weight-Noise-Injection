import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# COMPREHENSIVE QUANTIZATION EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE QUANTIZED MODEL EVALUATION")
print("=" * 80)
print("\nComparing FP16 Baseline vs INT8 vs AWQ 4-bit")
print("Metrics: Perplexity, KL Divergence, Memory, Quality Degradation")

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("LOADING EVALUATION RESULTS")
print("=" * 80)

# Load FP16 baseline results
print("\nLoading FP16 Baseline results...")
try:
    with open('baseline_fp16_results.json', 'r', encoding='utf-8') as f:
        fp16_results = json.load(f)
    print(f"✓ FP16 Baseline loaded")
    print(f"  PPL: {fp16_results['perplexity_results']['perplexity']:.2f}")
    print(f"  Prompts: {fp16_results['statistics']['successful_generations']}/{fp16_results['statistics']['total_prompts']}")
except FileNotFoundError:
    print("❌ FP16 baseline results not found!")
    exit(1)

# Load INT8 results
print("\nLoading INT8 (8-bit) results...")
try:
    with open('int8_quantized_results.json', 'r', encoding='utf-8') as f:
        int8_results = json.load(f)
    print(f"✓ INT8 loaded")
    print(f"  PPL: {int8_results['perplexity_results']['perplexity']:.2f}")
    print(f"  KL Divergence: {int8_results['kl_divergence']['overall_kl']:.6f}")
    int8_available = True
except FileNotFoundError:
    print("⚠ INT8 results not found - will skip INT8 comparison")
    int8_available = False

# Load AWQ 4-bit results
print("\nLoading AWQ 4-bit results...")
try:
    with open('awq_4bit_quantized_results.json', 'r', encoding='utf-8') as f:
        awq_results = json.load(f)
    print(f"✓ AWQ 4-bit loaded")
    print(f"  PPL: {awq_results['perplexity_results']['perplexity']:.2f}")
    print(f"  KL Divergence: {awq_results['kl_divergence']['overall_kl']:.6f}")
    awq_available = True
except FileNotFoundError:
    print("⚠ AWQ 4-bit results not found - will skip AWQ comparison")
    awq_available = False

# ============================================================================
# PERPLEXITY COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("PERPLEXITY ANALYSIS")
print("=" * 80)

fp16_ppl = fp16_results['perplexity_results']['perplexity']
fp16_loss = fp16_results['perplexity_results']['avg_loss']

print(f"\n{'Model':<20} {'PPL':>8} {'Loss':>8} {'ΔPPL%':>10} {'Quality':>12}")
print("-" * 70)
print(f"{'FP16 Baseline':<20} {fp16_ppl:>8.2f} {fp16_loss:>8.4f} {0:>9.2f}% {'Reference':<12}")

if int8_available:
    int8_ppl = int8_results['perplexity_results']['perplexity']
    int8_loss = int8_results['perplexity_results']['avg_loss']
    int8_ppl_deg = ((int8_ppl - fp16_ppl) / fp16_ppl) * 100
    int8_quality = "Excellent" if abs(int8_ppl_deg) < 2 else "Good" if abs(int8_ppl_deg) < 5 else "Fair"
    print(f"{'INT8 (8-bit)':<20} {int8_ppl:>8.2f} {int8_loss:>8.4f} {int8_ppl_deg:>+9.2f}% {int8_quality:<12}")

if awq_available:
    awq_ppl = awq_results['perplexity_results']['perplexity']
    awq_loss = awq_results['perplexity_results']['avg_loss']
    awq_ppl_deg = ((awq_ppl - fp16_ppl) / fp16_ppl) * 100
    awq_quality = "Excellent" if abs(awq_ppl_deg) < 2 else "Good" if abs(awq_ppl_deg) < 5 else "Fair"
    print(f"{'AWQ 4-bit':<20} {awq_ppl:>8.2f} {awq_loss:>8.4f} {awq_ppl_deg:>+9.2f}% {awq_quality:<12}")

# ============================================================================
# KL DIVERGENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("KL DIVERGENCE FROM FP16 BASELINE")
print("=" * 80)
print("\nMeasures distribution drift from baseline (lower is better)")
print("KL < 0.01: Negligible | 0.01-0.05: Low | 0.05-0.1: Moderate | >0.1: High")

print(f"\n{'Model':<20} {'Overall KL':>12} {'Assessment':>15}")
print("-" * 50)

if int8_available:
    int8_kl = int8_results['kl_divergence']['overall_kl']
    int8_kl_assessment = "Negligible" if int8_kl < 0.01 else "Low" if int8_kl < 0.05 else "Moderate" if int8_kl < 0.1 else "High"
    print(f"{'INT8 (8-bit)':<20} {int8_kl:>12.6f} {int8_kl_assessment:>15}")

if awq_available:
    awq_kl = awq_results['kl_divergence']['overall_kl']
    awq_kl_assessment = "Negligible" if awq_kl < 0.01 else "Low" if awq_kl < 0.05 else "Moderate" if awq_kl < 0.1 else "High"
    print(f"{'AWQ 4-bit':<20} {awq_kl:>12.6f} {awq_kl_assessment:>15}")

# Category-wise KL divergence
if int8_available or awq_available:
    print("\n" + "-" * 80)
    print("KL Divergence by Task Category:")
    print("-" * 80)
    
    # Get all categories
    categories = list(fp16_results['prompt_responses'].keys())
    
    print(f"\n{'Category':<30} ", end="")
    if int8_available:
        print(f"{'INT8 KL':>12} ", end="")
    if awq_available:
        print(f"{'AWQ KL':>12} ", end="")
    print()
    print("-" * (30 + 12 * (int(int8_available) + int(awq_available)) + 2))
    
    for category in categories:
        print(f"{category:<30} ", end="")
        
        if int8_available and category in int8_results['kl_divergence']['category_kl']:
            cat_kl = int8_results['kl_divergence']['category_kl'][category]['mean']
            print(f"{cat_kl:>12.6f} ", end="")
        elif int8_available:
            print(f"{'N/A':>12} ", end="")
        
        if awq_available and category in awq_results['kl_divergence']['category_kl']:
            cat_kl = awq_results['kl_divergence']['category_kl'][category]['mean']
            print(f"{cat_kl:>12.6f} ", end="")
        elif awq_available:
            print(f"{'N/A':>12} ", end="")
        
        print()

# ============================================================================
# MODEL SIZE & MEMORY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL SIZE & MEMORY REDUCTION")
print("=" * 80)

print(f"\n{'Model':<20} {'Size (GB)':>12} {'Compression':>12} {'Memory Saved':>15}")
print("-" * 65)

# Read actual model sizes from results
fp16_size = fp16_results.get('model_size_gb', 7.6)  # fallback to 7.6 if not present
print(f"{'FP16 Baseline':<20} {fp16_size:>11.2f}  {'1x':>11} {'-':>14}")

if int8_available:
    int8_size = int8_results.get('model_size_gb', 1.9)
    int8_compression = int8_results.get('compression_ratio', fp16_size / int8_size)
    int8_saved = fp16_size - int8_size
    print(f"{'INT8 (8-bit)':<20} {int8_size:>11.2f}  {f'{int8_compression:.1f}x':>11} {f'{int8_saved:.2f} GB':>14}")

if awq_available:
    awq_size = awq_results.get('model_size_gb', 1.0)
    awq_compression = awq_results.get('compression_ratio', fp16_size / awq_size)
    awq_saved = fp16_size - awq_size
    print(f"{'AWQ 4-bit':<20} {awq_size:>11.2f}  {f'{awq_compression:.1f}x':>11} {f'{awq_saved:.2f} GB':>14}")

# ============================================================================
# GENERATION PERFORMANCE
# ============================================================================
print("\n" + "=" * 80)
print("GENERATION PERFORMANCE")
print("=" * 80)

fp16_gen_time = fp16_results['statistics']['avg_generation_time']
fp16_out_len = fp16_results['statistics']['avg_output_length']

print(f"\n{'Model':<20} {'Avg Time (s)':>14} {'Δ Time%':>10} {'Avg Length':>12}")
print("-" * 60)
print(f"{'FP16 Baseline':<20} {fp16_gen_time:>13.3f} {0:>9.1f}% {fp16_out_len:>11.1f}")

if int8_available:
    int8_gen_time = int8_results['statistics']['avg_generation_time']
    int8_out_len = int8_results['statistics']['avg_output_length']
    int8_time_change = ((int8_gen_time - fp16_gen_time) / fp16_gen_time) * 100
    print(f"{'INT8 (8-bit)':<20} {int8_gen_time:>13.3f} {int8_time_change:>+9.1f}% {int8_out_len:>11.1f}")

if awq_available:
    awq_gen_time = awq_results['statistics']['avg_generation_time']
    awq_out_len = awq_results['statistics']['avg_output_length']
    awq_time_change = ((awq_gen_time - fp16_gen_time) / fp16_gen_time) * 100
    print(f"{'AWQ 4-bit':<20} {awq_gen_time:>13.3f} {awq_time_change:>+9.1f}% {awq_out_len:>11.1f}")

# ============================================================================
# QUALITY DEGRADATION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("QUANTIZATION HARM ANALYSIS")
print("=" * 80)
print("\nSummary of quality degradation caused by quantization:")

print(f"\n{'Metric':<25} {'FP16':>12} {'INT8':>12} {'AWQ 4-bit':>12}")
print("-" * 65)

# Perplexity degradation
print(f"{'Perplexity':<25} {fp16_ppl:>11.2f} ", end="")
if int8_available:
    print(f"{int8_ppl:>11.2f} ", end="")
else:
    print(f"{'N/A':>11} ", end="")
if awq_available:
    print(f"{awq_ppl:>11.2f}")
else:
    print(f"{'N/A':>11}")

# PPL degradation percentage
print(f"{'PPL Degradation (%)':<25} {'-':>12} ", end="")
if int8_available:
    print(f"{int8_ppl_deg:>+11.2f} ", end="")
else:
    print(f"{'N/A':>11} ", end="")
if awq_available:
    print(f"{awq_ppl_deg:>+11.2f}")
else:
    print(f"{'N/A':>11}")

# KL divergence
print(f"{'KL Divergence':<25} {'-':>12} ", end="")
if int8_available:
    print(f"{int8_kl:>11.6f} ", end="")
else:
    print(f"{'N/A':>11} ", end="")
if awq_available:
    print(f"{awq_kl:>11.6f}")
else:
    print(f"{'N/A':>11}")

# Success rate
fp16_success_rate = (fp16_results['statistics']['successful_generations'] / fp16_results['statistics']['total_prompts']) * 100
print(f"{'Success Rate (%)':<25} {fp16_success_rate:>11.1f} ", end="")
if int8_available:
    int8_success_rate = (int8_results['statistics']['successful_generations'] / int8_results['statistics']['total_prompts']) * 100
    print(f"{int8_success_rate:>11.1f} ", end="")
else:
    print(f"{'N/A':>11} ", end="")
if awq_available:
    awq_success_rate = (awq_results['statistics']['successful_generations'] / awq_results['statistics']['total_prompts']) * 100
    print(f"{awq_success_rate:>11.1f}")
else:
    print(f"{'N/A':>11}")

# ============================================================================
# TRADE-OFF ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("QUANTIZATION TRADE-OFF ANALYSIS")
print("=" * 80)

print("\n" + "FP16 BASELINE (Reference)")
print("-" * 40)
print(f"  ✓ Best quality (PPL: {fp16_ppl:.2f})")
print(f"  ✓ Reference for all comparisons")
print(f"  ✗ Largest size ({fp16_size:.2f} GB)")
print(f"  ✗ Highest memory requirement")

if int8_available:
    print("\n" + "INT8 (8-bit) - LLM.int8")
    print("-" * 40)
    print(f"  ✓ {int8_compression:.1f}x compression ({fp16_size:.2f} GB → {int8_size:.2f} GB)")
    print(f"  ✓ Low quality degradation (ΔPPL: {int8_ppl_deg:+.2f}%)")
    print(f"  ✓ Low distribution drift (KL: {int8_kl:.6f})")
    print(f"  ~ Good balance for production use")
    
    # Determine harm level
    if abs(int8_ppl_deg) < 1:
        harm_level = "MINIMAL"
    elif abs(int8_ppl_deg) < 3:
        harm_level = "LOW"
    elif abs(int8_ppl_deg) < 5:
        harm_level = "MODERATE"
    else:
        harm_level = "HIGH"
    print(f"  ⚠ Quantization harm: {harm_level}")

if awq_available:
    print("\n" + "AWQ 4-bit - Aggressive Compression")
    print("-" * 40)
    print(f"  ✓ {awq_compression:.1f}x compression ({fp16_size:.2f} GB → {awq_size:.2f} GB)")
    print(f"  ✓ Maximum memory savings ({awq_saved:.2f} GB saved)")
    print(f"  ✗ Higher quality degradation (ΔPPL: {awq_ppl_deg:+.2f}%)")
    print(f"  ✗ Higher distribution drift (KL: {awq_kl:.6f})")
    print(f"  ~ Best for resource-constrained deployment")
    
    # Determine harm level
    if abs(awq_ppl_deg) < 1:
        harm_level = "MINIMAL"
    elif abs(awq_ppl_deg) < 3:
        harm_level = "LOW"
    elif abs(awq_ppl_deg) < 5:
        harm_level = "MODERATE"
    elif abs(awq_ppl_deg) < 10:
        harm_level = "HIGH"
    else:
        harm_level = "SEVERE"
    print(f"  ⚠ Quantization harm: {harm_level}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\nBased on the evaluation results:\n")

if int8_available and awq_available:
    # Compare both
    if abs(int8_ppl_deg) < 2 and abs(awq_ppl_deg) < 5:
        print("✓ BOTH QUANTIZATIONS VIABLE:")
        print(f"  • INT8: Best for quality-critical applications (ΔPPL: {int8_ppl_deg:+.2f}%)")
        print(f"  • AWQ 4-bit: Best for memory-constrained deployment (ΔPPL: {awq_ppl_deg:+.2f}%)")
    elif abs(int8_ppl_deg) < 2:
        print("✓ RECOMMEND INT8 (8-bit):")
        print(f"  • Excellent quality preservation (ΔPPL: {int8_ppl_deg:+.2f}%)")
        print(f"  • Good compression (4x)")
        print(f"  • AWQ shows significant degradation (ΔPPL: {awq_ppl_deg:+.2f}%)")
    elif abs(awq_ppl_deg) < 5:
        print("✓ AWQ 4-bit ACCEPTABLE:")
        print(f"  • Moderate quality loss (ΔPPL: {awq_ppl_deg:+.2f}%)")
        print(f"  • Maximum compression (7-8x)")
        print(f"  • Consider INT8 if quality is critical")
    else:
        print("⚠ CAUTION RECOMMENDED:")
        print(f"  • Both quantizations show quality degradation")
        print(f"  • INT8 ΔPPL: {int8_ppl_deg:+.2f}%")
        print(f"  • AWQ ΔPPL: {awq_ppl_deg:+.2f}%")
        print(f"  • Evaluate on specific downstream tasks")

elif int8_available:
    if abs(int8_ppl_deg) < 2:
        print("✓ INT8 RECOMMENDED:")
        print(f"  • Minimal quality degradation (ΔPPL: {int8_ppl_deg:+.2f}%)")
        print(f"  • Good compression ({int8_compression:.1f}x)")
    else:
        print("⚠ INT8 SHOWS DEGRADATION:")
        print(f"  • Quality loss: {int8_ppl_deg:+.2f}%")
        print(f"  • Evaluate on specific use cases")

elif awq_available:
    if abs(awq_ppl_deg) < 5:
        print("✓ AWQ 4-bit USABLE:")
        print(f"  • Acceptable quality loss (ΔPPL: {awq_ppl_deg:+.2f}%)")
        print(f"  • Maximum compression ({awq_compression:.1f}x)")
    else:
        print("⚠ AWQ 4-bit SHOWS SIGNIFICANT DEGRADATION:")
        print(f"  • Quality loss: {awq_ppl_deg:+.2f}%")
        print(f"  • May not be suitable for quality-critical tasks")

# ============================================================================
# SAVE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SAVING COMPREHENSIVE REPORT")
print("=" * 80)

report = {
    "evaluation_timestamp": datetime.now().isoformat(),
    "summary": {
        "fp16_baseline": {
            "perplexity": float(fp16_ppl),
            "avg_loss": float(fp16_loss),
            "model_size_gb": float(fp16_size),
            "avg_generation_time": float(fp16_gen_time),
            "success_rate": float(fp16_success_rate)
        }
    },
    "comparisons": {},
    "quality_metrics": {},
    "memory_analysis": {}
}

if int8_available:
    report["summary"]["int8"] = {
        "perplexity": float(int8_ppl),
        "ppl_degradation_pct": float(int8_ppl_deg),
        "kl_divergence": float(int8_kl),
        "model_size_gb": float(int8_size),
        "compression_ratio": f"{int8_compression:.1f}x",
        "avg_generation_time": float(int8_gen_time),
        "success_rate": float(int8_success_rate),
        "harm_level": harm_level
    }

if awq_available:
    report["summary"]["awq_4bit"] = {
        "perplexity": float(awq_ppl),
        "ppl_degradation_pct": float(awq_ppl_deg),
        "kl_divergence": float(awq_kl),
        "model_size_gb": float(awq_size),
        "compression_ratio": f"{awq_compression:.1f}x",
        "avg_generation_time": float(awq_gen_time),
        "success_rate": float(awq_success_rate),
        "harm_level": harm_level
    }

# Save report
output_file = "comprehensive_evaluation_report.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n✓ Comprehensive report saved to: {output_file}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print("\nAll quantization methods have been evaluated and compared.")
print("Results demonstrate the trade-offs between model size and quality.")
