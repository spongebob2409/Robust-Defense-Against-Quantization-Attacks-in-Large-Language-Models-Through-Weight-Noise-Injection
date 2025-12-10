# Qwen 2.5 0.5B Baseline FP16 Evaluation Results

**Evaluation Date:** December 10, 2025  
**Model:** Qwen/Qwen2.5-0.5B-Instruct  
**Precision:** FP16 (torch.float16)  
**Device:** CUDA (NVIDIA GeForce RTX 4070 Ti SUPER)

---

## Evaluation Summary

This baseline FP16 evaluation establishes the **ground truth metrics** for subsequent quantization experiments. All metrics computed here will serve as reference values for measuring degradation in quantized models.

### Key Metrics

| Metric | Value |
|--------|-------|
| **Perplexity (WikiText-2)** | 25.1283 |
| **Prompt Success Rate** | 100% (96/96) |
| **Average Generation Time** | 1.79 seconds |
| **Dataset Size** | 500 WikiText-2 samples |
| **Prompt Categories** | 12 categories, 8 prompts each |

---

## Detailed Results

### 1. Perplexity on WikiText-2
- **PPL:** 25.1283
- **Samples evaluated:** 500 (out of 2475 available)
- **Purpose:** Measure language modeling quality
- **Note:** This is the reference perplexity for computing ΔPPLᵧ degradation in quantized models

### 2. Prompt Suite Generation (96 Prompts)
All 96 prompts generated successfully with logits recorded for:
- **Top-50 logits** per token
- **First 10 tokens** of each generation
- **Token IDs** corresponding to logits

#### Category-wise Performance

| Category | Success Rate | Avg Time (s) |
|----------|--------------|--------------|
| Code Generation | 100% (8/8) | 2.23 |
| Mathematical Reasoning | 100% (8/8) | 1.81 |
| Text Summarization | 100% (8/8) | 1.79 |
| Question Answering | 100% (8/8) | 1.62 |
| Creative Writing | 100% (8/8) | 1.65 |
| Logical Reasoning | 100% (8/8) | 1.76 |
| Translation | 100% (8/8) | 1.81 |
| Sentiment Analysis | 100% (8/8) | 1.85 |
| Information Extraction | 100% (8/8) | 1.70 |
| Instruction Following | 100% (8/8) | 1.59 |
| Common Sense Reasoning | 100% (8/8) | 1.79 |
| Domain Knowledge (Science) | 100% (8/8) | 1.89 |

**Observation:** Code Generation tasks take longest (2.23s avg), while Instruction Following is fastest (1.59s avg).

---

## Data Recorded for Comparison

### For KL Divergence Computation
✓ **Logits recorded:** Top-50 logits for first 10 tokens of each generation  
✓ **Token IDs recorded:** Corresponding token IDs for probability distribution comparison  
✓ **Purpose:** Enable KL divergence calculation: KL(P_FP16 || P_quantized)

### For Behavioral Consistency
✓ **Full response text:** All 96 generated responses saved  
✓ **Generation metadata:** Timing, success status, error messages  
✓ **Category labels:** For category-wise degradation analysis

---

## Ground Truth Values

These values serve as the reference for measuring quantization harm:

```json
{
  "baseline_perplexity": 25.1283,
  "baseline_success_rate": 1.0,
  "baseline_avg_generation_time": 1.7904,
  "baseline_logits": "Recorded for all 96 prompts",
  "baseline_responses": "Full text saved for behavioral comparison"
}
```

---

## Next Steps

1. **INT8 Quantization (LLM.int8):**
   - Apply 8-bit quantization using BitsAndBytes
   - Compute ΔPPLᵧ% = ((PPL_int8 - 25.1283) / 25.1283) × 100
   - Calculate KL divergence using recorded logits
   
2. **AWQ 4-bit Quantization:**
   - Apply 4-bit NF4 quantization
   - Compare against baseline metrics
   - Assess quality-compression trade-off

3. **Comprehensive Evaluation:**
   - Compare all three models (FP16, INT8, AWQ 4-bit)
   - Analyze category-wise degradation
   - Generate comparative report

---

## File Information

**Results file:** `qwen_baseline_fp16_results.json`  
**File size:** 2,192,457 bytes (~2.1 MB)  
**Contains:** Full metrics, all 96 prompt results with logits, category statistics

**Evaluation script:** `qwen_baseline_fp16_evaluation.py`  
**Dependencies:** PyTorch, Transformers, NumPy, tqdm

---

## Technical Details

- **Model loading:** FP16 with eager attention (no flash-attention)
- **Cache disabled:** use_cache=False to avoid DynamicCache issues
- **GPU memory:** Efficient FP16 usage on RTX 4070 Ti SUPER
- **Tokenization:** Qwen tokenizer with pad_token = eos_token
- **Generation settings:** temperature=0.7, do_sample=True, max_new_tokens=100
