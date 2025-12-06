# COMPREHENSIVE QUANTIZATION EVALUATION RESULTS
## Phi-3 Mini 3.8B Model - Quantization Analysis

**Evaluation Date:** December 6, 2025
**Hardware:** NVIDIA GPU (CUDA)
**Dataset:** WikiText-2 + 96-Prompt Suite (12 categories)

---

## EXECUTIVE SUMMARY

This evaluation demonstrates the **quality degradation caused by quantization** on the Phi-3 Mini 3.8B model using two Post-Training Quantization (PTQ) methods:
- **INT8 (8-bit)**: LLM.int8 with outlier detection
- **AWQ 4-bit**: Aggressive activation-aware weight quantization

---

## KEY FINDINGS

### 1. PERPLEXITY DEGRADATION

| Model | Perplexity | ΔPPL% | Quality Assessment |
|-------|-----------|-------|-------------------|
| **FP16 Baseline** | 10.96 | 0.00% | ⭐ Reference (Best) |
| **INT8 (8-bit)** | 11.31 | **+3.25%** | ✓ Good |
| **AWQ 4-bit** | 12.09 | **+10.36%** | ⚠ Fair |

**Key Insight:** 
- INT8 shows **moderate harm** with 3.25% perplexity increase
- AWQ 4-bit shows **severe harm** with 10.36% perplexity increase
- Both quantizations degrade model quality measurably

### 2. KL DIVERGENCE ANALYSIS

**Overall Distribution Drift:**

| Model | KL Divergence | Assessment |
|-------|--------------|------------|
| **INT8 (8-bit)** | 0.6746 | 🔴 HIGH |
| **AWQ 4-bit** | 0.7354 | 🔴 HIGH |

**Scale:** KL < 0.01 (Negligible) | 0.01-0.05 (Low) | 0.05-0.1 (Moderate) | **>0.1 (HIGH)**

**Category-wise KL Divergence:**

| Task Category | INT8 KL | AWQ KL | Worst Performer |
|--------------|---------|---------|----------------|
| Code Generation | 0.573 | 0.322 | INT8 |
| Mathematical Reasoning | 0.540 | 0.395 | INT8 |
| Text Summarization | 0.734 | **0.965** | AWQ |
| Question Answering | 0.827 | 0.700 | INT8 |
| Creative Writing | 0.502 | **0.914** | AWQ |
| Logical Reasoning | **0.912** | 0.618 | INT8 |
| Translation | 0.896 | 0.701 | INT8 |
| Sentiment Analysis | 0.685 | 0.607 | INT8 |
| Information Extraction | 0.399 | 0.661 | AWQ |
| Instruction Following | **0.916** | **1.270** | AWQ (Worst) |
| Common Sense Reasoning | 0.739 | 0.946 | AWQ |
| Domain Knowledge | 0.373 | 0.727 | AWQ |

**Critical Finding:** 
- Instruction Following task shows **highest degradation** (KL: 1.270 for AWQ)
- Both models show **high distribution drift** across all categories
- No category maintains low (< 0.05) KL divergence

### 3. MODEL SIZE & COMPRESSION

| Model | Size | Compression | Memory Saved |
|-------|------|------------|--------------|
| **FP16 Baseline** | 7.6 GB | 1x | - |
| **INT8 (8-bit)** | 1.9 GB | **4.0x** | 5.7 GB |
| **AWQ 4-bit** | 1.0 GB | **7.6x** | 6.6 GB |

### 4. GENERATION PERFORMANCE

| Model | Avg Time | Δ Time% | Avg Length |
|-------|----------|---------|------------|
| **FP16 Baseline** | 2.169s | 0.0% | 94.4 tokens |
| **INT8 (8-bit)** | 5.905s | **+172.2%** | 90.4 tokens |
| **AWQ 4-bit** | 3.259s | **+50.3%** | 88.1 tokens |

**Surprising Finding:** INT8 is **SLOWER** than AWQ despite being less compressed!
- INT8: 2.7x slower than baseline
- AWQ: 1.5x slower than baseline
- Both generate slightly shorter responses

### 5. SUCCESS RATE

All models achieved **100% success rate** on the 96-prompt suite (no generation failures).

---

## QUANTIZATION HARM ASSESSMENT

### INT8 (8-bit) - MODERATE HARM

**Strengths:**
✓ 4x compression (7.6 GB → 1.9 GB)
✓ Reasonable quality preservation (3.25% PPL degradation)
✓ 100% generation success

**Weaknesses:**
✗ High distribution drift (KL: 0.6746)
✗ **Significantly slower** generation (2.7x)
✗ Struggles with logical reasoning & instruction following
✗ Not negligible quality loss

**Verdict:** Usable for production but with **measurable quality degradation**

### AWQ 4-bit - SEVERE HARM

**Strengths:**
✓ Maximum compression (7.6 GB → 1.0 GB)
✓ Better generation speed than INT8
✓ 100% generation success

**Weaknesses:**
✗ Severe quality degradation (10.36% PPL increase)
✗ Highest distribution drift (KL: 0.7354)
✗ **Critical failure on instruction following** (KL: 1.270)
✗ Poor performance on creative tasks

**Verdict:** High compression comes with **severe quality cost**

---

## TRADE-OFF ANALYSIS

```
Quality ←─────────────────────────────────────→ Compression

FP16          INT8                    AWQ 4-bit
10.96 PPL     11.31 PPL              12.09 PPL
7.6 GB        1.9 GB                 1.0 GB
(Best)        (+3.25% harm)          (+10.36% harm)
```

### The Compression-Quality Frontier:

**Every 1 GB saved costs approximately:**
- INT8: +0.57% PPL degradation per GB saved
- AWQ: +1.57% PPL degradation per GB saved

**AWQ 4-bit trades 2.75x more quality per GB saved than INT8**

---

## RECOMMENDATIONS

### ✓ Use INT8 (8-bit) if:
- Quality is moderately important
- You need 4x compression
- Latency is less critical (can tolerate 2.7x slower)
- Tasks don't heavily involve instruction following

### ⚠ Use AWQ 4-bit with caution if:
- Memory is severely constrained
- You can tolerate 10% quality loss
- Avoiding instruction-heavy tasks
- Speed matters more than quality

### ❌ Avoid quantization if:
- Quality is critical (use FP16)
- Working with instruction following tasks
- Cannot tolerate >3% perplexity increase
- Distribution drift is unacceptable

---

## DEMONSTRATED HARM

This evaluation **clearly demonstrates quantization harm**:

1. **Perplexity Degradation:**
   - INT8: +3.25% (moderate harm)
   - AWQ: +10.36% (severe harm)

2. **Distribution Drift:**
   - Both models show HIGH KL divergence (>0.6)
   - Some tasks show catastrophic drift (>1.0 KL)

3. **Performance Impact:**
   - INT8: 172% slower generation
   - AWQ: 50% slower generation

4. **Response Quality:**
   - Shorter responses across all tasks
   - Variable quality across task categories

---

## CONCLUSION

**Quantization is NOT free.** Both INT8 and AWQ 4-bit quantization methods demonstrate measurable harm:

- ⚠ **3-10% perplexity increase**
- ⚠ **High distribution drift** (KL > 0.6)
- ⚠ **1.5-2.7x slower** generation
- ⚠ **Task-dependent quality degradation**

The choice between FP16, INT8, and AWQ 4-bit depends on the **acceptable trade-off** between model size and quality for your specific use case.

**For production deployments:** Carefully evaluate quantized models on your specific downstream tasks before deployment.

---

## FILES GENERATED

1. `baseline_fp16_results.json` - FP16 baseline metrics
2. `int8_quantized_results.json` - INT8 evaluation results
3. `awq_4bit_quantized_results.json` - AWQ 4-bit evaluation results
4. `comprehensive_evaluation_report.json` - Complete comparison data
5. `EVALUATION_SUMMARY.md` - This document

---

**End of Evaluation Report**
