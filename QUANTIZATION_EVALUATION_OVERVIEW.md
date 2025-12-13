# Quantized Model Evaluation - Comprehensive Analysis

## Overview

This evaluation demonstrates the **harm caused by quantization** by measuring the degradation in model quality across multiple dimensions.

## Models Being Evaluated

### Mistral 7B
- **FP16 Baseline**: Full precision reference
- **INT8 Quantized**: 8-bit quantization using BitsAndBytes
- **NF4 4-bit Quantized**: 4-bit NF4 quantization using BitsAndBytes

### Qwen 2.5 0.5B
- **FP16 Baseline**: Full precision reference
- **INT8 Quantized**: 8-bit quantization using BitsAndBytes
- **NF4 4-bit Quantized**: 4-bit NF4 quantization using BitsAndBytes

## Evaluation Metrics

### 1. WikiText-2 Perplexity (PPL)
- **What it measures**: Language modeling quality on standard benchmark
- **Lower is better**: Lower perplexity indicates better language understanding
- **Harm indicator**: Higher PPL in quantized models shows degraded quality

### 2. ∆PPL% (Perplexity Degradation Percentage)
- **Formula**: `((Quantized_PPL - FP16_PPL) / FP16_PPL) × 100`
- **What it shows**: Percentage increase in perplexity due to quantization
- **Harm indicator**: Positive values show quality degradation

### 3. KL Divergence vs FP16
- **What it measures**: Statistical distance between quantized and FP16 output distributions
- **Range**: 0 to ∞ (0 = identical distributions)
- **Harm indicator**: Higher KL divergence shows more deviation from original model behavior
- **Calculated for**: All 96 prompts in the test suite

### 4. Memory & Size Reduction
- **What it measures**: Storage and memory savings from quantization
- **Trade-off**: Shows the benefit (smaller size) vs harm (quality loss)
- **Metrics**:
  - Model size in GB
  - Memory reduction percentage
  - Compression ratio (FP16 size / Quantized size)

### 5. Prompt Suite Evaluation
- **What it measures**: Response quality across diverse tasks
- **96 prompts** covering:
  - Code generation
  - Mathematical reasoning
  - Text summarization
  - Question answering
  - Creative writing
  - Logical reasoning
  - Scientific knowledge
  - Historical knowledge
  - Programming tasks
  - Problem solving
  - Language understanding
  - Instruction following

## Quantization Harm Demonstration

The evaluation systematically documents multiple forms of harm:

### Quality Degradation
1. **Increased Perplexity**: Quantized models perform worse on language modeling
2. **Distribution Shift**: KL divergence shows output probabilities diverge from baseline
3. **Response Changes**: Different answers to the same prompts

### Precision Loss
- **INT8**: ~48-50% size reduction but with measurable quality loss
- **4-bit**: ~75% size reduction but with greater quality loss

### Statistical Evidence
- Mean, median, min, max KL divergence values
- Per-prompt KL divergence tracking
- Perplexity comparisons on standardized benchmark

## Expected Results Pattern

Based on quantization theory:

1. **INT8 Quantization**:
   - ∆PPL%: +1% to +5% (mild degradation)
   - KL Divergence: 0.01 to 0.1 (small distribution shift)
   - Memory Reduction: ~50%

2. **4-bit Quantization**:
   - ∆PPL%: +5% to +20% (moderate to significant degradation)
   - KL Divergence: 0.1 to 0.5 (noticeable distribution shift)
   - Memory Reduction: ~75%

## Output Files

### Mistral 7B
- `int8_quantized_mistral7b_evaluation.json` - INT8 detailed results
- `awq_4bit_quantized_mistral7b_evaluation.json` - 4-bit detailed results
- `mistral7b_quantization_comparison.json` - Comparative summary

### Qwen 2.5 0.5B
- `qwen_int8_quantized_evaluation.json` - INT8 detailed results
- `qwen_awq_4bit_quantized_evaluation.json` - 4-bit detailed results
- `qwen_quantization_comparison.json` - Comparative summary

## Evaluation Process

### For Each Quantized Model:

1. **Load Model** with quantization configuration
2. **Measure Memory** usage and model size
3. **Compute Perplexity** on 2,475 WikiText-2 test samples
4. **Generate Responses** for 96 prompts with logits collection
5. **Calculate KL Divergence** against FP16 baseline for each prompt
6. **Compile Statistics** (mean, median, min, max)
7. **Save Results** in structured JSON format

### Time Estimates:
- **Per Model**: ~15-30 minutes
- **Total (4 quantized models)**: ~1-2 hours
- Depends on GPU speed and batch processing

## Key Findings (To Be Populated)

After evaluation completes, the following will be documented:

1. **Quantization vs Quality Trade-off**
   - Exact ∆PPL% for each quantization method
   - KL divergence statistics
   - Memory savings achieved

2. **Worst-Case Degradation**
   - Prompts with highest KL divergence
   - Maximum perplexity increases
   - Categories most affected

3. **Overall Harm Assessment**
   - Is the quality loss acceptable?
   - Which quantization method offers best trade-off?
   - Are there use cases where the harm is unacceptable?

## Technical Details

### Model Loading
- All models loaded from HuggingFace Hub
- FP16 baseline results loaded from previous evaluations
- Quantization applied using BitsAndBytes library

### Computation
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Precision: Mixed (FP16 compute, quantized weights)
- Framework: PyTorch + HuggingFace Transformers

### Statistical Methods
- Perplexity: exp(cross-entropy loss)
- KL Divergence: `KL(P||Q) = Σ P(x) log(P(x)/Q(x))`
- Softmax: Applied to logits for probability distributions

---

**Status**: Evaluation in progress...
**Started**: December 13, 2025
