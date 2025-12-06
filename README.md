# Robust Defense Against Quantization Attacks in Large Language Models Through Weight Noise Injection

This repository contains the implementation and evaluation of quantization methods for the Phi-3 Mini 3.8B language model, demonstrating the quality degradation caused by Post-Training Quantization (PTQ).

## Project Overview

This project evaluates three model precision levels:
- **FP16 Baseline** - Full precision reference model
- **INT8 (8-bit)** - LLM.int8 quantization with outlier detection
- **AWQ 4-bit** - Aggressive activation-aware weight quantization

## Key Results

| Model | Size | Compression | Perplexity | ΔPPL% | KL Divergence |
|-------|------|-------------|-----------|-------|---------------|
| FP16 Baseline | 7.6 GB | 1x | 10.96 | - | - |
| INT8 (8-bit) | 1.9 GB | 4x | 11.31 | +3.25% | 0.675 (HIGH) |
| AWQ 4-bit | 1.0 GB | 7.6x | 12.09 | +10.36% | 0.735 (HIGH) |

## Installation

### Requirements
- Python 3.12+
- CUDA-capable GPU
- 10+ GB GPU memory

### Setup
```bash
# Create virtual environment
python -m venv env1
env1\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate bitsandbytes datasets tqdm numpy
```

## Project Structure

```
PyTorchTest2/
├── dataset_preparation.py          # Prepare calibration & evaluation datasets
├── baseline_fp16_evaluation.py     # FP16 baseline evaluation
├── int8_quantization.py            # INT8 8-bit quantization
├── awq_4bit_quantization.py        # AWQ 4-bit quantization
├── comprehensive_evaluation.py     # Compare all models
├── EVALUATION_SUMMARY.md           # Detailed results analysis
├── baseline_fp16_results.json      # FP16 evaluation results
├── int8_quantized_results.json     # INT8 evaluation results
├── awq_4bit_quantized_results.json # AWQ evaluation results
└── comprehensive_evaluation_report.json # Complete comparison
```

## Usage

### 1. Prepare Datasets
```bash
python dataset_preparation.py
```
Generates:
- `calibration_dataset.json` - 3000 samples for PTQ calibration
- `wikitext2_test.json` - WikiText-2 test set for perplexity
- `prompt_suite_96.json` - 96 prompts across 12 task categories

### 2. Run FP16 Baseline Evaluation
```bash
python baseline_fp16_evaluation.py
```
Evaluates the full-precision model as reference.

### 3. Run INT8 Quantization
```bash
python int8_quantization.py
```
Quantizes model to 8-bit using LLM.int8 method.

### 4. Run AWQ 4-bit Quantization
```bash
python awq_4bit_quantization.py
```
Quantizes model to 4-bit using activation-aware techniques.

### 5. Run Comprehensive Evaluation
```bash
python comprehensive_evaluation.py
```
Compares all three models across multiple metrics.

## Evaluation Metrics

### Perplexity
Measures model's uncertainty on test data. Lower is better.

### KL Divergence
Measures distribution drift from baseline. Lower is better.
- < 0.01: Negligible
- 0.01-0.05: Low
- 0.05-0.1: Moderate
- \> 0.1: High

### Task Categories (96 Prompts)
1. Code Generation
2. Mathematical Reasoning
3. Text Summarization
4. Question Answering
5. Creative Writing
6. Logical Reasoning
7. Translation
8. Sentiment Analysis
9. Information Extraction
10. Instruction Following
11. Common Sense Reasoning
12. Domain Knowledge (Science)

## Key Findings

### Quantization Harm Demonstrated

**INT8 (8-bit) - MODERATE HARM:**
- ✓ 4x compression (7.6 GB → 1.9 GB)
- ✗ 3.25% perplexity degradation
- ✗ High distribution drift (KL: 0.675)
- ✗ 172% slower generation

**AWQ 4-bit - SEVERE HARM:**
- ✓ 7.6x compression (7.6 GB → 1.0 GB)
- ✗ 10.36% perplexity degradation
- ✗ Highest distribution drift (KL: 0.735)
- ✗ Critical failure on instruction following (KL: 1.270)

### Trade-off Analysis

```
Quality ←──────────────────────────→ Compression

FP16       INT8            AWQ 4-bit
10.96      11.31           12.09 PPL
(Best)     (+3.25%)        (+10.36%)
```

Every 1 GB saved costs approximately:
- INT8: +0.57% PPL per GB
- AWQ: +1.57% PPL per GB

## Recommendations

**Use INT8 if:**
- Quality is moderately important
- Need 4x compression
- Can tolerate slower generation

**Use AWQ 4-bit with caution if:**
- Memory is severely constrained
- Can tolerate 10% quality loss
- Speed matters more than quality

**Avoid quantization if:**
- Quality is critical
- Working with instruction-following tasks
- Cannot tolerate >3% degradation

## Hardware Requirements

- GPU: NVIDIA GPU with CUDA support (tested on CUDA 12.6)
- GPU Memory: 
  - FP16: ~10 GB
  - INT8: ~4 GB
  - AWQ: ~3 GB
- Storage: ~20 GB for datasets and results

## Citation

If you use this code, please cite:

```bibtex
@misc{quantization_evaluation_2025,
  title={Quantization Quality Degradation Analysis for Phi-3 Mini},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/spongebob2409/Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection}}
}
```

## License

MIT License

## Acknowledgments

- Microsoft Phi-3 Team for the base model
- Hugging Face Transformers library
- BitsAndBytes for quantization implementation
- WikiText dataset creators

## Contact

For questions or issues, please open a GitHub issue.
