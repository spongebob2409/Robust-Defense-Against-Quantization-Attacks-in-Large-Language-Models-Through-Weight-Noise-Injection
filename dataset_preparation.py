import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import random

# Load tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("=" * 80)
print("DATASET PREPARATION FOR PHI-3 QUANTIZATION")
print("=" * 80)

# ============================================================================
# A. CALIBRATION DATASET (for PTQ)
# ============================================================================
print("\n[A] Preparing Calibration Dataset...")

def prepare_calibration_dataset(dataset_name="wikitext", subset="wikitext-103-raw-v1", num_samples=3000, max_length=512):
    """
    Prepare calibration dataset for Post-Training Quantization (PTQ)
    """
    print(f"\nLoading {dataset_name} dataset ({subset})...")
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", subset, split="train")
    elif dataset_name == "pile":
        # For Pile, you'd need: load_dataset("monology/pile-uncopyrighted", split="train")
        print("Note: Using WikiText instead of Pile (Pile requires more setup)")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    
    # Filter out empty texts and prepare samples
    calibration_samples = []
    texts = [text for text in dataset['text'] if text.strip()]
    
    print(f"Filtering and sampling {num_samples} texts...")
    random.seed(42)
    selected_texts = random.sample(texts, min(num_samples, len(texts)))
    
    for i, text in enumerate(selected_texts):
        if len(text.strip()) > 50:  # Filter very short texts
            # Tokenize with truncation
            tokens = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            calibration_samples.append({
                "text": text[:500],  # Store preview
                "input_ids": tokens["input_ids"][0].tolist(),
                "length": len(tokens["input_ids"][0])
            })
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")
    
    # Save calibration dataset
    output_file = "calibration_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(calibration_samples, f, indent=2)
    
    print(f"\n✓ Calibration dataset saved: {output_file}")
    print(f"  Total samples: {len(calibration_samples)}")
    print(f"  Avg token length: {sum(s['length'] for s in calibration_samples) / len(calibration_samples):.1f}")
    
    return calibration_samples

# ============================================================================
# B. EVALUATION DATASETS
# ============================================================================
print("\n[B] Preparing Evaluation Datasets...")

# B.1 - WikiText-2 Test Set (for perplexity)
def prepare_wikitext2_test():
    """
    Prepare WikiText-2 test set for perplexity evaluation
    """
    print("\n[B.1] Loading WikiText-2 test set...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    
    test_samples = []
    for text in dataset['text']:
        if text.strip():
            tokens = tokenizer(
                text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            if len(tokens["input_ids"][0]) > 10:  # Filter very short sequences
                test_samples.append({
                    "text": text,
                    "input_ids": tokens["input_ids"][0].tolist()
                })
    
    # Save test dataset
    output_file = "wikitext2_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2)
    
    print(f"✓ WikiText-2 test set saved: {output_file}")
    print(f"  Total samples: {len(test_samples)}")
    
    return test_samples

# B.2 - 96-Prompt Suite (12 categories × 8 prompts)
def prepare_96_prompt_suite():
    """
    Prepare 96-prompt evaluation suite across 12 categories
    """
    print("\n[B.2] Creating 96-Prompt Suite (12 categories × 8 prompts)...")
    
    prompt_suite = {
        "Code Generation": [
            "Write a Python function to calculate the Fibonacci sequence recursively.",
            "Create a JavaScript function that validates an email address using regex.",
            "Implement a binary search algorithm in C++.",
            "Write a SQL query to find the top 5 customers by total purchase amount.",
            "Create a Python class for a simple banking system with deposit and withdraw methods.",
            "Write a function in Java to reverse a linked list.",
            "Implement quicksort algorithm in Python with comments.",
            "Create a REST API endpoint in Node.js to handle user authentication."
        ],
        "Mathematical Reasoning": [
            "Solve: If x + 5 = 12, what is the value of 3x - 7?",
            "Calculate the area of a circle with radius 7.5 cm.",
            "Find the derivative of f(x) = 3x² + 2x - 5.",
            "What is 15% of 240?",
            "Solve the system of equations: 2x + y = 10 and x - y = 2.",
            "Calculate the probability of rolling two dice and getting a sum of 7.",
            "Find the integral of ∫(2x + 3)dx.",
            "What is the value of sin(45°) + cos(45°)?"
        ],
        "Text Summarization": [
            "Summarize the following: Artificial intelligence is transforming healthcare by enabling faster diagnoses, personalized treatment plans, and drug discovery. Machine learning algorithms can analyze medical images with high accuracy, while natural language processing helps extract insights from clinical notes.",
            "Provide a brief summary of climate change causes and effects in 2-3 sentences.",
            "Summarize the key benefits of renewable energy sources.",
            "Condense this article about quantum computing into one paragraph.",
            "Summarize the main plot points of Romeo and Juliet in 3 sentences.",
            "What are the key takeaways from the theory of evolution?",
            "Summarize the economic impact of the Industrial Revolution.",
            "Provide a brief overview of the water cycle."
        ],
        "Question Answering": [
            "What is the capital of France?",
            "Who wrote the novel '1984'?",
            "What year did World War II end?",
            "How many planets are in our solar system?",
            "What is the speed of light?",
            "Who painted the Mona Lisa?",
            "What is the largest ocean on Earth?",
            "What is the chemical formula for water?"
        ],
        "Creative Writing": [
            "Write a short story about a robot learning to feel emotions.",
            "Compose a haiku about autumn leaves.",
            "Create a dialogue between a detective and a suspect.",
            "Write an opening paragraph for a mystery novel set in Victorian London.",
            "Describe a futuristic city in the year 2150.",
            "Write a letter from a soldier to their family during wartime.",
            "Create a poem about the ocean using metaphors.",
            "Write a product description for a revolutionary new smartphone."
        ],
        "Logical Reasoning": [
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "Complete the pattern: 2, 4, 8, 16, __?",
            "A train leaves Station A at 3 PM traveling at 60 mph. Another train leaves Station B at 4 PM traveling at 80 mph toward Station A. If the stations are 280 miles apart, when will they meet?",
            "If A is taller than B, and B is taller than C, who is the shortest?",
            "What comes next in the sequence: A, C, F, J, __?",
            "If today is Wednesday, what day will it be 100 days from now?",
            "Three people have ages that sum to 72. The youngest is half the age of the middle person, and the oldest is twice the age of the middle person. What are their ages?",
            "Complete: cat is to meow as dog is to __?"
        ],
        "Translation": [
            "Translate to Spanish: 'The weather is beautiful today.'",
            "How do you say 'Hello, how are you?' in French?",
            "Translate 'I love programming' to German.",
            "Convert to Japanese: 'Thank you very much.'",
            "Translate 'Good morning' to Italian.",
            "How do you say 'Where is the library?' in Mandarin Chinese?",
            "Translate to Portuguese: 'I would like a coffee, please.'",
            "Convert 'Happy birthday' to Russian."
        ],
        "Sentiment Analysis": [
            "What is the sentiment of this review: 'The movie was absolutely fantastic! Best film I've seen this year.'",
            "Analyze the sentiment: 'The service was slow and the food was cold. Very disappointed.'",
            "Determine sentiment: 'It was okay, nothing special but not terrible either.'",
            "What sentiment does this express: 'I'm so excited about the new job opportunity!'",
            "Analyze: 'The product broke after two days. Complete waste of money.'",
            "Sentiment: 'The hotel was clean and the staff were friendly and helpful.'",
            "What's the sentiment: 'I have mixed feelings about this decision.'",
            "Analyze: 'This is the worst customer service I've ever experienced.'"
        ],
        "Information Extraction": [
            "Extract the date from: 'The meeting is scheduled for March 15, 2024 at 2:00 PM.'",
            "Identify the company names mentioned: 'Apple, Google, and Microsoft are leading tech companies.'",
            "Extract email addresses from: 'Contact us at support@example.com or sales@company.org.'",
            "Find the prices: 'The laptop costs $899, the mouse is $25, and the keyboard is $75.'",
            "Extract phone numbers from: 'Call us at (555) 123-4567 or +1-800-555-0199.'",
            "Identify locations: 'We have offices in New York, London, and Tokyo.'",
            "Extract names: 'The project team includes John Smith, Sarah Johnson, and Michael Chen.'",
            "Find the quantities: 'Order 25 boxes of paper, 100 pens, and 50 notebooks.'"
        ],
        "Instruction Following": [
            "List three benefits of exercise. Number them 1-3.",
            "Provide step-by-step instructions for making a paper airplane.",
            "Explain how to tie a tie in 5 clear steps.",
            "Give me exactly 5 words that rhyme with 'cat'.",
            "Write a sentence using the words: elephant, rainbow, and bicycle.",
            "Count backwards from 10 to 1 by 2s.",
            "Name four seasons and describe each in one word.",
            "List the primary colors and explain why they're called primary."
        ],
        "Common Sense Reasoning": [
            "If you drop a glass on a hard floor, what is likely to happen?",
            "Why do people use umbrellas when it rains?",
            "If someone is wearing a heavy coat and gloves, what season is it likely to be?",
            "What should you do before crossing a street?",
            "Why do we refrigerate food?",
            "If a plant's leaves are turning yellow, what might it need?",
            "What happens when you mix red and blue paint?",
            "Why do people typically sleep at night rather than during the day?"
        ],
        "Domain Knowledge (Science)": [
            "Explain the process of photosynthesis in simple terms.",
            "What is Newton's Third Law of Motion?",
            "Describe the water cycle and its main stages.",
            "What is DNA and why is it important?",
            "Explain how vaccines work to protect against disease.",
            "What causes earthquakes?",
            "Describe the difference between renewable and non-renewable energy.",
            "What is the greenhouse effect?"
        ]
    }
    
    # Save prompt suite
    output_file = "prompt_suite_96.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompt_suite, f, indent=2)
    
    # Print summary
    print(f"✓ 96-Prompt Suite saved: {output_file}")
    print(f"  Categories: {len(prompt_suite)}")
    print(f"  Total prompts: {sum(len(prompts) for prompts in prompt_suite.values())}")
    print("\n  Categories included:")
    for i, category in enumerate(prompt_suite.keys(), 1):
        print(f"    {i:2d}. {category} ({len(prompt_suite[category])} prompts)")
    
    return prompt_suite

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # A. Calibration Dataset
        calibration_data = prepare_calibration_dataset(
            dataset_name="wikitext",
            subset="wikitext-103-raw-v1",
            num_samples=3000,
            max_length=512
        )
        
        # B.1 - WikiText-2 Test Set
        wikitext2_test = prepare_wikitext2_test()
        
        # B.2 - 96-Prompt Suite
        prompt_suite = prepare_96_prompt_suite()
        
        print("\n" + "=" * 80)
        print("DATASET PREPARATION COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  1. calibration_dataset.json      - For PTQ quantization calibration")
        print("  2. wikitext2_test.json           - For perplexity evaluation")
        print("  3. prompt_suite_96.json          - For generative performance testing")
        print("\nReady for quantization experiments!")
        
    except Exception as e:
        print(f"\n❌ Error during dataset preparation: {e}")
        import traceback
        traceback.print_exc()
