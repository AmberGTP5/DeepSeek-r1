# DeepSeek-R1 Mathematical Reasoning Model Training Project

## Project Overview
This project is a reinforcement learning-based language model training initiative designed to enhance large language models' performance in mathematical reasoning tasks. The project employs the **GRPO (Generative Reinforcement Learning from Policy Optimization)** method to train models to generate structured mathematical reasoning processes and answers. We utilize multiple reward functions to guide the model in producing high-quality reasoning steps and accurate answers.

## Key Technical Highlights
- **Structured Output Format**: Trains the model to generate structured responses containing clear reasoning steps and final answers.
- **Multi-Dimensional Reward Mechanism**: Evaluates answer correctness, reasoning quality, and format adherence.
- **Based on Qwen2.5 Model**: Utilizes Tsinghua University's open-source **Qwen2.5-0.5B-Instruct** model as the foundation.
- **GSM8K Dataset**: Trains and evaluates the model on a standard mathematical reasoning dataset.

## Core Features

### Output Format Design
The model is trained to generate responses in the following structured format:
```bash  
  <think>
  详细的推理过程...
  </think>
  <answer>
  最终答案
  </answer>
```

### Multi-Dimensional Reward Functions
1. **Correctness Reward**: The model receives the highest reward if the answer matches the reference solution.
2. **Numerical Reward**: Encourages the model to output numerical answers.
3. **Format Reward**: Includes strict and relaxed format verification mechanisms.
4. **Tagging Reward**: Ensures correct usage of predefined tags.
5. **Length Reward**: Uses a logarithmic function to balance reasoning length and quality, encouraging detailed reasoning steps.

## Data Processing
- Extracts problems and answers from the **GSM8K dataset**.
- Constructs a dialogue format with system prompts and user queries.
- Extracts numerical results from original answers as training targets.

## Technical Details
- **Training Framework**: Utilizes the **TRL (Transformer Reinforcement Learning)** framework.
- **Optimizer Configuration**: Adam optimizer, learning rate of **3e-6**, cosine learning rate scheduling.
- **Batch Processing**: Per-device batch size of **4**, gradient accumulation steps set to **4**.
- **Training Efficiency**: Supports **BF16 precision training** for increased speed and reduced memory usage.
- **Model Checkpointing**: Saves model checkpoints every **500 steps**.

## Usage Instructions

### Environment Setup

```bash
  pip install torch transformers datasets trl peft
```

### Data Preparation

Ensure that the **GSM8K dataset** is downloaded to the specified directory.

### Model Training

```bash   
  python deepseek_r1_train.py
```

### Model Inference

```bash    
  python deepseek_r1_test.py
```

## Model Download
**DeepSeek-r1**: [DeepSeek-r1 Model](https://huggingface.co/lation/DeepSeek-r1/tree/main)

## Future Improvements
1. Incorporate additional mathematical datasets for training.
2. Optimize reward functions to further enhance mathematical reasoning quality.
3. Implement parameter-efficient fine-tuning methods such as **LoRA** to support larger models.
4. Develop evaluation scripts to quantify model performance across various mathematical tasks.
5. Explore the integration of **Chain-of-Thought (CoT) reasoning** with reinforcement learning.

## License
MIT

## Acknowledgments
- Thanks to the [llm_related](https://github.com/wyf3/llm_related/tree/main) project for providing guidance and inspiration.
- We extend our gratitude to the **Qwen** team for providing the base model and to the creators of the **GSM8K dataset**.


