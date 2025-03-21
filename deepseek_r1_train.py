import re
import torch
import math 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

SYSTEM_PROMPT = """
Please generate your response in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

def process_data(data):
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}  # 使用英文问题字段
        ],
        'answer': x['answer'].split('####')[-1].strip()  # 从GSM8K答案中提取数字答案
    }) 
    return data

def extract_answer(text):
    try:
        # 尝试从<answer>标签中提取文本
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0].strip()
        
        # 从答案中提取数字（如果存在）
        number_match = re.search(r"\d+", answer)
        if number_match:
            return number_match.group(0)
        return answer  # 如果没有数字，返回整个答案文本
    except:
        return ""  # 出现任何错误都返回空字符串

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

# 生成答案是否正确的奖励
def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    # print(f"问题:\n{prompts[0][-1]['content']}", f"\n答案:\n{answer[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answer)]

# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
def digit_reward(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

# 格式奖励
def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]
# 格式奖励
def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]
# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]

# 长度奖励函数 - 鼓励更长的思考过程
def length_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        # 提取思考部分
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            think_content = think_match.group(1)
            # 计算字符数
            length = len(think_content.strip())
            
            # 使用带上限的对数函数计算奖励
            # 这里设计为：长度100以下线性增长，超过100后增长变缓
            # 最大奖励限制在1.0左右
            if length <= 100:
                reward = length / 100.0  # 线性增长，最大0.5
            else:
                # 对数增长，但有上限
                reward = 0.5 + 0.5 * min(1.0, math.log(length/100.0 + 1) / math.log(10))
                
            rewards.append(min(1.0, reward))  # 确保奖励不超过1.0
        else:
            rewards.append(0.0)  # 如果没有思考部分，奖励为0
    
    return rewards


if __name__ == '__main__':
    model_name = "/root/autodl-tmp/deepseek-r1/models/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,  # Qwen模型需要这个参数
        torch_dtype=torch.bfloat16  # 使用bf16格式减少内存使用
    )
#     # 使用lora方法训练
#     lora_config = LoraConfig(
#     r=8,  # 降低rank可以减少参数量
#     lora_alpha=16,  # 可以调小以减少内存使用
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.05, 
#     task_type=TaskType.CAUSAL_LM
# )
#     # 使用lora方法训练
#     model = get_peft_model(model, lora_config)
    model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ds = load_dataset('/root/autodl-tmp/deepseek-r1/dataset/gsm8k')
    data = process_data(ds['train'])
    
    output_dir="output"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=3e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=10,
        bf16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=500,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )
    
    # 打印实际的参数值以验证
    print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"num_generations: {training_args.num_generations}")
    print(f"Global batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        mark_reward,
        soft_format_reward,
        hard_format_reward,
        digit_reward,
        length_reward,
        correctness_reward
        ],
    args=training_args,
    train_dataset=data,

)
    trainer.train()
    trainer.save_model(output_dir)
