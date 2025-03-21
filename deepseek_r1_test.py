import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from tqdm import tqdm

# 定义答案提取函数(与训练时相同)
def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# 加载测试数据集
test_data = load_dataset('parquet', data_files={'test': 'test-00000-of-00001.parquet'})['test']

# 数据预处理
def process_test_data(data):
    processed = []
    for item in data:
        processed.append({
            'question': item['question'],
            'answer': item['answer'].split('####')[-1].strip()  # 提取数字答案
        })
    return processed

# 加载训练好的模型和分词器
model_path = "output"  # 您保存模型的路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.cuda()
model.eval()  # 设置为评估模式

# 系统提示
SYSTEM_PROMPT = """
Please generate your response in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

# 评估函数
def evaluate_model(model, tokenizer, test_data, num_samples=None):
    results = []
    correct = 0
    total = 0
    
    # 如果指定了样本数，则只评估部分测试集
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    for item in tqdm(test_data):
        total += 1
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': item['question']}
        ]
        
        # 将提示转换为模型输入格式
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=False  # 使用贪婪解码以获得确定性结果
            )
        
        # 解码生成的回答
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # 提取答案
        extracted_answer = extract_answer(generated_text)
        
        # 检查是否正确
        is_correct = extracted_answer == item['answer']
        if is_correct:
            correct += 1
        
        # 保存结果
        results.append({
            'question': item['question'],
            'reference_answer': item['answer'],
            'generated_text': generated_text,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct
        })
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'detailed_results': results
    }

# 主评估流程
test_data_processed = process_test_data(test_data)
eval_results = evaluate_model(model, tokenizer, test_data_processed)

# 打印评估结果
print(f"准确率: {eval_results['accuracy']:.4f} ({eval_results['correct']}/{eval_results['total']})")

# 保存详细结果
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

# 错误分析 - 打印一些错误的例子
print("\n错误案例分析:")
errors = [r for r in eval_results['detailed_results'] if not r['is_correct']]
for i, error in enumerate(errors[:5]):  # 展示前5个错误
    print(f"\n错误 #{i+1}:")
    print(f"问题: {error['question']}")
    print(f"正确答案: {error['reference_answer']}")
    print(f"模型答案: {error['extracted_answer']}")