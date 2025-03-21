import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="output", checkpoint=None):
    """加载模型和分词器"""
    if checkpoint:
        full_path = f"{model_path}/checkpoint-{checkpoint}"
        print(f"正在加载检查点: {full_path}")
    else:
        full_path = model_path
        print(f"正在加载最终模型: {full_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    model = AutoModelForCausalLM.from_pretrained(
        full_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    print("模型加载完成!")
    return model, tokenizer

def clean_model_output(output_text):
    """增强版清理模型输出函数，提取完整答案句子"""
    # 1. 查找多种可能的干扰内容起始点
    cutoff_patterns = [
        r'Human:', r'Human：', r'Human\b',  # 标准对话格式及单独的Human单词
        r'Human beings', r'People',  # 常见转移到非数学内容的起点
        r'In conclusion', r'To summarize',  # 结束后可能开始的新话题
        r'Let me', r'I hope',  # 模型自我指代常用语
        r'Now let\'s', r'Let\'s now'  # 可能引入新话题的过渡短语
    ]
    
    # 查找最早出现的干扰内容
    earliest_pos = len(output_text)
    for pattern in cutoff_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    
    # 截取到干扰内容之前
    if earliest_pos < len(output_text):
        cleaned_text = output_text[:earliest_pos].strip()
    else:
        cleaned_text = output_text.strip()
    
    # 2. 寻找结论句子作为答案
    conclusion_patterns = [
        r'(因此[，,].*?[.。])',
        r'(所以[，,].*?[.。])',
        r'(答案[是为:：].*?[.。])',
        r'(结果[是为:：].*?[.。])',
        r'(计算得.*?[.。])',
        r'(结论[是为:：].*?[.。])',
        r'(最终.*?等于.*?[.。])',
        r'(综上所述[，,].*?[.。])'
    ]
    
    answer = ""
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, cleaned_text, re.DOTALL)
        if matches:
            # 使用最后一个匹配的结论句子
            answer = matches[-1].strip()
            break
    
    # 3. 如果没找到结论句子，尝试找含有数字的最后一句话
    if not answer:
        # 按句子分割
        sentences = re.split(r'[.。!！?？]', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            # 从最后一句往前找，直到找到包含数字的句子
            for sentence in reversed(sentences):
                if re.search(r'\d+', sentence):
                    answer = sentence.strip()
                    break
    
    # 4. 格式化最终输出
    if answer:
        # 格式化输出
        formatted_output = f"<think>\n{cleaned_text}\n</think>\n\n<answer>\n{answer}\n</answer>"
    else:
        # 如果没有找到明确的答案
        formatted_output = f"<think>\n{cleaned_text}\n</think>\n\n<answer>\n不能确定答案\n</answer>"
    
    return formatted_output

def get_clean_output(question, model, tokenizer, max_tokens=200):
    """获取清理后的模型输出"""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 只解码模型生成的部分
    input_length = inputs.input_ids.shape[1]
    raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # 清理输出
    cleaned_output = clean_model_output(raw_output)
    
    return cleaned_output

# 其余函数保持不变

def interactive_mode():
    """交互式问答模式"""
    model, tokenizer = load_model()
    
    print("\n===== 清理后的模型输出 =====")
    print("输入'exit'或'quit'退出")
    
    while True:
        question = input("\n请输入问题: ")
        
        if question.lower() in ['exit', 'quit']:
            print("再见!")
            break
            
        if not question.strip():
            continue
            
        try:
            cleaned_output = get_clean_output(question, model, tokenizer)
            print("\n--- 清理后的输出 ---")
            print(cleaned_output)
            
        except Exception as e:
            print(f"发生错误: {e}")

def test_examples():
    """测试示例问题"""
    model, tokenizer = load_model()
    
    example_questions = [
        "如果一个苹果重150克，一个香蕉重120克，那么3个苹果和2个香蕉一共重多少克?",
        "小明有25颗糖果，他给了小红8颗，又给了小刚5颗，然后他妈妈又给了他10颗，他现在有多少颗糖果?",
        "一个长方形的长是8米，宽是6米，求它的面积。"
    ]
    
    for i, question in enumerate(example_questions):
        print(f"\n\n===== 示例 {i+1} =====")
        print(f"问题: {question}")
        
        try:
            cleaned_output = get_clean_output(question, model, tokenizer)
            print("\n--- 清理后的输出 ---")
            print(cleaned_output)
            print("-" * 50)
            
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    print("选择模式:")
    print("1. 交互式模式")
    print("2. 测试示例问题")
    print("3. 加载特定检查点")
    
    choice = input("选择 (1/2/3): ")
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        test_examples()
    elif choice == "3":
        checkpoint = input("输入检查点编号 (例如: 500, 1000, 1500, 1868): ")
        model, tokenizer = load_model(checkpoint=checkpoint)
        
        print("\n===== 清理后的模型输出 =====")
        print("输入'exit'或'quit'退出")
        
        while True:
            question = input("\n请输入问题: ")
            
            if question.lower() in ['exit', 'quit']:
                print("再见!")
                break
                
            if not question.strip():
                continue
                
            try:
                cleaned_output = get_clean_output(question, model, tokenizer)
                print("\n--- 清理后的输出 ---")
                print(cleaned_output)
                
            except Exception as e:
                print(f"发生错误: {e}")
    else:
        print("无效选择")