#!/usr/bin/env python3
"""
将ChatGPT生成的JSONL文件整合并转换为BIO格式
"""

import json
import os
import re
from typing import List, Dict, Tuple


def is_punctuation(token: str) -> bool:
    """判断token是否为标点符号"""
    return re.fullmatch(r'[^\w\s]', token) is not None


def load_jsonl_files(directory: str) -> List[Dict]:
    """加载指定目录下的所有JSONL文件"""
    all_data = []
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件:")
    for filename in sorted(jsonl_files):
        print(f"  - {filename}")
    
    for filename in sorted(jsonl_files):
        filepath = os.path.join(directory, filename)
        print(f"\n正在处理: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            file_data = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        file_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"  警告: 第{line_num}行JSON解析错误: {e}")
            
            print(f"  加载了 {len(file_data)} 条记录")
            all_data.extend(file_data)
    
    print(f"\n总共加载了 {len(all_data)} 条记录")
    return all_data


def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    对文本进行分词，返回(token, start_pos, end_pos)的列表
    使用简单的空格和标点分词
    """
    tokens = []
    # 使用正则表达式分词，保留标点符号
    pattern = r'\w+|[^\w\s]'
    
    for match in re.finditer(pattern, text):
        token = match.group()
        start_pos = match.start()
        end_pos = match.end()
        tokens.append((token, start_pos, end_pos))
    
    return tokens


def convert_to_bio(text: str, spans: List[Dict]) -> List[Tuple[str, str]]:
    """
    将文本和标注信息转换为BIO格式
    
    Args:
        text: 原文本
        spans: 标注信息列表，每个包含start, end, label
    
    Returns:
        (token, label)对的列表
    """
    # 分词
    tokens = tokenize_text(text)
    
    # 初始化所有token的标签为'O'
    token_labels = ['O'] * len(tokens)
    
    # 为每个span分配标签
    for span in spans:
        start_char = span['start']
        end_char = span['end']
        label = span['label']
        
        # 找到与span重叠的tokens
        first_token_idx = None
        last_token_idx = None
        
        for i, (token, token_start, token_end) in enumerate(tokens):
            # 检查token是否与span有重叠
            if token_end > start_char and token_start < end_char:
                if first_token_idx is None:
                    first_token_idx = i
                last_token_idx = i
        
        # 分配BIO标签
        if first_token_idx is not None:
            # 第一个token用B-标签
            token_labels[first_token_idx] = f'B-{label}'
            # 后续token用I-标签
            for i in range(first_token_idx + 1, last_token_idx + 1):
                token_labels[i] = f'I-{label}'
    
    # 返回(token, label)对，但过滤掉标点符号
    result = []
    for (token, _, _), label in zip(tokens, token_labels):
        if not is_punctuation(token):  # 只保留非标点符号的token
            result.append((token, label))
    
    return result


def save_bio_format(data: List[Dict], output_file: str):
    """将数据保存为BIO格式"""
    print(f"\n正在转换为BIO格式并保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_sentences = len(data)
        total_tokens = 0
        
        for i, item in enumerate(data, 1):
            if i % 100 == 0 or i == total_sentences:
                print(f"  处理进度: {i}/{total_sentences}")
            
            text = item['text']
            spans = item.get('spans', [])
            
            # 转换为BIO格式
            bio_tokens = convert_to_bio(text, spans)
            
            # 写入文件
            for token, label in bio_tokens:
                f.write(f"{token} {label}\n")
                total_tokens += 1
            
            # 句子间用空行分隔
            f.write("\n")
        
        print(f"  转换完成: {total_sentences} 个句子, {total_tokens} 个token")


def main():
    """主函数"""
    # 设置路径
    input_dir = "silver_data/Chatgpt/JSONL"
    output_file = "chatgpt_integrated_bio_no_punctuation.txt"
    
    print("=" * 60)
    print("ChatGPT JSONL文件整合和BIO格式转换工具")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 加载所有JSONL文件
    all_data = load_jsonl_files(input_dir)
    
    if not all_data:
        print("错误: 没有找到有效的数据")
        return
    
    # 转换为BIO格式并保存
    save_bio_format(all_data, output_file)
    
    print(f"\n处理完成! 结果已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n统计信息:")
    print(f"  总记录数: {len(all_data)}")
    
    # 统计标签类型
    label_counts = {}
    for item in all_data:
        for span in item.get('spans', []):
            label = span['label']
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  标签类型: {len(label_counts)}")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")


if __name__ == "__main__":
    main()