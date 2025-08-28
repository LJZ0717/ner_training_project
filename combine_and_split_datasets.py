#!/usr/bin/env python3
"""
合并Silver和Gold数据集，并按8:2划分训练集和测试集
"""

import random
import os
from typing import List, Tuple


def read_bio_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """
    读取BIO格式文件，返回句子列表
    每个句子是(token, label)对的列表
    """
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    token, label = parts
                    current_sentence.append((token, label))
        
        # 处理文件末尾的句子
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def normalize_labels(sentences: List[List[Tuple[str, str]]], label_mapping: dict = None) -> List[List[Tuple[str, str]]]:
    """
    标准化标签格式
    """
    if label_mapping is None:
        # 默认将HPO_term标准化为HPO_TERM
        label_mapping = {
            'B-HPO_term': 'B-HPO_TERM',
            'I-HPO_term': 'I-HPO_TERM'
        }
    
    normalized_sentences = []
    for sentence in sentences:
        normalized_sentence = []
        for token, label in sentence:
            # 标准化标签
            normalized_label = label_mapping.get(label, label)
            normalized_sentence.append((token, normalized_label))
        normalized_sentences.append(normalized_sentence)
    
    return normalized_sentences


def save_bio_file(sentences: List[List[Tuple[str, str]]], filepath: str):
    """
    保存BIO格式文件
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            
            # 句子间用空行分隔（最后一个句子除外）
            if i < len(sentences) - 1:
                f.write("\n")


def split_dataset(sentences: List[List[Tuple[str, str]]], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List, List]:
    """
    按指定比例划分数据集
    """
    # 设置随机种子确保可重现
    random.seed(random_seed)
    
    # 打乱数据
    sentences_copy = sentences.copy()
    random.shuffle(sentences_copy)
    
    # 计算划分点
    total_sentences = len(sentences_copy)
    train_size = int(total_sentences * train_ratio)
    
    train_sentences = sentences_copy[:train_size]
    test_sentences = sentences_copy[train_size:]
    
    return train_sentences, test_sentences


def print_statistics(sentences: List[List[Tuple[str, str]]], dataset_name: str):
    """
    打印数据集统计信息
    """
    total_sentences = len(sentences)
    total_tokens = sum(len(sentence) for sentence in sentences)
    
    # 统计标签
    label_counts = {}
    for sentence in sentences:
        for token, label in sentence:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n=== {dataset_name} 统计信息 ===")
    print(f"句子数: {total_sentences}")
    print(f"Token数: {total_tokens}")
    print("标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据集合并和划分工具")
    print("=" * 60)
    
    # 文件路径
    gold_file = "gold_data/bio_dataset_cleaned.txt"
    silver_file = "chatgpt_integrated_bio_no_punctuation.txt"
    
    # 输出文件路径
    combined_file = "combined_dataset.txt"
    train_file = "train_dataset.txt"
    test_file = "test_dataset.txt"
    
    # 检查输入文件
    if not os.path.exists(gold_file):
        print(f"错误: Gold数据文件不存在: {gold_file}")
        return
    
    if not os.path.exists(silver_file):
        print(f"错误: Silver数据文件不存在: {silver_file}")
        return
    
    # 读取数据集
    print(f"\n正在读取Gold数据: {gold_file}")
    gold_sentences = read_bio_file(gold_file)
    print_statistics(gold_sentences, "Gold数据")
    
    print(f"\n正在读取Silver数据: {silver_file}")
    silver_sentences = read_bio_file(silver_file)
    print_statistics(silver_sentences, "Silver数据（原始）")
    
    # 标准化Silver数据的标签
    print("\n正在标准化Silver数据标签...")
    silver_sentences = normalize_labels(silver_sentences)
    print_statistics(silver_sentences, "Silver数据（标准化后）")
    
    # 合并数据集
    print("\n正在合并数据集...")
    combined_sentences = gold_sentences + silver_sentences
    print_statistics(combined_sentences, "合并数据集")
    
    # 保存合并数据集
    print(f"\n保存合并数据集到: {combined_file}")
    save_bio_file(combined_sentences, combined_file)
    
    # 划分训练集和测试集
    print("\n正在划分训练集和测试集 (8:2)...")
    train_sentences, test_sentences = split_dataset(combined_sentences, train_ratio=0.8)
    
    print_statistics(train_sentences, "训练集")
    print_statistics(test_sentences, "测试集")
    
    # 保存训练集和测试集
    print(f"\n保存训练集到: {train_file}")
    save_bio_file(train_sentences, train_file)
    
    print(f"保存测试集到: {test_file}")
    save_bio_file(test_sentences, test_file)
    
    print("\n" + "=" * 60)
    print("数据集处理完成!")
    print("=" * 60)
    
    # 最终统计
    print(f"\n最终文件:")
    print(f"  合并数据集: {combined_file}")
    print(f"  训练集: {train_file} ({len(train_sentences)} 句子)")
    print(f"  测试集: {test_file} ({len(test_sentences)} 句子)")


if __name__ == "__main__":
    main()