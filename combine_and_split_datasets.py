import random
import os
from typing import List, Tuple


def read_bio_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """
    Read a BIO format file and return a list of sentences. Each sentence is a list of (token, label) pairs.
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
        
        # Process the sentence at the end of the file
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def normalize_labels(sentences: List[List[Tuple[str, str]]], label_mapping: dict = None) -> List[List[Tuple[str, str]]]:
    """
    标准化标签格式
    """
    if label_mapping is None:

        label_mapping = {
            'B-HPO_term': 'B-HPO_TERM',
            'I-HPO_term': 'I-HPO_TERM'
        }
    
    normalized_sentences = []
    for sentence in sentences:
        normalized_sentence = []
        for token, label in sentence:
            # Standardized labels
            normalized_label = label_mapping.get(label, label)
            normalized_sentence.append((token, normalized_label))
        normalized_sentences.append(normalized_sentence)
    
    return normalized_sentences


def save_bio_file(sentences: List[List[Tuple[str, str]]], filepath: str):
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            
            
            if i < len(sentences) - 1:
                f.write("\n")


def split_dataset(sentences: List[List[Tuple[str, str]]], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List, List]:
    
    random.seed(random_seed)
    
    # Shuffle the data
    sentences_copy = sentences.copy()
    random.shuffle(sentences_copy)
    
    # Calculate partition points
    total_sentences = len(sentences_copy)
    train_size = int(total_sentences * train_ratio)
    
    train_sentences = sentences_copy[:train_size]
    test_sentences = sentences_copy[train_size:]
    
    return train_sentences, test_sentences


def print_statistics(sentences: List[List[Tuple[str, str]]], dataset_name: str):
    
    total_sentences = len(sentences)
    total_tokens = sum(len(sentence) for sentence in sentences)
    
    # Statistics Tags
    label_counts = {}
    for sentence in sentences:
        for token, label in sentence:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n=== {dataset_name} Statistics ===")
    print(f"Number of sentences: {total_sentences}")
    print(f"Number of Tokens: {total_tokens}")
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


def main():
    print("=" * 60)
    print("Dataset merging and partitioning tools")
    print("=" * 60)
    
    gold_file = "gold_data/bio_dataset_cleaned.txt"
    silver_file = "chatgpt_integrated_bio_no_punctuation.txt"
    
    combined_file = "combined_dataset.txt"
    train_file = "train_dataset.txt"
    test_file = "test_dataset.txt"
    
    if not os.path.exists(gold_file):
        print(f"Error: Gold data file does not exist: {gold_file}")
        return
    
    if not os.path.exists(silver_file):
        print(f"Error: Silver data file does not exist: {silver_file}")
        return
    
   
    print(f"\nReading Gold data: {gold_file}")
    gold_sentences = read_bio_file(gold_file)
    print_statistics(gold_sentences, "Gold Data")
    
    print(f"\nReading Silver data: {silver_file}")
    silver_sentences = read_bio_file(silver_file)
    print_statistics(silver_sentences, "Silver data (raw)")
    
    # Standardize labels for Silver data
    print("\nStandardizing Silver data labels...")
    silver_sentences = normalize_labels(silver_sentences)
    print_statistics(silver_sentences, "Silver data (normalized)")
    
    # Merge datasets
    print("\nMerging datasets")
    combined_sentences = gold_sentences + silver_sentences
    print_statistics(combined_sentences, "Merge datasets")
    
    # Save the merged dataset
    print(f"\nSave the merged dataset to: {combined_file}")
    save_bio_file(combined_sentences, combined_file)
    
    # Divide the training set and test set
    print("\nSplitting training and testing sets (8:2)")
    train_sentences, test_sentences = split_dataset(combined_sentences, train_ratio=0.8)
    
    print_statistics(train_sentences, "training set")
    print_statistics(test_sentences, "Test set")
    
    # Save the training and test sets
    print(f"\nSave the training set to: {train_file}")
    save_bio_file(train_sentences, train_file)
    
    print(f"Save test set to: {test_file}")
    save_bio_file(test_sentences, test_file)
    
    print("\n" + "=" * 60)
    print("Dataset processing completed")
    print("=" * 60)
    
    # 最终统计
    print(f"\nFinal Document:")
    print(f"  Merge datasets: {combined_file}")
    print(f"  Training set: {train_file} ({len(train_sentences)} sentence)")
    print(f"  Test set: {test_file} ({len(test_sentences)} sentence)")


if __name__ == "__main__":
    main()