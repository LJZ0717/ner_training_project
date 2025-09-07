import json
import os
import re
from typing import List, Dict, Tuple


def is_punctuation(token: str) -> bool:
    """Determine whether the token is a punctuation mark"""
    return re.fullmatch(r'[^\w\s]', token) is not None


def load_jsonl_files(directory: str) -> List[Dict]:
    """Load all JSONL files in the specified directory"""
    all_data = []
    jsonl_files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    
    print(f"find {len(jsonl_files)} JSONL documents:")
    for filename in sorted(jsonl_files):
        print(f"  - {filename}")
    
    for filename in sorted(jsonl_files):
        filepath = os.path.join(directory, filename)
        print(f"\nProcessing: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            file_data = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        file_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: JSON parsing error at line {line_num}: {e}")
            
            print(f"  Warning: JSON parsing error at line {line_num} {len(file_data)} 条记录")
            all_data.extend(file_data)
    
    print(f"\nTotal loaded {len(all_data)} Records")
    return all_data


def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
   Tokenizes the text and returns a list of (token, start_pos, end_pos)
Tokenizes using simple spaces and punctuation
    """
    tokens = []
    # Use regular expressions to segment words and retain punctuation
    pattern = r'\w+|[^\w\s]'
    
    for match in re.finditer(pattern, text):
        token = match.group()
        start_pos = match.start()
        end_pos = match.end()
        tokens.append((token, start_pos, end_pos))
    
    return tokens


def convert_to_bio(text: str, spans: List[Dict]) -> List[Tuple[str, str]]:
    """
    Convert text and annotation information into BIO format
    
    Args:
        text: original text
        spans: a list of annotation information, each containing start, end, label
    
    Returns:
        A list of (token, label) pairs
    """
    
    tokens = tokenize_text(text)
    
    # Initialize all token labels to 'O'
    token_labels = ['O'] * len(tokens)
    
    # Assign a label to each span
    for span in spans:
        start_char = span['start']
        end_char = span['end']
        label = span['label']
        
        # Find tokens that overlap with span
        first_token_idx = None
        last_token_idx = None
        
        for i, (token, token_start, token_end) in enumerate(tokens):
            # Check if token overlaps with span
            if token_end > start_char and token_start < end_char:
                if first_token_idx is None:
                    first_token_idx = i
                last_token_idx = i
        
        # Assign BIO labels
        if first_token_idx is not None:
            # The first token is labeled with B-
            token_labels[first_token_idx] = f'B-{label}'
            # Subsequent tokens use I-tags
            for i in range(first_token_idx + 1, last_token_idx + 1):
                token_labels[i] = f'I-{label}'
    
    # Return (token, label) pairs, but filter out punctuation
    result = []
    for (token, _, _), label in zip(tokens, token_labels):
        if not is_punctuation(token): # Only keep non-punctuation tokens
            result.append((token, label))
    
    return result


def save_bio_format(data: List[Dict], output_file: str):
    """Save data in BIO format"""
    print(f"\nConverting to BIO format and saving to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_sentences = len(data)
        total_tokens = 0
        
        for i, item in enumerate(data, 1):
            if i % 100 == 0 or i == total_sentences:
                print(f"  Processing progress: {i}/{total_sentences}")
            
            text = item['text']
            spans = item.get('spans', [])
            
            # Convert to BIO format
            bio_tokens = convert_to_bio(text, spans)
            
            # Write to file
            for token, label in bio_tokens:
                f.write(f"{token} {label}\n")
                total_tokens += 1
            
            
            f.write("\n")
        
        print(f"  Conversion Completed: {total_sentences} sentences, {total_tokens} token")


def main():
    """main"""
    
    input_dir = "silver_data/Chatgpt/JSONL"
    output_file = "chatgpt_integrated_bio_no_punctuation.txt"
    
    print("=" * 60)
    print("ChatGPT JSONL file integration and BIO format conversion tool")
    print("=" * 60)
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
   # Load all JSONL files
    all_data = load_jsonl_files(input_dir)
    
    if not all_data:
        print("Error: No valid data found")
        return
    
    # Convert to BIO format and save
    save_bio_format(all_data, output_file)
    
    print(f"\nProcessing completed! Results saved to: {output_file}")
    
    # Display some statistics
    print("\nStatistics:")
    print(f"  Total number of records: {len(all_data)}")
    
    # Statistics Tag Type
    label_counts = {}
    for item in all_data:
        for span in item.get('spans', []):
            label = span['label']
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  Tag Type: {len(label_counts)}")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")


if __name__ == "__main__":
    main()