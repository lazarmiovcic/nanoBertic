from transformers import BertTokenizerFast
import os

def read_first_n_lines(file_path, n):
    """Reads the first n lines from a file using a loop.

    Args:
        file_path (str): The path to the file.
        n (int): The number of lines to read from the beginning.

    Returns:
        str: A single string containing the first n lines joined together,
             or an empty string if the file is empty or n is 0.
    """
    result = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(n):
                line = f.readline()
                if not line:  # Reached end of file
                    break
                result += line
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    return result

# Example usage:
n = 100
first_few_lines = read_first_n_lines('classla-sr-dir/classla-sr', n)
print(first_few_lines)


# Load the pre-trained multilingual BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')


tokens = tokenizer(first_few_lines, return_attention_mask=False)['input_ids']

print(f"Number of tokens: {len(tokens)}")

print(f"First 10 tokens: {tokens}")

print(f"First 10 token strings: {tokenizer.convert_ids_to_tokens(tokens)}")