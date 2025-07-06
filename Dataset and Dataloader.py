import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

print("Tokenizer loaded successfully!")
print(f"Vocab size: {len(tokenizer)}")

print(tokenizer("ovo je rečenica"))

class BertDataset_v1(Dataset):
    def __init__(self, path_to_data_dir, tokenizer, seq_len,):
        self.paths = [str(x) for x in Path(path_to_data_dir).glob('**/*.txt')] # Use path_to_data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = self.tokenizer.vocab['[PAD]'] # Store PAD ID for convenience

        self.total_samples = 0
        self.file_line_counts = [] # Stores (file_path, num_lines_in_file)
        self.cumulative_line_offsets = [0] # Stores cumulative sums of lines for quick lookup

        print("\n[Dataset Init] Counting lines across all batch files...")
        for p in self.paths:
            with open(p, 'r', encoding='utf-8') as f:
                num_lines_in_file = sum(1 for _ in f) # Efficiently count lines
            self.file_line_counts.append((p, num_lines_in_file))
            self.total_samples += num_lines_in_file
            self.cumulative_line_offsets.append(self.total_samples) # Add cumulative sum

        if self.total_samples == 0:
            raise ValueError(f"No lines found in any files in {path_to_data_dir}. Check your data.")
        print(f"[Dataset Init] Found {self.total_samples} total samples across {len(self.paths)} files.")


    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        # Handle index out of bounds (shouldn't happen if __len__ is correct, but good for robustness)
        if not (0 <= index < self.total_samples):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {self.total_samples}")

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1_str, t2_str, is_next_label = self.get_sent(index)

        # Tokenize and get input IDs for t1 and t2 (after stripping)
        # Note: self.tokenizer('string')['input_ids'][1:-1]
        # This will remove [CLS] and [SEP] added by the tokenizer for single sentences.
        t1_token_ids = self.tokenizer(t1_str)['input_ids'][1:-1]
        t2_token_ids = self.tokenizer(t2_str)['input_ids'][1:-1]

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1_token_ids)  # Pass token_ids, not string
        t2_random, t2_label = self.random_word(t2_token_ids)  # Pass token_ids, not string

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        cls_id = self.tokenizer.vocab['[CLS]']
        sep_id = self.tokenizer.vocab['[SEP]']
        pad_id = self.tokenizer.vocab['[PAD]']  # Use stored pad_id

        t1_final = [cls_id] + t1_random + [sep_id]
        t2_final = t2_random + [sep_id]  # T2 does not start with CLS in BERT

        # Labels for MLM must correspond to the final input sequence.
        # Pad t1_label and t2_label with PAD_ID for tokens that are not masked.
        t1_label_final = [pad_id] + t1_label + [pad_id]
        t2_label_final = t2_label + [pad_id]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        # The segment label for t1 is 0, for t2 is 1.
        segment_label = [1 for _ in range(len(t1_final))] + [2 for _ in range(len(t2_final))]

        bert_input = (t1_final + t2_final)[:self.seq_len]
        bert_label = (t1_label_final + t2_label_final)[:self.seq_len]
        segment_label = segment_label[:self.seq_len]  # Ensure segment_label is also truncated

        # Calculate padding needed
        padding_length = self.seq_len - len(bert_input)
        if padding_length < 0:  # Should not happen with [:self.seq_len] but as a safeguard
            padding_length = 0

        padding_list = [pad_id for _ in range(padding_length)]

        bert_input.extend(padding_list)
        bert_label.extend(padding_list)
        segment_label.extend(padding_list)

        output = {"input_ids": bert_input,
                  "input_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, token_ids):  # Now expects list of token_ids
        output = []
        output_label = []
        for tok_id in token_ids:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:  # 80% chance change token to mask token
                    output.append(self.tokenizer.vocab['[MASK]'])
                elif prob < 0.9:  # 10% chance change token to random token
                    output.append(random.randrange(len(self.tokenizer.vocab)))
                else:  # 10% chance change token to current token
                    output.append(tok_id)
                output_label.append(tok_id)  # Original token ID is the label for these 15%
            else:  # 85% chance: token left unchanged
                output.append(tok_id)
                output_label.append(self.pad_token_id)  # Label is PAD for unmasked tokens (don't predict)
        return output, output_label

    def get_file_lines(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            return f.readlines()

    def get_sent(self, index):
        # Find which file and line number within that file the index corresponds to
        # Uses cumulative_line_offsets to efficiently find the file
        file_idx = 0
        for i in range(len(self.cumulative_line_offsets) - 1):
            if self.cumulative_line_offsets[i] <= index < self.cumulative_line_offsets[i+1]:
                file_idx = i
                break
        line_idx_in_file = index - self.cumulative_line_offsets[file_idx]

        lines = self.get_file_lines(self.paths[file_idx])
        # Get t1
        t1_str = lines[line_idx_in_file]

        prob = random.random()
        if prob > 0.5: # Positive Pair
            # Try to get the next line from the same file
            if line_idx_in_file < self.file_line_counts[file_idx][1] - 1: # [1] is num_lines_in_file
                t2_str = lines[line_idx_in_file+1]
                return t1_str, t2_str, 1
            else: # Last line of current file, try to get from next file
                if file_idx < len(self.paths) - 1: # Check if there's a next file
                    with open(self.paths[file_idx+1], 'r', encoding='utf-8') as next_file:
                        t2_str = next_file.readline()
                    return t1_str, t2_str, 1
                else: # Last line of the entire corpus
                    random_line_from_file = lines[random.randrange(len(lines))]
                    return t1_str, random_line_from_file, 0 # Now it's a negative pair

        else: # Negative Pair
            random_line_from_file = lines[random.randrange(len(lines))]
            return t1_str, random_line_from_file, 0


# # --- Instantiate and use the Dataset ---
# # Ensure './text_batches' is the correct path to your batch files
# train_data = BertDataset_v1('text_batches', tokenizer, 64)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
#
# # Test getting a sample batch
# print("\n--- Testing DataLoader (first batch) ---")
# try:
#     sample_data = next(iter(train_loader))
#     print("Successfully loaded a batch of data.")
#     for key, value in sample_data.items():
#         print(f"  {key}: shape={value.shape}, type={value.dtype}")
# except Exception as e:
#     print(f"Error loading a sample batch: {e}")
#
# # Test getting a random single sample
# print("\n--- Testing Dataset (random sample) ---")
# try:
#     random_sample = train_data[random.randrange(len(train_data))]
#     print("Successfully loaded a random single sample.")
#     for key, value in random_sample.items():
#         print(f"  {key}: {value} ")
#     # Decode a part of the sample to verify content
#     decoded_bert_input = tokenizer.decode(random_sample['bert_input'].tolist())
#     print(f"  Decoded bert_input: {decoded_bert_input}")
# except Exception as e:
#     print(f"Error loading a random sample: {e}")
