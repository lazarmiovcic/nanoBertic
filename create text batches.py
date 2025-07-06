import os
import tqdm
import math

# --- Helper functions (add these if you don't have them) ---
def count_lines(file_path):
    """Counts the number of non-empty lines in a file."""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # Only count non-empty lines, consistent with your buffer logic
                count += 1
    return count


def count_batches(total_lines, batch_size):
    """Calculates the total number of batches."""
    return math.ceil(total_lines / batch_size)


def split_text_into_batches(input_file_path, output_base_dir, batch_size=10000,
                            tr_split=0.8, val_split=0.1, test_split=0.1):
    # Ensure splits sum up correctly
    if not (0 <= tr_split <= 1 and 0 <= val_split <= 1 and (tr_split + val_split) <= 1):
        raise ValueError("tr_split and val_split must be between 0 and 1, and their sum must be <= 1.")

    test_split = 1.0 - tr_split - val_split
    if test_split < 0:  # Handle cases where tr_split + val_split > 1
        test_split = 0

    print(f"Counting lines in '{input_file_path}'...")
    total_lines = count_lines(input_file_path)
    total_batches = count_batches(total_lines, batch_size)
    print(f"Total lines: {total_lines}, Total estimated batches: {total_batches}")

    num_tr_batches = math.floor(total_batches * tr_split)
    num_val_batches = math.ceil(total_batches * val_split)
    num_test_batches = total_batches - num_tr_batches - num_val_batches
    print(f"Distribution: Train batches: {num_tr_batches}, Val batches: {num_val_batches}, Test batches: {num_test_batches}")

    train_dir = os.path.join(output_base_dir, 'train')
    val_dir = os.path.join(output_base_dir, 'val')
    test_dir = os.path.join(output_base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    text_data_buffer = []
    file_count = 0
    tr_file_count = 0
    val_file_count = 0
    test_file_count = 0

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm.tqdm(infile, desc=f"Preparing data from '{input_file_path}'"):
            cleaned_line = line.strip()
            if cleaned_line: # Only add non-empty lines to the buffer
                text_data_buffer.append(cleaned_line)

            # Check if the buffer has reached the desired batch size
            if len(text_data_buffer) >= batch_size:
                file_count += 1
                if file_count <= num_tr_batches:
                    tr_file_count += 1
                    output_file_path = os.path.join(train_dir, f'train_batch_{tr_file_count}.txt')
                elif num_tr_batches < file_count <= num_tr_batches + num_val_batches:
                    val_file_count += 1
                    output_file_path = os.path.join(val_dir, f'val_batch_{val_file_count}.txt')
                else:
                    test_file_count += 1
                    output_file_path = os.path.join(test_dir, f'test_batch_{test_file_count}.txt')

                # Write all accumulated lines to a new file, separated by newlines
                with open(output_file_path, 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data_buffer))

                text_data_buffer = [] # Clear the buffer for the next batch

    # This block handles any remaining lines in the buffer after the loop
    if text_data_buffer:
        file_count += 1
        test_file_count += 1
        with open(os.path.join(test_dir, f'test_batch_{file_count}.txt'), 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data_buffer))

    print(f"\nData preparation complete. Saved {file_count} files to '{output_base_dir}'.")
    print(f"Breakdown: Train: {tr_file_count} batches, Val: {val_file_count} batches, Test: {test_file_count} batches.")


input_file_path = "classla-sr-dir/classla-sr"
output_base_dir = "all_text_batches"

batch_size = 10000

train_split = 0.8
val_split = 0.1
test_split = 0.1

split_text_into_batches(input_file_path, output_base_dir, batch_size, train_split, val_split)
