import os
from pathlib import Path
import tqdm
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


# paths = [str(x) for x in Path('./text_batches').glob('**/*.txt')]
#
# tokenizer = BertWordPieceTokenizer(
#     clean_text=True,
#     handle_chinese_chars=False,
#     strip_accents=False,
#     lowercase=True
# )
#
# print("Starting tokenizer training...")
# tokenizer.train(
#     files=paths,
#     vocab_size=30_000,
#     min_frequency=5,
#     limit_alphabet=1000,
#     wordpieces_prefix='##',
#     special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
# )
# print("Tokenizer training complete.")
#
# os.makedirs('./bert-it-1', exist_ok=True)
# tokenizer.save_model('./bert-it-1', 'bert-it')

