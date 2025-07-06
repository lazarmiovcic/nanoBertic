import numpy as np
import torch

import math
from torch import nn
from torch.optim import Adam
import tqdm

n_embd = 64
max_len = 128
batch_size = 32
n_layers = 2
n_heads = 2
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self):
        return self.pe


class BERTEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size, seq_len, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=0)    # (seq_len, d_model)
        self.segment = nn.Embedding(3, d_model, padding_idx=0)           # (seq_len, d_model)
        self.position = PositionEmbedding(d_model, seq_len)              # (seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, segment_label):
        x = self.token(ids) + self.segment(segment_label) + self.position()
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_size = d_model // n_heads

        self.key = nn.Linear(d_model, d_model)    # (d_model, n_heads * head_size)
        self.query = nn.Linear(d_model, d_model)  # (d_model, n_heads * head_size)
        self.value = nn.Linear(d_model, d_model)  # (d_model, n_heads * head_size)
        self.proj = nn.Linear(d_model, d_model)   # (d_model, n_heads * head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # mask.shape = (batch_size, 1, 1, max_len)

        B, T, D = x.shape  # B - batch size, T - max_len, D - n_embd

        k = self.key(x).view(B, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)    # (B, n_heads, T, head_size)
        q = self.query(x).view(B, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)  # (B, n_heads, T, head_size)
        v = self.value(x).view(B, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)  # (B, n_heads, T, head_size)

        # (B, n_heads, T, head_size) @ (B, n_heads, head_size, T) -> (B, n_heads, T, T)
        att_scores = (q @ k.permute(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_size))

        att_scores = att_scores.masked_fill(mask == 0, float('-inf'))

        att_scores = nn.functional.softmax(att_scores, -1)
        att_scores = self.dropout(att_scores)  # suppose adding dropout layer here is the original way

        # (B, n_heads, T, T) * (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)
        out = att_scores @ v
        out = out.transpose(1, 2).contiguous().view(B, T,
                                                    self.n_heads * self.head_size)  # (B, T, n_heads * head_size)

        return self.proj(out)  # (B, T, n_heads * head_size) - original shape


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (batch_size, max_len, n_embd)
        # encoder mask: (batch_size, 1, max_len, max_len)
        # result: (batch_size, max_len, d_model)
        x = x + self.dropout(self.sa(self.ln1(x), mask))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return self.dropout(x)


class MaskedLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.lin = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, x):
        return self.softmax(self.lin(x))  # (batch_size, seq_len, vocab_size)

class NextSentencePrediction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        # (B, T, D) -> (B, 2)
        return self.softmax(self.lin(x[:,0]))  # x[:,0] <=> x[:,0, :]


class BERT(nn.Module):
    def __init__(self, n_layers, vocab_size, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = BERTEmbeddings(d_model, vocab_size, max_len, dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.mlm = MaskedLanguageModel(d_model, vocab_size)
        self.nsp = NextSentencePrediction(d_model)

    def forward(self, ids, segment_label):
        # shape of ids is (batch_size, seq_len)
        # attention masking for padded token
        # mask = (ids > 0).unsqueeze(1).repeat(1, ids.size(1), 1).unsqueeze(1)  # (batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
        mask = (ids != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

        x = self.embed(ids, segment_label)
        for block in self.blocks:
            x = block(x, mask)

        # (B,T,vocab_size), (B,2)
        return self.mlm(x), self.nsp(x)


class ScheduledOptim:
    """Wrapper for optimizer with warmup scheduling and checkpoint support."""
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self._step = 0  # current step count
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Increment step, update LR, then step optimizer."""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return min(
            self._step ** -0.5,
            self.n_warmup_steps ** -1.5 * self._step
        )

    def _update_learning_rate(self):
        """Update LR on each step (warmup + decay)."""
        self._step += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            '_step': self._step,
            'warmup_steps': self.n_warmup_steps,
            'optimizer': self._optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state (step count, warmup, and optimizer state)."""
        self._step = state_dict['_step']
        self.n_warmup_steps = state_dict['warmup_steps']
        self._optimizer.load_state_dict(state_dict['optimizer'])


class BertTrainer:
    def __init__(
            self,
            model,
            train_dataloader,
            test_dataloader=None,
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            warmup_steps=10000,
            log_freq=10,
            device='cuda'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.log_freq = log_freq
        self.device = device

        self.model.to(self.device)

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.d_model, warmup_steps)

        self.mlm_criterion = nn.NLLLoss(ignore_index=0)  # ignore_index=0 tells the program to ignore [PAD] tokens
        self.nsp_criterion = nn.NLLLoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def test(self, epoch):
        self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, data_loader, train=True):

        # Reset statistics for the current epoch
        total_mlm_loss = 0
        total_nsp_loss = 0
        total_loss = 0

        # For NSP accuracy
        total_nsp_correct = 0
        total_nsp_elements = 0 # Number of samples for NSP prediction

        # For MLM accuracy (more complex, requires ignoring padded tokens and non-masked tokens)
        total_mlm_correct = 0
        total_mlm_elements = 0 # Number of masked tokens

        mode = 'train' if train else 'test'

        # Set model to train/eval mode
        if train:
            self.model.train()
        else:
            self.model.eval() # Use eval mode for test/validation to disable dropout/batchnorm


        # progress bar
        # Add `leave=True` if you want the progress bar to remain on screen after completion
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
            leave=True
        )

        for i, batch in data_iter:
            batch = {key: value.to(self.device) for key, value in batch.items()}

            # Set gradients to zero only if training
            if train:
                self.optim_schedule.zero_grad()

            mlm_output, nsp_output = self.model(batch["input_ids"], batch["segment_label"])

            # Calculate MLM Loss
            # mlm_output: (B, T, V), target: (B, T)
            # NLLLoss expects (B, V, T) for input
            mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), batch["input_label"])

            # Calculate NSP Loss
            # nsp_output: (B, 2), target: (B,)
            nsp_loss = self.nsp_criterion(nsp_output, batch["is_next"])

            loss = mlm_loss + nsp_loss

            if train:
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # Update total losses
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            total_loss += loss.item()

            # Calculate NSP Accuracy
            predicted_next_sentence_labels = nsp_output.argmax(dim=-1) # Use dim=-1 for argmax
            current_nsp_correct = (predicted_next_sentence_labels == batch["is_next"]).sum().item()
            current_nsp_elements = batch["is_next"].nelement() # Total NSP samples in current batch

            total_nsp_correct += current_nsp_correct
            total_nsp_elements += current_nsp_elements

            # Calculate MLM Accuracy
            predicted_mlm_labels = mlm_output.argmax(dim=-1) # (B, T)
            # Find which labels are actually masked (i.e., not 0 / PAD)
            masked_positions = (batch["input_label"] != 0) # Boolean mask for masked tokens
            current_mlm_correct = (predicted_mlm_labels[masked_positions] == batch["input_label"][masked_positions]).sum().item()
            current_mlm_elements = masked_positions.sum().item() # Count of actual masked tokens

            total_mlm_correct += current_mlm_correct
            total_mlm_elements += current_mlm_elements


            # Calculate metrics for the *current* iteration for `post_fix`
            current_avg_loss = total_loss / (i + 1)
            current_nsp_acc = (total_nsp_correct / total_nsp_elements * 100) if total_nsp_elements > 0 else 0
            current_mlm_acc = (total_mlm_correct / total_mlm_elements * 100) if total_mlm_elements > 0 else 0


            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss": loss.item(), # Current batch loss
                "avg_loss": current_avg_loss, # Average loss for epoch so far
                "mlm_loss": mlm_loss.item(),
                "nsp_loss": nsp_loss.item(),
                "nsp_acc": current_nsp_acc, # Average NSP accuracy for epoch so far
                "mlm_acc": current_mlm_acc # Average MLM accuracy for epoch so far
            }

            # Update the progress bar's postfix
            data_iter.set_postfix(post_fix)

            # Old way of writing to console, `set_postfix` is more integrated with tqdm
            # if i % self.log_freq == 0:
            #     data_iter.write(str(post_fix))

            # SAVE CHECKPOINT EVERY 30 steps
            if train and (i + 1) % 30 == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim_schedule._optimizer.state_dict(),
                    'scheduler_state_dict': self.optim_schedule.state_dict(),
                    'epoch': epoch,
                    'step': self.optim_schedule._step
                }
                checkpoint_path = '/content/drive/MyDrive/bert-checkpoint/bert_checkpoint.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch}, step {i+1}")

        # print information after each epoch
        # Calculate final epoch metrics
        final_avg_loss = total_loss / len(data_iter)
        final_nsp_acc = (total_nsp_correct / total_nsp_elements * 100) if total_nsp_elements > 0 else 0
        final_mlm_acc = (total_mlm_correct / total_mlm_elements * 100) if total_mlm_elements > 0 else 0

        print(
            f"EP{epoch}, {mode}: \
            avg_loss={final_avg_loss:.4f}, \
            nsp_acc={final_nsp_acc:.2f}%, \
            mlm_acc={final_mlm_acc:.2f}%"
        )
