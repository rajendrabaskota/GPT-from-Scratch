import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import re
import os
import json
import wandb
import subprocess
from copy import deepcopy
from tqdm import tqdm
torch.manual_seed(1337)

os.environ["WANDB_API_KEY"] = "<your_api_key>"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend='nccl', world_size=world_size, rank=rank)


batch_size = 32
block_size = 256 # window length
n_embd = 512 # dimension of embedding vector
n_head = 8 # number of heads in multi_head attention
n_layers = 8 # number of decoder layers Nx
dropout_rate = 0.1
eval_iters = 200 # take average of eval_iters output while evaluating
eval_interval = 500 # evaluate every eval_interval
save_interval = 1000
total_iters = 20000
learning_rate = 3e-2
train_ratio = 0.98 # train-test-split
num_gradient_accumulation = 16
lr_scheduling_rate = 0.1


def load_tokenized():
    tokenized_data = torch.load("/kaggle/input/bpe-using-library/tokenized_data.pt")
    input_ids = tokenized_data['input_ids']
    vocab_size = tokenized_data['vocab_size']

    return input_ids, vocab_size


def train_test_split(data):
    n = int(train_ratio * len(data))
    x_train = data[:n]
    x_val = data[n:]

    return x_train.view(-1), x_val.view(-1)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        super(CustomDataset, self).__init__()
        self.x = x

    def __len__(self):
        return self.x.shape[0] - block_size - 1

    def __iter__(self):
        return iter(self.x)

    def __getitem__(self, idx):
        return self.x[idx : idx+block_size], self.x[idx+1 : idx+block_size+1]


def prepare_dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      sampler=DistributedSampler(dataset)
                                            )
    return dataloader


class Head(nn.Module):
    # ONE HEAD OF SELF-ATTENTION

    def __init__(self, head_size, gpu_id):
        # The head size is denoted by dk in the paper
        # The dimension of head size is generally equal to the dimension of the embedding vector
        super().__init__()
        self.gpu_id = gpu_id
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
                             # Registers a buffer that is not considered a model parameter
                            # Here tril isn't a model parameter to learn. so we register it as a buffer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape # C is equal to the head size
        k = self.key(x) # gives output (B, T, C)
        q = self.query(x) # output of (B, T, C)
        v = self.value(x) # output of (B, T, C)
        wei = q @ k.transpose(-2, -1) # (B, T, C) @ (B, C, T) --> (B, T, T) this (B, T, T) represents the amount of affinity each token has
                                        # with other tokens defined inside that block_size(window)
        wei = wei / C**0.5 # normalizing the weights; controls the variance; for more explanation look in the rough section
        tril = torch.tril(torch.ones((T, T), device=self.gpu_id))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, gpu_id, num_heads, head_size):
        super().__init__()
        self.gpu_id = gpu_id
        self.heads = nn.ModuleList([Head(head_size, self.gpu_id) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd), # the projection layer
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, gpu_id):
        # n_head gives the number of heads for multi-head attention
        super().__init__()
        self.gpu_id = gpu_id
        head_size = n_embd // n_head # gives the output dimension of one head
        self.sa_head = MultiHeadAttention(self.gpu_id, num_heads=n_head, head_size=head_size) # since we concat at the end in multihead attention,
                                                                            # the head size for one attention head = n_embd / num_heads
                                                                            # kind of like a group conv in CNN, instead of a large filter we
        self.FFN = FeedForwardNetwork(n_embd=n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.layer_norm1(x)) # implementing skip connection/skip connection
        x = x + self.FFN(self.layer_norm2(x)) # implementing skip connection/skip connection

        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, gpu_id, vocab_size):
        super().__init__()
        self.gpu_id = gpu_id
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(self.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, gpu_id) for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(n_embd) # the final layer norm
        self.lm_head = nn.Linear(n_embd, self.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #token embeddings, gives output of shape (B, T, C) here C = n_embd = 32
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.gpu_id)) # gives output of shape (T, C)
        x = tok_emb + pos_emb # (B, T, C) + (T, C) --> (B, T, C)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) # output of shape (B, T, vocab_size)


        # Explanation of B, T, C
        # B- batch dimension
        # T - time dimension (timestep) in this project one character=one timestep
        # C - channel dimension ie, the dimension of the embedding vector
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # logits of shape B*T, C
#             print(f"Before: {target}")
            target = target.view(B*T)
#             print(f"After: {target}, its shape is {target.shape}")
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens): # generates output given idx ie, the starting token
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            idx_trunc = idx[:, -block_size:] # making sure to truncate tokens if we receive more number of tokens than block_size
                                             # just to make sure position embedding doesn't throw any error
            logits, loss = self(idx_trunc) # logits of shape B, T, C
#             print(f"logits shape: {logits.shape}")
#             print(f"logits: {logits}")
            logits = logits[:, -1, :] #logits of shape B, C
#             print(f"logits after filter shape: {logits.shape}")
#             print(f"logits after filter: {logits}")
            probs = F.softmax(logits, dim=-1) # probs of shape
#             print(f"probs shape: {probs.shape}")
#             print(f"probs: {probs}")
            idx_next = torch.multinomial(probs, num_samples=1)
#             print(f"next index shape: {idx_next.shape}")
#             print(f"next index: {idx_next}")
            idx = torch.cat((idx, idx_next), dim=1)
#             print(f"index shape: {idx.shape}")
#             print(f"index tensor: {idx}")
        return idx


class Trainer:
    def __init__(self, rank, world_size, model, optimizer, learning_rate, num_iter, num_gradient_accum, lr_scheduling_rate):
        self.gpu_id = rank
        self.world_size = world_size
        self.model = model
        self.optimizer = optimizer
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.num_gradient_accum = num_gradient_accum
        self.lr_scheduling_rate = lr_scheduling_rate

    def lr_scheduling(self):
        temp = deepcopy(self.optimizer.state_dict())
        self.learning_rate = self.learning_rate * self.lr_scheduling_rate
        temp['param_groups'][0]['lr'] = self.learning_rate
        self.optimizer.load_state_dict(temp)

    def push_to_hub(self, current_iter):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_iter': current_iter
        }
        torch.save(state, "gpt-model-embed-dim-512-batch-32-layers-8-lr-1.2e-3.pt")

        result = subprocess.run('git add .', shell=True)
        result = subprocess.run('git commit -m new-checkpoint', shell=True)
        result = subprocess.run('git push', shell=True)

    def setup_wandb(self):
        wandb.init(
        project = "embed-dim-512-batch-32-layers-8-lr-1.2e-3",
        config = {
            "learning_rate": self.learning_rate,
            "architecture": "GPT",
            "dataset": "law-hearings-machine-translation-danish",
            "steps": self.num_iter
        }
    )

    def train(self, X_train, X_val):
        X_train = X_train.view(-1)
        X_val = X_val.view(-1)
        train_dl = prepare_dataloader(CustomDataset(X_train))
        val_dl = prepare_dataloader(CustomDataset(X_val))
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        if self.gpu_id == 0:
            progress_bar = tqdm(total=self.num_iter, desc="Loss", dynamic_ncols=True)
            self.setup_wandb()

        train_dl_iter = iter(train_dl)
        val_dl_iter = iter(val_dl)

        train_loss = torch.tensor([], device=self.gpu_id)
        num_no_consecutive_improvement = 0
        val_losses = []
        self.model.train()
        for i in range(self.num_iter):
            # each loop gives a batch of data as specified in the dataloader
            try:
                data, target = next(train_dl_iter)
            except:
                train_dl_iter = iter(train_dl)
                data, target = next(train_dl_iter)

            logits, loss = self.model(data.to(self.gpu_id), target.to(self.gpu_id))
            train_loss = torch.cat((train_loss, torch.tensor([loss.item()], device=self.gpu_id)))
            loss = loss / self.num_gradient_accum
            loss.backward()

            if (i+1)%self.num_gradient_accum == 0 or i == self.num_iter-1:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if i % eval_interval == 0:
                self.model.eval()
                val_loss = torch.tensor([], device=self.gpu_id)

                for _ in range(eval_iters):
                    try:
                        data, target = next(val_dl_iter)
                    except:
                        val_dl_iter = iter(val_dl)
                        data, target = next(val_dl_iter)

                    with torch.no_grad():
                        logits, loss = self.model(data.to(self.gpu_id), target.to(self.gpu_id))

                    val_loss = torch.cat((val_loss, torch.tensor([loss.item()], device=self.gpu_id)))

                dist.reduce(train_loss, dst=0)
                dist.reduce(val_loss, dst=0)
                if dist.get_rank() == 0:
                    avg_train_loss = torch.sum(train_loss) / (len(train_loss)*self.world_size)
                    avg_val_loss = torch.sum(val_loss) / (len(val_loss)*self.world_size)

                    progress_bar.set_postfix({'Train': avg_train_loss.item(), 'Test': avg_val_loss.item()}, refresh=True)
                    wandb.log({"train_loss": avg_train_loss.item(), "test_loss": avg_val_loss.item()}, step=i)

                    val_losses.append(avg_val_loss.item())
                    try:
                        if min(test_losses) <= avg_val_loss.item():
                            num_no_consecutive_improvement += 1
                        elif num_no_consecutive_improvement > 0:
                            num_no_consecutive_improvement = 0
                    except:
                        pass

                train_loss = torch.tensor([], device=self.gpu_id)
                self.model.train()

            if i % save_interval == 0 and not i==0:
                self.push_to_hub(i)

            if dist.get_rank() == 0:
                progress_bar.update(1)

            if num_no_consecutive_improvement == 2:
                self.lr_scheduling()

        progress_bar.close()


def main(rank, world_size, num_iter, num_gradient_accum, lr_scheduling_rate):
    ddp_setup(rank, world_size)
#     text = load_data()
#     input_ids, vocab_size = tokenize(text)
    input_ids, vocab_size = load_tokenized()
    X_train, X_val = train_test_split(input_ids)
    model = GPTLanguageModel(rank, vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())/1e6} M")
    trainer = Trainer(rank, world_size, model, optimizer, learning_rate, num_iter, num_gradient_accum, lr_scheduling_rate)
    trainer.train(X_train, X_val)
    destroy_process_group()

if __name__ == "__main__":
    num_iter = total_iters
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_iters, num_gradient_accumulation, lr_scheduling_rate), nprocs=world_size)
