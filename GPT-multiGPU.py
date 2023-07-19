import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import re
import os
from tqdm import tqdm
torch.manual_seed(1337)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    init_process_group(backend='nccl', world_size=world_size, rank=rank)

batch_size = 64
block_size = 256 # window length
n_embd = 384 # dimension of embedding vector
n_head = 6 # number of heads in multi_head attention
n_layers = 6 # number of decoder layers Nx
dropout_rate = 0.2
eval_iters = 200 # take average of eval_iters output while evaluating
eval_interval = 500 # evaluate every eval_interval
learning_rate = 3e-4
train_ratio = 0.9 # train-test-split

# prev_model = False
# if prev_model:
#     checkpoint_path = "/kaggle/working/checkpoint-500.pth.tar"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_data():
    text = ""
    # for book in books:
    #     with open("/kaggle/input/bookcropus/books1/epubtxt/"+book, 'r', encoding='utf-8') as f:
    #         a_book = f.read()
    #         a_book = a_book.lower()
    #     text += a_book

    with open("/kaggle/input/tiny-shakespeare/input.txt", 'r', encoding='utf-8') as f:
        a_book = f.read()
        a_book = a_book.lower()
    text += a_book

    print(f"Number of characters: {len(text)}")
    print(f"Number of words: {len(text.split())}") #number of words

    vocab_list = re.findall(r'\w+|[^\w\S]+|[^\w]', text)
    vocab_list = sorted(list(set(vocab_list)))
    vocab_size = len(vocab_list)
    print(f"Vocab size: {vocab_size}")

    stoi = {value: key for key, value in enumerate(vocab_list)}
    itos = {key: value for value, key in stoi.items()}
    encode = lambda x: [stoi[_] for _ in x]
    decode = lambda x: ''.join(itos[_] for _ in x)

    data = torch.tensor(encode(re.findall(r'\w+|[^\w\S]+|[^\w]', text)))

    n = int(train_ratio * len(data))
    x_train = data[:n]
    x_val = data[n:]

    return x_train.view(1, -1), x_val.view(1, -1), vocab_size


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        super(CustomDataset, self).__init__()
        self.x = x
        
    def __len__(self):
        return self.x.shape[0] - 2*block_size - 1
    
    def __getitem__(self, idx):
        return self.x[idx : idx+block_size], self.x[idx+1 : idx+block_size+1]

def prepare_dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      sampler=DistributedSampler(dataset))
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
            nn.ReLU(),
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
    def __init__(self, rank, world_size, model, optimizer, epochs):
        self.gpu_id = rank
        self.world_size = world_size
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        # self.model = DDP(self.model, device_ids=[self.gpu_id])
        
    def train(self, X_train, X_val):
        X_train = X_train.view(-1)
        X_val = X_val.view(-1)
        train_dl = prepare_dataloader(CustomDataset(X_train))
        val_dl = prepare_dataloader(CustomDataset(X_val))
        self.model = self.model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        for epoch in tqdm(range(self.epochs)):
            # self.train_data.sampler.set_epoch(epoch)
            print(f"Epoch: {epoch}")
            for data, target in train_dl: # each loop gives a batch of data as specified in the dataloader
                logits, loss = self.model(data.to(self.gpu_id), target.to(self.gpu_id))
                print(f"GPU: {self.gpu_id}, loss: {loss.item()}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def main(rank, world_size, epochs):
    ddp_setup(rank, world_size)
    X_train, X_val, vocab_size = prepare_data()
    model = GPTLanguageModel(rank, vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())/1e6} M")
    trainer = Trainer(rank, world_size, model, optimizer, epochs)
    trainer.train(X_train, X_val)
    destroy_process_group()

if __name__ == "__main__":
    epochs = 2
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, epochs), nprocs=world_size)
