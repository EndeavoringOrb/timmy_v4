import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
from datetime import datetime as dt

# hyperparameters
val_split = 0.9
batch_size = 64 # for training
dropout = 0.2

max_iters = 1_000_000_000
eval_interval = 1 #evaluates and saves the model
eval_iters = 5

learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_head = 2
n_embd = 128
n_layer = 2
block_size = 32 # max context length

vocab_size = int(input("Enter vocab size: ")) # this needs to be the same size as the vocab size. so if you load "wiki.model", go to "wiki.vocab" and see how many lines there are. that is the vocab size you should use
load_amount = 0 # set to 0 if you want to load everything
vocab_list_path = input("Enter vocab path: ")
sentence_piece_model = input("Enter vocab model path: ")

model_version = 2
# ---------
torch.manual_seed(42)

sp = spm.SentencePieceProcessor()
sp.load(f'{sentence_piece_model}')

def encode(pieces):
    return sp.encode_as_ids(pieces)
def decode(pieces):
    thing = pieces[0]
    ret_val = [sp.DecodeIds(i.item()) for i in pieces[0]]
    return ret_val
def format_nicely(lookup_arr,ids,pieces):
    text = ""
    for i in range(len(pieces)):
        thing = lookup_arr[ids[i]][0]
        thing2 = lookup_arr[ids[i]]
        if lookup_arr[ids[i]][0] == "▁":
            text += " " + pieces[i]
        else:
            text += pieces[i]
    return text
def make_lookup_arr(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.readlines()
    vocab_arr = [i.split("\t")[0] for i in text]
    return vocab_arr

vocab_arr = make_lookup_arr(vocab_list_path)


"""
print("reading text")
with open(f"{text_file}", "r", encoding="utf-8") as f:
    text = f.readlines()


# Load the trained model
print("loading model")
sp = spm.SentencePieceProcessor()
sp.load(f'{sentence_piece_model}.model')

#pieces = []

print("characterizing text")
#pieces = sp.encode_as_pieces(text)
#print("finished")
if load_amount != 0:
    text = text[:load_amount]
text_len = len(text)
start = dt.now()
for num, i in enumerate(text):
    # Piece a sentence
    pieces.extend(sp.encode_as_ids(i))
    if num%100000 == 0:
        end = dt.now()
        elapsed = end-start
        percent = num/text_len + 0.00000001 # messy way to avoid zero division error
        print(f"{num}/{text_len} - {percent*100:.3f}% - ETA: {(1/percent)*(elapsed)-elapsed}                     ",end="\r")
    #print(pieces)
    #input("")
print(f"{text_len}/{text_len} - 100.00%")

print(f"length of dataset in pieces: {len(pieces)}")

#str_int = {ch:i for i,ch in enumerate(chars)}
#int_str = {i:ch for i,ch in enumerate(chars)}
def encode(pieces):
    return sp.encode_as_ids(pieces)
def decode(pieces):
    return sp.decode_pieces(pieces)
#encode = lambda s: [str_int[c] for c in s]
#decode = lambda l: "".join([int_str[i] for i in l])

print(encode("hello world"))
print(decode(encode("hello world")))

#data = torch.tensor(pieces, dtype=torch.long)
print(data.shape, data.dtype)

#train_num = int(val_split*len(data))
train_data = data[:train_num]
val_data = data[train_num:]

train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    #print(f"when input is {context} the target is: {target}")

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
"""

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # single head perform self-attention
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        
        wei = q @ k.transpose(-2,-1) * (C**-0.5) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BatchNorm1d:

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        #calc forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embds = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embds + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get preds
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #decoded_thing = decode(idx_next)
            #decoded_thing2 = decode(idx)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            print(format_nicely(vocab_arr,idx[0],decode(idx)),end="\r")
        return idx






model = BigramLanguageModel()
model = model.to(device)
