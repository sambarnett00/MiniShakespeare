import torch
import torch.nn as nn
from torch.nn import functional as F
import requests as r
from os.path import exists
from tqdm import tqdm


batch_size = 64  ## how many indep seq will we processin parallel
block_size = 256 # what is the max context length for predictions?
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384  # 384 / 6 = every head is 64 dim (standard)
n_head = 6
n_layer = 6 ## blocks
dropout = 0.2
## for less performant, bring down n_layers and n_embd, etc
# conversion between GPT3 paper param names to above
#   n_layers = n_layer
#   d_model  = n_embd
#   n_heads  = n_head 
#   d_head   = head_size

# -------------

torch.manual_seed(1337)

## load dataset
def get_data(url: str):
    if not exists("../data/input.txt"):
        res = r.get(url)
        with open("../data/input.txt", "w", encoding="utf-8") as f:
            f.write(res.text)
        return res.text
    else:
        with open("../data/input.txt", "r", encoding="utf-8") as f:
            return f.read()
    
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = get_data(url)

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("".join(chars))
# print(f"{vocab_size=}")

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# When we load the data, we need to make sure we load it to device
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head) -> None:
        super(CausalSelfAttention, self).__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        
        ## head_size = n_embd // n_head
        ## single Head, q,k,v = nn.Linear(n_embd, head_size, bias=False)
        ## 3 * n_embd = |{q,k,v}| * (n_embd * head_size) * n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # nh: num head, hs: head size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        y: torch.Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y)) ## (B, T, C)
        return y
        

class FeedForward(nn.Module):
    """a simple Linear Layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(n_embd, 4 * n_embd) ## the 4* is from the paper
        self.lin2 = nn.Linear(4 * n_embd, n_embd) ## projection layer going back into residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer Bock: comm followed by computation"""
    
    def __init__(self, n_embd, n_head):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd) # normalise features? at initialisation
        self.csa = CausalSelfAttention(n_embd, n_head) ## communication
        self.ln2 = nn.LayerNorm(n_embd) # B,T dims are considered both batch layers
        self.ffwd = FeedForward(n_embd)                 ## computation

    def forward(self, x):
        x = x + self.csa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # each token directly reads off the logitcs for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) ## std layer norm before final projection
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T,C
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T) ## targets.view(-1)
        
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # the current implementation of gen is rediculous, as we only need the last token
        #   but we are currently providing the entire context
        # idx is (B, T) array of indices in the current context
        for _ in tqdm(range(max_new_tokens)):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            logits, _ = self(idx_cond)
            ## only care about the last token, last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

