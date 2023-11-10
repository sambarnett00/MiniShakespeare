import torch
from transformer_eff import Transformer, get_batch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200


## tells pytorch that we will not call .backward on anything in this function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Transformer()
# model = torch.compile(model)
m = model.to(device)
## model parameters must be moved to device
##   embedding table is moved to device, calculations happen on gpu
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
prog_bar = tqdm(range(max_iters))
for iter in prog_bar:
    if iter % eval_interval == 0:
        losses = estimate_loss()
        prog_bar.desc = f"train={losses['train']:.4f}, val={losses['val']:.4f}"
        torch.save(model.state_dict(), "saved/ckpt_transformer.pt")
        # print(f"{iter=}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "saved/transformer-eff.pt")