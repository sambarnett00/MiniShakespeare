import torch
from transformer_eff import Transformer, decode

max_new_tokens = 5000
in_path = "saved/transformer-eff.pt"
out_path = "more.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

model: Transformer = Transformer()
model.load_state_dict(torch.load(in_path))
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long).to(device)
output = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)
