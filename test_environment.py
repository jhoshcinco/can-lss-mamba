# test the new environment to see if it works

#write docker files, requirements.txt so it's a one run thing. Also cross platform compatibility
# you need torch,pandas

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

device = "cuda"

model = MambaLMHeadModel(
    d_model=256
    n_layer=4,
    vocab_size=32000,
).to(device)

x = torch.randint(0, 32000, (2, 128), device=device)
y = model(x)

print("Forward pass OK:", y.logits.shape)
