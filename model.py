import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba


class ECANet(nn.Module):
    """
    Efficient Channel Attention (ECA) Module.
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channels, Length]
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LSS_CAN_Mamba(nn.Module):
    def __init__(self, num_unique_ids, num_continuous_feats=9, d_model=128, d_state=16):
        super().__init__()

        # 1. Embed Discrete IDs
        self.emb_dim = 32
        self.id_embedding = nn.Embedding(num_unique_ids, self.emb_dim)

        # 2. Input Projection (Embeddings + 9 Continuous Feats)
        self.input_proj = nn.Linear(self.emb_dim + num_continuous_feats, d_model)

        # 3. Local Branch (CNN) - Texture/Jitter
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.LayerNorm(d_model)

        # 4. Feature Attention (ECA)
        self.eca = ECANet(channels=d_model)

        # 5. Global Branch (Mamba) - Long Context
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(d_model)

        # 6. Classifier
        self.classifier = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_ids, x_feats):
        # x_ids: [Batch, Seq_Len]
        # x_feats: [Batch, Seq_Len, 9]

        # Embed and Concatenate
        x_emb = self.id_embedding(x_ids)
        x = torch.cat([x_emb, x_feats], dim=-1)
        x = self.input_proj(x)

        # Local CNN (Transpose for Conv1d)
        residual = x
        x_t = x.transpose(1, 2)
        x_t = self.act(self.local_conv(x_t))
        x_t = self.eca(x_t)  # Apply Attention
        x = x_t.transpose(1, 2)
        x = self.norm1(x + residual)

        # Global Mamba
        residual = x
        x = self.mamba(x)
        x = self.norm2(x + residual)

        # Classification (Global Average Pooling)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits