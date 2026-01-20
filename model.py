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
    def __init__(self, num_unique_ids, num_continuous_feats=9, d_model=256, d_state=32):
        super().__init__()

        # 1. Embed Discrete IDs with larger capacity
        self.emb_dim = 64  # Increased from 32 for richer ID representations
        self.id_embedding = nn.Embedding(num_unique_ids, self.emb_dim)

        # Add uncertainty representation for unknown IDs
        self.num_unique_ids = num_unique_ids
        self.unk_idx = num_unique_ids - 1  # Assumes <UNK> is last token

        # 2. Input Projection (Embeddings + 9 Continuous Feats)
        self.input_proj = nn.Linear(self.emb_dim + num_continuous_feats, d_model)

        # Add separate payload feature extractor (ID-independent path)
        self.payload_proj = nn.Linear(num_continuous_feats, d_model // 2)
        self.payload_norm = nn.LayerNorm(d_model // 2)

        # 3. Local Branch (CNN) - Texture/Jitter
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.LayerNorm(d_model)

        # 4. Feature Attention (ECA)
        self.eca = ECANet(channels=d_model)

        # 5. Global Branch (Mamba) - Long Context
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(d_model)

        # 6. Classifier with intermediate layer for better capacity
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x_ids, x_feats):
        # x_ids: [Batch, Seq_Len]
        # x_feats: [Batch, Seq_Len, 9]

        # Dual-path processing for better unknown ID handling
        # Path 1: ID-dependent (standard)
        x_emb = self.id_embedding(x_ids)
        x_combined = torch.cat([x_emb, x_feats], dim=-1)
        x_id_path = self.input_proj(x_combined)

        # Path 2: ID-independent (payload only) - helps with unknown IDs
        x_payload_path = self.payload_proj(x_feats)
        x_payload_path = self.payload_norm(x_payload_path)

        # Detect UNK tokens and blend paths adaptively
        is_unk = (x_ids == self.unk_idx).float().unsqueeze(-1)  # [Batch, Seq_Len, 1]

        # For UNK: rely more on payload path, for known: use ID path
        # Expand payload path to match d_model dimension
        x_payload_expanded = torch.cat([x_payload_path, x_payload_path], dim=-1)  # [B, S, d_model]
        x = (1 - is_unk) * x_id_path + is_unk * x_payload_expanded

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
        logits = self.classifier(x)

        return logits