"""
Baseline models for comparison with LSS-CAN-Mamba.

These baseline models provide simple architectures to demonstrate
the effectiveness of the LSS-CAN-Mamba approach.
"""

import torch
import torch.nn as nn


class SimpleMLPBaseline(nn.Module):
    """
    Simple Multi-Layer Perceptron baseline.
    
    Flattens the sequence and processes with fully connected layers.
    This is the simplest possible baseline.
    """
    
    def __init__(self, num_unique_ids, num_continuous_feats=9, d_model=256, seq_len=100):
        super().__init__()
        
        self.seq_len = seq_len
        self.emb_dim = 32
        self.id_embedding = nn.Embedding(num_unique_ids, self.emb_dim)
        
        # Input dimension: (emb_dim + num_continuous_feats) * seq_len
        input_dim = (self.emb_dim + num_continuous_feats) * seq_len
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 2)
        )
    
    def forward(self, x_ids, x_feats):
        # x_ids: [Batch, Seq_Len]
        # x_feats: [Batch, Seq_Len, 9]
        
        # Embed IDs
        x_emb = self.id_embedding(x_ids)  # [Batch, Seq_Len, emb_dim]
        
        # Concatenate embeddings and features
        x = torch.cat([x_emb, x_feats], dim=-1)  # [Batch, Seq_Len, emb_dim + 9]
        
        # Flatten
        x = x.flatten(start_dim=1)  # [Batch, Seq_Len * (emb_dim + 9)]
        
        # Pass through MLP
        logits = self.mlp(x)
        
        return logits


class SimpleLSTMBaseline(nn.Module):
    """
    Simple LSTM baseline.
    
    Processes the sequence with a bidirectional LSTM and classifies
    based on the final hidden state.
    """
    def __init__(self, num_unique_ids, num_continuous_feats, d_model, seq_len=None, num_layers=2, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_unique_ids, d_model - num_continuous_feats)
        self.cont_proj = nn.Linear(num_continuous_feats, num_continuous_feats)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers, # Now this works because it refers to the argument above
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, ids, feats):
        # (Same forward pass code as before)
        x_emb = self.embedding(ids)
        x_cont = self.cont_proj(feats)
        x = torch.cat([x_emb, x_cont], dim=-1)
        
        output, _ = self.lstm(x)
        # Take the last time step
        last_output = output[:, -1, :]
        return self.classifier(last_output)
        
        return logits


class SimpleCNNBaseline(nn.Module):
    """
    Simple CNN baseline.
    
    Processes the sequence with 1D convolutions and classifies
    based on global average pooling.
    """
    
    def __init__(self, num_unique_ids, num_continuous_feats, d_model, seq_len=None, **kwargs):
        super().__init__()
        
        self.emb_dim = 32
        self.id_embedding = nn.Embedding(num_unique_ids, self.emb_dim)
        
        # Input dimension: emb_dim + num_continuous_feats
        input_dim = self.emb_dim + num_continuous_feats
        
        # Project to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # CNN layers
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 2)
        )
    
    def forward(self, x_ids, x_feats):
        # x_ids: [Batch, Seq_Len]
        # x_feats: [Batch, Seq_Len, 9]
        
        # Embed IDs
        x_emb = self.id_embedding(x_ids)  # [Batch, Seq_Len, emb_dim]
        
        # Concatenate embeddings and features
        x = torch.cat([x_emb, x_feats], dim=-1)  # [Batch, Seq_Len, emb_dim + 9]
        
        # Project to d_model
        x = self.input_proj(x)  # [Batch, Seq_Len, d_model]
        
        # Transpose for Conv1d: [Batch, d_model, Seq_Len]
        x = x.transpose(1, 2)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # [Batch, d_model]
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class SimpleGRUBaseline(nn.Module):
    # FIXED: Added seq_len (to ignore it) and num_layers (to define it)
    def __init__(self, num_unique_ids, num_continuous_feats, d_model, seq_len=None, num_layers=2, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(num_unique_ids, d_model - num_continuous_feats)
        self.cont_proj = nn.Linear(num_continuous_feats, num_continuous_feats)
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,  # Now this variable is defined
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, ids, feats):
        x_emb = self.embedding(ids)
        x_cont = self.cont_proj(feats)
        x = torch.cat([x_emb, x_cont], dim=-1)
        
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)
        
        return logits


# Dictionary mapping model names to classes for easy instantiation
BASELINE_MODELS = {
    'mlp': SimpleMLPBaseline,
    'lstm': SimpleLSTMBaseline,
    'cnn': SimpleCNNBaseline,
    'gru': SimpleGRUBaseline,
}


def get_baseline_model(model_name, **kwargs):
    """
    Factory function to get baseline model by name.
    
    Args:
        model_name: Name of baseline model ('mlp', 'lstm', 'cnn', 'gru')
        **kwargs: Arguments to pass to model constructor
    
    Returns:
        Instantiated baseline model
    """
    if model_name not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline model: {model_name}. "
                        f"Available models: {list(BASELINE_MODELS.keys())}")
    
    return BASELINE_MODELS[model_name](**kwargs)
