import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureFusionMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[2048, 1024, 768], output_dim=768, dropout=0.3):
        super(FeatureFusionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.output_layer = nn.Linear(hidden_dims[2], output_dim)
        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dims[0])
        self.layer_norm2 = nn.LayerNorm(hidden_dims[1])
        self.layer_norm3 = nn.LayerNorm(hidden_dims[2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  
        x = self.fc1(x)
        x = self.activation(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.layer_norm3(x)
        x = self.dropout(x)
        x = x + residual  
        x = self.output_layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=768, output_dim=768, dropout=0.3):
        super(MLP, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dims)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual 
        return x

class Img_State_Fusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=1024, output_dim=512):
        super(Img_State_Fusion, self).__init__()
        self.fusion = nn.Sequential(nn.Linear(input_dim, hidden_dims), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dims, output_dim))

    def forward(self, image_features, state_embeddings):
        combined_features = torch.einsum('bd,bd->bd', image_features, state_embeddings)
        fused_features = self.fusion(combined_features)
        return fused_features  




