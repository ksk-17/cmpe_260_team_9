import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

MAX_DEMONSTRATIONS = 5
MAX_H = 30
MAX_W = 30
COLOR_VOCAB_SIZE = 11
D_MODEL = 256


class GridPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_h=MAX_H, max_w=MAX_W):
        super().__init__()
        h_indices = torch.arange(max_h, dtype=torch.float32) / (max_h - 1)
        w_indices = torch.arange(max_w, dtype=torch.float32) / (max_w - 1)
        h_map_2d, w_map_2d = torch.meshgrid(h_indices, w_indices, indexing='ij')
        pos_enc_2d = torch.stack([h_map_2d, w_map_2d], dim=-1)
        self.register_buffer('pos_enc_2d', pos_enc_2d)
        self.proj = nn.Linear(2, d_model)

    def forward(self, x: torch.Tensor):
        pos_map = self.pos_enc_2d.to(x.device)
        projected_map = self.proj(pos_map)
        return projected_map.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    
class GridEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model

        self.color_embed = nn.Embedding(COLOR_VOCAB_SIZE, d_model)

        self.pos_encoder = GridPositionalEncoding(d_model)

        self.cnn = nn.Sequential(
            # Input: (D_MODEL, 30, 30)
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # (D_MODEL/2, 15, 15)
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # (D_MODEL/4, 7, 7) - (H/W is 7 or 8 depending on padding)
            nn.Conv2d(d_model // 4, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
            # Output: (D_MODEL, 7, 7) (approx)
        )

        self.proj = nn.Linear(d_model * 7 * 7, d_model)

    def forward(self, grid_long: torch.Tensor) -> torch.Tensor:
        B, H, W = grid_long.shape

        grid_indices = torch.clamp(grid_long, min=0, max=COLOR_VOCAB_SIZE - 2) 
        grid_indices[grid_long == -1] = COLOR_VOCAB_SIZE - 1

        embedded_grid = self.color_embed(grid_indices)

        embedded_grid = embedded_grid + self.pos_encoder(embedded_grid)
        
        cnn_input = embedded_grid.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(cnn_input)
    
        flattened = cnn_output.flatten(start_dim=1)
        feature_vector = self.proj(flattened)
        
        return feature_vector
    
class ARCStateEncoder(nn.Module):
    """
    Encodes the entire task (demonstrations + test input) into a single 
    Task Embedding V_context.
    """
    def __init__(self, d_model=D_MODEL, num_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.grid_encoder = GridEncoder(d_model=d_model)
        
        # FFN to combine I_features and O_features for a single pair vector
        self.pair_combiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Transformer Encoder for set-based context aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable token for the test input feature (used as the query later)
        self.test_query_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)  # Output a single scalar (the Value V(s))
        )

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x: return torch.zeros(1, self.d_model, device=self.grid_encoder.cnn[0].weight.device)
        
        all_grids_stacked = torch.stack(x)
        all_features = self.grid_encoder(all_grids_stacked)
        
        pair_features = []
        for i in range(MAX_DEMONSTRATIONS):
            I_feat = all_features[2 * i]
            O_feat = all_features[2 * i + 1]
            
            combined = torch.cat([I_feat, O_feat], dim=-1)
            pair_vector = self.pair_combiner(combined)
            pair_features.append(pair_vector)

        I_test_feature = all_features[-1]
        
        context_sequence = torch.stack(pair_features + [I_test_feature], dim=0).unsqueeze(0) # (B=1, SeqLen=6, D_MODEL)
        
        transformer_output = self.transformer_encoder(context_sequence)
        
        V_context = transformer_output[:, -1, :]

        value = self.value_head(V_context.detach())
        
        return V_context, value.squeeze(1)


if __name__ == "__main__":
    dummy_input_grids = []
    for i in range(2 * MAX_DEMONSTRATIONS + 1):
        grid_data = torch.randint(0, 10, (MAX_H, MAX_W), dtype=torch.long)
        grid_data[20:, 20:] = -1
        dummy_input_grids.append(grid_data)

    encoder = ARCStateEncoder(d_model=128)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    dummy_input_grids = [g.to(device) for g in dummy_input_grids]

    print(f"Running encoder on device: {device}")
    
    try:
        V_context = encoder(dummy_input_grids)
        
        print("\n--- State Encoder Output ---")
        print(f"Input Sequence Length (11 grids): {len(dummy_input_grids)}")
        print(f"Output Task Embedding V_context Shape: {V_context.shape}") # Should be (1, D_MODEL)
        
        print(f"V_context mean magnitude: {V_context.abs().mean().item():.4f}")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")