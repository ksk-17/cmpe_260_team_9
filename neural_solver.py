import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from arc_dataset import pad_grid

def encode_context(model, input_grid, output_grid, device):
    if not isinstance(input_grid, torch.Tensor):
        input_grid = torch.tensor(pad_grid(input_grid), device=device).unsqueeze(0)
    if not isinstance(output_grid, torch.Tensor):
        output_grid = torch.tensor(pad_grid(output_grid), device=device).unsqueeze(0)
    if len(input_grid.shape) == 2: input_grid = input_grid.unsqueeze(0)
    if len(output_grid.shape) == 2: output_grid = output_grid.unsqueeze(0)
    return model.relation_encoder(input_grid, output_grid)

class RelationEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(11, 16)
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.rel_fc = nn.Sequential(nn.Linear(128, d_model), nn.ReLU())

    def forward(self, inp, out):
        def feats(x):
            x = self.embedding(x.long()).permute(0, 3, 1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x

        f_in = feats(inp)
        f_out = feats(out)
        f_cat = torch.cat([f_in, f_out], dim=1)
        # Adaptive pool to handle variable grid sizes if needed, or just flatten 30x30
        x = F.adaptive_max_pool2d(f_cat, (1, 1)).flatten(1)
        return self.rel_fc(x)


class NeuroSolver(nn.Module):
    def __init__(self, dsl, d_model=128, max_len=10):
        super().__init__()
        self.dsl = dsl
        self.d_model = d_model
        self.max_len = max_len

        self.specials = ['<PAD>', '<SOS>', '<EOS>']
        self.digits = list(range(10))
        self.vocab = self.specials + self.digits + self.dsl.op_names
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.relation_encoder = RelationEncoder(d_model)
        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.gru = nn.GRUCell(d_model * 2, d_model)
        self.actor = nn.Linear(d_model, self.vocab_size)
        self.critic = nn.Linear(d_model, 1)  # Critical for PPO

    def forward(self, prev_token, hx, context):
        emb = self.token_embed(prev_token)
        gru_input = torch.cat([emb, context], dim=1)
        hx = self.gru(gru_input, hx)
        logits = self.actor(hx)
        value = self.critic(hx)
        return logits, value, hx

    def predict_program(self, input_grid, output_grid, device):
        # Wraps forward for inference (returns 2 values usually, but we keep consistency)
        ctx = encode_context(self, input_grid, output_grid, device)
        hx = ctx
        curr = torch.tensor([self.token_to_id['<SOS>']], device=device)
        tokens = []
        for _ in range(self.max_len):
            logits, _, hx = self.forward(curr, hx, ctx)
            action = Categorical(logits=logits).sample()
            tok = self.vocab[action.item()]
            tokens.append(tok)
            if tok == '<EOS>': break
            curr = action
        return tokens, []