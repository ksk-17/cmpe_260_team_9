import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RelationEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(11, 16)

        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # Relation: Concatenate Input(64) + Output(64)
        self.rel_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(128, d_model)

    def forward(self, inp, out):
        def feats(x):
            x = self.embedding(x.long()).permute(0, 3, 1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x

        f_in = feats(inp)
        f_out = feats(out)

        f_cat = torch.cat([f_in, f_out], dim=1)
        x = F.relu(self.rel_conv(f_cat))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class NeuroSolver(nn.Module):
    def __init__(self, dsl, d_model=128, max_len=10):
        super().__init__()
        self.dsl = dsl
        self.d_model = d_model
        self.max_len = max_len

        self.specials = ['<PAD>', '<SOS>', '<EOS>']
        self.digits = list(range(10))
        self.vocab = self.specials + self.digits + self.dsl.op_names
        self.vocab_size = len(self.vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}

        self.relation_encoder = RelationEncoder(d_model)
        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.gru = nn.GRUCell(d_model * 2, d_model)
        self.actor = nn.Linear(d_model, self.vocab_size)

    def forward(self, prev_token, hx, context):
        emb = self.token_embed(prev_token)
        gru_input = torch.cat([emb, context], dim=1)
        hx = self.gru(gru_input, hx)
        logits = self.actor(hx)
        return logits, hx

    def predict_program(self, input_grid, output_grid, device):
        if len(input_grid.shape) == 2:
            input_grid = input_grid.unsqueeze(0)
            output_grid = output_grid.unsqueeze(0)

        context = self.relation_encoder(input_grid, output_grid)
        hx = context

        current_token = torch.tensor([self.token_to_id['<SOS>']], device=device)
        tokens_out = []

        stack_depth = 1
        args_needed = 0

        for _ in range(self.max_len):
            logits, hx = self.forward(current_token, hx, context)

            # Masking
            mask = torch.full_like(logits, -1e9)
            valid_ids = []

            if args_needed > 0:
                for d in self.digits: valid_ids.append(self.token_to_id[d])
            else:
                valid_ids.append(self.token_to_id['<EOS>'])
                for op in self.dsl.ops:
                    if stack_depth >= op['stack_in']:
                        valid_ids.append(self.token_to_id[op['name']])

            if not valid_ids: valid_ids = [self.token_to_id['<EOS>']]
            mask[0, valid_ids] = 0

            probs = F.softmax(logits + mask, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            tok_str = self.vocab[action.item()]
            tokens_out.append(tok_str)
            if tok_str == '<EOS>': break

            if args_needed > 0:
                args_needed -= 1
            elif tok_str in self.dsl.op_map:
                op = self.dsl.op_map[tok_str]
                stack_depth = stack_depth - op['stack_in'] + op['stack_out']
                args_needed = op['n_args']

            current_token = action

        return tokens_out, []

    def train_step_supervised(self, input_grid, output_grid, target_tokens, device):
        context = self.relation_encoder(input_grid, output_grid)
        hx = context
        current_token = torch.tensor([self.token_to_id['<SOS>']], device=device)

        loss = 0
        full_target = target_tokens + ['<EOS>']

        for t_str in full_target:
            logits, hx = self.forward(current_token, hx, context)
            target_id = self.token_to_id[t_str]
            loss += F.cross_entropy(logits, torch.tensor([target_id], device=device))
            current_token = torch.tensor([target_id], device=device)

        return loss / len(full_target)