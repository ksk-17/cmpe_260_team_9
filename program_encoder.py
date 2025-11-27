import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from dsl import DSL_PRIMITIVES

D_MODEL = 256
MAX_PROGRAM_LENGTH = 50


class ProgramVocabulary:

    def __init__(self, dsl_primitives: Dict[str, Any]):
        self.dsl_primitives = dsl_primitives
        self.tokens: List[str] = []

        # 1. Special Tokens
        self.tokens.extend(['<PAD>', '<START>', '<END>', '<UNK>'])

        # 2. Primitives (from DSL)
        self.tokens.extend(list(dsl_primitives.keys()))

        # 3. Numeric Arguments (Colors, Factors, Thickness, Angles)
        self.tokens.extend([str(i) for i in range(10)])  # Colors 0-9
        self.tokens.extend([str(i) for i in [10, 90, 180, 270]])  # Common constants

        # 4. Symbolic Arguments (for MapColor, Count, FindObjects)
        self.tokens.extend([
            '*',  # For FindObjects(color='*')
            'unique_colors',  # For Count(property)
            'pixels',  # For Count(property)
            'objects',  # For Count(property)
            'most_frequent',  # For MapColor(rule)
            'background',  # For MapColor(rule)
            'opposite',  # For MapColor(rule)
            'axis_0',  # For MirrorGrid(axis=0)
            'axis_1'  # For MirrorGrid(axis=1)
        ])

        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.vocab_size = len(self.tokens)

    def encode(self, program: List[str]) -> torch.Tensor:
        ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in program]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> List[str]:
        return [self.id_to_token.get(i.item(), '<UNK>') for i in ids]


class ProgramGenerator(nn.Module):
    def __init__(self, vocab: ProgramVocabulary, d_model=D_MODEL, n_layers=4, n_heads=4):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab.vocab_size, d_model)

        self.sequence_pos_encoder = nn.Embedding(MAX_PROGRAM_LENGTH, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_head = nn.Linear(d_model, vocab.vocab_size)

    def forward(self,
                program_tokens: torch.Tensor,
                v_context: torch.Tensor,
                target_mask: torch.Tensor = None) -> torch.Tensor:
        B, T = program_tokens.shape

        token_embed = self.token_embedding(program_tokens)

        positions = torch.arange(T, device=program_tokens.device).unsqueeze(0)  # (1, T)
        pos_embed = self.sequence_pos_encoder(positions)  # (1, T, D_MODEL)
        decoder_input = token_embed + pos_embed

        memory = v_context.unsqueeze(1)  # (B, 1, D_MODEL)

        decoder_output = self.transformer_decoder(
            tgt=decoder_input,  # Query (the partial program)
            memory=memory,  # Key/Value (the task context)
            tgt_mask=target_mask  # Causal Mask to prevent looking ahead
        )  # Output: (B, T, D_MODEL)

        logits = self.output_head(decoder_output)  # (B, T, VocabSize)
        return logits

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

if __name__ ==  "__main__":
    vocab = ProgramVocabulary(DSL_PRIMITIVES)

    generator = ProgramGenerator(vocab, d_model=D_MODEL, n_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)

    v_context_dummy = torch.randn(2, D_MODEL, device=device)

    program_strings = [
        ['<START>', 'Recolor', '5'],
        ['<START>', 'Shift', '1']
    ]
    program_tokens_dummy = torch.stack([
        vocab.encode(p).to(device) for p in program_strings
    ])

    T = program_tokens_dummy.size(1)
    causal_mask = generate_square_subsequent_mask(T).to(device)

    logits = generator(program_tokens_dummy, v_context_dummy, causal_mask)
    probabilities = F.softmax(logits[:, -1, :], dim=-1)  # (B, VocabSize)

    print(f"Program Generator Output Logits Shape: {logits.shape}")
    print(f"Probabilities for next token (Batch 1): {probabilities[0].shape}")
    print(f"Next Best Token (Batch 1): {vocab.id_to_token[probabilities[0].argmax().item()]}")