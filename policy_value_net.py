from dataclasses import dataclass
from typing import List, Dict, Optional, Literal, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


ArgType = Literal["categorical", "pointer"]

@dataclass
class ArgSpec:
    """Specification for one argument factor of the action."""
    name: str
    kind: ArgType                 # "categorical" or "pointer"
    size: int                     # categorical vocab size or max pointer slots

@dataclass
class PolicyValueConfig:
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1

    # Tokenization of the state (grid/object tokens you already build)
    vocab_size: int = 512         # token vocab for state encoder (edit to your tokenizer)
    max_seq_len: int = 1024

    # Object / region memory (for pointer args). Provide embeddings for M slots.
    max_object_slots: int = 128
    object_feat_dim: int = 64     # if you pass raw object features to be projected

    # Action space (factorized): first factor MUST be op_id (categorical)
    n_ops: int = 64               # number of DSL ops
    arg_specs: Tuple[ArgSpec, ...] = (
        # example layout; replace with your DSL’s factors (beyond op_id)
        # ArgSpec(name="obj_ptr", kind="pointer", size=128),
        # ArgSpec(name="color", kind="categorical", size=10),
    )

    # Initialization
    value_hidden: int = 256
    policy_hidden: int = 256

    # Optional: tie object encoder
    use_object_encoder: bool = True


# -----------------------------
# Positional embedding
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# -----------------------------
# Transformer encoder trunk
# -----------------------------
def build_transformer_encoder(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


# -----------------------------
# Core Network
# -----------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, cfg: PolicyValueConfig):
        super().__init__()
        self.cfg = cfg

        # State encoder (token embeddings + positional + transformer)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.trunk = build_transformer_encoder(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

        # Optional object encoder to project features to model dim (for pointer heads)
        if cfg.use_object_encoder:
            self.obj_proj = nn.Linear(cfg.object_feat_dim, cfg.d_model)
        else:
            self.obj_proj = nn.Identity()

        # Shared latent pooling (CLS-like): mean-pool + layernorm
        self.lat_norm = nn.LayerNorm(cfg.d_model)

        # Policy heads
        self.op_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.policy_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.policy_hidden, cfg.n_ops)
        )

        self.cat_heads = nn.ModuleDict()
        self.ptr_queries = nn.ModuleDict()  # query vectors for pointer attention

        for spec in cfg.arg_specs:
            if spec.kind == "categorical":
                self.cat_heads[spec.name] = nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.policy_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(cfg.policy_hidden, spec.size)
                )
            elif spec.kind == "pointer":
                # a learned query to attend over object memory
                self.ptr_queries[spec.name] = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            else:
                raise ValueError(f"Unknown arg kind: {spec.kind}")

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1)
        )

    # ---- utilities ----
    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] (True=keep)
        if mask is None:
            return x.mean(dim=1)
        w = mask.float().unsqueeze(-1)  # [B,T,1]
        s = (x * w).sum(dim=1)
        d = w.sum(dim=1).clamp_min(1.0)
        return s / d

    def _pointer_scores(
        self,
        query: torch.Tensor,        # [B, D]
        mem: torch.Tensor,          # [B, M, D]
        mem_mask: Optional[torch.Tensor],  # [B, M] (True=valid)
        scale: Optional[float] = None
    ) -> torch.Tensor:
        # scaled dot-product attention scores over memory slots (no softmax yet)
        # returns [B, M] with -inf where masked
        B, M, D = mem.shape
        q = query.unsqueeze(1)      # [B,1,D]
        scores = torch.einsum("bid,bjd->bij", q, mem).squeeze(1)  # [B,M]
        if scale is None:
            scale = 1.0 / math.sqrt(D)
        scores = scores * scale
        if mem_mask is not None:
            scores = scores.masked_fill(~mem_mask, float("-inf"))
        return scores

    # ---- forward ----
    def forward(
        self,
        state_tokens: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,     # Bool [B,T]
        object_feats: Optional[torch.Tensor] = None,  # Float [B,M,F]
        object_mask: Optional[torch.Tensor] = None,   # Bool [B,M]
    ) -> Dict[str, Any]:
        cfg = self.cfg
        B, T = state_tokens.shape

        x = self.tok_emb(state_tokens)                # [B,T,D]
        x = self.pos(x)
        x = self.dropout(x)

        # key_padding_mask expects True for PAD positions; we have True=keep. Convert if provided.
        kpm = None
        if attn_mask is not None:
            kpm = ~attn_mask  # True=pad

        x = self.trunk(x, src_key_padding_mask=kpm)   # [B,T,D]

        # pooled latent
        latent = self._masked_mean(x, attn_mask)
        latent = self.lat_norm(latent)                # [B,D]

        # Prepare memory for pointer arguments
        ptr_mem = None
        if len(self.ptr_queries) > 0:
            if object_feats is None:
                # If you use pointer args, you must pass object_feats/object_mask
                raise ValueError("Pointer args present but object_feats is None.")
            mem = self.obj_proj(object_feats)         # [B,M,D]
            ptr_mem = mem

        # Policy: op logits
        op_logits = self.op_head(latent)              # [B, n_ops]

        cat_logits: Dict[str, torch.Tensor] = {}
        ptr_scores: Dict[str, torch.Tensor] = {}

        for spec in self.cfg.arg_specs:
            if spec.kind == "categorical":
                head = self.cat_heads[spec.name]
                cat_logits[spec.name] = head(latent)  # [B, n_cat]
            else:  # pointer
                q = self.ptr_queries[spec.name](latent)  # [B,D]
                scores = self._pointer_scores(q, ptr_mem, object_mask)  # [B,M] (-inf masked)
                # Optionally clamp to avoid all -inf (when no valid objects)
                if object_mask is not None:
                    all_invalid = (~object_mask).all(dim=1)  # [B]
                    if all_invalid.any():
                        # If no valid slot, make a dummy uniform over first slot (won't be used if you also mask downstream)
                        scores[all_invalid] = torch.zeros(scores.size(1), device=scores.device)
                ptr_scores[spec.name] = scores  # softmax later with mask

        # Value in [0,1]
        value_logit = self.value_head(latent)         # [B,1]
        value = torch.sigmoid(value_logit)

        return {
            "policy": {
                "op_logits": op_logits,
                "cat_logits": cat_logits,
                "ptr_scores": ptr_scores,   # raw scores; apply masked softmax when sampling/lossing
            },
            "value": value,
            "latent": latent,
        }


# -----------------------------
# Loss: AlphaGo-style (value MSE + policy CE vs. search policy)
# -----------------------------
def alphago_loss(
    outputs: Dict[str, Any],
    *,
    # Value targets
    z: torch.Tensor,                               # [B] or [B,1], final outcome in {0,1}
    weight_decay: float = 0.0,
    model: Optional[nn.Module] = None,
    cat_pis: Optional[Dict[str, torch.Tensor]] = None,
    ptr_pis: Optional[Dict[str, torch.Tensor]] = None,
    op_pi: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    policy = outputs["policy"]
    value = outputs["value"].squeeze(-1)  # [B]

    losses = []
    logs: Dict[str, float] = {}

    # Value loss
    z = z.view_as(value)
    v_loss = F.mse_loss(value, z)
    losses.append(v_loss)
    logs["value_mse"] = float(v_loss.detach().item())

    # Policy: op_id
    if op_pi is not None:
        op_logp = F.log_softmax(policy["op_logits"], dim=-1)         # [B, n_ops]
        op_loss = -(op_pi * op_logp).sum(dim=-1).mean()
        losses.append(op_loss)
        logs["op_ce"] = float(op_loss.detach().item())

    # Policy: categorical args
    if cat_pis is not None:
        for name, targ in cat_pis.items():
            logits = policy["cat_logits"][name]                       # [B, n_cat]
            logp = F.log_softmax(logits, dim=-1)
            ce = -(targ * logp).sum(dim=-1).mean()
            losses.append(ce)
            logs[f"cat_ce/{name}"] = float(ce.detach().item())

    # Policy: pointer args (masked softmax over memory)
    if ptr_pis is not None:
        for name, targ in ptr_pis.items():
            scores = policy["ptr_scores"][name]                       # [B, M] (-inf masked)
            logp = F.log_softmax(scores, dim=-1)                      # masked by -inf
            ce = -(targ * logp).sum(dim=-1).mean()
            losses.append(ce)
            logs[f"ptr_ce/{name}"] = float(ce.detach().item())

    total = sum(losses)

    # L2/weight decay (AlphaGo uses L2). If using AdamW with weight_decay, you can skip this.
    if weight_decay > 0.0 and model is not None:
        l2 = 0.0
        for p in model.parameters():
            l2 = l2 + p.pow(2).sum()
        l2 = weight_decay * l2
        total = total + l2
        logs["l2"] = float(l2.detach().item())

    logs["total"] = float(total.detach().item())
    return total, logs


# -----------------------------
# Sampling utilities (for MCTS root policy, etc.)
# -----------------------------
@torch.no_grad()
def masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # scores: [B,M], mask: [B,M] with True=valid
    if mask is None:
        return scores.softmax(dim=-1)
    masked = scores.masked_fill(~mask, float("-inf"))
    return masked.softmax(dim=-1)

@torch.no_grad()
def sample_action_factors(
    outputs: Dict[str, Any],
    *,
    object_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> Dict[str, torch.Tensor]:
    policy = outputs["policy"]
    B = policy["op_logits"].size(0)

    # Op
    op_logits = policy["op_logits"] / max(1e-6, temperature)
    op_probs = F.softmax(op_logits, dim=-1)
    op_id = torch.multinomial(op_probs, num_samples=1).squeeze(1)

    # Categorical args
    categorical: Dict[str, torch.Tensor] = {}
    for name, logits in policy["cat_logits"].items():
        probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
        categorical[name] = torch.multinomial(probs, num_samples=1).squeeze(1)

    # Pointer args
    pointer: Dict[str, torch.Tensor] = {}
    for name, scores in policy["ptr_scores"].items():
        if object_mask is not None:
            probs = masked_softmax(scores / max(1e-6, temperature), object_mask)
        else:
            probs = F.softmax(scores / max(1e-6, temperature), dim=-1)
        pointer[name] = torch.multinomial(probs, num_samples=1).squeeze(1)

    return {"op_id": op_id, "categorical": categorical, "pointer": pointer}


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from policy_value_net import PolicyValueNet, PolicyValueConfig, ArgSpec, alphago_loss

    # ---- 1. config & network ----
    cfg = PolicyValueConfig(
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        vocab_size=256,
        max_seq_len=256,
        max_object_slots=64,
        object_feat_dim=32,
        n_ops=32,
        arg_specs=(
            ArgSpec(name="obj_ptr", kind="pointer", size=64),
            ArgSpec(name="color", kind="categorical", size=10),
        ),
    )

    net = PolicyValueNet(cfg)
    print("✅ Network built:", sum(p.numel() for p in net.parameters()) / 1e6, "M params")

    # ---- 2. dummy batch ----
    B, T, M, D = 8, 64, 32, 32
    state_tokens = torch.randint(0, cfg.vocab_size, (B, T))
    attn_mask = torch.rand(B, T) > 0.1
    object_feats = torch.randn(B, M, D)
    object_mask = torch.rand(B, M) > 0.2

    # ---- 3. forward ----
    out = net(state_tokens, attn_mask, object_feats, object_mask)

    print("latent:", out["latent"].shape)
    print("op_logits:", out["policy"]["op_logits"].shape)
    print("color logits:", out["policy"]["cat_logits"]["color"].shape)
    print("pointer scores:", out["policy"]["ptr_scores"]["obj_ptr"].shape)
    print("value:", out["value"].shape)

    # ---- 4. fake MCTS targets ----
    op_pi = F.one_hot(torch.randint(0, cfg.n_ops, (B,)), num_classes=cfg.n_ops).float()
    cat_pis = {"color": F.one_hot(torch.randint(0, 10, (B,)), num_classes=10).float()}
    ptr_pis = {"obj_ptr": F.one_hot(torch.randint(0, M, (B,)), num_classes=M).float() * object_mask.float()}
    z = torch.randint(0, 2, (B,)).float()

    # ---- 5. loss & backward ----
    loss, logs = alphago_loss(out, z=z, op_pi=op_pi, cat_pis=cat_pis, ptr_pis=ptr_pis, model=net, weight_decay=1e-4)
    loss.backward()
    print("✅ forward/backward successful")
    print("loss logs:", logs)