# dse_core.py — Differentiable Symbolic Executor over grids & object tables
# Depends on dse_helpers.py placed alongside this file.
# torch >= 2.0

from __future__ import annotations
from typing import NamedTuple, Dict, Any, List, Tuple, Callable, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import helpers (assume same package)
from dsl import (
    one_hot, grid_sample_ndhwc,
    affine_grid_rotation, affine_grid_translate,
    neighbor_affinity, init_seeds_soft, soft_message_pass,
    soft_pool_features, four_neigh_conv, merge_by_prob,
    flip_h, flip_v, scale, color_replace,
    mask_union, mask_intersection, mask_difference,
    dilate, erode, paint, relate, select_largest,
    permute_colors, draw_rect, crop_around_object,
    color_eq, filter_by_color, paste_object_mean_color
)
from dsl import obj_to_px, px_to_obj

# ============================
# Data containers
# ============================
class DSEState(NamedTuple):
    G: torch.Tensor   # (B,H,W,C) — soft one-hot grid
    A: torch.Tensor   # (B,H,W,K) — soft pixel->object assignment
    T: torch.Tensor   # (B,K,F)   — object table
    aux: Dict[str, torch.Tensor]  # scratchpad: {"last_mask_px":(B,H,W), "last_mask_obj":(B,K), "selection_obj":(B,K)}


# ============================
# Small networks (pluggable)
# ============================
class SmallCNN(nn.Module):
    def __init__(self, in_ch=10, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1))
    def forward(self, x):  # x: (B,C,H,W)
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_out))
    def forward(self, x):
        return self.net(x)


# ============================
# DSE Core
# ============================
class DSE(nn.Module):
    """Differentiable Symbolic Executor
    - Holds differentiable semantics for a small ARC-style DSL
    - All ops are pure, side-effect-free, and accept/return DSEState
    - Provides registry so a controller (policy/program) can call ops by name
    """
    def __init__(self, C=10, K=32, F_attr=0, embed_dim=32):
        super().__init__()
        self.C, self.K = C, K
        # Perception / pixel embedding for objectization
        self.embed = SmallCNN(in_ch=C, out_ch=embed_dim)
        # Attribute heads for filtering on object table
        # We keep a generic attribute MLP; specific filters can be learned concepts
        self.attr_mlp = MLP(self.feature_dim(C), 1)
        # Prototype decoder from object features -> color logits (for recon loss)
        self.color_proto = MLP(self.feature_dim(C), C)
        self.op_registry: Dict[str, Callable[[DSEState, Dict[str, Any]], DSEState]] = {}
        self._register_builtin_ops()

    # ------------------------------------------------------
    # Object feature layout must match helpers.soft_pool_features
    # [size(1), cx(1), cy(1), color_hist(C), bbox(4)] -> 1+2+C+4
    # ------------------------------------------------------
    def feature_dim(self, C: int) -> int:
        return 1 + 2 + C + 4

    # ============================
    # Public API
    # ============================
    def init_state(self, X: torch.Tensor) -> DSEState:
        """X: (B,H,W) integer colors in {0..C-1}. Returns soft-init state."""
        B, H, W = X.shape
        G = F.one_hot(X, self.C).float()
        # differentiable object extraction (soft CC)
        phi = self.embed(G.permute(0,3,1,2)).permute(0,2,3,1)  # (B,H,W,D)
        w = neighbor_affinity(phi)
        A = init_seeds_soft(G, self.K)
        for _ in range(8):
            A = soft_message_pass(A, w, beta=2.0)
        T = soft_pool_features(G, A)
        aux: Dict[str, torch.Tensor] = {}
        return DSEState(G=G, A=A, T=T, aux=aux)

    def register(self, name: str, fn: Callable[[DSEState, Dict[str, Any]], DSEState]):
        self.op_registry[name] = fn

    def forward(self, state: DSEState, program: List[Tuple[str, Dict[str, Any]]]) -> DSEState:
        """Execute a program = list of (op_name, kwargs) with differentiable semantics."""
        for op_name, kwargs in program:
            state = self._call(op_name, state, kwargs)
        return state

    # ============================
    # Built-in ops
    # ============================
    def _register_builtin_ops(self):
        self.register("ExtractObjects", self.op_extract_objects)
        self.register("Filter", self.op_filter)
        self.register("Rotate", self.op_rotate)
        self.register("Translate", self.op_translate)
        self.register("FloodFill", self.op_flood_fill)
        self.register("Count", self.op_count)
        self.register("IfElse", self.op_ifelse)
        self.register("Compose", self.op_compose)
        # extras
        self.register("FlipH", self.op_flip_h)
        self.register("FlipV", self.op_flip_v)
        self.register("Scale", self.op_scale)
        self.register("ColorReplace", self.op_color_replace)
        self.register("Dilate", self.op_dilate)
        self.register("Erode", self.op_erode)
        self.register("Relate", self.op_relate)
        self.register("FilterByColor", self.op_filter_by_color)
        self.register("Paint", self.op_paint)
        self.register("SelectLargest", self.op_select_largest)
        self.register("CropAround", self.op_crop_around)
        self.register("SelectMaskPxFromObj", self.op_objmask_to_px)
        self.register("ObjMaskToPx", self.op_objmask_to_px)  # alias
        self.register("PxMaskToObj", self.op_pxmask_to_obj)
        self.register("SelectMaskObjFromPx", self.op_pxmask_to_obj)  # alias

    # --------- Ops (each returns a new state) ---------
    def op_extract_objects(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        steps = kw.get("steps", 8)
        beta = kw.get("beta", 2.0)
        phi = self.embed(s.G.permute(0,3,1,2)).permute(0,2,3,1)
        w = neighbor_affinity(phi, tau=kw.get("tau", 0.5))
        A = s.A
        for _ in range(steps):
            A = soft_message_pass(A, w, beta=beta)
        T = soft_pool_features(s.G, A)
        return DSEState(G=s.G, A=A, T=T)

    def op_filter(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        """Filter returns mask over objects in T; store in aux['last_mask_obj'] for later ops."""
        logits = self.attr_mlp(s.T)  # (B,K,1)
        pmask = torch.sigmoid(logits).squeeze(-1)   # (B,K)
        aux = dict(s.aux); aux["last_mask_obj"] = pmask
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)

    def op_filter_by_color(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        color_idx = kw["color"]
        pmask = filter_by_color(s.T, color_idx)  # (B,K)
        aux = dict(s.aux); aux["last_mask_obj"] = pmask
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)

    def op_objmask_to_px(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        """Convert an object mask (kw['mask_obj'] or aux['last_mask_obj']) to a pixel mask and stash in aux['last_mask_px']."""
        m_obj = kw.get("mask_obj", s.aux.get("last_mask_obj", None))
        if m_obj is None:
            B, K = s.T.shape[0], s.T.shape[1]
            m_obj = torch.ones(B, K, device=s.T.device, dtype=s.T.dtype) / max(K, 1)
        M = obj_to_px(s.A, m_obj)  # (B,H,W)
        aux = dict(s.aux); aux["last_mask_px"] = M
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)
    
    def op_pxmask_to_obj(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        """
        Convert a pixel mask (kw['mask_px'] or aux['last_mask_px']) to an object mask and
        stash it in aux['last_mask_obj'] (and also 'selection_obj').
        kw:
        - mask_px: (B,H,W) optional; if absent, uses aux['last_mask_px']
        - normalize: bool (default True) -> fraction of object covered vs raw mass
        - sharpen_tau: float (optional) -> if provided, apply softmax sharpening over objects
        """
        M_px = kw.get("mask_px", s.aux.get("last_mask_px", None))
        if M_px is None:
            # fallback: no-op uniform mask to avoid crash
            B, H, W, K = s.G.size(0), s.G.size(1), s.G.size(2), s.A.size(-1)
            M_px = torch.zeros(B, H, W, device=s.G.device, dtype=s.G.dtype)

        m_obj = px_to_obj(s.A, M_px, normalize=kw.get("normalize", True))  # (B,K)

        # Optional sharpening to emphasize the most-covered objects
        tau = kw.get("sharpen_tau", None)
        if tau is not None and tau > 0:
            m_obj = torch.softmax(m_obj / tau, dim=-1)

        aux = dict(s.aux)
        aux["last_mask_obj"] = m_obj
        aux["selection_obj"] = m_obj
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)


    def op_rotate(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        theta = kw.get("theta")  # (B,) in degrees or scalar
        if not torch.is_tensor(theta):
            theta = torch.tensor([theta], device=s.G.device, dtype=s.G.dtype).repeat(s.G.size(0))
        grid = affine_grid_rotation(s.G.shape[1:3], theta)
        G = grid_sample_ndhwc(s.G, grid, mode=kw.get("mode", "nearest"))
        T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)

    def op_translate(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        dx = kw.get("dx", 0.0); dy = kw.get("dy", 0.0)
        if not torch.is_tensor(dx):
            dx = torch.tensor([dx], device=s.G.device, dtype=s.G.dtype).repeat(s.G.size(0))
        if not torch.is_tensor(dy):
            dy = torch.tensor([dy], device=s.G.device, dtype=s.G.dtype).repeat(s.G.size(0))
        grid = affine_grid_translate(s.G.shape[1:3], dx, dy)
        G = grid_sample_ndhwc(s.G, grid, mode=kw.get("mode", "nearest"))
        T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)

    def op_flood_fill(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        # Seed can be: pixel mask (B,H,W) or object mask (B,K) — here we support pixel mask.
        seed_px: torch.Tensor = kw["seed_mask"]  # (B,H,W) in [0,1]
        color_idx: int = kw["color"]
        T_iter = kw.get("steps", 6)
        alpha = kw.get("alpha", 2.0); beta = kw.get("beta", 1.0)
        R = seed_px
        for _ in range(T_iter):
            neigh = four_neigh_conv(R)  # (B,H,W)
            R = torch.sigmoid(alpha * neigh + beta * seed_px)
        G = paint(s.G, R, color_idx)
        aux = dict(s.aux); aux["last_mask_px"] = R
        return DSEState(G=G, A=s.A, T=soft_pool_features(G, s.A), aux=aux)

    def op_paint(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        mask: torch.Tensor = kw.get("mask") or s.aux.get("last_mask_px")  # (B,H,W)
        color_idx: int = kw["color"]
        G = paint(s.G, mask, color_idx)
        return DSEState(G=G, A=s.A, T=soft_pool_features(G, s.A), aux=s.aux)

    def op_count(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        """Count over last stored mask (object or pixel). Store scalar in aux['last_count']."""
        if kw.get("use_px", False) and ("last_mask_px" in s.aux):
            count = s.aux["last_mask_px"].sum(dim=(1,2), keepdim=True)  # (B,1)
        else:
            m = s.aux.get("last_mask_obj", torch.ones_like(s.T[:,:,0]))  # (B,K)
            count = m.sum(dim=1, keepdim=True)
        aux = dict(s.aux); aux["last_count"] = count.squeeze(-1)  # (B,)
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)

    def op_ifelse(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        cond_logit: torch.Tensor = kw["cond_logit"]  # (B,)
        then_prog: List[Tuple[str, Dict[str, Any]]] = kw["then"]
        else_prog: List[Tuple[str, Dict[str, Any]]] = kw["else"]
        p = torch.sigmoid(kw.get("lambda", 4.0) * cond_logit)  # (B,)
        s_then = self.forward(s, then_prog)
        s_else = self.forward(s, else_prog)
        # Blend states
        G = merge_by_prob(s_then.G, s_else.G, p)
        A = merge_by_prob(s_then.A, s_else.A, p)
        T = merge_by_prob(s_then.T, s_else.T, p)
        # aux: blend keys present in either branch
        aux = dict()
        keys = set(s_then.aux.keys()) | set(s_else.aux.keys()) | set(s.aux.keys())
        for k in keys:
            if (k in s_then.aux) and (k in s_else.aux):
                aux[k] = merge_by_prob(s_then.aux[k], s_else.aux[k], p)
            elif k in s_then.aux:
                aux[k] = merge_by_prob(s_then.aux[k], s_else.aux.get(k, s.aux.get(k)), p)
            elif k in s_else.aux:
                aux[k] = merge_by_prob(s_then.aux.get(k, s.aux.get(k)), s_else.aux[k], p)
        return DSEState(G=G, A=A, T=T, aux=aux)

    def op_compose(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        ops: List[Tuple[str, Dict[str, Any]]] = kw["ops"]
        return self.forward(s, ops)

    # Extras wired to helpers
    def op_flip_h(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        G = flip_h(s.G); T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)
    def op_flip_v(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        G = flip_v(s.G); T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)
    def op_scale(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        sx = kw.get("sx", 1.0); sy = kw.get("sy", 1.0)
        if not torch.is_tensor(sx): sx = torch.tensor([sx], device=s.G.device, dtype=s.G.dtype).repeat(s.G.size(0))
        if not torch.is_tensor(sy): sy = torch.tensor([sy], device=s.G.device, dtype=s.G.dtype).repeat(s.G.size(0))
        G = scale(s.G, sx, sy, mode=kw.get("mode", "nearest"))
        T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)
    def op_color_replace(self, s, kw):
        alpha = kw.get("alpha", 1.0)
        # color ids must be Python ints for the helper
        c_from = int(kw["from"])
        c_to   = int(kw["to"])
        G = color_replace(s.G, c_from, c_to, alpha=alpha)   # helper now handles alpha shapes
        T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)
    def op_dilate(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        mask = kw.get("mask") or s.aux.get("last_mask_px"); k = kw.get("k", 3)
        M = dilate(mask, k=k)
        aux = dict(s.aux); aux["last_mask_px"] = M
        G = paint(s.G, M, kw.get("color", 1)) if kw.get("paint", False) else s.G
        return DSEState(G=G, A=s.A, T=s.T, aux=aux)
    def op_erode(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        mask = kw.get("mask") or s.aux.get("last_mask_px"); k = kw.get("k", 3)
        M = erode(mask, k=k)
        aux = dict(s.aux); aux["last_mask_px"] = M
        G = paint(s.G, M, kw.get("color", 1)) if kw.get("paint", False) else s.G
        return DSEState(G=G, A=s.A, T=s.T, aux=aux)
    def op_relate(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        direction = kw.get("dir", "right")
        R = relate(s.T, direction)
        pmask = R.max(dim=2).values  # (B,K)
        aux = dict(s.aux); aux["last_mask_obj"] = pmask
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)
    def op_select_largest(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        prior = s.aux.get("last_mask_obj", s.T[:, :, 0].new_ones(s.T.size(0), s.T.size(1)))
        sel = select_largest(prior, s.T)  # (B,K)
        aux = dict(s.aux); aux["selection_obj"] = sel; aux["last_mask_obj"] = sel
        return DSEState(G=s.G, A=s.A, T=s.T, aux=aux)
    def op_crop_around(self, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        sel = s.aux.get("selection_obj", s.aux.get("last_mask_obj"))
        if sel is None:
            # default: uniform selection
            sel = torch.ones(s.T.size(0), s.T.size(1), device=s.T.device, dtype=s.T.dtype)
            sel = sel / sel.sum(dim=1, keepdim=True)
        G = crop_around_object(s.G, s.T, sel, margin=kw.get("margin", 0.05))
        T = soft_pool_features(G, s.A)
        return DSEState(G=G, A=s.A, T=T, aux=s.aux)

    # ============================
    # Private
    # ============================
    def _call(self, name: str, s: DSEState, kw: Dict[str, Any]) -> DSEState:
        if name not in self.op_registry:
            raise KeyError(f"Op {name} not registered.")
        return self.op_registry[name](s, kw)


# ============================
# Example: small loss heads (optional)
# ============================
class DSELosses(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.C = C

    def forward(self, state: DSEState, target_X: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if target_X is not None:
            # Use KLDiv between log-probabilities and one-hot targets (more stable
            # because state.G is a probability tensor, not raw logits).
            GT = F.one_hot(target_X, self.C).float()                      # (B,H,W,C)
            logP = state.G.clamp_min(1e-6).log()                          # (B,H,W,C)
            losses["grid_ce"] = F.kl_div(
                logP.reshape(-1, self.C),
                GT.reshape(-1, self.C),
                reduction="batchmean",
            )
        # Encourage sharp but smooth assignments
        A = state.A.clamp(1e-6, 1.0)
        ent = -(A * A.log()).sum(dim=-1).mean()
        losses["assign_entropy"] = 0.01 * ent
        # Total variation on assignments (spatial smoothness)
        tv = (A[:, 1:, :, :] - A[:, :-1, :, :]).abs().mean() + (A[:, :, 1:, :] - A[:, :, :-1, :]).abs().mean()
        losses["assign_tv"] = 0.01 * tv
        return losses


# ============================
# Usage snippet (for reference)
# ============================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, C, K = 2, 16, 16, 10, 9
    X = torch.randint(0, C, (B, H, W))
    dse = DSE(C=C, K=K)
    s0 = dse.init_state(X)
    program = [
        ("FilterByColor", {"color": 2}),          # append mask col to T
        ("SelectLargest", {}),                      # append soft selection col
        ("CropAround", {"margin": 0.05}),          # crop & resample around selected
        ("Rotate", {"theta": 90}),                 # rotate grid
        ("ColorReplace", {"from": 2, "to": 3}),   # swap colors softly
    ]
    s1 = dse(s0, program)
    crit = DSELosses(C)
    losses = crit(s1, target_X=X)
    total = sum(losses.values())
    total.backward()
    print("Ran program; total loss:", float(total))

    torch.manual_seed(0)
    B, H, W, C, K = 2, 16, 16, 10, 9
    X = torch.randint(0, C, (B, H, W))
    dse = DSE(C=C, K=K)
    s0 = dse.init_state(X)
    program = [
        ("FilterByColor", {"color": 2}),          # append mask col to T
        ("SelectLargest", {}),                      # append soft selection col
        ("CropAround", {"margin": 0.05}),          # crop & resample around selected
        ("Rotate", {"theta": 90}),                 # rotate grid
        ("ColorReplace", {"from": 2, "to": 3}),   # swap colors softly
    ]
    s1 = dse(s0, program)
    crit = DSELosses(C)
    losses = crit(s1, target_X=X)
    total = sum(losses.values())
    total.backward()
    print("Ran program; total loss:", float(total))

    torch.manual_seed(0)
    B, H, W, C, K = 2, 16, 16, 10, 9
    X = torch.randint(0, C, (B, H, W))
    dse = DSE(C=C, K=K)
    s0 = dse.init_state(X)
    program = [
        ("SelectMaskPxFromObj", {}),            # if you already had an obj mask and turned it into px
        ("PxMaskToObj", {"normalize": True}),   # now convert px back to object weights
        ("SelectLargest", {}),                  # optional: pick the largest among the selected
    ]
    s1 = dse(s0, program)
    crit = DSELosses(C)
    losses = crit(s1, target_X=X)
    total = sum(losses.values())
    total.backward()
    print("Ran program; total loss:", float(total))
