# dse_helpers.py — Differentiable Symbolic Executor helpers and extra ops
# Requirements: torch >= 2.0

from __future__ import annotations
import math
from typing import Tuple, NamedTuple, Callable, List, Dict, Any

import torch
import torch.nn.functional as F

# -----------------------------
# Small utilities
# -----------------------------

def one_hot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot for integer tensor idx of shape (...,) -> (..., C) as float.
    Accepts negative idx as 'no class' (returns all zeros).
    """
    assert num_classes > 0
    idx_long = idx.long().clamp(min=-1)
    out_shape = (*idx_long.shape, num_classes)
    out = idx_long.new_zeros(out_shape, dtype=torch.float32)
    mask = idx_long >= 0
    if mask.any():
        out[mask, idx_long[mask]] = 1.0
    return out


def grid_sample_ndhwc(G: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Wrapper for grid_sample that accepts G in (B,H,W,C) and returns same shape.
    Grid is (B,H,W,2) in [-1,1]."""
    B, H, W, C = G.shape
    x = G.permute(0, 3, 1, 2)  # (B,C,H,W)
    y = F.grid_sample(x, grid, mode=mode, align_corners=False)
    return y.permute(0, 2, 3, 1)


# -----------------------------
# Affine grids
# -----------------------------

def affine_grid_rotation(hw: Tuple[int, int], theta_deg: torch.Tensor) -> torch.Tensor:
    """Create a sampling grid that rotates by theta (degrees, per-batch) around image center.
    hw: (H,W); theta_deg: (B,) tensor.
    Returns grid (B,H,W,2).
    """
    H, W = hw
    B = theta_deg.shape[0]
    theta_rad = theta_deg * math.pi / 180.0
    cos_t = torch.cos(theta_rad)
    sin_t = torch.sin(theta_rad)
    A = torch.zeros(B, 2, 3, device=theta_deg.device, dtype=theta_deg.dtype)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    grid = F.affine_grid(A, size=(B, 1, H, W), align_corners=False)
    return grid


def affine_grid_translate(hw: Tuple[int, int], dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Create a sampling grid for translation by (dx,dy) pixels (per-batch). dx,dy: (B,).
    Returns grid (B,H,W,2)."""
    H, W = hw
    B = dx.shape[0]
    # Convert pixels -> normalized coords in [-1,1]
    tx = 2.0 * dx / max(W - 1, 1)
    ty = 2.0 * dy / max(H - 1, 1)
    A = torch.zeros(B, 2, 3, device=dx.device, dtype=dx.dtype)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 0, 2] = tx
    A[:, 1, 2] = ty
    grid = F.affine_grid(A, size=(B, 1, H, W), align_corners=False)
    return grid


# -----------------------------
# Neighborhood handling & message passing
# -----------------------------

def _shift(x: torch.Tensor, dy: int, dx: int, pad_val: float = 0.0) -> torch.Tensor:
    """Shift 2D map x (B,H,W,...) by (dy,dx) with zero padding. Positive dy shifts down."""
    B, H, W = x.shape[:3]
    pad_t = max(dy, 0)
    pad_b = max(-dy, 0)
    pad_l = max(dx, 0)
    pad_r = max(-dx, 0)
    x2 = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b), value=pad_val)
    y = x2[:, pad_t:pad_t+H, pad_l:pad_l+W, ...]
    return y


def neighbor_affinity(phi: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Compute 4-neighborhood affinities from per-pixel embeddings phi (B,H,W,D).
    Returns w (B,H,W,4) for directions [up, down, left, right]."""
    B, H, W, D = phi.shape
    # neighbor embeddings
    phi_up = _shift(phi, -1, 0)
    phi_down = _shift(phi, 1, 0)
    phi_left = _shift(phi, 0, -1)
    phi_right = _shift(phi, 0, 1)
    def sim(a, b):
        d2 = ((a - b) ** 2).sum(dim=-1)
        return torch.sigmoid(-d2 / max(tau, 1e-6))
    w_up = sim(phi, phi_up)
    w_down = sim(phi, phi_down)
    w_left = sim(phi, phi_left)
    w_right = sim(phi, phi_right)
    w = torch.stack([w_up, w_down, w_left, w_right], dim=-1)
    return w


def init_seeds_soft(G: torch.Tensor, K: int) -> torch.Tensor:
    """Produce initial soft assignments A0 (B,H,W,K) using evenly spaced Gaussian seeds.
    This is parameter-free and stable; the model can refine via message passing.
    """
    B, H, W, C = G.shape
    device = G.device
    # Place K centers on a sqrt(K) x sqrt(K) grid
    n = int(math.ceil(math.sqrt(K)))
    ys = torch.linspace(0.1, 0.9, steps=n, device=device)
    xs = torch.linspace(0.1, 0.9, steps=n, device=device)
    centers = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).view(-1, 2)  # (n*n,2) in [0,1]
    centers = centers[:K]
    # Pixel coordinate grid in [0,1]
    yy = torch.linspace(0, 1, steps=H, device=device)
    xx = torch.linspace(0, 1, steps=W, device=device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")  # (H,W)
    coords = torch.stack([Y, X], dim=-1)  # (H,W,2)
    # RBF distance to centers
    sigma2 = 0.03
    d2 = ((coords[None, :, :, None, :] - centers[None, None, None, :, :]) ** 2).sum(dim=-1)  # (1,H,W,K)
    logits = -d2 / sigma2
    A0 = F.softmax(logits.expand(B, -1, -1, -1), dim=-1)
    return A0


def soft_message_pass(A: torch.Tensor, w: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """One step of soft message passing for assignments A (B,H,W,K) with 4-neigh affinities w (B,H,W,4).
    Returns updated A with per-pixel softmax over K.
    """
    # Gather neighbor-weighted assignments per direction
    A_up = _shift(A, -1, 0)
    A_down = _shift(A, 1, 0)
    A_left = _shift(A, 0, -1)
    A_right = _shift(A, 0, 1)
    msg = (
        w[..., 0:1] * A_up +
        w[..., 1:2] * A_down +
        w[..., 2:3] * A_left +
        w[..., 3:4] * A_right
    )
    logits = beta * msg
    return F.softmax(logits, dim=-1)


# -----------------------------
# Feature pooling for object table
# -----------------------------

def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.sum(dim=(1, 2), keepdim=True) + eps)


def soft_pool_features(G: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Build object table T from grid G (B,H,W,C) and assignments A (B,H,W,K).
    Features per object (concatenated):
    [size, cx, cy, color_hist(C), bbox_soft(4)]  -> F = 1 + 2 + C + 4
    All coords normalized to [0,1].
    """
    B, H, W, C = G.shape
    K = A.shape[-1]
    device = G.device
    # Normalize per-object pixel weights
    Wk = _safe_normalize(A)  # (B,H,W,K)
    # Coordinates in [0,1]
    yy = torch.linspace(0, 1, steps=H, device=device).view(1, H, 1, 1)
    xx = torch.linspace(0, 1, steps=W, device=device).view(1, 1, W, 1)
    # Size
    size = A.sum(dim=(1, 2)) / float(H * W)  # (B,K)
    # Centroids
    cy = (Wk * yy).sum(dim=(1, 2))  # (B,K)
    cx = (Wk * xx).sum(dim=(1, 2))  # (B,K)
    # Color histogram / mean color
    color_hist = (A.unsqueeze(-1) * G.unsqueeze(-2)).sum(dim=(1, 2))  # (B,K,C)
    denom = A.sum(dim=(1, 2)).unsqueeze(-1) + 1e-6
    color_hist = color_hist / denom  # mean color per object
    # Soft bbox via soft-(min,max) using log-sum-exp as a smooth maximum
    t = 20.0  # temperature; higher -> sharper
    # For min, use -max on negative coords with weights
    def soft_max_w(x, w):
        # x: (B,H,W,1), w: (B,H,W,K)
        z = t * x
        m = (w * z).amax(dim=(1, 2), keepdim=False)  # (B,K)
        # log-sum-exp with weights approximated by max (stable & fast)
        return m / t
    def soft_min_w(x, w):
        return -soft_max_w(-x, w)
    Y = yy.expand(B, H, W, 1)
    X = xx.expand(B, H, W, 1)
    top = soft_min_w(Y, A)
    bottom = soft_max_w(Y, A)
    left = soft_min_w(X, A)
    right = soft_max_w(X, A)
    bbox = torch.stack([top, left, bottom, right], dim=-1)  # (B,K,4)
    # Assemble features
    feats = [
        size.unsqueeze(-1),           # (B,K,1)
        cx.unsqueeze(-1),             # (B,K,1)
        cy.unsqueeze(-1),             # (B,K,1)
        color_hist,                   # (B,K,C)
        bbox,                         # (B,K,4)
    ]
    T = torch.cat(feats, dim=-1)      # (B,K,F)
    return T


# -----------------------------
# Region ops
# -----------------------------

def four_neigh_conv(mask: torch.Tensor) -> torch.Tensor:
    """Apply 4-neighborhood sum conv to a (B,1,H,W) or (B,H,W) mask and return (B,1,H,W).
    """
    squeezed = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
        squeezed = True
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=mask.device, dtype=mask.dtype
    ).view(1, 1, 3, 3)
    y = F.conv2d(mask, kernel, padding=1)
    if squeezed:
        y = y.squeeze(1)
    return y


def merge_by_prob(a: Any, b: Any, p: torch.Tensor) -> Any:
    """Blend two tensors/tuples of tensors with scalar or (B,)-shaped prob p in [0,1].
    Supports structures (G,A,T) as tuples or NamedTuple with .G/.A/.T.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(merge_by_prob(x, y, p) for x, y in zip(a, b))
    if hasattr(a, "_fields") and hasattr(b, "_fields"):
        return a.__class__(*(merge_by_prob(x, y, p) for x, y in zip(a, b)))
    # broadcast p
    while p.dim() < a.dim():
        p = p.view(-1, *([1] * (a.dim() - 1)))
    return p * a + (1.0 - p) * b


# -----------------------------
# EXTRA PRIMITIVES (useful for ARC-AGI)
# -----------------------------

# 1) Flip operations (exact, differentiable)

def flip_h(G: torch.Tensor) -> torch.Tensor:
    return torch.flip(G, dims=[2])  # flip width


def flip_v(G: torch.Tensor) -> torch.Tensor:
    return torch.flip(G, dims=[1])  # flip height


# 2) Scale (nearest/bilinear via grid_sample)

def scale(G: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    B, H, W, C = G.shape
    # Build affine with scaling factors
    A = torch.zeros(B, 2, 3, device=G.device, dtype=G.dtype)
    A[:, 0, 0] = sx
    A[:, 1, 1] = sy
    grid = F.affine_grid(A, size=(B, 1, H, W), align_corners=False)
    return grid_sample_ndhwc(G, grid, mode=mode)


# 3) ColorReplace (soft): replace probability mass of color a with color b

def color_replace(G: torch.Tensor, color_from: int, color_to: int, alpha=1.0) -> torch.Tensor:
    """
    Replace soft mass of color_from with color_to. Supports alpha as float or (B,) or (B,1,1,1).
    """
    B, H, W, C = G.shape
    device, dtype = G.device, G.dtype

    def _to_alpha(a):
        if torch.is_tensor(a):
            if a.dim() == 1:   # (B,)
                return a.view(-1, 1, 1, 1).to(dtype)
            if a.dim() == 4:   # (B,1,1,1)
                return a.to(dtype)
            # fallback: make scalar
            return a.mean().view(1, 1, 1, 1).to(dtype)
        else:
            return torch.tensor(float(a), device=device, dtype=dtype).view(1, 1, 1, 1)

    a = _to_alpha(alpha)

    cf = F.one_hot(torch.tensor(color_from, device=device), C).view(1, 1, 1, C).float()
    ct = F.one_hot(torch.tensor(color_to,   device=device), C).view(1, 1, 1, C).float()
    w = (G * cf).sum(dim=-1, keepdim=True)  # (B,H,W,1)
    return (1 - a * w) * G + a * w * ct


# 4) Mask set ops

def mask_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b - a * b, 0.0, 1.0)


def mask_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def mask_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * (1.0 - b)


# 5) Morphology (soft) on masks using log-sum-exp pooling

def dilate(mask: torch.Tensor, k: int = 3, t: float = 6.0) -> torch.Tensor:
    if mask.dim() == 3:
        x = mask.unsqueeze(1)
    else:
        x = mask
    pad = k // 2
    # Unfold and softmax over window
    patches = F.unfold(x, kernel_size=k, padding=pad)  # (B, k*k, H*W)
    patches = patches.transpose(1, 2)  # (B, H*W, k*k)
    y = (patches * F.softmax(t * patches, dim=-1)).sum(dim=-1)  # (B, H*W)
    y = y.view(x.size(0), 1, mask.size(-2), mask.size(-1))
    return y.squeeze(1) if mask.dim() == 3 else y


def erode(mask: torch.Tensor, k: int = 3, t: float = 6.0) -> torch.Tensor:
    return 1.0 - dilate(1.0 - mask, k=k, t=t)


# 6) Paint with mask (already implied by FloodFill, but handy)

def paint(G: torch.Tensor, mask: torch.Tensor, color_idx: int) -> torch.Tensor:
    C = G.size(-1)
    paint_vec = F.one_hot(torch.tensor(color_idx, device=G.device), C).view(1, 1, 1, C).float()
    M = mask.unsqueeze(-1)
    return (1 - M) * G + M * paint_vec


# 7) Relate(dir) — directional neighbor selection between objects

def relate(T: torch.Tensor, direction: str = "right", kappa: float = 12.0) -> torch.Tensor:
    """Return relation scores R_ij (B,K,K) indicating j is in direction of i.
    Uses centroids from T: [size, cx, cy, ...]."""
    cx = T[:, :, 1]
    cy = T[:, :, 2]
    dx = cx.unsqueeze(2) - cx.unsqueeze(1)
    dy = cy.unsqueeze(2) - cy.unsqueeze(1)
    if direction == "right":
        s = torch.sigmoid(kappa * (dx))
    elif direction == "left":
        s = torch.sigmoid(kappa * (-dx))
    elif direction == "down":
        s = torch.sigmoid(kappa * (dy))
    elif direction == "up":
        s = torch.sigmoid(kappa * (-dy))
    else:
        raise ValueError("direction ∈ {right,left,down,up}")
    return s  # (B,K,K)


# 8) SelectLargestObject — soft one-hot over objects

def select_largest(M_obj: torch.Tensor, T: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    size = T[:, :, 0]  # (B,K)
    logits = (M_obj * size)
    logits = logits / max(tau, 1e-6)
    return F.softmax(logits, dim=-1)  # (B,K)


# 9) Permute colors by index mapping (e.g., [2,0,1,...])

def permute_colors(G: torch.Tensor, perm: List[int]) -> torch.Tensor:
    P = torch.tensor(perm, device=G.device, dtype=torch.long)
    return G.index_select(-1, P)


# 10) Draw line/rect (soft rasterization)

def draw_rect(G: torch.Tensor, top: float, left: float, bottom: float, right: float, color_idx: int, edge: float = 0.01) -> torch.Tensor:
    B, H, W, C = G.shape
    yy = torch.linspace(0, 1, steps=H, device=G.device).view(1, H, 1)
    xx = torch.linspace(0, 1, steps=W, device=G.device).view(1, 1, W)
    # Soft edges using sigmoid ramps
    tmask = torch.sigmoid((yy - top) / edge) * (1 - torch.sigmoid((yy - (top + edge)) / edge))
    bmask = torch.sigmoid(((bottom) - yy) / edge) * (1 - torch.sigmoid(((bottom - edge) - yy) / edge))
    lmask = torch.sigmoid((xx - left) / edge) * (1 - torch.sigmoid((xx - (left + edge)) / edge))
    rmask = torch.sigmoid(((right) - xx) / edge) * (1 - torch.sigmoid(((right - edge) - xx) / edge))
    border_y = (tmask + bmask).clamp(0, 1).unsqueeze(-1)
    border_x = (lmask + rmask).clamp(0, 1).unsqueeze(-1)
    border = 1 - (1 - border_y) * (1 - border_x)  # union
    return paint(G, border.squeeze(-1), color_idx)


# 11) Center-crop around an object (soft attention crop -> same size via resample)

def crop_around_object(G: torch.Tensor, T: torch.Tensor, sel: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Select an object via soft weights sel (B,K) and crop its bbox with margin, resampling back to original size.
    """
    B, H, W, C = G.shape
    bbox = (sel.unsqueeze(-1) * T[:, :, -4:]).sum(dim=1)  # (B,4)
    top, left, bottom, right = bbox.unbind(dim=-1)
    # Expand by margin
    h = (bottom - top)
    w = (right - left)
    top = (top - margin * h).clamp(0, 1)
    left = (left - margin * w).clamp(0, 1)
    bottom = (bottom + margin * h).clamp(0, 1)
    right = (right + margin * w).clamp(0, 1)
    # Build affine that maps target output to this bbox
    sx = (right - left)
    sy = (bottom - top)
    tx = (left + right - 1)
    ty = (top + bottom - 1)
    A = torch.zeros(B, 2, 3, device=G.device, dtype=G.dtype)
    A[:, 0, 0] = sx
    A[:, 1, 1] = sy
    A[:, 0, 2] = tx
    A[:, 1, 2] = ty
    grid = F.affine_grid(A, size=(B, 1, H, W), align_corners=False)
    return grid_sample_ndhwc(G, grid, mode="nearest")


# 12) Soft equality of colors (returns mask of pixels matching color_idx)

def color_eq(G: torch.Tensor, color_idx: int, tau: float = 0.3) -> torch.Tensor:
    B, H, W, C = G.shape
    target = F.one_hot(torch.tensor(color_idx, device=G.device), C).view(1, 1, 1, C).float()
    logits = (G * target).sum(dim=-1) / max(tau, 1e-6)
    return torch.sigmoid(12.0 * (logits - 0.5))  # sharpened


# 13) Select objects by color

def filter_by_color(T: torch.Tensor, color_idx: int, tau: float = 0.2) -> torch.Tensor:
    # color histogram starts at index 3 in our T layout: [size, cx, cy, color_hist(C), bbox(4)]
    C = T.size(-1) - 3 - 4
    color = T[:, :, 3:3+C]
    logits = color[:, :, color_idx] / max(tau, 1e-6)
    return torch.sigmoid(logits)


# 14) Glue/paste object mask from A back to grid color (using its mean color)

def paste_object_mean_color(G: torch.Tensor, A: torch.Tensor, T: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    C = G.size(-1)
    color = (sel.unsqueeze(-1) * T[:, :, 3:3+C]).sum(dim=1).view(-1, 1, 1, C)
    mask = (A * sel.view(-1, 1, 1, A.size(-1))).sum(dim=-1)  # (B,H,W)
    return (1 - mask.unsqueeze(-1)) * G + mask.unsqueeze(-1) * color


# -----------------------------
# Minimal sanity tests (optional)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, C, K = 2, 16, 16, 10, 9
    X = torch.randint(0, C, (B, H, W))
    G = F.one_hot(X, C).float()
    # Fake embeddings
    phi = torch.randn(B, H, W, 16)
    w = neighbor_affinity(phi)
    A0 = init_seeds_soft(G, K)
    A1 = soft_message_pass(A0, w)
    T = soft_pool_features(G, A1)
    # Ops smoke tests
    G2 = grid_sample_ndhwc(G, affine_grid_rotation((H, W), torch.zeros(B)))
    G3 = color_replace(G2, 1, 2)
    m = color_eq(G3, 2)
    G4 = paint(G3, m, 3)
    sel = select_largest(torch.ones(B, K), T)
    G5 = crop_around_object(G4, T, sel)
    print("OK — helpers executed without error.")


def obj_to_px(A: torch.Tensor, m_obj: torch.Tensor) -> torch.Tensor:
    """
    Convert an object mask m_obj (B,K) to a pixel mask (B,H,W)
    via soft assignments A (B,H,W,K).
    """
    return (A * m_obj[:, None, None, :]).sum(dim=-1).clamp(0.0, 1.0)

def px_to_obj(A: torch.Tensor, M_px: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Convert a pixel mask M_px (B,H,W) into an object mask (B,K) using soft assignments A (B,H,W,K).
    If normalize=True, returns fraction of each object covered by the mask; else returns raw mass.
    """
    # raw mass per object under the pixel mask
    mass = (A * M_px[:, :, :, None]).sum(dim=(1, 2))  # (B,K)

    if not normalize:
        return mass.clamp(min=0.0)

    # normalize by total mass of each object so result ~ fraction in [0,1]
    denom = A.sum(dim=(1, 2)).clamp_min(1e-6)         # (B,K)
    return (mass / denom).clamp(0.0, 1.0)