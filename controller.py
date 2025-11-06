# controller_runner.py — Controller (policy/value) and a simple program runner over the DSE
# Depends on: dse_core.py (DSE), dse_helpers.py

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dse import DSE, DSEState

# -----------------------------
# Action space definition (discrete ops + discrete args; simple continuous args too)
# -----------------------------

OP_LIST = [
    # grid geometry
    "Rotate", "Translate", "FlipH", "FlipV", "Scale",
    # colors / painting
    "ColorReplace", "Paint",
    # selection & masks
    "FilterByColor", "SelectLargest", "Relate",
    "SelectMaskPxFromObj", "PxMaskToObj",
    # cropping
    "CropAround",
]
OP2ID = {n:i for i,n in enumerate(OP_LIST)}

# Discrete argument vocabularies
NUM_COLORS = 10
DIR_LIST = ["right","left","down","up"]
DIR2ID = {n:i for i,n in enumerate(DIR_LIST)}

# -----------------------------
# Encoders
# -----------------------------
class GridEncoder(nn.Module):
    def __init__(self, C: int, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(64, d_out)
    def forward(self, G_bhwc: torch.Tensor) -> torch.Tensor:
        x = G_bhwc.permute(0,3,1,2)
        x = self.net(x).flatten(1)
        return self.proj(x)

class TableEncoder(nn.Module):
    def __init__(self, F: int, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, 128), nn.ReLU(inplace=True),
            nn.Linear(128, d_out))
    def forward(self, T: torch.Tensor) -> torch.Tensor:
        # mean-pool over objects
        return self.net(T).mean(dim=1)

# -----------------------------
# Controller with heads for: op, color, dir, obj, and simple continuous args
# -----------------------------
class Controller(nn.Module):
    def __init__(self, C: int, K: int, Fdim: int, d_model: int = 256):
        super().__init__()
        self.C, self.K = C, K
        self.enc_g = GridEncoder(C, d_out=128)
        self.enc_t = TableEncoder(Fdim, d_out=128)
        self.fuse = nn.Sequential(nn.Linear(256, d_model), nn.ReLU(inplace=True))
        # Heads
        self.h_op    = nn.Linear(d_model, len(OP_LIST))
        self.h_color = nn.Linear(d_model, NUM_COLORS)
        self.h_dir   = nn.Linear(d_model, len(DIR_LIST))
        self.h_obj   = nn.Linear(d_model, K)
        self.h_cont  = nn.Linear(d_model, 4)   # theta, dx, dy, alpha
        self.h_value = nn.Linear(d_model, 1)

    def forward(self, state: DSEState) -> Dict[str, torch.Tensor]:
        g = self.enc_g(state.G)
        # infer feature dim from T
        Fdim = state.T.size(-1)
        t = self.enc_t(state.T)
        h = self.fuse(torch.cat([g, t], dim=-1))
        return {
            "logits_op": self.h_op(h),
            "logits_color": self.h_color(h),
            "logits_dir": self.h_dir(h),
            "logits_obj": self.h_obj(h),
            "cont": torch.tanh(self.h_cont(h)),  # in [-1,1]
            "V": self.h_value(h).squeeze(-1),
        }

# -----------------------------
# Sampling helpers
# -----------------------------

def sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=1.0, hard=True).argmax(dim=-1)

def sample_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.gumbel_softmax(logits, tau=1.0, hard=True)

# -----------------------------
# Program runner (single step + rollout of L steps)
# -----------------------------
class ProgramRunner:
    def __init__(self, dse: DSE):
        self.dse = dse

    @torch.no_grad()
    def step(self, ctrl_out: Dict[str, torch.Tensor], state: DSEState) -> Tuple[str, Dict[str, Any]]:
        B = state.G.size(0)
        op_id = sample_categorical(ctrl_out["logits_op"])  # (B,)
        # decode continuous args
        cont = ctrl_out["cont"]
        theta = 180.0 * cont[:, 0]
        dx    =  4.0 * cont[:, 1]
        dy    =  4.0 * cont[:, 2]
        alpha =  1.0 * (cont[:, 3] * 0.5 + 0.5)  # map to [0,1]
        # decode discrete args
        color = sample_categorical(ctrl_out["logits_color"])  # (B,)
        direc = sample_categorical(ctrl_out["logits_dir"])    # (B,)
        obj_w = F.softmax(ctrl_out["logits_obj"], dim=-1)     # (B,K)

        # Build a single batched action by majority op (simple; replace with per-batch loop if needed)
        # For simplicity, pick the mode op across batch to form one batched call.
        with torch.no_grad():
            op_counts = torch.bincount(op_id, minlength=len(OP_LIST))
            op_pick = int(op_counts.argmax().item())
        op_name = OP_LIST[op_pick]

        # kwargs by op
        kw: Dict[str, Any] = {}
        if op_name == "Rotate":
            kw = {"theta": theta}
        elif op_name == "Translate":
            kw = {"dx": dx, "dy": dy}
        elif op_name == "FlipH":
            kw = {}
        elif op_name == "FlipV":
            kw = {}
        elif op_name == "Scale":
            sx = (cont[:, 1] * 0.5 + 0.5) * 1.5  # [0.5, 1.5]
            sy = (cont[:, 2] * 0.5 + 0.5) * 1.5
            kw = {"sx": sx, "sy": sy}
        elif op_name == "ColorReplace":
            frm = int(color[0].item())         # choose batch[0] for now
            to  = (frm + 1) % NUM_COLORS
            kw = {"from": frm, "to": to, "alpha": alpha}  # pass alpha tensor; helper handles it
        elif op_name == "Paint":
            kw = {"color": color.clamp_max(NUM_COLORS-1)}
        elif op_name == "FilterByColor":
            kw = {"color": color.clamp_max(NUM_COLORS-1)}
        elif op_name == "SelectLargest":
            kw = {}
        elif op_name == "Relate":
            # map id->dir string
            dir_idx = int(direc[0].item()) if B==1 else 0
            kw = {"dir": DIR_LIST[dir_idx]}
        elif op_name == "SelectMaskPxFromObj":
            kw = {}
        elif op_name == "PxMaskToObj":
            kw = {"normalize": True, "sharpen_tau": 0.2}
        elif op_name == "CropAround":
            kw = {"margin": 0.05}
        else:
            kw = {}

        return op_name, kw

    def run(self, controller: Controller, state: DSEState, L: int = 6) -> Tuple[DSEState, List[Tuple[str, Dict[str, Any]]]]:
        trace: List[Tuple[str, Dict[str, Any]]] = []
        for _ in range(L):
            out = controller(state)
            op_name, kw = self.step(out, state)
            state = self.dse._call(op_name, state, kw)
            trace.append((op_name, kw))
        return state, trace

# -----------------------------
# Simple behavior-cloning loss (if you have teacher traces)
# -----------------------------
class BCLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ctrl_out: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = F.cross_entropy(ctrl_out["logits_op"], target["op"])  # op ids
        if "color" in target:
            loss = loss + 0.2 * F.cross_entropy(ctrl_out["logits_color"], target["color"]) 
        if "dir" in target:
            loss = loss + 0.2 * F.cross_entropy(ctrl_out["logits_dir"], target["dir"]) 
        if "obj" in target:
            loss = loss + 0.2 * F.cross_entropy(ctrl_out["logits_obj"], target["obj"]) 
        return loss

# -----------------------------
# Smoke test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B,H,W,C,K = 2, 16, 16, 10, 9
    X = torch.randint(0, C, (B,H,W))
    dse = DSE(C=C, K=K)
    s = dse.init_state(X)
    ctrl = Controller(C=C, K=K, Fdim=dse.feature_dim(C))
    runner = ProgramRunner(dse)
    s2, trace = runner.run(ctrl, s, L=4)
    print("Trace:", trace)