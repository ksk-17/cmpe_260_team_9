from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, NamedTuple
import math
import torch
import torch.nn.functional as F

from policy_value_net import (
    PolicyValueNet, PolicyValueConfig, masked_softmax
)

class Action(NamedTuple):
    op_id: int
    cat_args: Dict[str, int]
    ptr_args: Dict[str, int]

    def key(self) -> Tuple:
        # hashable key for dicts
        return ("a", self.op_id,
                tuple(sorted(self.cat_args.items())),
                tuple(sorted(self.ptr_args.items())))

class EdgeStats:
    __slots__ = ("P","N","W","Q")
    def __init__(self, prior: float):
        self.P = float(prior)
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

class Node:
    def __init__(self, state: Any, latent: Optional[torch.Tensor] = None):
        self.state = state
        self.latent = latent           # optional cached latent (not required)
        self.children: Dict[Tuple, "Node"] = {}
        self.edges: Dict[Tuple, EdgeStats] = {}
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_reward = 0.0

# ---------- Config ----------

@dataclass
class MCTSConfig:
    num_simulations: int = 128
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature: float = 1.0
    max_candidates: int = 128
    topk_ops: int = 16
    topk_cat: int = 6
    topk_ptr: int = 8


# helper to convert a hashable key back to an Action
def action_from_key(akey: Tuple) -> Action:
    # akey == ("a", op_id, tuple(sorted(cat_items)), tuple(sorted(ptr_items)))
    _, op_id, cat_t, ptr_t = akey
    return Action(
        op_id=op_id,
        cat_args=dict(cat_t),
        ptr_args=dict(ptr_t),
    )

def action_from_key(akey: tuple) -> Action:
    # akey = ("a", op_id, tuple(sorted(cat_items)), tuple(sorted(ptr_items)))
    _, op_id, cat_t, ptr_t = akey
    return Action(op_id=op_id, cat_args=dict(cat_t), ptr_args=dict(ptr_t))

@torch.no_grad()
def build_action_candidates(
    policy_out: Dict[str, Any],
    object_mask: Optional[torch.Tensor],
    topk_ops: int,
    topk_cat: int,
    topk_ptr: int,
    max_candidates: int
) -> Tuple[List[Action], torch.Tensor]:
    """
    Create a pruned set of joint action candidates with joint priors.
    Returns:
      candidates: list[Action]
      priors:     [K] tensor summing to 1
    """
    op_logits = policy_out["op_logits"]           # [1, n_ops]
    cat_logits = policy_out["cat_logits"]         # dict name->[1, n_cat]
    ptr_scores = policy_out["ptr_scores"]         # dict name->[1, M]

    # Ops
    op_probs = F.softmax(op_logits, dim=-1)       # [1, n_ops]
    k_ops = min(topk_ops, op_probs.size(1))
    op_prob_vals, op_idx = torch.topk(op_probs, k_ops, dim=-1)  # [1,k_ops]

    # Categorical args
    cat_tops: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, logits in cat_logits.items():
        probs = F.softmax(logits, dim=-1)         # [1, n_cat]
        k = min(topk_cat, probs.size(1))
        pv, ix = torch.topk(probs, k, dim=-1)
        cat_tops[name] = (pv.squeeze(0), ix.squeeze(0))  # [k], [k]

    # Pointer args
    ptr_tops: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, scores in ptr_scores.items():
        probs = masked_softmax(scores, object_mask)  # [1, M]
        k = min(topk_ptr, probs.size(1))
        pv, ix = torch.topk(probs, k, dim=-1)
        ptr_tops[name] = (pv.squeeze(0), ix.squeeze(0))  # [k], [k]

    # Cartesian product with pruning
    candidates: List[Action] = []
    priors: List[float] = []

    # Build lists to iterate
    op_list = list(zip(op_prob_vals.squeeze(0).tolist(), op_idx.squeeze(0).tolist()))

    # Prepare factor option lists
    cat_opt_lists: Dict[str, List[Tuple[float,int]]] = {
        n: list(zip(pv.tolist(), ix.tolist())) for n,(pv,ix) in cat_tops.items()
    }
    ptr_opt_lists: Dict[str, List[Tuple[float,int]]] = {
        n: list(zip(pv.tolist(), ix.tolist())) for n,(pv,ix) in ptr_tops.items()
    }

    def recurse_build(i_op: int, cat_items: List[Tuple[str,int,float]],
                      ptr_items: List[Tuple[str,int,float]]):
        # Convert partial to Action/priors
        pass

    # Since we may have multiple argument factors, do a compact nested loop.
    # Start with ops, then greedy mix & match arguments by taking topk per factor
    for p_op, op in op_list:
        # Start with 1 candidate; multiply in each factor, pruning by max_candidates
        partial: List[Tuple[Action, float]] = [(Action(op, {}, {}), p_op)]
        # Categorical args
        for n, opts in cat_opt_lists.items():
            new: List[Tuple[Action, float]] = []
            for act, w in partial:
                for p, idx in opts:
                    if len(new) >= max_candidates: break
                    aa = Action(act.op_id, {**act.cat_args, n: idx}, dict(act.ptr_args))
                    new.append((aa, w * p))
            partial = sorted(new, key=lambda x: -x[1])[:max_candidates]
        # Pointer args
        for n, opts in ptr_opt_lists.items():
            new = []
            for act, w in partial:
                for p, idx in opts:
                    if len(new) >= max_candidates: break
                    aa = Action(act.op_id, dict(act.cat_args), {**act.ptr_args, n: idx})
                    new.append((aa, w * p))
            partial = sorted(new, key=lambda x: -x[1])[:max_candidates]

        # Append to global
        for act, w in partial:
            if len(candidates) >= max_candidates: break
            candidates.append(act)
            priors.append(w)
        if len(candidates) >= max_candidates:
            break

    if len(candidates) == 0:
        # Fallback: pick arg=0 for everything
        fallback = Action(int(op_idx[0,0].item()), {n:0 for n in cat_logits.keys()}, {n:0 for n in ptr_scores.keys()})
        candidates = [fallback]
        priors = [1.0]

    priors_t = torch.tensor(priors, dtype=torch.float32)
    priors_t = priors_t / priors_t.sum().clamp_min(1e-8)
    return candidates, priors_t

class MCTS:
    def __init__(
        self,
        net: PolicyValueNet,
        cfg: MCTSConfig,
        # encoder: function mapping env state -> (state_tokens, attn_mask, object_feats, object_mask)
        encode_state_fn,
        # executor: function applying action -> (next_state, terminal:bool, reward:float)
        step_fn,
        device: torch.device = torch.device("cpu"),
    ):
        self.net = net.to(device)
        self.cfg = cfg
        self.encode_state = encode_state_fn
        self.step = step_fn
        self.device = device

    @torch.no_grad()
    def _eval_state(self, state) -> Tuple[Dict[str,Any], torch.Tensor, Dict[str,torch.Tensor]]:
        st = self.encode_state(state)
        # Expect tuple: (state_tokens[B=1,T], attn_mask[B=1,T], object_feats[B=1,M,F], object_mask[B=1,M])
        out = self.net(*[x.to(self.device) if torch.is_tensor(x) else x for x in st])
        return out, st[3], st  # policy/value, object_mask, raw encodings

    def _select(self, node: Node) -> Tuple[List[Tuple[Node, Action]], Node]:
        path: List[Tuple[Node, Action]] = []
        cur = node
        while cur.is_expanded and not cur.is_terminal:
            # choose action maximizing Q + U
            total_N = sum(e.N for e in cur.edges.values()) + 1
            best_a, best_score = None, -1e9
            for akey, edge in cur.edges.items():
                U = self.cfg.cpuct * edge.P * math.sqrt(total_N) / (1 + edge.N)
                score = edge.Q + U
                if score > best_score:
                    best_score = score
                    best_a = akey
            action_tuple = best_a
            action = Action(action_tuple[1], dict(action_tuple[2]), dict(action_tuple[3]))
            child = cur.children[action_tuple]
            path.append((cur, action))
            cur = child
        return path, cur

    def _expand(self, node: Node):
        if node.is_terminal or node.is_expanded:
            return

        out, object_mask, enc = self._eval_state(node.state)  # B=1
        node.latent = out["latent"]  # cache

        # Build candidates and priors
        cands, pri = build_action_candidates(
            out["policy"], object_mask=object_mask,
            topk_ops=self.cfg.topk_ops, topk_cat=self.cfg.topk_cat,
            topk_ptr=self.cfg.topk_ptr, max_candidates=self.cfg.max_candidates
        )

        # Optional Dirichlet noise at root-like nodes (apply at first expansion)
        noise = torch.distributions.Dirichlet(
            torch.full_like(pri, self.cfg.dirichlet_alpha)
        ).sample()
        pri = (1 - self.cfg.dirichlet_eps) * pri + self.cfg.dirichlet_eps * noise

        for a, p in zip(cands, pri.tolist()):
            akey = a.key()
            if akey not in node.edges:
                node.edges[akey] = EdgeStats(prior=p)
                node.children[akey] = Node(state=None)  # will set after stepping
        node.is_expanded = True

        # Evaluate leaf value
        value = float(out["value"].squeeze(0).item())
        return value, cands, pri

    def _backup(self, path: List[Tuple[Node, Action]], leaf_value: float):
        # Zero-sum not needed; we directly predict probability of success (same player)
        v = leaf_value
        for parent, action in reversed(path):
            edge = parent.edges[action.key()]
            edge.N += 1
            edge.W += v
            edge.Q = edge.W / edge.N

    def run(self, root_state) -> Tuple[Action, Dict[Action, float], Node]:
        root = Node(root_state)

        # If root is terminal, short-circuit via executor (caller should usually check)
        # Expand once to create children/prior
        for sim in range(self.cfg.num_simulations):
            # 1) Selection
            path, leaf = self._select(root)

            # 2) If child state unknown for a parent->child along the path, step it now
            # (We compute child states lazily during selection)
            if path:
                # ensure each step has a materialized child state
                st = root_state
                # replay path from root to assign states where missing
                cursor = root
                for (parent, a) in path:
                    akey = a.key()
                    child = parent.children[akey]
                    if child.state is None:
                        ns, terminal, reward = self.step(parent.state, a)
                        child.state = ns
                        child.is_terminal = terminal
                        if terminal:
                            child.terminal_reward = float(reward)
                    cursor = child

            # 3) Expansion / Evaluation
            if leaf.is_terminal:
                v = leaf.terminal_reward
                self._backup(path, v)
                continue

            res = self._expand(leaf)
            if res is None:
                # Already expanded (rare race), evaluate value directly
                out, _, _ = self._eval_state(leaf.state)
                v = float(out["value"].squeeze(0).item())
            else:
                v, _, _ = res

            self._backup(path, v)

        # Build root policy π̂ from visit counts
        visit_counts = []
        actions = []
        for akey, edge in root.edges.items():
            visit_counts.append(edge.N)
            actions.append(Action(akey[1], dict(akey[2]), dict(akey[3])))

        if len(actions) == 0:
            # no children => return a dummy no-op
            return Action(0, {}, {}), {}, root

        vc = torch.tensor(visit_counts, dtype=torch.float32)
        tau = max(1e-6, self.cfg.temperature)
        pi = (vc ** (1.0 / tau))
        pi = (pi / pi.sum()).tolist()

        # Build π̂ over hashable action-keys
        akeys = [a.key() for a in actions]
        search_policy = {akey: p for akey, p in zip(akeys, pi)}  # dict[Tuple, float]

        # pick action (sample)
        idx = int(torch.multinomial(torch.tensor(pi), 1).item())
        return actions[idx], search_policy, root


# ---------- Minimal demo ----------

if __name__ == "__main__":
    import torch
    from policy_value_net import PolicyValueConfig, ArgSpec, PolicyValueNet

    # 1) Build network
    cfg_net = PolicyValueConfig(
        d_model=192, n_heads=3, n_layers=3, dropout=0.1,
        vocab_size=256, max_seq_len=128,
        max_object_slots=32, object_feat_dim=32,
        n_ops=16,
        arg_specs=(
            ArgSpec(name="color", kind="categorical", size=8),
            ArgSpec(name="obj_ptr", kind="pointer", size=32),
        ),
    )
    net = PolicyValueNet(cfg_net).eval()

    # 2) Dummy encoder: state -> tensors (B=1)
    class DummyState(NamedTuple):
        step: int
        max_steps: int
        M: int

    def encode_state(state: DummyState):
        B, T, M, F = 1, 32, state.M, 32
        state_tokens = torch.randint(0, cfg_net.vocab_size, (B, T))
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        object_feats = torch.randn(B, M, F)
        object_mask = torch.ones(B, M, dtype=torch.bool)
        return state_tokens, attn_mask, object_feats, object_mask

    # 3) Dummy executor: applies action; terminal at max_steps; reward 1 at the end 30% of time
    import random
    def step_fn(state: DummyState, action: Action):
        ns = DummyState(step=state.step + 1, max_steps=state.max_steps, M=state.M)
        terminal = ns.step >= ns.max_steps
        reward = 1.0 if (terminal and random.random() < 0.3) else 0.0
        return ns, terminal, reward

    # 4) Run MCTS
    mcfg = MCTSConfig(num_simulations=64, cpuct=1.25, temperature=1.0,
                      topk_ops=8, topk_cat=4, topk_ptr=4, max_candidates=64)

    mcts = MCTS(net, mcfg, encode_state_fn=encode_state, step_fn=step_fn)

    root = DummyState(step=0, max_steps=6, M=24)
    action, pi, tree = mcts.run(root)
    print("✅ MCTS produced action:", action)
    top = sorted(pi.items(), key=lambda kv: -kv[1])[:5]
    pretty = [(action_from_key(akey), p) for akey, p in top]
    print("π̂ size:", len(pi), "top-5:", pretty)