import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from collections import Counter

# Imports assuming these files exist in your environment
from neural_solver import NeuroSolver, encode_context

def compute_reward(pred, target, inp):
    if pred.shape == target.shape and np.array_equal(pred, target): return 5.0
    if pred.shape != target.shape: return 0.0
    base = (inp == target).mean() if inp.shape == target.shape else 0
    curr = (pred == target).mean()
    return max(0, (curr - base) / (1 - base + 1e-6))

def run_bsil(model, engine, task, dev):
    train_pairs = task['train']
    if not train_pairs: return torch.tensor(0.0, requires_grad=True, device=dev), 0.0, None

    ctx = encode_context(model, train_pairs[0]['input'], train_pairs[0]['output'], dev)

    candidates = []
    for _ in range(20):
        tok = torch.tensor([model.token_to_id['<SOS>']], device=dev)
        hx = ctx
        seq = []
        log_probs = []
        for _ in range(10):
            # Expecting 3 values: logits, value, hx
            ret = model(tok, hx, ctx)
            if len(ret) == 3:
                logits, _, hx = ret
            else:
                logits, hx = ret  # Fallback if model is old

            dist = Categorical(logits=logits)
            action = dist.sample()
            seq.append(model.vocab[action.item()])
            log_probs.append(dist.log_prob(action))
            tok = action
            if seq[-1] == '<EOS>': break
        candidates.append((seq, torch.stack(log_probs).sum()))

    best_prog, best_r = None, -1
    for prog, lp in candidates:
        clean = [p for p in prog if p not in ['<SOS>', '<EOS>', '<PAD>']]
        r = 0
        try:
            for tp in train_pairs:
                res = engine.execute(clean, tp['input'])
                r += compute_reward(res, tp['output'], tp['input'])
            r /= len(train_pairs)
        except:
            r = 0
        if r > best_r: best_r, best_prog = r, (prog, lp)

    if best_r > 0.1:
        loss = -best_prog[1]
        return loss, best_r, best_prog[0]
    return torch.tensor(0.0, requires_grad=True, device=dev), 0.0, None


# ==========================================
# 2. PPO (Proximal Policy Optimization)
# ==========================================

def run_ppo(model, engine, task, dev):
    train_pairs = task['train']
    if not train_pairs: return torch.tensor(0.0, requires_grad=True, device=dev), 0.0

    ctx = encode_context(model, train_pairs[0]['input'], train_pairs[0]['output'], dev)

    traj_log_probs, traj_values, traj_rewards, traj_entropies = [], [], [], []

    for _ in range(4):
        tok = torch.tensor([model.token_to_id['<SOS>']], device=dev)
        hx = ctx
        seq_log_probs, seq_values, seq_entropies, seq_actions = [], [], [], []

        for _ in range(10):
            logits, val, hx = model(tok, hx, ctx)
            dist = Categorical(logits=logits)
            action = dist.sample()

            seq_log_probs.append(dist.log_prob(action))
            seq_values.append(val)
            seq_entropies.append(dist.entropy())
            seq_actions.append(action)

            tok = action
            if model.vocab[action.item()] == '<EOS>': break

        prog = [model.vocab[a.item()] for a in seq_actions]
        clean = [p for p in prog if p not in ['<SOS>', '<EOS>', '<PAD>']]

        r_val = 0
        try:
            for tp in train_pairs:
                res = engine.execute(clean, tp['input'])
                r_val += compute_reward(res, tp['output'], tp['input'])
            r_val /= len(train_pairs)
        except:
            pass

        R = r_val
        returns = []
        for _ in range(len(seq_values)):
            returns.insert(0, R)
            R = R * 0.95

        traj_log_probs.extend(seq_log_probs)
        traj_values.extend(seq_values)
        traj_rewards.extend(returns)
        traj_entropies.extend(seq_entropies)

    if not traj_rewards: return torch.tensor(0.0, requires_grad=True, device=dev), 0.0

    # Explicitly cast to float32 to prevent Double dtype errors
    rewards_t = torch.tensor(traj_rewards, dtype=torch.float32, device=dev)
    if rewards_t.std() > 1e-6:
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    log_probs_t = torch.stack(traj_log_probs)
    values_t = torch.stack(traj_values).squeeze()
    entropies_t = torch.stack(traj_entropies)

    advantage = rewards_t - values_t.detach()
    ratio = torch.exp(log_probs_t - log_probs_t.detach())
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage

    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(values_t, rewards_t)
    entropy_loss = -entropies_t.mean()

    return actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss, np.mean(traj_rewards)


grammar_stats = Counter()

def generate_random_grid_dream():
    h, w = random.randint(3, 10), random.randint(3, 10)
    g = np.zeros((h, w), int)
    for _ in range(random.randint(1, 3)):
        c, r, k = random.randint(1, 9), random.randint(0, h - 1), random.randint(0, w - 1)
        g[r:min(r + 2, h), k:min(k + 2, w)] = c
    return g


def run_dreamcoder(model, engine, task, dev):
    loss, r, prog = run_bsil(model, engine, task, dev)

    if prog and r > 4.0:
        clean = [p for p in prog if p not in ['<SOS>', '<EOS>', '<PAD>']]
        for i in range(len(clean) - 1):
            if isinstance(clean[i], str) and isinstance(clean[i + 1], str):
                grammar_stats[(clean[i], clean[i + 1])] += 1

    if len(grammar_stats) > 0 and random.random() < 0.3:
        bigram = random.choice(list(grammar_stats.keys()))
        syn_in = generate_random_grid_dream()
        full_prog = []
        for p in bigram:
            full_prog.append(p)
            if p in engine.dsl.op_map:
                op = engine.dsl.op_map[p]
                for _ in range(op['n_args']): full_prog.append(random.randint(0, 9))

        try:
            syn_out = engine.execute(full_prog, syn_in)
            if not np.array_equal(syn_in, syn_out):
                d_ctx = encode_context(model, syn_in, syn_out, dev)
                d_hx = d_ctx
                d_tok = torch.tensor([model.token_to_id['<SOS>']], device=dev)
                d_loss = 0
                for token in full_prog + ['<EOS>']:
                    if token not in model.vocab: continue
                    # Handle tuple return safely
                    ret = model(d_tok, d_hx, d_ctx)
                    if len(ret) == 3:
                        l, _, d_hx = ret
                    else:
                        l, d_hx = ret

                    target = torch.tensor([model.token_to_id[token]], device=dev)
                    d_loss += F.cross_entropy(l, target)
                    d_tok = target

                loss = loss + d_loss if isinstance(loss, torch.Tensor) else d_loss
        except:
            pass

    return loss, r


class MCTSNode:
    def __init__(self, parent=None, token=None):
        self.parent = parent
        self.token = token
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0


def run_mcts(model, engine, task, dev):
    train_pairs = task['train']
    if not train_pairs: return torch.tensor(0.0, requires_grad=True, device=dev), 0.0

    ctx = encode_context(model, train_pairs[0]['input'], train_pairs[0]['output'], dev)
    root = MCTSNode(token='<SOS>')

    # 20 Simulations
    for _ in range(20):
        node = root
        hx = ctx
        depth = 0
        seq = []

        # 1. Selection
        while node.children and depth < 10:
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                u = child.value / (child.visits + 1e-6) + 1.0 * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                if u > best_score:
                    best_score = u
                    best_child = child
            node = best_child
            seq.append(node.token)

            # Update RNN state
            inp_tok = torch.tensor([model.token_to_id[node.token]], device=dev)
            _, _, hx = model(inp_tok, hx, ctx)
            depth += 1

        # 2. Expansion
        if depth < 10 and node.token != '<EOS>':
            inp_tok = torch.tensor([model.token_to_id[node.token if node.token else '<SOS>']], device=dev)
            logits, val, hx = model(inp_tok, hx, ctx)
            probs = F.softmax(logits, dim=1).squeeze()

            # Expand top 5 actions
            topk = torch.topk(probs, 5)
            for i in range(5):
                idx = topk.indices[i].item()
                tok_str = model.vocab[idx]
                child = MCTSNode(parent=node, token=tok_str)
                child.prior = topk.values[i].item()
                node.children[tok_str] = child

            # 3. Simulation (Value Estimate from Critic)
            reward = val.item()
        else:
            # Terminal State - True Reward
            clean = [p for p in seq if p not in ['<SOS>', '<EOS>']]
            r_sum = 0
            try:
                for tp in train_pairs:
                    res = engine.execute(clean, tp['input'])
                    r_sum += compute_reward(res, tp['output'], tp['input'])
                reward = r_sum / len(train_pairs)
            except:
                reward = 0

        # 4. Backprop
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Select best path based on visits
    seq = []
    node = root
    while node.children:
        node = max(node.children.values(), key=lambda n: n.visits)
        seq.append(node.token)
        if node.token == '<EOS>': break

    clean = [p for p in seq if p not in ['<SOS>', '<EOS>']]
    final_r = 0
    try:
        for tp in train_pairs:
            res = engine.execute(clean, tp['input'])
            final_r += compute_reward(res, tp['output'], tp['input'])
        final_r /= len(train_pairs)
    except:
        pass

    # Train Supervised on MCTS path if it was good
    loss = torch.tensor(0.0, requires_grad=True, device=dev)
    if final_r > 0.1:
        loss = 0
        curr = torch.tensor([model.token_to_id['<SOS>']], device=dev)
        hx = ctx
        for t in seq:
            l, _, hx = model(curr, hx, ctx)
            tgt = torch.tensor([model.token_to_id[t]], device=dev)
            loss += F.cross_entropy(l, tgt)
            curr = tgt

    return loss, final_r