import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List
from arc_dataset import ARCDataset
from dsl import Grid, DSL_PRIMITIVES, execute_program
from state_encoder import ARCStateEncoder
from program_encoder import ProgramGenerator, ProgramVocabulary, generate_square_subsequent_mask
import os

ARC_DATA_ROOT = "arc-prize-2025"
TRAIN_CHALLENGES = os.path.join(ARC_DATA_ROOT, "arc-agi_training_challenges.json")
TRAIN_SOLUTIONS = os.path.join(ARC_DATA_ROOT, "arc-agi_training_solutions.json")

D_MODEL = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 1
MAX_PROGRAM_LENGTH = 50


def get_reward(predicted_output: Grid, target_output: torch.Tensor) -> float:
    # Make sure we are dealing with a Grid
    if not isinstance(predicted_output, Grid):
        return 0.0

    # Move to CPU and align dtypes
    pred_data = predicted_output.data.cpu().to(torch.long).clone()
    target_data = target_output.cpu().clone().to(torch.long)

    # Shapes must match
    if pred_data.shape != target_data.shape:
        return 0.0

    # Treat padding (-1) as background (0)
    target_data[target_data < 0] = 0
    pred_data[pred_data < 0] = 0  # in case something weird appears

    # Optional: clamp predicted colors to ARC range [0, 9]
    pred_data[pred_data > 9] = 0

    # Exact match gets full reward
    if torch.equal(pred_data, target_data):
        return 1.0

    # IoU over non-zero pixels
    pred_mask = (pred_data != 0).float()
    target_mask = (target_data != 0).float()

    intersection = (pred_mask * target_mask).sum()
    union = (pred_mask + target_mask - pred_mask * target_mask).sum()

    if union == 0:
        return 0.0

    iou_reward = (intersection / union).item()
    return iou_reward

def sample_program(
        generator: ProgramGenerator,
        v_context: torch.Tensor,
        max_len: int = 50,
        device: torch.device = torch.device('cpu')
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Samples a program token-by-token from the ProgramGenerator using
    an autoregressive decoding loop.

    Returns:
      - program_sequence: List[str] of tokens (no <START>/<END>)
      - total_log_probs: 1D tensor of log π(a_t | s) for each step t
      - policy_sequence: 2D tensor (T, vocab_size) of policies per step
    """
    log_probs_list = []
    policies_list = []
    program_sequence = []

    start_token_id = generator.vocab.token_to_id['<START>']
    end_token_id = generator.vocab.token_to_id['<END>']

    current_tokens = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    v_context = v_context.to(device)

    for _ in range(max_len):
        T = current_tokens.size(1)
        causal_mask = generate_square_subsequent_mask(T).to(device)

        # Forward through decoder
        logits = generator(current_tokens, v_context, causal_mask)  # (1, T, V)
        next_token_logits = logits[:, -1, :].squeeze(0)            # (V,)

        # Policy and sample
        policy = F.softmax(next_token_logits, dim=-1)              # (V,)
        policies_list.append(policy.unsqueeze(0))                  # (1, V)

        next_token_id = torch.multinomial(policy, num_samples=1)   # (1,)

        # Log-prob of sampled token (scalar)
        log_prob = torch.log(policy[next_token_id]).squeeze()      # ()
        log_probs_list.append(log_prob)

        # Append token to sequence
        next_token_tensor = next_token_id.unsqueeze(0)             # (1, 1)
        current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)

        if next_token_id.item() == end_token_id:
            break

        program_sequence.append(generator.vocab.id_to_token[next_token_id.item()])

    # (T,) – one scalar log-prob per step
    total_log_probs = torch.stack(log_probs_list, dim=0)

    # (T, V) – one policy distribution per step
    policy_sequence = torch.cat(policies_list, dim=0)

    return program_sequence, total_log_probs, policy_sequence


def train_reinforce(encoder, generator, dataloader, optimizer, device, gamma=0.99, entropy_coeff=0.01, value_coeff=0.5):
    encoder.train()
    generator.train()
    total_reward = 0
    total_loss = 0
    num_tasks = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        stacked_grids_tensor = inputs.squeeze(0).to(device)
        task_grids = [t for t in stacked_grids_tensor]
        target_grid = targets.squeeze(0).to(device)

        # 1. Encode State (S) and Predict Value (V)
        try:
            # CRITICAL CHANGE: Encoder now returns V_context and predicted_value
            V_context, predicted_value = encoder(task_grids)
        except Exception as e:
            # ... (error handling)
            continue

        # 2. Sample Program (A) and Get Log-Probs and Policies
        # CRITICAL CHANGE: sample_program returns log_probs AND policy_sequence
        try:
            # Assuming 'vocab' and 'MAX_PROGRAM_LENGTH' are defined in the module scope
            program_tokens, log_probs_tensor, policy_sequence = sample_program(
                generator, V_context, max_len=MAX_PROGRAM_LENGTH, device=device
            )
        except Exception as e:
            print(f"Skipping task due to sampler error: {e}")
            continue

        # 3. Execute Program and Get Reward (R)
        test_input_tensor = task_grids[-1]

        # Map padding (-1) to background (0) before creating a Grid
        test_input_clean = test_input_tensor.clone()
        test_input_clean[test_input_clean < 0] = 0

        initial_grid = Grid(test_input_clean.cpu().to(dtype=torch.uint8))

        # NOTE: execute_program should handle errors and return a Grid object or None/Error
        final_result = execute_program(program_tokens, initial_grid)

        # Assuming get_reward returns float, potentially using IoU (0.0 to 1.0)
        R = get_reward(final_result, target_grid)
        R_tensor = torch.tensor([R], dtype=torch.float32, device=device)

        # 4. Calculate Losses

        # A. Advantage (A = R - V(s))
        # Detach the predicted_value to ensure gradients from policy loss
        # do not flow back into the value head (standard practice).
        advantage = R_tensor - predicted_value.detach()

        # B. Policy Loss (Gradient Ascent on Expected Advantage)
        # Policy Gradient: -log(pi(a|s)) * A
        policy_loss = -(torch.sum(log_probs_tensor) * advantage)

        # C. Value Loss (MSE on Baseline Update)
        # Value Loss: (R - V(s))^2. Use R_tensor as the target.
        value_loss = F.mse_loss(predicted_value, R_tensor)

        # D. Entropy Loss (Regularization for Exploration)
        # Entropy H(pi) = -sum(pi * log(pi)). We want to maximize this (minimize -H).
        # Note: torch.sum is needed here as the result is a sum over all time steps.
        entropy = -torch.sum(policy_sequence * torch.log(policy_sequence + 1e-8))
        entropy_loss = -entropy_coeff * entropy  # Negative sign to MAXIMIZE entropy (i.e., penalize certainty)

        # E. Total Loss
        total_task_loss = policy_loss + value_coeff * value_loss + entropy_loss

        # 5. Backpropagation
        optimizer.zero_grad()
        total_task_loss.backward()
        # You may want to add gradient clipping here:
        # nn.utils.clip_grad_norm_(encoder.parameters() + generator.parameters(), max_norm=1.0)
        optimizer.step()

        # 6. Metrics Update
        total_reward += R
        total_loss += total_task_loss.item()
        num_tasks += 1

        # ... (rest of the printout and summary remain the same)
        if num_tasks % 10 == 0:
            print(
                f"  Task {num_tasks}: R={R:.2f}, L_P={policy_loss.item():.4f}, L_V={value_loss.item():.4f}, L_E={entropy_loss.item():.4f}, L_Total={total_task_loss.item():.4f}, Prog={program_tokens[:5]}...")

    avg_reward = total_reward / num_tasks if num_tasks > 0 else 0
    avg_loss = total_loss / num_tasks if num_tasks > 0 else 0

    print(f"\n--- Epoch Summary ---")
    print(f"Average Reward: {avg_reward:.4f} ({int(total_reward)}/{num_tasks} solved)")
    print(f"Average Loss: {avg_loss:.4f}")

    return avg_reward, avg_loss

GAMMA = 0.99
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # full_dataset = ARCDataset(challenges_path=TRAIN_CHALLENGES, solutions_path=TRAIN_SOLUTIONS)
        full_dataset = ARCDataset(folder_path='ARC-AGI/data/training')
        dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded {len(full_dataset)} training tasks.")
    except Exception as e:
        print(f"Error loading dataset: {e}. Please ensure file paths and ARCDataset implementation are correct.")
        return

    vocab = ProgramVocabulary(DSL_PRIMITIVES)

    encoder = ARCStateEncoder(d_model=D_MODEL).to(device)
    generator = ProgramGenerator(vocab=vocab, d_model=D_MODEL).to(device)

    params = list(encoder.parameters()) + list(generator.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # 5. Training Loop
    print("\n--- Starting REINFORCE Training ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- EPOCH {epoch + 1}/{NUM_EPOCHS} ---")

        # Call the REINFORCE training function
        avg_reward, avg_loss = train_reinforce(
            encoder=encoder,
            generator=generator,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            # CRITICAL CORRECTION: Pass the new RL hyperparameters
            gamma=GAMMA,
            entropy_coeff=ENTROPY_COEFF,
            value_coeff=VALUE_COEFF
            # Removed max_program_length as it is not a parameter of train_reinforce
            # and is assumed to be a global constant used by sample_program.
        )

        if avg_reward > 0.5:
            print("Model showing strong performance. Consider saving checkpoint.")


if __name__ == "__main__":
    main()