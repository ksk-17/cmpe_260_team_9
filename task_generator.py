import random
import numpy as np
from arc_dataset import pad_grid, generate_random_grid

def generate_synthetic_batch(dsl, engine, batch_size=16):
    """
    Generates SEQUENTIAL programs (length 1 to 4) to teach composition.
    """
    inputs, outputs, programs = [], [], []
    for _ in range(batch_size):
        inp = generate_random_grid()

        # 1. Decide sequence length (1 to 4 ops)
        seq_len = random.randint(1, 4)
        prog_tokens = []
        current_grid = inp.copy()
        valid_seq = True

        for _ in range(seq_len):
            # Weighted sampling for ops
            weights = [1.0] * len(dsl.ops)
            for i, op in enumerate(dsl.ops):
                if 'Color' in op['name'] or 'Symm' in op['name'] or 'Scale' in op['name'] or 'Tile' in op['name']:
                    weights[i] = 3.0

            op = random.choices(dsl.ops, weights=weights, k=1)[0]

            step_tokens = [op['name']]
            args = []
            for i in range(op['n_args']):
                # Special handling for Recolor to pick existing colors
                if op['name'] == 'Recolor' and i == 0:
                    present = np.unique(current_grid)
                    c = random.choice(present) if len(present) > 0 else 0
                    args.append(int(c))
                # Special handling for Upscale/Tile to avoid 0
                elif op['name'] in ['Upscale', 'Tile']:
                    args.append(random.randint(1, 3))  # Force small valid factors
                else:
                    args.append(random.randint(0, 9))

            step_tokens.extend(args)

            # Execute step
            try:
                temp_prog = prog_tokens + step_tokens
                res = engine.execute(temp_prog, inp)
                current_grid = res
                prog_tokens.extend(step_tokens)
            except:
                valid_seq = False
                break

        if valid_seq:
            try:
                out = engine.execute(prog_tokens, inp)
                if not np.array_equal(inp, out):  # Must do SOMETHING
                    inputs.append(pad_grid(inp))
                    outputs.append(pad_grid(out))
                    programs.append(prog_tokens)
            except:
                pass

    return inputs, outputs, programs