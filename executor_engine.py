import numpy as np

class ExecutionEngine:
    def __init__(self, dsl):
        self.dsl = dsl

    def execute(self, program_tokens, input_grid):
        stack = [np.array(input_grid, dtype=int)]
        idx = 0
        steps = 0

        while idx < len(program_tokens) and steps < 50:
            token = program_tokens[idx]
            idx += 1
            steps += 1

            if token in self.dsl.op_map:
                op_def = self.dsl.op_map[token]
                if len(stack) < op_def['stack_in']:
                    return stack[-1] if stack else np.array(input_grid)

                inputs = []
                for _ in range(op_def['stack_in']):
                    inputs.append(stack.pop())

                imm_args = []
                for _ in range(op_def['n_args']):
                    if idx < len(program_tokens) and isinstance(program_tokens[idx], int):
                        imm_args.append(program_tokens[idx])
                        idx += 1
                    else:
                        return stack[-1] if stack else np.array(input_grid)

                try:
                    res = op_def['func'](*inputs, *imm_args)
                    stack.append(res)
                except Exception:
                    if inputs: stack.append(inputs[-1])
            else:
                pass

        return stack[-1] if stack else np.array(input_grid)