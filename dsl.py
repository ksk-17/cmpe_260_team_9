import numpy as np
import torch
from scipy.ndimage import label
from typing import List, Tuple, Union, Dict


# representing grid as an object
class Grid:
    def __init__(self, data: torch.tensor):
        if data.dtype != torch.uint8:
            # Ensure the underlying data is always a byte tensor (0-255 colors)
            data = data.to(dtype=torch.uint8)
        self.data = data

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Grid({self.shape}, unique_colors={np.unique(self.data.cpu())})"


# base class for DSL operations
class Primitive:
    def __init__(self, name: str, arity: int):
        self.name = name
        self.arity = arity  # no of inputs

    def __call__(self, *args) -> Union[Grid, List[Grid], int, Tuple]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class Recolor(Primitive):
    def __init__(self):
        super().__init__("Recolor", arity=3)

    def __call__(self, grid: Grid, old_color: int, new_color: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"Recolor expects Grid, got {type(grid)}")
        if not isinstance(old_color, int) or not isinstance(new_color, int): raise TypeError("Colors must be integers.")

        new_data = grid.data.clone()
        mask = (new_data == old_color)
        new_data[mask] = new_color
        return Grid(new_data)


class Truncate(Primitive):
    def __init__(self):
        super().__init__("Truncate", arity=2)

    def __call__(self, grid: Grid, target_color: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"Truncate expects Grid, got {type(grid)}")
        if not isinstance(target_color, int): raise TypeError("Target color must be an integer.")

        new_data = grid.data.clone()
        mask = (new_data == target_color)
        new_data[mask] = 0
        return Grid(new_data)


class Rotate90(Primitive):
    def __init__(self):
        super().__init__("Rotate90", arity=1)

    def __call__(self, grid: Grid) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"Rotate90 expects Grid, got {type(grid)}")
        return Grid(torch.rot90(grid.data, k=-1, dims=(0, 1)))


class PaintCell(Primitive):
    def __init__(self):
        super().__init__("PaintCell", arity=4)

    def __call__(self, grid: Grid, r: int, c: int, color: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"PaintCell expects Grid, got {type(grid)}")
        if not all(isinstance(x, int) for x in [r, c, color]): raise TypeError("r, c, color must be integers.")

        new_data = grid.data.clone()
        rows, cols = new_data.shape

        if 0 <= r < rows and 0 <= c < cols:
            new_data[r, c] = color
            return Grid(new_data)
        else:
            raise ValueError(f"Cell coordinates ({r}, {c}) out of bounds {grid.shape}.")


class UpscaleGrid(Primitive):
    def __init__(self):
        super().__init__("UpscaleGrid", arity=2)

    def __call__(self, grid: Grid, factor: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"UpscaleGrid expects Grid, got {type(grid)}")
        if not isinstance(factor, int) or factor <= 0: raise ValueError("Factor must be a positive integer.")

        # use repeat_interleave
        sclaed_data = torch.repeat_interleave(
            torch.repeat_interleave(grid.data, factor, dim=0),
            factor, dim=1
        )

        return Grid(sclaed_data)


class Shift(Primitive):
    def __init__(self):
        super().__init__("Shift", arity=3)

    def __call__(self, grid: Grid, dr: int, dc: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"Shift expects Grid, got {type(grid)}")
        if not all(isinstance(x, int) for x in [dr, dc]): raise TypeError("Shift factors must be integers.")

        # torch.roll for cyclic shift
        shifted_data = torch.roll(grid.data, shifts=(dr, dc), dims=(0, 1))
        return Grid(shifted_data)


class MirrorGrid(Primitive):
    def __init__(self):
        super().__init__("MirrorGrid", arity=2)

    def __call__(self, grid: Grid, axis: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"MirrorGrid expects Grid, got {type(grid)}")
        if not isinstance(axis, int): raise TypeError("Axis must be an integer.")
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (row/H-flip) or 1 (col/V-flip).")

        mirrored_data = torch.flip(grid.data, dims=[axis])
        return Grid(mirrored_data)


class Overlay(Primitive):
    def __init__(self):
        super().__init__("Overlay", arity=2)

    def __call__(self, grid_a: Grid, grid_b: Grid) -> Grid:
        if not isinstance(grid_a, Grid) or not isinstance(grid_b, Grid):
            raise TypeError(f"Overlay expects two Grid objects, got {type(grid_a)} and {type(grid_b)}")
        if grid_a.shape != grid_b.shape:
            raise ValueError(f"Overlay requires grids of the same shape, got {grid_a.shape} and {grid_b.shape}")

        new_data = grid_a.data.clone()
        mask = (grid_b.data != 0)  # not the background color
        new_data[mask] = grid_b.data[mask]
        return Grid(new_data)


class IsolateColor(Primitive):
    def __init__(self):
        super().__init__("IsolateColor", arity=2)

    def __call__(self, grid: Grid, color: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"IsolateColor expects Grid, got {type(grid)}")
        if not isinstance(color, int): raise TypeError("Color must be an integer.")

        new_data = torch.zeros_like(grid.data)
        mask = (grid.data == color)

        new_data[mask] = color
        return Grid(new_data)


class GetBoundingBox(Primitive):
    def __init__(self):
        super().__init__("GetBoundingBox", arity=1)

    def __call__(self, grid: Grid) -> Tuple[int, int, int, int]:
        if not isinstance(grid, Grid): raise TypeError(f"GetBoundingBox expects Grid, got {type(grid)}")

        coords = torch.nonzero(grid.data)

        if coords.numel() == 0:
            return (0, 0, 0, 0)

        r1, c1 = torch.min(coords, dim=0).values.tolist()
        r2, c2 = torch.max(coords, dim=0).values.tolist()

        return (r1, c1, r2 + 1, c2 + 1)


class ApplyBorder(Primitive):
    def __init__(self):
        super().__init__("ApplyBorder", arity=3)

    def __call__(self, grid: Grid, color: int, thickness: int) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"ApplyBorder expects Grid, got {type(grid)}")
        if not all(isinstance(x, int) for x in [color, thickness]): raise TypeError(
            "Color and thickness must be integers.")
        if thickness <= 0:
            return grid

        H, W = grid.shape
        new_H, new_W = H + 2 * thickness, W + 2 * thickness

        new_data = torch.full((new_H, new_W), color, dtype=torch.uint8, device=grid.data.device)

        r_start, c_start = thickness, thickness
        r_end, c_end = r_start + H, c_start + W

        new_data[r_start:r_end, c_start:c_end] = grid.data

        return Grid(new_data)


class Crop(Primitive):
    def __init__(self):
        super().__init__("Crop", arity=2)

    def __call__(self, grid: Grid, box: Tuple[int, int, int, int]) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"Crop expects Grid, got {type(grid)}")
        if not isinstance(box, tuple) or len(box) != 4: raise TypeError("Box must be a 4-element tuple.")

        r1, c1, r2, c2 = box

        # Clip coordinates to grid boundaries to prevent out-of-bounds error
        H, W = grid.shape
        r1, c1 = max(0, r1), max(0, c1)
        r2, c2 = min(H, r2), min(W, c2)

        cropped_data = grid.data[r1:r2, c1:c2].clone()
        return Grid(cropped_data)


class GridUnion(Primitive):
    def __init__(self):
        super().__init__("GridUnion", arity=2)

    def __call__(self, grid_a: Grid, grid_b: Grid) -> Grid:
        if not isinstance(grid_a, Grid) or not isinstance(grid_b, Grid):
            raise TypeError(f"GridUnion expects two Grid objects, got {type(grid_a)} and {type(grid_b)}")
        if grid_a.shape != grid_b.shape:
            raise ValueError("GridUnion requires grids of the same shape.")

        # Logic is sound: copy A, then overlay B where A is background (0)
        result_data = grid_a.data.clone()
        mask_only_b = (grid_a.data == 0) & (grid_b.data != 0)
        result_data[mask_only_b] = grid_b.data[mask_only_b]
        return Grid(result_data)


class GridIntersection(Primitive):
    def __init__(self):
        super().__init__("GridIntersection", arity=2)

    def __call__(self, grid_a: Grid, grid_b: Grid) -> Grid:
        if not isinstance(grid_a, Grid) or not isinstance(grid_b, Grid):
            raise TypeError(f"GridIntersection expects two Grid objects, got {type(grid_a)} and {type(grid_b)}")
        if grid_a.shape != grid_b.shape:
            raise ValueError("GridIntersection requires grids of the same shape.")

        # Logic is sound: only keep non-zero elements present in both A and B
        result_data = torch.zeros_like(grid_a.data)
        mask_both = (grid_a.data != 0) & (grid_b.data != 0)
        result_data[mask_both] = grid_a.data[mask_both]
        return Grid(result_data)


class FindObjects(Primitive):
    def __init__(self):
        super().__init__("FindObjects", arity=2)

    def __call__(self, grid: Grid, target_color: Union[int, str] = '*') -> List[Grid]:
        if not isinstance(grid, Grid): raise TypeError(f"FindObjects expects Grid, got {type(grid)}")
        if not isinstance(target_color, (int, str)): raise TypeError("Target color must be an integer or string '*'")

        data_np = grid.data.cpu().numpy()

        if target_color == '*':
            target_mask = data_np != 0
        elif isinstance(target_color, int):
            target_mask = data_np == target_color
        else:
            raise ValueError(f"Invalid target color argument: {target_color}")

        # perform connected
        labeled_array, num_features = label(target_mask, structure=np.ones((3, 3)))

        if num_features == 0:
            return []

        # extract and convert
        object_list = []
        for i in range(1, num_features + 1):
            object_mask = (labeled_array == i)
            object_data = np.where(object_mask, data_np, 0)
            object_tensor = torch.tensor(object_data, dtype=torch.uint8, device=grid.data.device)
            object_list.append(Grid(object_tensor))

        return object_list


class Partition(Primitive):
    def __init__(self):
        super().__init__("Partition", arity=3)

    def __call__(self, grid: Grid, dim: int, split: int) -> List[Grid]:
        if not isinstance(grid, Grid): raise TypeError(f"Partition expects Grid, got {type(grid)}")
        if dim not in (0, 1): raise ValueError("Dimension must be 0 (row) or 1 (column).")
        if not isinstance(split, int) or split <= 0: raise ValueError("Split must be a positive integer.")

        size = grid.shape[dim]

        if size % split != 0:
            raise ValueError(f"Grid size {size} not divisible by split {split}.")

        if dim == 0:
            chunks = torch.chunk(grid.data, split, dim=0)
        elif dim == 1:
            chunks = torch.chunk(grid.data, split, dim=1)

        return [Grid(chunk) for chunk in chunks]


class Count(Primitive):
    def __init__(self):
        super().__init__("Count", arity=2)

    def __call__(self, grid: Grid, property_arg: str) -> int:
        if not isinstance(grid, Grid): raise TypeError(f"Count expects Grid, got {type(grid)}")

        if property_arg == 'unique_colors':
            return len(torch.unique(grid.data))
        elif property_arg == "pixels":
            return torch.sum(grid.data != 0).item()
        elif property_arg == "objects":
            data_np = grid.data.cpu().numpy()
            target_mask = data_np != 0
            # Ensure structure is explicitly defined for connectivity
            _, num_features = label(target_mask, structure=np.ones((3, 3)))
            return num_features
        else:
            raise ValueError(f"Unknown property argument for Count: {property_arg}")


class MapColor(Primitive):
    def __init__(self):
        super().__init__("MapColor", arity=3)

    def __call__(self, grid: Grid, old_color: int, color_rule: str) -> Grid:
        if not isinstance(grid, Grid): raise TypeError(f"MapColor expects Grid, got {type(grid)}")
        if not isinstance(old_color, int): raise TypeError("Old color must be an integer.")
        if not isinstance(color_rule, str): raise TypeError("Color rule must be a string.")

        target_color = -1  # Sentinel value

        if color_rule == 'most_frequent':
            target_color = self.get_most_frequent_color(grid.data)
        elif color_rule == 'background':
            target_color = 0
        elif color_rule == 'opposite':
            counts = torch.bincount(grid.data.flatten())
            if counts.shape[0] < 2:
                target_color = 0
            else:
                # Find the least frequent non-zero color
                non_zero_counts = counts[1:]
                if non_zero_counts.numel() == 0:
                    target_color = 0
                else:
                    target_color = torch.argmin(non_zero_counts).item() + 1
        else:
            raise ValueError(f"Unknown color rule: {color_rule}")

        if target_color >= 0:
            new_data = grid.data.clone()
            new_data[new_data == old_color] = target_color
            return Grid(new_data)

        return grid

    def get_most_frequent_color(self, grid_data: torch.Tensor) -> int:
        counts = torch.bincount(grid_data.flatten())
        if counts.shape[0] < 2:
            return 0

        counts[0] = 0  # Ignore background color (0) for 'most_frequent'
        return torch.argmax(counts).item()


DSL_PRIMITIVES: Dict[str, Primitive] = {
    # Geometric
    'Recolor': Recolor(),
    'Partition': Partition(),
    'PaintCell': PaintCell(),
    'UpscaleGrid': UpscaleGrid(),
    'Shift': Shift(),
    'MirrorGrid': MirrorGrid(),
    # Color
    'Rotate90': Rotate90(),
    'IsolateColor': IsolateColor(),
    'MapColor': MapColor(),
    # Composition
    'Overlay': Overlay(),
    'GridUnion': GridUnion(),
    'GridIntersection': GridIntersection(),
    # Feature Extraction & Cropping
    'GetBoundingBox': GetBoundingBox(),
    'FindObjects': FindObjects(),
    'Crop': Crop(),
    'Count': Count(),
    'ApplyBorder': ApplyBorder(),
    'Truncate': Truncate(),
}


# --- New Robust Execute Program Function ---

def execute_program(program_tokens: List[Union[str, int]], initial_grid: Grid) -> Union[Grid, List[Grid], int, Tuple]:
    stack = [initial_grid]
    i = 0

    # Define which arguments are immediate (literals) by their index in the primitive's signature.
    IMMEDIATE_ARG_INDICES = {
        'Recolor': [1, 2],  # grid, old_color, new_color
        'PaintCell': [1, 2, 3],  # grid, r, c, color
        'Shift': [1, 2],  # grid, dr, dc
        'MirrorGrid': [1],  # grid, axis
        'UpscaleGrid': [1],  # grid, factor
        'ApplyBorder': [1, 2],  # grid, color, thickness
        'IsolateColor': [1],  # grid, color
        'FindObjects': [1],  # grid, target_color
        'Partition': [1, 2],  # grid, dim, split
        'Count': [1],  # grid, property_arg
        'MapColor': [1, 2],  # grid, old_color, color_rule
        'Truncate': [1]  # grid, target_color
    }

    def resolve_token(token: Union[str, int]) -> Union[str, int, None]:
        if isinstance(token, int): return token
        if isinstance(token, str):
            if token.isdigit(): return int(token)
            if token == 'axis_0': return 0
            if token == 'axis_1': return 1
            if token in ['pixels', 'objects', 'unique_colors', 'most_frequent', 'background', 'opposite', '*']:
                return token
        return None

    while i < len(program_tokens):
        token = program_tokens[i]

        if isinstance(token, str) and token in DSL_PRIMITIVES:
            primitive = DSL_PRIMITIVES[token]
            arity = primitive.arity

            immediate_indices = IMMEDIATE_ARG_INDICES.get(primitive.name, [])
            num_immediate_args = len(immediate_indices)

            # --- 1. Consume IMMEDIATE arguments from program list ---
            tokens_to_consume = 0
            program_args = {}

            try:
                if i + 1 + num_immediate_args > len(program_tokens):
                    raise IndexError(
                        f"Insufficient immediate tokens for {primitive.name}. Expected {num_immediate_args}.")

                # Consume and resolve tokens
                for idx, arg_index in enumerate(immediate_indices):
                    token_val = program_tokens[i + 1 + idx]
                    resolved_val = resolve_token(token_val)

                    if resolved_val is None:
                        # Cannot use an unresolvable token as an immediate argument
                        raise TypeError(f"Invalid token '{token_val}' for immediate argument of {primitive.name}.")

                    program_args[arg_index] = resolved_val
                    tokens_to_consume += 1

                i += 1 + tokens_to_consume  # Advance index past primitive and immediate args

                # --- 2. Consume STACK arguments ---
                num_stack_args = arity - num_immediate_args

                if len(stack) < num_stack_args:
                    raise IndexError(
                        f"Insufficient items on stack for {primitive.name}. Expected {num_stack_args} stack args, got {len(stack)}")

                # Pop arguments in reverse order
                stack_args_raw = [stack.pop() for _ in range(num_stack_args)]
                stack_args = stack_args_raw[::-1]  # Reverse to correct order

                # --- 3. Assemble FINAL arguments list ---
                final_args = [None] * arity

                stack_arg_counter = 0
                for k in range(arity):
                    if k not in immediate_indices:
                        final_args[k] = stack_args[stack_arg_counter]
                        stack_arg_counter += 1
                    else:
                        final_args[k] = program_args[k]

                # --- 4. Execute and Push Result ---
                result = primitive(*final_args)

                # Push list results (like from FindObjects, Partition) individually
                if isinstance(result, list) and result and all(isinstance(g, Grid) for g in result):
                    for g in result:
                        stack.append(g)
                else:
                    stack.append(result)

            except Exception as e:
                # Log the error and immediately return a dummy grid to skip the program
                # The execution error handling in reinforcement_solver.py expects this to be caught.
                print(f"Execution Error in program step {i}: {e}. Skipping program.")
                return Grid(torch.zeros(1, 1, dtype=torch.uint8))

        else:
            # Skip literal tokens that appear outside of an argument position or special tokens
            i += 1

            # The final result must be the last item on the stack
    return stack[-1] if stack and isinstance(stack[-1], (Grid, List, int, Tuple)) else Grid(
        torch.zeros(1, 1, dtype=torch.uint8))


if __name__ == "__main__":
    example_data = torch.tensor([
        [1, 1, 0],
        [0, 2, 0],
        [0, 0, 1]
    ], dtype=torch.uint8)
    initial_grid = Grid(example_data)

    recolor_op = Recolor()
    grid_b = recolor_op(initial_grid, 1, 5)
    print("\nAfter Recolor (1->5):\n", grid_b.data)

    rotate_op = Rotate90()
    grid_c = rotate_op(initial_grid)
    print("\nAfter Rotate90:\n", grid_c.data)

    program = [
        'Recolor', 1, 5,
        'Rotate90'
    ]

    ep_output = execute_program(program_tokens=program, initial_grid=initial_grid)
    print("\nFinal Result of Program executor (Recolor(1->5) then Rotate90):\n", ep_output.data)

    temp_grid = recolor_op(initial_grid, 1, 5)
    final_grid = rotate_op(temp_grid)
    print("\nFinal Result of Manual execution (Recolor(1->5) then Rotate90):\n", final_grid.data)