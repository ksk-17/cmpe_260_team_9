import numpy as np

class ARCDSL:
    def __init__(self):
        # n_args: integers needed (0-9)
        self.ops = [
            # --- Global Ops ---
            {'name': 'Identity', 'func': self.identity, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'Rotate90', 'func': self.rot90, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'Rotate180', 'func': self.rot180, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'Rotate270', 'func': self.rot270, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'FlipH', 'func': self.flip_h, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'FlipV', 'func': self.flip_v, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'CropNonZero', 'func': self.crop_nonzero, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},
            {'name': 'CropColor', 'func': self.crop_color, 'n_args': 1, 'stack_in': 1, 'stack_out': 1},  # New
            {'name': 'Recolor', 'func': self.recolor, 'n_args': 2, 'stack_in': 1, 'stack_out': 1},
            {'name': 'FloodFill', 'func': self.flood_fill, 'n_args': 1, 'stack_in': 1, 'stack_out': 1},
            {'name': 'Border', 'func': self.add_border, 'n_args': 1, 'stack_in': 1, 'stack_out': 1},
            {'name': 'InvertColor', 'func': self.invert_color, 'n_args': 0, 'stack_in': 1, 'stack_out': 1},

            # --- Scaling & Tiling Ops (New) ---
            {'name': 'Upscale', 'func': self.upscale, 'n_args': 1, 'stack_in': 1, 'stack_out': 1},
            {'name': 'Tile', 'func': self.tile, 'n_args': 2, 'stack_in': 1, 'stack_out': 1},

            # --- Symmetry/Mirror Ops (Partial Mirroring) ---
            {'name': 'Symm_L_to_R', 'func': lambda g: self.symmetrize(g, 0), 'n_args': 0, 'stack_in': 1,
             'stack_out': 1},
            {'name': 'Symm_R_to_L', 'func': lambda g: self.symmetrize(g, 1), 'n_args': 0, 'stack_in': 1,
             'stack_out': 1},
            {'name': 'Symm_T_to_B', 'func': lambda g: self.symmetrize(g, 2), 'n_args': 0, 'stack_in': 1,
             'stack_out': 1},
            {'name': 'Symm_B_to_T', 'func': lambda g: self.symmetrize(g, 3), 'n_args': 0, 'stack_in': 1,
             'stack_out': 1},

            # --- Object/Color Specific Ops (Partial Rotation/Shift) ---
            # Rotate ONLY pixels of specific color around their centroid
            {'name': 'RotateColor90', 'func': self.rotate_color_90, 'n_args': 1, 'stack_in': 1, 'stack_out': 1},

            # Shift ONLY pixels of specific color
            {'name': 'ShiftColorN', 'func': lambda g, c: self.shift_color(g, c, -1, 0), 'n_args': 1, 'stack_in': 1,
             'stack_out': 1},
            {'name': 'ShiftColorS', 'func': lambda g, c: self.shift_color(g, c, 1, 0), 'n_args': 1, 'stack_in': 1,
             'stack_out': 1},
            {'name': 'ShiftColorW', 'func': lambda g, c: self.shift_color(g, c, 0, -1), 'n_args': 1, 'stack_in': 1,
             'stack_out': 1},
            {'name': 'ShiftColorE', 'func': lambda g, c: self.shift_color(g, c, 0, 1), 'n_args': 1, 'stack_in': 1,
             'stack_out': 1},
        ]

        self.op_names = [op['name'] for op in self.ops]
        self.op_map = {op['name']: op for op in self.ops}

    # --- Basic Primitives ---
    def identity(self, grid, *args):
        return grid.copy()

    def rot90(self, grid, *args):
        return np.rot90(grid, k=1)

    def rot180(self, grid, *args):
        return np.rot90(grid, k=2)

    def rot270(self, grid, *args):
        return np.rot90(grid, k=3)

    def flip_h(self, grid, *args):
        return np.fliplr(grid)

    def flip_v(self, grid, *args):
        return np.flipud(grid)

    def crop_nonzero(self, grid, *args):
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)
        if not np.any(rows) or not np.any(cols): return grid
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return grid[rmin:rmax + 1, cmin:cmax + 1]

    def crop_color(self, grid, color):
        rows, cols = np.where(grid == color)
        if len(rows) == 0: return grid
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        return grid[rmin:rmax + 1, cmin:cmax + 1]

    def recolor(self, grid, c_old, c_new):
        g = grid.copy()
        g[g == c_old] = c_new
        return g

    def invert_color(self, grid, *args):
        g = grid.copy()
        g[g == 0] = 99  # Temp
        g[g != 99] = 0
        g[g == 99] = 1
        return g

    def add_border(self, grid, color):
        return np.pad(grid, pad_width=1, mode='constant', constant_values=color)

    def flood_fill(self, grid, color):
        g = grid.copy()
        g[g == 0] = color
        return g

    # --- Scaling & Tiling ---
    def upscale(self, grid, factor):
        if factor < 1: factor = 1
        return np.kron(grid, np.ones((factor, factor), dtype=int))

    def tile(self, grid, r_reps, c_reps):
        if r_reps < 1: r_reps = 1
        if c_reps < 1: c_reps = 1
        return np.tile(grid, (r_reps, c_reps))

    # --- Symmetry Primitives ---
    def symmetrize(self, grid, mode):
        """
        Completes the grid by mirroring one half onto the other.
        0: Left -> Right
        1: Right -> Left
        2: Top -> Bottom
        3: Bottom -> Top
        """
        h, w = grid.shape
        g = grid.copy()

        if mode == 0:  # L -> R
            mid = w // 2
            g[:, mid:] = np.fliplr(g[:, :mid]) if w % 2 == 0 else np.fliplr(g[:, :mid])[:, :-1]
        elif mode == 1:  # R -> L
            mid = w // 2
            g[:, :mid] = np.fliplr(g[:, mid:]) if w % 2 == 0 else np.fliplr(g[:, mid + 1:])
        elif mode == 2:  # T -> B
            mid = h // 2
            g[mid:, :] = np.flipud(g[:mid, :]) if h % 2 == 0 else np.flipud(g[:mid, :])[:-1, :]
        elif mode == 3:  # B -> T
            mid = h // 2
            g[:mid, :] = np.flipud(g[mid:, :]) if h % 2 == 0 else np.flipud(g[mid + 1:, :])

        return g

    # --- Object Primitives ---
    def shift_color(self, grid, color, dr, dc):
        """ Moves ONLY pixels of 'color' by (dr, dc). Non-destructive to others if possible. """
        h, w = grid.shape
        g = grid.copy()
        mask = (g == color)
        if not np.any(mask): return g

        # Clear original pos
        g[mask] = 0

        # Calculate new positions
        rows, cols = np.where(mask)
        new_rows = rows + dr
        new_cols = cols + dc

        # Filter bounds
        valid = (new_rows >= 0) & (new_rows < h) & (new_cols >= 0) & (new_cols < w)

        # Set new pos
        g[new_rows[valid], new_cols[valid]] = color
        return g

    def rotate_color_90(self, grid, color):
        """ Rotates ONLY the object defined by 'color' 90 degrees around its bounding box center. """
        rows, cols = np.where(grid == color)
        if len(rows) == 0: return grid.copy()

        # 1. Extract Bounding Box
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        obj_h = rmax - rmin + 1
        obj_w = cmax - cmin + 1

        # Subgrid of just that object
        subgrid = np.zeros((obj_h, obj_w), dtype=int)
        subgrid[rows - rmin, cols - cmin] = 1  # Binary mask of object

        # 2. Rotate Subgrid
        rot_sub = np.rot90(subgrid, k=1)
        rh, rw = rot_sub.shape

        # 3. Place back (centered)
        g = grid.copy()
        g[g == color] = 0  # Erase old

        # Calculate new top-left to keep center roughly same
        center_r = rmin + obj_h // 2
        center_c = cmin + obj_w // 2

        new_rmin = center_r - rh // 2
        new_cmin = center_c - rw // 2

        # Paste
        for r in range(rh):
            for c in range(rw):
                if rot_sub[r, c] == 1:
                    target_r = new_rmin + r
                    target_c = new_cmin + c
                    if 0 <= target_r < g.shape[0] and 0 <= target_c < g.shape[1]:
                        g[target_r, target_c] = color
        return g

if __name__ == "__main__":
    dsl = ARCDSL()