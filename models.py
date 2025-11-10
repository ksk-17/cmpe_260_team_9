import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)

class UNetARC(nn.Module):
    """
    U-Net that takes a tall input [B, 10, (1+2*K)*30, 30] and returns logits for a 30x30 target window.
    By default, it returns the TOP 30 rows (where the test input block is placed).
    """
    def __init__(
        self,
        in_ch: int = 10,
        base: int = 64,
        num_classes: int = 10,
        block_h: int = 30,
        target_block: int | Literal["top","bottom"] = "top",
        total_blocks: int | None = None,   # if you want "bottom", set total_blocks=(1+2*K_MAX)
    ):
        super().__init__()
        self.block_h = block_h
        self.target_block = target_block
        self.total_blocks = total_blocks  # needed only if target_block="bottom"

        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, 2)
        self.dec1 = DoubleConv(base*2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def _slice_target(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Extract the 30x30 (block_h x block_h) window for supervision.
        Default: TOP block (rows [0:block_h]).
        If target_block is an int k (0-based), uses rows [k*block_h:(k+1)*block_h].
        If 'bottom', uses the last block window (requires total_blocks to be set).
        """
        H = logits.shape[-2]
        W = logits.shape[-1]
        bh = self.block_h
        assert W >= bh and H >= bh, f"Logits too small (H={H},W={W}) for block_h={bh}"

        if self.target_block == "top":
            start = 0
        elif self.target_block == "bottom":
            if self.total_blocks is None:
                raise ValueError("total_blocks must be provided when target_block='bottom'")
            start = (self.total_blocks - 1) * bh
        elif isinstance(self.target_block, int):
            start = int(self.target_block) * bh
        else:
            raise ValueError(f"Invalid target_block={self.target_block}")

        end = start + bh
        # guard in case of minor rounding due to pad/crop
        end = min(end, H)
        start = max(0, end - bh)
        return logits[..., start:end, :]

    def forward(self, x: torch.Tensor, return_full: bool = False):
        """
        x: [B, 10, tallH, 30]
        return_full=False -> return [B, num_classes, 30, 30] sliced target window
        return_full=True  -> also return full logits [B, num_classes, tallH, 30]
        """
        x = x.float()
        B, C, H, W = x.shape

        # pad to multiples of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)

        # encoder
        e1 = self.enc1(x)               # H×W
        e2 = self.enc2(self.pool(e1))   # H/2
        e3 = self.enc3(self.pool(e2))   # H/4
        e4 = self.enc4(self.pool(e3))   # H/8

        # decoder (match skip sizes explicitly)
        d3 = self.up3(e4)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode='nearest')
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode='nearest')
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode='nearest')
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        full = self.head(d1)  # [B, num_classes, H_pad, W_pad]

        # crop to original tallH x W
        if pad_h or pad_w:
            full = full[..., :H, :W]

        if return_full:
            return full, self._slice_target(full)

        return self._slice_target(full)