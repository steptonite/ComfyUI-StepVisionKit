"""Cut By Mask — replaces the five-node black-background workaround.

Plain torch arithmetic over the whole batch. No PIL: the WAS node this replaces
(Image Blend by Mask) calls tensor2pil() with no batch loop, so it dies on any
video with 'Cannot handle this data type'.
"""

import numpy as np
import torch

from .boxes import hex_to_rgb

BACKGROUND = ["black", "white", "transparent", "blur", "original", "color"]


class VKCutByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "frames to cut"}),
                "mask": ("MASK", {"tooltip": "White = keep. From SAM2, MediaPipe, anything."}),
                "background": (BACKGROUND, {"default": "black", "tooltip": "What replaces everything outside the mask. transparent gives a black image plus a mask — send both to Join Image With Alpha and save as webp or png, because h264 has no alpha."}),
                "color": ("STRING", {"default": "#000000", "tooltip": "For background=color."}),
                "blur_amount": ("INT", {"default": 24, "min": 1, "max": 200, "tooltip": "For background=blur."}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Soften the mask edge by N pixels."}),
                "grow": ("INT", {"default": 0, "min": -200, "max": 200, "tooltip": "Expand (+) or contract (-) the mask."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Keep the background instead of the object."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "cut"
    CATEGORY = "VisionKit"
    DESCRIPTION = "Cut the masked object out onto a chosen background. Works on a whole video batch."

    def cut(self, images, mask, background, color, blur_amount, feather, grow, invert_mask):
        import cv2

        n, H, W = images.shape[0], images.shape[1], images.shape[2]
        m = mask.detach().cpu().float().numpy()
        if m.ndim == 2:
            m = m[None]

        proc = []
        for i in range(n):
            a = m[min(i, m.shape[0] - 1)]
            if a.shape[0] != H or a.shape[1] != W:
                a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
            if invert_mask:
                a = 1.0 - a
            if grow:
                k = np.ones((abs(grow) * 2 + 1,) * 2, np.uint8)
                a = cv2.dilate(a, k) if grow > 0 else cv2.erode(a, k)
            if feather:
                k = feather * 2 + 1
                a = cv2.GaussianBlur(a, (k, k), 0)
            proc.append(np.clip(a, 0, 1).astype(np.float32))

        alpha = torch.from_numpy(np.stack(proc))                  # (n,H,W)
        a4 = alpha.unsqueeze(-1)
        img = images.detach().cpu().float()

        if background == "original":
            bg = img
        elif background == "white":
            bg = torch.ones_like(img)
        elif background == "color":
            r, g, b = [v / 255.0 for v in hex_to_rgb(color)]
            bg = torch.zeros_like(img)
            bg[..., 0], bg[..., 1], bg[..., 2] = r, g, b
        elif background == "blur":
            k = blur_amount * 2 + 1
            arr = img.numpy()
            bg = torch.from_numpy(np.stack([cv2.GaussianBlur(arr[i], (k, k), 0) for i in range(n)]))
        else:                                                     # black, transparent
            bg = torch.zeros_like(img)

        out = img * a4 + bg * (1.0 - a4)
        print(f"[VKCutByMask] frames {n} · background {background} · "
              f"mean coverage {float(alpha.mean()):.3f}")
        return (torch.clamp(out, 0, 1), alpha)


NODES = {"VKCutByMask": VKCutByMask}
DISPLAY = {"VKCutByMask": "Cut By Mask"}
