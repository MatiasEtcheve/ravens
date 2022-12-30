from pathlib import Path
from typing import Tuple

import mmcv
import numpy as np
import torch
from depth.models import build_depther
from mmcv.runner import load_checkpoint
from torchvision import transforms


def load_model(
    depth_config_file: Path, depth_checkpoint_file: Path
) -> Tuple[str, transforms.Compose, torch.nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(0.1, device)
    preprocess = transforms.Compose(
        [
            transforms.Normalize(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
            )
        ]
    )
    cfg = mmcv.Config.fromfile(depth_config_file)
    model = build_depther(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(depth_checkpoint_file), map_location="cpu")
    model = model.to(device)
    model.eval()
    return device, preprocess, model


def infer_depth(
    color_images: np.ndarray,
    device: str,
    preprocess: transforms.Compose,
    model: torch.nn.Module,
) -> np.ndarray:
    image_shape = color_images.shape[-3:]
    assert tuple(image_shape) == (
        480,
        640,
        3,
    ), f"Expecting depth images of size (480, 640, 3) got {tuple(image_shape)}"
    batch = (
        torch.Tensor(color_images)
        .reshape(-1, *image_shape)
        .permute(0, 3, 1, 2)
        .to(device)
    )
    img_metas = [
        {
            "pad_shape": tuple(_.shape),
            "img_shape": tuple(_.shape),
            "ori_shape": tuple(_.shape),
            "scale_factor": 1,
            "flip": False,
            "img_norm_cfg": {
                "mean": _.mean(axis=(1, 2)),
                "std": _.mean(axis=(1, 2)),
            },
        }
        for _ in batch
    ]
    model.eval()
    with torch.no_grad():
        output_batch = model.forward(
            [preprocess(batch)], [img_metas], return_loss=False
        )
    data = np.concatenate(output_batch, axis=0)
    data_min = np.min(data, axis=(1, 2))
    data_max = np.max(data, axis=(1, 2))
    data = (
        255
        * (data - data_min[:, None, None])
        / (data_max[:, None, None] - data_min[:, None, None])
    )
    return data.reshape(color_images.shape[:-1])
