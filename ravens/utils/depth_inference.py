import pickle

import numpy as np
import torch


def infer_depth(
    color_images: np.ndarray,
    device,
    infer_helper,
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
    _, output_batch = infer_helper.predict(batch / 255)
    data = output_batch.squeeze()
    # data_min = np.min(data, axis=(1, 2))
    # data_max = np.max(data, axis=(1, 2))
    # data = (
    #     255
    #     * (data - data_min[:, None, None])
    #     / (data_max[:, None, None] - data_min[:, None, None])
    # )
    return data.reshape(color_images.shape[:-1])
