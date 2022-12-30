# coding=utf-8
# Copyright 2022 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ravens main training script."""

# coding=utf-8
# Copyright 2022 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ravens main training script."""

import os
import pickle
import shutil
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import tensorflow as tf
import torch
from absl import app, flags
from depth.apis import multi_gpu_test, single_gpu_test
from depth.datasets import build_dataloader, build_dataset
from depth.datasets.pipelines import Compose
from depth.models import build_depther
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils import DictAction
from torchvision import transforms

from ravens import agents, dataset, tasks
from ravens.environments.environment import Environment
from ravens.tasks import cameras

flags.DEFINE_string("root_dir", ".", "")
flags.DEFINE_string("data_dir", ".", "Directory to dataset.")
flags.DEFINE_string("assets_root", "./assets/", "Path to assets")
flags.DEFINE_bool("disp", False, "")
flags.DEFINE_bool("shared_memory", False, "")
flags.DEFINE_string("task", "hanoi", "task name")
flags.DEFINE_string("agent", "transporter", "Agent type")
flags.DEFINE_integer("n_demos", 100, "Number of training demos used.")
flags.DEFINE_integer("n_steps", 40000, "Maximum number of training steps.")
flags.DEFINE_integer("n_runs", 1, "Number of training. Done sequentially.")
flags.DEFINE_integer("gpu", 0, "Index of used GPU")
flags.DEFINE_integer("gpu_limit", None, "")
flags.DEFINE_string(
    "depth_config_file",
    None,
    "Path to the config file if depth estimation is done.",
)
flags.DEFINE_string(
    "depth_checkpoint_file",
    None,
    "Path to the checkpoint pretrained model if depth estimation is done.",
)

FLAGS = flags.FLAGS


def load_model(depth_config_file, depth_checkpoint_file):
    torch.cuda.set_per_process_memory_fraction(0.1, 0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def infer_depth(color_images, device, preprocess, model):
    image_shape = color_images[0].shape
    assert tuple(image_shape) == (
        480,
        640,
        3,
    ), f"Expecting depth images of size (480, 640, 3) got {tuple(image_shape)}"
    batch = torch.Tensor(color_images).permute(0, 3, 1, 2).to(device)
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
    data = np.concatenate(output_batch, axis=0).reshape(color_images.shape[:-1])
    data_min = np.min(data, axis=(1, 2))
    data_max = np.max(data, axis=(1, 2))
    data = (
        255
        * (data - data_min[:, None, None])
        / (data_max[:, None, None] - data_min[:, None, None])
    )
    return data


def main(unused_argv):
    # Configure which GPU to use.
    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices("GPU")
    if not gpus:
        print("No GPUs detected. Running with CPU.")
    else:
        cfg.set_visible_devices(gpus[FLAGS.gpu], "GPU")

    # Configure how much GPU to use (in Gigabytes).
    if FLAGS.gpu_limit is not None:
        mem_limit = 1024 * FLAGS.gpu_limit
        dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
        cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    # Initialize environment and task.
    env = Environment(
        FLAGS.assets_root, disp=FLAGS.disp, shared_memory=FLAGS.shared_memory, hz=480
    )
    task = tasks.names[FLAGS.task]()
    task.mode = "test"

    # Load test dataset.
    ds = dataset.Dataset(
        os.path.join(FLAGS.data_dir, f"{FLAGS.task}-test"),
        FLAGS.depth_config_file,
        FLAGS.depth_checkpoint_file,
    )
    if ds.estimate_depth:
        device, preprocess, model = load_model(
            FLAGS.depth_config_file,
            FLAGS.depth_checkpoint_file,
        )
    # Run testing for each training run.
    for train_run in range(FLAGS.n_runs):
        name = f"{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}"

        # Initialize agent.
        np.random.seed(train_run)
        tf.random.set_seed(train_run)
        agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)

        # # Run testing every interval.
        # for train_step in range(0, FLAGS.n_steps + 1, FLAGS.interval):

        # Load trained agent.
        if FLAGS.n_steps > 0:
            agent.load(FLAGS.n_steps)

        # Run testing and save total rewards with last transition info.
        results = []
        for i in range(ds.n_episodes):
            print(f"Test: {i + 1}/{ds.n_episodes}")
            episode, seed = ds.load(i)
            goal = episode[-1]
            total_reward = 0
            np.random.seed(seed)
            env.seed(seed)
            env.set_task(task)
            obs = env.reset()
            if ds.estimate_depth:
                obs["depth"] = list(
                    infer_depth(
                        np.stack(obs["color"], axis=0), device, preprocess, model
                    )
                )
            info = None
            reward = 0
            n_steps = 0
            infos = []
            for _ in range(task.max_steps):
                act = agent.act(obs, info, goal)
                obs, reward, done, info = env.step(act)
                if ds.estimate_depth:
                    obs["depth"] = list(
                        infer_depth(
                            np.stack(obs["color"], axis=0), device, preprocess, model
                        )
                    )
                total_reward += reward
                n_steps += 1
                infos.append(info)
                print(f"Total Reward: {total_reward} Done: {done}")
                if done:
                    break
            results.append((total_reward, n_steps, infos))

            # Save results.
            folder_name = (
                "predictions"
                if FLAGS.depth_config_file is None
                else "prections-estimated-depth"
            )
            if not tf.io.gfile.exists(
                os.path.join(
                    FLAGS.root_dir,
                    folder_name,
                    name,
                )
            ):
                tf.io.gfile.makedirs(
                    os.path.join(
                        FLAGS.root_dir,
                        folder_name,
                        name,
                    )
                )
            with tf.io.gfile.GFile(
                os.path.join(
                    FLAGS.root_dir, folder_name, name, f"{name}-{FLAGS.n_steps}.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    app.run(main)
