#!/usr/bin/env python

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.grounding import (
    GROUNDING_BBOX_KEY,
    GROUNDING_DIM_FACTOR_KEY,
    GROUNDING_POLICY_PROMPT_KEY,
    GROUNDING_SOURCE_CAMERA_KEY,
    GROUNDING_SUCCESS_KEY,
    grounding_frame,
)
from lerobot.grounding.pipeline import ellipse_mask_from_box
from lerobot.utils.constants import DEFAULT_FEATURES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class SegmentDatasetConfig:
    source_repo_id: str
    output_repo_id: str
    source_root: str | Path | None = None
    output_root: str | Path | None = None
    scene_camera_key: str = "observation.images.scene"
    dim_factor: float | None = None
    push_to_hub: bool = False
    private: bool = False
    video: bool = True
    batch_encoding_size: int = 1


def _to_numpy_image(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    else:
        value = np.asarray(value)

    if value.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {value.shape}.")
    if value.shape[0] in {1, 3} and value.shape[-1] not in {1, 3}:
        value = np.moveaxis(value, 0, -1)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value, 0.0, 1.0 if value.max() <= 1.0 else 255.0)
        if value.max() <= 1.0:
            value = (value * 255.0).round()
    return value.astype(np.uint8)


def _to_numpy_feature(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return value


def _episode_column(meta, name: str):
    episodes = meta.episodes
    column_names = getattr(episodes, "column_names", None)
    if column_names is not None and name not in column_names:
        return None
    try:
        return episodes[name]
    except Exception:
        return None


def _episode_value(meta, name: str, episode_index: int, default=None):
    column = _episode_column(meta, name)
    if column is None:
        return default
    value = column[episode_index]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@parser.wrap()
def segment_dataset(cfg: SegmentDatasetConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    src = LeRobotDataset(
        cfg.source_repo_id,
        root=cfg.source_root,
        return_uint8=True,
    )

    robot_type = getattr(getattr(src.meta, "info", None), "robot_type", "robot")
    dst = LeRobotDataset.create(
        cfg.output_repo_id,
        src.fps,
        root=cfg.output_root,
        robot_type=robot_type,
        features=src.features,
        use_videos=cfg.video and len(src.meta.video_keys) > 0,
        batch_encoding_size=cfg.batch_encoding_size,
    )

    with VideoEncodingManager(dst):
        for episode_index in range(src.num_episodes):
            start_index = int(_episode_value(src.meta, "dataset_from_index", episode_index))
            end_index = int(_episode_value(src.meta, "dataset_to_index", episode_index))
            bbox_xyxy = _episode_value(src.meta, GROUNDING_BBOX_KEY, episode_index)
            if bbox_xyxy is None:
                raise ValueError(f"Missing {GROUNDING_BBOX_KEY} for episode {episode_index}.")

            dim_factor = cfg.dim_factor
            if dim_factor is None:
                dim_factor = float(_episode_value(src.meta, GROUNDING_DIM_FACTOR_KEY, episode_index, default=0.35))

            for frame_index in range(start_index, end_index):
                item = src[frame_index]
                frame = {"task": item["task"]}

                for key, feature in src.features.items():
                    # DEFAULT_FEATURES (index, episode_index, task_index, timestamp,
                    # frame_index) are auto-populated by add_frame and rejected as
                    # extra features if supplied by the caller.
                    if key in DEFAULT_FEATURES:
                        continue

                    value = item[key]
                    if key == cfg.scene_camera_key:
                        image = _to_numpy_image(value)
                        mask = ellipse_mask_from_box(image.shape, [int(v) for v in bbox_xyxy])
                        frame[key] = grounding_frame(image, mask, dim_factor=dim_factor)
                    elif feature["dtype"] in {"image", "video"}:
                        frame[key] = _to_numpy_image(value)
                    else:
                        frame[key] = _to_numpy_feature(value)

                dst.add_frame(frame)

            episode_metadata = {
                GROUNDING_BBOX_KEY: [int(v) for v in bbox_xyxy],
                GROUNDING_DIM_FACTOR_KEY: float(dim_factor),
                GROUNDING_POLICY_PROMPT_KEY: _episode_value(src.meta, GROUNDING_POLICY_PROMPT_KEY, episode_index),
                GROUNDING_SOURCE_CAMERA_KEY: _episode_value(src.meta, GROUNDING_SOURCE_CAMERA_KEY, episode_index),
                GROUNDING_SUCCESS_KEY: bool(_episode_value(src.meta, GROUNDING_SUCCESS_KEY, episode_index, default=False)),
                "grounding_source_dataset_repo_id": cfg.source_repo_id,
                "grounding_source_episode_index": int(episode_index),
                "grounding_segmentation_backend": "box_fallback",
            }
            dst.save_episode(episode_metadata=episode_metadata)

    dst.finalize()

    if cfg.push_to_hub and dst.num_episodes > 0:
        dst.push_to_hub(private=cfg.private)

    return dst


def main():
    register_third_party_plugins()
    segment_dataset()


if __name__ == "__main__":
    main()
