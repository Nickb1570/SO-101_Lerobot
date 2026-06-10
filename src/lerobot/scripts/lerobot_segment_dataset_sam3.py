#!/usr/bin/env python
"""Derive a SAM3.1-grounded dataset from a recorded grounded dataset.

For each source episode this reads the stored frame-0 click point
(``grounding_click_xy``) from ``meta/episodes``, decodes the episode's scene
frames out of the MP4 videos, seeds SAM3.1 on frame 0 with that point, then
``propagate_in_video`` tracks the target mask across every frame. Each frame's
``observation.images.scene`` is rewritten to the grounded view (target bright,
rest dimmed by ``dim_factor``); all other features (wrist image, state, action)
are copied unchanged. The result is a new LeRobot dataset ready for policy
training (e.g. pi0.5).

Runs on a CUDA GPU (SAM video propagation is not practical on CPU). When SAM
returns no mask for a frame (e.g. brief occlusion), it falls back to the static
click-circle mask so the frame is still grounded.

NOTE: this script intentionally does NOT use ``from __future__ import
annotations`` — ``@parser.wrap()`` reads the function's raw annotation, and a
stringized annotation breaks draccus.
"""

import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
from PIL import Image

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.grounding import (
    GROUNDING_BBOX_KEY,
    GROUNDING_CLICK_XY_KEY,
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
class SegmentSam3Config:
    source_repo_id: str
    output_repo_id: str
    source_root: str | Path | None = None
    output_root: str | Path | None = None
    # Dataset-feature name of the scene stream (NOT the bare obs key "scene").
    scene_camera_key: str = "observation.images.scene"
    # Optional text concept to bias SAM toward; the stored click point is the
    # primary, per-episode disambiguator (which strawberry was selected).
    sam_prompt: str = "strawberry"
    # Path to a SAM3.1 checkpoint. None -> download from Hugging Face.
    sam_checkpoint: str | None = None
    sam_version: str = "sam3.1"
    # Override the per-episode dim_factor stored at record time.
    dim_factor: float | None = None
    push_to_hub: bool = False
    private: bool = False
    video: bool = True
    batch_encoding_size: int = 1
    device: str = "cuda"


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


# ── SAM3.1 video predictor ────────────────────────────────────────────────────
# These helpers wrap the SAM3.1 video predictor. The call pattern below mirrors
# the SAM2 video API that SAM3 follows (init_state / add_new_points_or_box /
# propagate_in_video). If your installed `sam3` build names these differently,
# match them to the repo's video-predictor example notebook — only this section
# should need touching.


def _build_sam3_video_predictor(cfg: SegmentSam3Config):
    import torch
    from sam3.model_builder import build_sam3_video_predictor, download_ckpt_from_hf

    checkpoint = cfg.sam_checkpoint
    if checkpoint is None:
        checkpoint = download_ckpt_from_hf(version=cfg.sam_version)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    predictor = build_sam3_video_predictor(checkpoint_path=checkpoint, device=device)
    return predictor, device


def _propagate_episode(predictor, frames_dir: str, num_frames: int, point_xy, device) -> list:
    """Seed SAM on frame 0 with the click point and propagate across the episode.

    Returns a list of length ``num_frames``; entries are bool HxW masks or None
    where SAM produced nothing.
    """
    import torch

    masks: list = [None] * num_frames
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else _nullctx()
    with torch.inference_mode(), autocast:
        state = predictor.init_state(video_path=frames_dir)
        predictor.reset_state(state)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=np.array([point_xy], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            logits = mask_logits[0]
            arr = (logits > 0.0).squeeze().detach().cpu().numpy().astype(bool)
            if arr.ndim == 2 and 0 <= frame_idx < num_frames:
                masks[frame_idx] = arr
    return masks


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@parser.wrap()
def segment_dataset_sam3(cfg: SegmentSam3Config) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    src = LeRobotDataset(cfg.source_repo_id, root=cfg.source_root, return_uint8=True)

    predictor, device = _build_sam3_video_predictor(cfg)
    logging.info(f"SAM3.1 video predictor ready on {device}.")

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
            num_frames = end_index - start_index

            bbox_xyxy = _episode_value(src.meta, GROUNDING_BBOX_KEY, episode_index)
            if bbox_xyxy is None:
                raise ValueError(f"Missing {GROUNDING_BBOX_KEY} for episode {episode_index}.")
            bbox_xyxy = [int(v) for v in bbox_xyxy]

            click_xy = _episode_value(src.meta, GROUNDING_CLICK_XY_KEY, episode_index)
            if click_xy:
                point_xy = [float(click_xy[0]), float(click_xy[1])]
            else:  # fall back to bbox center if this episode was recorded in box mode
                x1, y1, x2, y2 = bbox_xyxy
                point_xy = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

            dim_factor = cfg.dim_factor
            if dim_factor is None:
                dim_factor = float(_episode_value(src.meta, GROUNDING_DIM_FACTOR_KEY, episode_index, default=0.35))

            # Decode scene frames for this episode and dump to a temp dir for SAM.
            images = []
            for frame_index in range(start_index, end_index):
                images.append(_to_numpy_image(src[frame_index][cfg.scene_camera_key]))

            tmp_dir = tempfile.mkdtemp(prefix=f"sam_ep{episode_index:06d}_")
            try:
                for i, img in enumerate(images):
                    Image.fromarray(img).save(Path(tmp_dir) / f"{i:05d}.jpg")

                masks = _propagate_episode(predictor, tmp_dir, num_frames, point_xy, device)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Static click-circle fallback for any frame SAM left empty.
            fallback_mask = ellipse_mask_from_box(images[0].shape, bbox_xyxy)
            n_fallback = 0

            for i, frame_index in enumerate(range(start_index, end_index)):
                item = src[frame_index]
                frame = {"task": item["task"]}

                mask = masks[i]
                if mask is None or not mask.any():
                    mask = fallback_mask
                    n_fallback += 1

                for key, feature in src.features.items():
                    if key in DEFAULT_FEATURES:
                        continue
                    value = item[key]
                    if key == cfg.scene_camera_key:
                        frame[key] = grounding_frame(images[i], mask, dim_factor=dim_factor)
                    elif feature["dtype"] in {"image", "video"}:
                        frame[key] = _to_numpy_image(value)
                    else:
                        frame[key] = _to_numpy_feature(value)

                dst.add_frame(frame)

            logging.info(
                f"Episode {episode_index}: {num_frames} frames, "
                f"{n_fallback} fell back to click-circle ({100 * n_fallback / max(num_frames, 1):.0f}%)."
            )

            episode_metadata = {
                GROUNDING_BBOX_KEY: bbox_xyxy,
                GROUNDING_CLICK_XY_KEY: [int(round(point_xy[0])), int(round(point_xy[1]))],
                GROUNDING_DIM_FACTOR_KEY: float(dim_factor),
                GROUNDING_POLICY_PROMPT_KEY: _episode_value(src.meta, GROUNDING_POLICY_PROMPT_KEY, episode_index),
                GROUNDING_SOURCE_CAMERA_KEY: _episode_value(src.meta, GROUNDING_SOURCE_CAMERA_KEY, episode_index),
                GROUNDING_SUCCESS_KEY: bool(_episode_value(src.meta, GROUNDING_SUCCESS_KEY, episode_index, default=False)),
                "grounding_source_dataset_repo_id": cfg.source_repo_id,
                "grounding_source_episode_index": int(episode_index),
                "grounding_segmentation_backend": cfg.sam_version,
                "grounding_sam_fallback_frames": int(n_fallback),
            }
            dst.save_episode(episode_metadata=episode_metadata)

    dst.finalize()

    if cfg.push_to_hub and dst.num_episodes > 0:
        dst.push_to_hub(private=cfg.private)

    return dst


def main():
    register_third_party_plugins()
    segment_dataset_sam3()


if __name__ == "__main__":
    main()
