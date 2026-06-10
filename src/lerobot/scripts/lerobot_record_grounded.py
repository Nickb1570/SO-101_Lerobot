#!/usr/bin/env python

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.grounding import (
    GROUNDING_BBOX_KEY,
    GROUNDING_CLICK_RADIUS_KEY,
    GROUNDING_CLICK_XY_KEY,
    GROUNDING_DIM_FACTOR_KEY,
    GROUNDING_IMAGE_SIZE_KEY,
    GROUNDING_POLICY_PROMPT_KEY,
    GROUNDING_PREVIEW_PATH_KEY,
    GROUNDING_RAW_PATH_KEY,
    GROUNDING_SELECTION_MODE_KEY,
    GROUNDING_SOURCE_CAMERA_KEY,
    GROUNDING_SUCCESS_KEY,
    select_box_for_episode,
)
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.utils import init_logging, log_say
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config


@dataclass
class GroundedPromptConfig:
    policy_prompt: str = "pick the highlighted strawberry"
    # Bare camera name as it appears in robot.get_observation() (e.g. "scene"),
    # NOT the dataset-feature name "observation.images.scene".
    scene_camera_key: str = "scene"
    # "click": click a pixel, a circle of radius click_radius_px is highlighted.
    # "box": draw a bounding box (needs a non-headless OpenCV build).
    selection_mode: str = "click"
    click_radius_px: int = 32
    dim_factor: float = 0.35
    stamp_dataset_repo_id: bool = False
    # Seconds of live-teleop "get set" countdown after the click, before the
    # episode starts recording. The operator can position the arm during it.
    get_set_seconds: int = 5


@dataclass
class GroundedRecordConfig(RecordConfig):
    grounding: GroundedPromptConfig = field(default_factory=GroundedPromptConfig)

    def __post_init__(self):
        super().__post_init__()
        if not self.dataset.single_task:
            self.dataset.single_task = self.grounding.policy_prompt


def _connect_teleop(teleop) -> None:
    if isinstance(teleop, list):
        for item in teleop:
            item.connect()
        return
    teleop.connect()


def _disconnect_teleop(teleop) -> None:
    if isinstance(teleop, list):
        for item in teleop:
            if item.is_connected:
                item.disconnect()
        return
    if teleop.is_connected:
        teleop.disconnect()


def _to_numpy_image(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    else:
        value = np.asarray(value)

    if value.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {value.shape}.")

    if value.shape[0] in {1, 3} and value.shape[-1] not in {1, 3}:
        value = np.moveaxis(value, 0, -1)

    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value, 0.0, 1.0 if value.max() <= 1.0 else 255.0)
        if value.max() <= 1.0:
            value = (value * 255.0).round()
    return value.astype(np.uint8)


def _capture_selection(
    robot,
    robot_observation_processor,
    dataset_root: str | Path,
    episode_index: int,
    cfg: GroundedRecordConfig,
):
    observation = robot.get_observation()
    observation = robot_observation_processor(observation)
    scene_key = cfg.grounding.scene_camera_key
    if scene_key not in observation:
        available = ", ".join(sorted(observation.keys()))
        raise KeyError(f"Scene camera key '{scene_key}' not found. Available keys: {available}")

    image = _to_numpy_image(observation[scene_key])
    return select_box_for_episode(
        image,
        dataset_root,
        episode_index,
        scene_camera_key=scene_key,
        prompt=cfg.dataset.single_task,
        dim_factor=cfg.grounding.dim_factor,
        click_radius_px=cfg.grounding.click_radius_px,
        mode=cfg.grounding.selection_mode,
    )


def _get_set_countdown(
    *,
    robot,
    events,
    fps,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    teleop,
    seconds: int,
    play_sounds: bool,
) -> None:
    """Live-teleop 'get set' window after the click, before recording starts.

    Runs the teleop pipeline (so the operator can position the arm) for ~1s per
    tick while printing a countdown, but with ``dataset=None`` so nothing is
    saved. A tapped right-arrow just trims one tick; it is cleared afterwards so
    it does not leak into the real recording.
    """
    log_say("Get set", play_sounds)
    for remaining in range(int(seconds), 0, -1):
        print(f"Recording starts in {remaining}...")
        record_loop(
            robot=robot,
            events=events,
            fps=fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=1.0,
        )
        if events["stop_recording"]:
            return
    events["exit_early"] = False


def _approve_or_deny() -> bool:
    """Ask whether to keep the episode just recorded.

    Returns ``True`` to approve (save it) or ``False`` to deny (discard and
    re-record this episode). Asked at the end of the reset window.
    """
    prompt = "Approve this episode? [y]es keep / [n]o re-record: "
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes", "a", "approve"}:
            return True
        if answer in {"n", "no", "x", "reject"}:
            return False
        print("Please enter 'y' (approve) or 'n' (re-record).")


@parser.wrap()
def record_grounded(cfg: GroundedRecordConfig) -> LeRobotDataset | None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            if cfg.grounding.stamp_dataset_repo_id:
                cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        _connect_teleop(teleop)
        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            num_episodes = cfg.dataset.num_episodes
            approved_episodes = 0
            while approved_episodes < num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                selection = _capture_selection(
                    robot,
                    robot_observation_processor,
                    dataset.root,
                    episode_index,
                    cfg,
                )

                # Click -> get-set countdown -> record. Teleop stays live during
                # the countdown so the operator can position the arm.
                _get_set_countdown(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    seconds=cfg.grounding.get_set_seconds,
                    play_sounds=cfg.play_sounds,
                )
                if events["stop_recording"]:
                    break

                log_say(f"Recording grounded episode {episode_index}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=cfg.display_compressed_images,
                )

                # Esc during the episode: stop the whole session, drop the buffer.
                if events["stop_recording"]:
                    dataset.clear_episode_buffer()
                    break

                # Left-arrow during the episode is an instant deny: skip the reset
                # window and re-record straight away.
                instant_deny = events["rerecord_episode"]
                events["rerecord_episode"] = False
                events["exit_early"] = False

                # Reset timer: teleop stays live so you can reset the scene / move
                # the arm home. Press the right arrow when done to end it early.
                if not instant_deny:
                    log_say(
                        f"Reset the environment - up to {int(cfg.dataset.reset_time_s)}s, press the right arrow when ready",
                        cfg.play_sounds,
                    )
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )
                    events["exit_early"] = False

                if events["stop_recording"]:
                    dataset.clear_episode_buffer()
                    break

                # Approve/deny: decided now, at the end of the reset window.
                approve = False if instant_deny else _approve_or_deny()
                if not approve:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    print(f"Episode denied. Approved episodes: {approved_episodes}/{num_episodes}")
                    continue

                episode_metadata = {
                    GROUNDING_BBOX_KEY: selection.bbox_xyxy,
                    GROUNDING_CLICK_XY_KEY: selection.center_xy if selection.center_xy is not None else [],
                    GROUNDING_CLICK_RADIUS_KEY: selection.radius_px if selection.radius_px is not None else 0,
                    GROUNDING_DIM_FACTOR_KEY: float(cfg.grounding.dim_factor),
                    GROUNDING_IMAGE_SIZE_KEY: selection.image_size_hw,
                    GROUNDING_POLICY_PROMPT_KEY: cfg.dataset.single_task,
                    GROUNDING_PREVIEW_PATH_KEY: selection.preview_path,
                    GROUNDING_RAW_PATH_KEY: selection.raw_path,
                    GROUNDING_SELECTION_MODE_KEY: selection.selection_mode,
                    GROUNDING_SOURCE_CAMERA_KEY: cfg.grounding.scene_camera_key,
                    GROUNDING_SUCCESS_KEY: True,
                }
                dataset.save_episode(episode_metadata=episode_metadata)
                approved_episodes += 1
                print(f"Approved episodes: {approved_episodes}/{num_episodes}")
                log_say(f"Approved {approved_episodes} of {num_episodes}", cfg.play_sounds)
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        _disconnect_teleop(teleop)

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved - skipping push to hub")

        log_say("Exiting", cfg.play_sounds)

    return dataset


def main():
    # Silence the SVT-AV1 encoder's per-episode version/config banner that PyAV
    # prints straight to the terminal. SVT_LOG=1 = errors only. setdefault so an
    # explicit env override still wins.
    os.environ.setdefault("SVT_LOG", "1")
    register_third_party_plugins()
    record_grounded()


if __name__ == "__main__":
    main()
