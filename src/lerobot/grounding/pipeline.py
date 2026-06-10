from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

GROUNDING_BBOX_KEY = "grounding_bbox_xyxy"
GROUNDING_CLICK_XY_KEY = "grounding_click_xy"
GROUNDING_CLICK_RADIUS_KEY = "grounding_click_radius_px"
GROUNDING_DIM_FACTOR_KEY = "grounding_dim_factor"
GROUNDING_IMAGE_SIZE_KEY = "grounding_image_size_hw"
GROUNDING_POLICY_PROMPT_KEY = "grounding_policy_prompt"
GROUNDING_PREVIEW_PATH_KEY = "grounding_preview_path"
GROUNDING_RAW_PATH_KEY = "grounding_raw_path"
GROUNDING_SELECTION_MODE_KEY = "grounding_selection_mode"
GROUNDING_SOURCE_CAMERA_KEY = "grounding_source_camera_key"
GROUNDING_SUCCESS_KEY = "episode_success"


@dataclass(frozen=True)
class SelectionResult:
    bbox_xyxy: list[int]
    raw_path: str
    preview_path: str
    image_size_hw: list[int]
    selection_mode: str
    center_xy: list[int] | None = None
    radius_px: int | None = None


def grounding_frame(image: np.ndarray, mask: np.ndarray, dim_factor: float = 0.35) -> np.ndarray:
    out = (image.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)
    out[mask.astype(bool)] = image[mask.astype(bool)]
    return out


def ellipse_mask_from_box(image_shape: tuple[int, int, int] | tuple[int, int], bbox_xyxy: list[int]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, y1 + 1, height))

    ys, xs = np.ogrid[:height, :width]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = max((x2 - x1) / 2.0, 1.0)
    ry = max((y2 - y1) / 2.0, 1.0)
    mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    return mask


def bbox_from_center(
    image_shape: tuple[int, int, int] | tuple[int, int], center_xy: tuple[float, float], radius_px: int
) -> list[int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r = max(int(radius_px), 1)
    x1 = int(np.clip(round(cx - r), 0, width - 1))
    y1 = int(np.clip(round(cy - r), 0, height - 1))
    x2 = int(np.clip(round(cx + r), x1 + 1, width))
    y2 = int(np.clip(round(cy + r), y1 + 1, height))
    return [x1, y1, x2, y2]


def select_box_for_episode(
    image: np.ndarray,
    dataset_root: str | Path,
    episode_index: int,
    *,
    scene_camera_key: str,
    prompt: str,
    dim_factor: float,
    click_radius_px: int = 32,
    mode: str = "click",
) -> SelectionResult:
    dataset_root = Path(dataset_root)

    rel_dir = Path("grounding") / f"episode-{episode_index:06d}"
    abs_dir = dataset_root / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = rel_dir / "selection_raw.png"
    preview_path = rel_dir / "selection_preview.png"

    # Save the raw frame first so a terminal fallback can open it to read pixels.
    Image.fromarray(image).save(abs_dir / raw_path.name)

    center_xy: list[int] | None = None
    radius_px: int | None = None
    if mode == "box":
        bbox_xyxy = _select_box(image, prompt=prompt)
        mode = "box"
    else:
        center_xy = _select_click(
            image, prompt=prompt, radius_px=click_radius_px, raw_image_path=abs_dir / raw_path.name
        )
        radius_px = int(click_radius_px)
        bbox_xyxy = bbox_from_center(image.shape, center_xy, radius_px)
        mode = "click"

    preview = grounding_frame(image, ellipse_mask_from_box(image.shape, bbox_xyxy), dim_factor=dim_factor)
    Image.fromarray(preview).save(abs_dir / preview_path.name)

    return SelectionResult(
        bbox_xyxy=bbox_xyxy,
        raw_path=raw_path.as_posix(),
        preview_path=preview_path.as_posix(),
        image_size_hw=[int(image.shape[0]), int(image.shape[1])],
        selection_mode=mode,
        center_xy=center_xy,
        radius_px=radius_px,
    )


def terminal_binary_label(prompt: str = "Episode success? [y/n]: ") -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


# ── Click selection (point + circle) ──────────────────────────────────────────


def _select_click(image: np.ndarray, *, prompt: str, radius_px: int, raw_image_path: Path) -> list[int]:
    """Show the frame and capture a single click. Returns [x, y] in pixels.

    Tries a Tkinter window first (stdlib + Pillow, no OpenCV GUI needed), then
    matplotlib, then falls back to typing coordinates in the terminal.
    """
    for backend in (_click_tk, _click_matplotlib):
        try:
            xy = backend(image, prompt, radius_px)
        except Exception:
            continue
        if xy is not None:
            return [int(round(xy[0])), int(round(xy[1]))]
        # A GUI opened but was closed without a click: go straight to terminal.
        break
    return _click_terminal(image, prompt, raw_image_path)


def _click_tk(image: np.ndarray, prompt: str, radius_px: int) -> tuple[float, float] | None:
    import tkinter as tk

    from PIL import ImageTk

    pil = Image.fromarray(image)
    root = tk.Tk()
    root.title(f"Click the target: {prompt}  —  click, then press Enter (Esc to skip)")
    photo = ImageTk.PhotoImage(pil)
    canvas = tk.Canvas(root, width=pil.width, height=pil.height, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)

    state: dict[str, object] = {"xy": None, "marker": None}

    def on_click(event: "tk.Event") -> None:
        state["xy"] = (event.x, event.y)
        if state["marker"] is not None:
            canvas.delete(state["marker"])
        r = max(int(radius_px), 1)
        state["marker"] = canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r, outline="#39ff14", width=2
        )

    def confirm(_event: "tk.Event | None" = None) -> None:
        if state["xy"] is not None:
            root.quit()

    def skip(_event: "tk.Event | None" = None) -> None:
        state["xy"] = None
        root.quit()

    canvas.bind("<Button-1>", on_click)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", skip)
    root.protocol("WM_DELETE_WINDOW", confirm)
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    root.mainloop()
    root.destroy()
    return state["xy"]  # type: ignore[return-value]


def _click_matplotlib(image: np.ndarray, prompt: str, radius_px: int) -> tuple[float, float] | None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Click the target: {prompt}\nthen close this window")
    state: dict[str, object] = {"xy": None, "patch": None}

    def on_click(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        state["xy"] = (float(event.xdata), float(event.ydata))
        if state["patch"] is not None:
            state["patch"].remove()
        patch = Circle((event.xdata, event.ydata), max(int(radius_px), 1), fill=False, color="#39ff14", linewidth=2)
        ax.add_patch(patch)
        state["patch"] = patch
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return state["xy"]  # type: ignore[return-value]


def _click_terminal(image: np.ndarray, prompt: str, raw_image_path: Path) -> list[int]:
    height, width = image.shape[:2]
    print(f"No display available. Open this frame to read pixel coordinates:\n  {raw_image_path}")
    print(f"Target: '{prompt}'. Image size: {width}x{height}.")
    print("Enter the click point as x,y")
    while True:
        raw = input("> ").strip()
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            print("Need exactly two comma-separated integers (x,y).")
            continue
        try:
            x, y = (int(part) for part in parts)
        except ValueError:
            print("Both coordinates must be integers.")
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            print("The point must stay within the image bounds.")
            continue
        return [x, y]


# ── Box selection (kept for `--grounding.selection_mode=box`) ──────────────────


def _select_box(image: np.ndarray, *, prompt: str) -> list[int]:
    try:
        return _select_box_opencv(image, prompt)
    except Exception:
        pass
    return _select_box_terminal(image, prompt)


def _select_box_opencv(image: np.ndarray, prompt: str) -> list[int]:
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    title = f"Draw a box around target: {prompt}"
    x, y, w, h = cv2.selectROI(title, bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w <= 0 or h <= 0:
        raise ValueError("No selection made.")
    return [int(x), int(y), int(x + w), int(y + h)]


def _select_box_terminal(image: np.ndarray, prompt: str) -> list[int]:
    height, width = image.shape[:2]
    print(f"Select target box for '{prompt}'. Image size: {width}x{height}.")
    print("Enter box as x1,y1,x2,y2")
    while True:
        raw = input("> ").strip()
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4:
            print("Need exactly four comma-separated integers.")
            continue
        try:
            x1, y1, x2, y2 = (int(part) for part in parts)
        except ValueError:
            print("All coordinates must be integers.")
            continue
        if x2 <= x1 or y2 <= y1:
            print("The lower-right corner must be strictly below and right of the upper-left corner.")
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print("The box must stay within the image bounds.")
            continue
        return [x1, y1, x2, y2]
