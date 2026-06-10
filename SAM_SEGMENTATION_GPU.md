# SAM3.1 Grounding on the GPU box → π0.5 training data

Run this on the **Linux / RTX 5090** machine (CUDA, has Hugging Face access).
It turns the recorded raw dataset into a **SAM3.1-grounded** dataset that π0.5
trains on.

- **Input:**  `Atabaku/so101-strawberry-raw` (50 episodes, on the Hub)
- **Output:** `Atabaku/so101-strawberry-raw-grounded`
- **What it does:** for each episode, reads the stored frame-0 **click point**,
  decodes the scene frames from the MP4s, seeds SAM3.1 on frame 0 with that
  point, **propagates the mask across the whole episode** (so the highlight
  tracks the strawberry as it's picked and placed), and rewrites
  `observation.images.scene` to the grounded view (target bright, rest dimmed).
  Wrist image, state, and action are copied unchanged.

## How the data is stored (so the approach makes sense)

The camera streams are **MP4 video** (`videos/`), one file per camera per
episode-chunk — **not** per-frame images. `LeRobotDataset[i]` decodes the frame
on demand, which is how the script reads them (it dumps each episode's decoded
scene frames to a temp dir of JPEGs for SAM's video predictor, then deletes
them). The only loose PNGs in the repo are `grounding/episode-*/selection_*.png`
(a single frame-0 snapshot + preview). The per-episode **click point** lives in
`meta/episodes` (`grounding_click_xy`) and is what seeds SAM.

---

## 1. Environment

The output dataset writes custom grounding columns via the fork's extended
`save_episode(episode_metadata=...)`, so you **must** use this fork (not stock
`lerobot`). SAM3.1 deps can clash with robot deps, so keep them in this one env
(no robot/camera packages needed here).

```bash
# clone the SO-101 fork (the one with src/lerobot/grounding + the SAM script)
git clone <your-fork-url> SO-101_Lerobot
cd SO-101_Lerobot

conda create -n sam31 python=3.10 -y
conda activate sam31

# CUDA torch for the 5090 (Blackwell → use a recent cu12x wheel; cu124+ recommended)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# the fork (dataset IO, grounding module, the SAM segment script)
pip install -e ".[dataset]"

# SAM3.1 (Meta) — editable install per its README
pip install git+https://github.com/facebookresearch/sam3.git
#   or: git clone https://github.com/facebookresearch/sam3 && pip install -e ./sam3
```

> RTX 5090 (Blackwell, sm_120) needs a recent CUDA torch build. If you see
> `no kernel image is available for execution`, your torch is too old for the
> GPU — install a newer cu12x/cu128 wheel (or a nightly) and retry.

## 2. Hugging Face login

SAM3.1 weights are gated — accept the model license on its HF page first, then:

```bash
hf auth login        # needs read (gated weights + private dataset) + write (push output)
hf auth whoami       # confirm it's the Atabaku account
```

## 3. Sanity check on ONE episode first

Don't process all 50 blind. Run a 1-episode slice and eyeball the result. The
quickest check: temporarily point at the dataset and process, then open a few
grounded frames. (The script processes the whole dataset; to spot-check fast,
make a 1-episode copy or just run it and inspect the first episode's frames.)

```bash
python -m lerobot.scripts.lerobot_segment_dataset_sam3 \
  --source_repo_id=Atabaku/so101-strawberry-raw \
  --output_repo_id=Atabaku/so101-strawberry-grounded-TEST \
  --scene_camera_key=observation.images.scene \
  --sam_prompt="strawberry" \
  --push_to_hub=false
```

It logs per episode, e.g. `Episode 0: 512 frames, 7 fell back to click-circle (1%)`.
A **low fallback %** means SAM tracked the strawberry well. A **high fallback %**
(say >40%) means SAM lost the object — see Troubleshooting.

Inspect frames from the local output (default cache
`~/.cache/huggingface/lerobot/Atabaku/so101-strawberry-grounded-TEST`):

```python
python - <<'PY'
from lerobot.datasets import LeRobotDataset
from PIL import Image
import numpy as np
d = LeRobotDataset("Atabaku/so101-strawberry-grounded-TEST")
for i in [0, 50, 150, 300]:            # start, mid, later frames of episode 0
    img = d[i]["observation.images.scene"]
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.shape[0] in (1, 3): arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8: arr = (arr*255).astype("uint8") if arr.max()<=1 else arr.astype("uint8")
    Image.fromarray(arr).save(f"check_{i:04d}.png")
print("wrote check_*.png — the strawberry should stay BRIGHT as it moves; rest dimmed")
PY
```

Open `check_*.png`. The strawberry should remain highlighted **through the pick
and into the bin** — that's the whole point of SAM propagation vs. the static
click-circle. If it looks right, delete the TEST dataset and do the full run:

```bash
python -c "from huggingface_hub import delete_repo; delete_repo('Atabaku/so101-strawberry-grounded-TEST', repo_type='dataset', missing_ok=True)"
rm -rf ~/.cache/huggingface/lerobot/Atabaku/so101-strawberry-grounded-TEST
```

## 4. Full run — all 50 episodes, push to the Hub

```bash
python -m lerobot.scripts.lerobot_segment_dataset_sam3 \
  --source_repo_id=Atabaku/so101-strawberry-raw \
  --output_repo_id=Atabaku/so101-strawberry-raw-grounded \
  --scene_camera_key=observation.images.scene \
  --sam_prompt="strawberry" \
  --push_to_hub=true \
  --private=true
```

Result: `https://huggingface.co/datasets/Atabaku/so101-strawberry-raw-grounded`
(private), same structure as the raw set but with the grounded scene stream.
Each episode's `meta/episodes` records `grounding_segmentation_backend=sam3.1`
and `grounding_sam_fallback_frames` so you can audit mask quality later.

## 5. Train π0.5 on the grounded dataset

π0.5 is language-conditioned; the dataset's task string is already
`"pick up the highlighted strawberry and place into the green bin"`. Starter
command (tune against `src/lerobot/policies/pi05/README.md`):

```bash
lerobot-train \
  --dataset.repo_id=Atabaku/so101-strawberry-raw-grounded \
  --policy.type=pi05 \
  --policy.device=cuda \
  --output_dir=outputs/train/pi05_strawberry \
  --job_name=pi05_strawberry \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000
# add --wandb.enable=true to log curves
```

Train on the **grounded** repo, not the raw one. Keep the wrist image and state
in the policy inputs if π0.5's SO-101 config expects them.

---

## Troubleshooting

- **SAM API names differ.** The script uses `init_state` /
  `add_new_points_or_box` / `propagate_in_video` / `build_sam3_video_predictor`
  (the SAM2 video pattern SAM3 follows). If your `sam3` build names them
  differently, edit only `_build_sam3_video_predictor` and `_propagate_episode`
  in `src/lerobot/scripts/lerobot_segment_dataset_sam3.py` to match the repo's
  video-predictor example notebook.
- **High fallback % / SAM loses the strawberry.** Options: (a) add the text
  concept — it already passes `--sam_prompt="strawberry"`; try
  `"ripe red strawberry"`; (b) the click point may be slightly off-object on
  frame 0 — check `selection_preview.png` for that episode; (c) heavy occlusion
  by the gripper is expected late in the pick — the click-circle fallback covers
  those frames so the episode is still usable.
- **CUDA OOM.** Episodes are short (~500 frames @ 640×480) so a 5090 is ample;
  if it ever OOMs, process fewer episodes per run or lower the SAM image size in
  the predictor config.
- **`no kernel image` on the 5090.** torch too old for Blackwell — install a
  newer cu12x/cu128 (or nightly) torch wheel.
- **Wrong codec / can't decode.** The raw videos are h264; ensure PyAV/torch
  decode backends are installed (they come with `.[dataset]`).

## Validation checklist before training

- [ ] TEST episode inspected: strawberry stays highlighted across the pick+place
- [ ] Fallback % is low on most episodes (check the per-episode log)
- [ ] `Atabaku/so101-strawberry-raw-grounded` exists on the Hub with 50 episodes
- [ ] `meta/episodes` has `grounding_segmentation_backend=sam3.1`
- [ ] π0.5 training points at the **grounded** repo
