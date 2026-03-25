# BoxMOT Custom Tracking Scripts

This folder contains custom tracking scripts built on top of `boxmot` fork.
They are intended to reproduce tracking results on datasets organized with separate `Annotation/` and `Data/` folders.

## Included scripts

### Single-case scripts
runs tracking for a single detections JSON file and its corresponding frames folder.
- `my_boosttrack.py`
- `my_botsort.py`
- `my_bytetrack.py`
- `my_deepocsort.py`
- `my_ocsort.py`
- `my_strongsort.py`

### Batch scripts
scans a dataset root, finds valid scenario/camera combinations, runs the specific tracker for each one, and stores all outputs automatically.
- `batch_my_boosttrack.py`
- `batch_my_botsort.py`
- `batch_my_bytetrack.py`
- `batch_my_deepocsort.py`
- `batch_my_ocsort.py`
- `batch_my_strongsort.py`

## Repository purpose

The scripts in this folder are intended to:

- parse the detections JSON using the same policy in all trackers at the repository fork,
- process frames in the exact order defined by the JSON file,
- export tracking results in MOT format,
- generate an annotated MP4 video,
- save annotated PNG frames,
- support both single-case execution and batch processing across multiple scenarios/cameras,
- keep a homogeneous structure across the main BoxMOT trackers.

## Trackers covered

### ReID-based trackers

These scripts require `--model_path`:

- BoostTrack
- BotSort
- DeepOCSort
- StrongSort

### Motion-only trackers

These scripts do **not** require `--model_path`:

- ByteTrack
- OCSort

## Requirements

This project uses the Conda environment defined in `environment.yml`.

Pinned base versions:

- Python 3.10
- numpy 1.26

The `environment.yml` file intentionally defines only the base environment. The BoxMOT package and its project dependencies are installed afterwards from the local cloned fork with `pip install -e .`.

After installing the local fork from the repository root, move to the custom scripts folder:

```bash
cd Boxmot/LINCOLN_PROJECT
```
Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate Boxmot
```

Then install the local fork in editable mode from the repository root:

```bash
python -m pip install --upgrade pip
pip install -e .
```

This is important: the custom scripts are meant to run against the **local cloned fork**, not against a separate PyPI installation.

For the complete Ubuntu setup, editable installation, reproducibility notes, and troubleshooting, see `UBUNTU_SETUP.md`.

## Quick start
Check the available CLI arguments:

```bash
python my_boosttrack.py --help
python my_botsort.py --help
python my_bytetrack.py --help
python my_deepocsort.py --help
python my_ocsort.py --help
python my_strongsort.py --help
```

And for batch execution:

```bash
python batch_my_boosttrack.py --help
python batch_my_botsort.py --help
python batch_my_bytetrack.py --help
python batch_my_deepocsort.py --help
python batch_my_ocsort.py --help
python batch_my_strongsort.py --help
```

## Expected dataset structure

```text
<root>/
  Annotation/
    <scenario>_json_files/
      fisheye_images_12/cam_fish_left_ann.json
      fisheye_images_13/cam_fish_front_ann.json
      fisheye_images_14/cam_fish_right_ann.json
      output_images/cam_zed_rgb_ann.json
  Data/
    <scenario>_label/
      fisheye_images_12/ ...frames...
      fisheye_images_13/ ...frames...
      fisheye_images_14/ ...frames...
      output_images/     ...frames...
```

## Main parameters

### Single-case scripts

Common arguments:

- `--image_dir`: path to the frames folder.
- `--json_path`: path to the detections JSON file.
- `--output_dir`: path to the output folder.

Additional argument for ReID-based trackers:

- `--model_path`: path to the ReID weights file.

### Batch scripts

Common argument:

- `--root`: dataset root containing `Annotation/` and `Data/`.

Additional argument for ReID-based trackers:

- `--model_path`: path to the ReID weights file.

## Outputs

### Single-case execution

Each single-case script generates:

- `results.txt`
- `tracking.mp4`
- `result_frames/` (annotated PNG frames)

### Batch execution

Each batch script generates one output folder per scenario/camera under a tracker-specific directory:

```text
<root>/Results/BoostTrack/<scenario_label>/<cam>/
<root>/Results/BotSort/<scenario_label>/<cam>/
<root>/Results/ByteTrack/<scenario_label>/<cam>/
<root>/Results/DeepOCSort/<scenario_label>/<cam>/
<root>/Results/OCSort/<scenario_label>/<cam>/
<root>/Results/StrongSort/<scenario_label>/<cam>/
```

Typical outputs are:

- `results.txt`
- `tracking.mp4`
- `result_frames/` (annotated PNG frames)

## Notes

- All custom scripts follow the same general structure.
- The JSON parsing policy is homogeneous across trackers.
- The frame index is driven by the JSON order, not by filename sorting.
- Missing or unreadable frames are replaced with a black canvas so the temporal index remains consistent.
- The MOT file is written using the bounding box returned by the tracker.
- Batch scripts do not change the tracking logic of the single-case scripts; they only automate repeated executions.
- ReID-based trackers require a valid ReID checkpoint, for example `osnet_x0_25_msmt17.pt`.
- Make sure you have enough disk space if frame export is enabled for long sequences.
