# Quick guide for Ubuntu (clean fork)

Steps to reproduce the results using the custom BoxMOT scripts located in `LINCOLN_PROJECT/`.

**Prerequisite:** Conda (Miniconda or Anaconda) must be installed before running the commands below.

## 1. Clone the fork

```bash
git clone https://github.com/<user>/<fork>.git
cd <fork>/<custom_scripts_folder>

### Example (in this case)
git clone https://github.com/MichelSanzPerez/boxmot.git Boxmot
cd Boxmot/LINCOLN_PROJECT
```

## 2. Prepare system dependencies

OpenCV with video support needs FFmpeg and GL; install:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## 3. Create the Conda environment

Use the included `environment.yml`:

```bash
conda env create -f environment.yml
conda activate Boxmot
```

The `environment.yml` file intentionally creates only the base Conda environment. The BoxMOT package and its project dependencies are installed afterwards from the local cloned fork.

## 4. Install the local fork in editable mode

From the repository root, install BoxMOT from the cloned source tree:

```bash
python -m pip install --upgrade pip
pip install -e .
```
> If you need a specific CUDA build of PyTorch, install that build in the environment before running `pip install -e .`.

## 5. Expected data structure

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

## 6. Run a single case

### ReID-based trackers

#### BoostTrack

```bash
python my_boosttrack.py \
  --image_dir /path/to/frames/fisheye_images_12 \
  --json_path /path/to/json/cam_fish_left_ann.json \
  --output_dir /path/to/results/BoostTrack \
  --model_path /path/to/osnet_x0_25_msmt17.pt
```

#### BotSort

```bash
python my_botsort.py \
  --image_dir /path/to/frames/fisheye_images_12 \
  --json_path /path/to/json/cam_fish_left_ann.json \
  --output_dir /path/to/results/BotSort \
  --model_path /path/to/osnet_x0_25_msmt17.pt
```

#### DeepOCSort

```bash
python my_deepocsort.py \
  --image_dir /path/to/frames/fisheye_images_12 \
  --json_path /path/to/json/cam_fish_left_ann.json \
  --output_dir /path/to/results/DeepOCSort \
  --model_path /path/to/osnet_x0_25_msmt17.pt
```

#### StrongSort

```bash
python my_strongsort.py \
  --image_dir /path/to/frames/fisheye_images_12 \
  --json_path /path/to/json/cam_fish_left_ann.json \
  --output_dir /path/to/results/StrongSort \
  --model_path /path/to/osnet_x0_25_msmt17.pt
```

### Motion-only trackers

#### ByteTrack

```bash
python my_bytetrack.py \
  --image_dir /path/to/frames/fisheye_images_12 \
  --json_path /path/to/json/cam_fish_left_ann.json \
  --output_dir /path/to/results/ByteTrack
```

#### OCSort

```bash
python my_ocsort.py \
  --image_dir /path/to/frames/fisheye_images_12 \
  --json_path /path/to/json/cam_fish_left_ann.json \
  --output_dir /path/to/results/OCSort
```

### Example

#### ReID-based tracker example

```bash
python my_boosttrack.py \
  --image_dir "/media/michel/DATASET/Data/footpath2_3walk_st_11_20_2024_label/fisheye_images_12" \
  --json_path "/media/michel/DATASET/Annotation/footpath2_3walk_st_11_20_2024_json_files/fisheye_images_12/cam_fish_left_ann.json" \
  --output_dir "/media/michel/DATASET/Results/BoostTrack/footpath2_3walk_st_11_20_2024_label/fisheye_images_12" \
  --model_path "/home/michel/Boxmot/LINCOLN_PROJECT/osnet_x0_25_msmt17.pt"
```

#### Motion-only tracker example

```bash
python my_bytetrack.py \
  --image_dir "/media/michel/DATASET/Data/footpath2_3walk_st_11_20_2024_label/fisheye_images_12" \
  --json_path "/media/michel/DATASET/Annotation/footpath2_3walk_st_11_20_2024_json_files/fisheye_images_12/cam_fish_left_ann.json" \
  --output_dir "/media/michel/DATASET/Results/ByteTrack/footpath2_3walk_st_11_20_2024_label/fisheye_images_12"
```

## 7. Run all scenarios/cameras in batch

### ReID-based trackers

```bash
python batch_my_boosttrack.py \
  --root /path/to/root \
  --model_path /path/to/osnet_x0_25_msmt17.pt

python batch_my_botsort.py \
  --root /path/to/root \
  --model_path /path/to/osnet_x0_25_msmt17.pt

python batch_my_deepocsort.py \
  --root /path/to/root \
  --model_path /path/to/osnet_x0_25_msmt17.pt

python batch_my_strongsort.py \
  --root /path/to/root \
  --model_path /path/to/osnet_x0_25_msmt17.pt
```

### Motion-only trackers

```bash
python batch_my_bytetrack.py \
  --root /path/to/root

python batch_my_ocsort.py \
  --root /path/to/root
```

### Examples

#### ReID-based tracker batch example

```bash
python batch_my_boosttrack.py \
  --root "/media/michel/DATASET" \
  --model_path "/home/michel/Boxmot/LINCOLN_PROJECT/osnet_x0_25_msmt17.pt"
```

#### Motion-only tracker batch example

```bash
python batch_my_bytetrack.py \
  --root "/media/michel/DATASET"
```

Generates tracker-specific output folders such as:

```text
<root>/Results/BoostTrack/<scenario_label>/<cam>/
  results.txt      # MOT format file
  tracking.mp4     # video with bounding boxes
  result_frames/   # PNG with bounding boxes

<root>/Results/BotSort/<scenario_label>/<cam>/
  results.txt
  tracking.mp4
  result_frames/

<root>/Results/ByteTrack/<scenario_label>/<cam>/
  results.txt
  tracking.mp4
  result_frames/

<root>/Results/DeepOCSort/<scenario_label>/<cam>/
  results.txt
  tracking.mp4
  result_frames/

<root>/Results/OCSort/<scenario_label>/<cam>/
  results.txt
  tracking.mp4
  result_frames/

<root>/Results/StrongSort/<scenario_label>/<cam>/
  results.txt
  tracking.mp4
  result_frames/
```

## 8. Reproducibility notes

- The Conda environment in `environment.yml` fixes the base Python and numpy versions.
- The BoxMOT package itself is installed from the local cloned fork with `pip install -e .`.
- ReID-based trackers require a valid ReID weights file, for example `osnet_x0_25_msmt17.pt`.
- The custom scripts share the same JSON parsing policy, frame indexing policy, and MOT export policy.
- All scripts write the MOT file from the bounding box returned by the tracker.
- Make sure you have enough disk space: `result_frames/` can be large for long sequences.

## 9. Quick troubleshooting

- `File not found`
  In batch mode: verify that the scenario name in `Annotation/` ends with `_json_files` and in `Data/` with `_label`; the mapping is automatic.

- `Model file not found`
  ReID-based trackers require `--model_path`. Verify that the checkpoint file exists and that the path is correct.

- `conda: command not found`
  This usually means that Conda is either not installed or not initialized in the current shell.

  **Possible solutions:**
  - Check whether Conda is installed:
    ```bash
    conda --version
    ```
  - If Conda is not available, install **Miniconda** or **Anaconda** first.
  - If Conda is installed but still not recognized, initialize your shell:

    ```bash
    conda init bash
    ```

    Then close and reopen the terminal.

- `Errors caused by paths with spaces`
  If a path contains spaces and is not enclosed in quotes, the shell splits it into multiple arguments and the script may fail with errors such as `unrecognized arguments`.

  **Recommendation:** always wrap paths containing spaces in double quotes.
