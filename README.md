# 👣 Workout Footstep Counter

Counts footsteps in a workout video using **YOLO pose estimation**. Every time a foot touches the floor, one step is counted. The result is a **side-by-side video** (original | annotated) showing the total steps and the **steps per 10 seconds** (speed), plus CSV files.

Use it from a **web UI** (upload, progress bar, preview, download) or from the **command line**.

![layout](https://img.shields.io/badge/output-original%20%7C%20YOLO%20pose%20%2B%20step%20count-green)

---

## Features

- YOLO11 pose model tracks 17 body keypoints in every frame
- Counts a step each time an ankle comes back down to the floor (left and right counted separately)
- Works when the person moves closer to or farther from the camera (lift is measured in torso lengths)
- Side-by-side output video with:
  - skeleton overlay and a green **STEP** flash on each touchdown
  - total steps, steps in the current 10 s window and live steps/min
  - a bar chart of steps per 10 s at the bottom
  - the original audio
- CSV export of every step and of the per-10-second counts
- Web UI with drag-and-drop upload, a live progress bar, result preview and downloads

---

## Requirements

- macOS, Linux or Windows
- Python **3.10–3.12**
- **ffmpeg** (used to encode the output video)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: `winget install ffmpeg`

Uses the Apple GPU (MPS) on Apple Silicon Macs or CUDA on NVIDIA GPUs automatically, and falls back to CPU otherwise.

---

## Installation

```bash
cd workout-analytics
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The YOLO model (`yolo11m-pose.pt`, ~40 MB) downloads automatically on the first run.

---

## Usage

### Option 1: Web UI (recommended)

```bash
source .venv/bin/activate
python app.py
```

Your browser opens at **http://127.0.0.1:7860**. Then:

1. **Upload** a video (mp4 / mov / webm)
2. Click **Count steps** and watch the progress bar:
   - Detecting body pose (YOLO) → Counting footsteps → Rendering side-by-side video
3. See the results: the video, total steps, average and fastest speed, the steps-per-10 s chart, and a table of every step
4. **Download** the video and CSV files from the Downloads box

Each upload is saved in its own folder: `output/jobs/<date-time>_<video-name>/`.

Stop the server with **Ctrl + C**.

### Option 2: Command line

Put your video in the `video/` folder, then run:

```bash
source .venv/bin/activate
python step_counter.py "video/morning-excercise copy.mp4"
```

Use quotes when the file name contains spaces. Tip: type `python step_counter.py "video/` and press **Tab** to auto-complete the name.

Example output:

```
   total steps: 205  (left 103, right 102)
        0-10s:  28 steps  (168 steps/min)
       10-20s:  28 steps  (168 steps/min)
       20-30s:  39 steps  (234 steps/min)
       ...
done: output/morning-excercise_copy_steps_side_by_side.mp4
```

#### Command-line options

| Option | Default | Description |
|---|---|---|
| `--model` | `yolo11m-pose.pt` | YOLO pose model. `yolo11n-pose.pt` is faster, `yolo11x-pose.pt` is more accurate |
| `--up` | `0.10` | How high (in torso lengths) a foot must rise to count. Lower = more sensitive |
| `--down` | `0.04` | Lift below which the foot always counts as on the floor |
| `--down-frac` | `0.4` | Touchdown when the foot drops below this fraction of its peak lift |
| `--min-gap` | `0.2` | Minimum seconds between two steps of the same foot |
| `--panel-height` | `1280` | Height in pixels of each half of the output video |
| `--out-dir` | `output` | Folder for results |

The web UI has the same main settings under **Detection settings**.

---

## Output files

For a video named `my_workout.mp4`:

| File | Contents |
|---|---|
| `my_workout_steps_side_by_side.mp4` | Original and annotated video side by side, with step counts |
| `my_workout_steps.csv` | Every step: `step_no, time_sec, frame, foot` |
| `my_workout_steps_per_10s.csv` | Per window: `window, steps, steps_per_min` |
| `my_workout_keypoints.npz` | Cached pose data. Re-runs skip YOLO and take seconds instead of minutes |

> If you change the video but keep the same file name, delete the old `_keypoints.npz` first. Otherwise the cached pose data is reused.

---

## How it works

1. **Pose estimation.** YOLO11-pose runs on every frame and keeps the largest person in view.
2. **Foot-lift signal.** For each ankle:
   - The floor level is the ankle's lowest position over a rolling 2-second window.
   - Lift is the height above the floor divided by torso length, so the person's distance from the camera doesn't matter.
   - The signal is smoothed to remove jitter.
3. **Step detection.** A foot is *up* once it rises above the lift threshold. It *touches down* when it falls below 40% of that lift's peak. Each touchdown counts as **one step**.
4. **Speed.** Steps are grouped into 10-second windows: steps/min = steps × 6. The last window may be shorter and is scaled to its real length.
5. **Rendering.** The original and annotated frames are placed side by side with the counters and chart, then encoded to H.264 MP4 with ffmpeg.

---

## Tips & limitations

- **Camera.** Works best with a fixed camera that sees the whole body, feet included, front-on or from the side.
- **One person.** Only the largest person in the frame is tracked.
- **Jumps.** In jumping jacks or two-footed jumps both feet land together, so each landing counts as **2 steps** (one per foot).
- **Missed or extra steps.**
  - Small shuffles while changing exercise may not be counted. Lower **Lift threshold** (for example 0.07).
  - Wobbles counted as steps: raise **Lift threshold** or **Min seconds between steps**.
- **Speed.** Processing takes roughly 1.5× the video length on an Apple Silicon Mac with `yolo11m-pose`. Use `--model yolo11n-pose.pt` for faster runs.

---

## Project structure

```
workout-analytics/
├── app.py                         # Web UI (Gradio): upload, progress, preview, download
├── step_counter.py                # Pose → step detection → side-by-side video + CSV
├── requirements.txt
├── README.md
├── video/                         # Put input videos here (command-line use)
├── output/                        # Command-line results
│   └── jobs/                      # Web UI results, one folder per upload
└── workout_video_analysis.ipynb   # Google Colab notebook: Gemini video analysis (separate tool)
```

### Colab notebook (Gemini)

`workout_video_analysis.ipynb` is a separate Google Colab notebook. It sends a workout video to Gemini for a written analysis: exercises, reps, form tips. It needs a Gemini API key stored in Colab **Secrets** as `GOOGLE_API_KEY`. The step counter above does **not** need any API key.
