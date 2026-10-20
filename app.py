"""Web UI for the footstep counter: upload a video, watch progress, preview and download results.

Usage:
    .venv/bin/python app.py      # then open http://127.0.0.1:7860
"""
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from step_counter import process_video

JOBS_DIR = Path(__file__).parent / "output" / "jobs"
# share of the progress bar for each pipeline stage: (start, width)
STAGES = {"pose": (0.0, 0.75, "Detecting body pose (YOLO)"),
          "steps": (0.75, 0.02, "Counting footsteps"),
          "render": (0.77, 0.23, "Rendering side-by-side video")}


def analyze(video_path, up, down_frac, min_gap, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    video = Path(video_path)
    job_dir = JOBS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{video.stem.replace(' ', '_')}"

    def report(stage, done, total):
        start, width, desc = STAGES[stage]
        progress(start + width * done / max(total, 1), desc=f"{desc} · {done}/{total}" if total > 1 else desc)

    progress(0, desc="Loading YOLO model")
    try:
        r = process_video(video, job_dir, up=up, down_frac=down_frac, min_gap=min_gap, progress=report)
    except ValueError as e:
        raise gr.Error(str(e))

    best = max(r["windows"], key=lambda w: w[2]) if r["windows"] else ("-", 0, 0)
    summary = (
        f"### 👣 Total steps: **{r['total']}**\n"
        f"Left foot: **{r['left']}** · Right foot: **{r['right']}** · "
        f"Video length: **{r['duration']:.1f} s** · "
        f"Average speed: **{round(r['total'] * 60 / max(r['duration'], 1e-6))} steps/min** · "
        f"Fastest 10 s: **{best[0]}** ({best[2]} steps)"
    )
    windows = pd.DataFrame(r["windows"], columns=["Window", "Steps/min", "Steps"])[["Window", "Steps", "Steps/min"]]
    steps = pd.DataFrame(r["steps"], columns=["Time (s)", "Foot"])
    steps.insert(0, "Step #", range(1, len(steps) + 1))
    files = [str(r["video"]), str(r["steps_csv"]), str(r["windows_csv"])]
    return str(r["video"]), summary, windows, windows, steps, files


with gr.Blocks(title="Footstep Counter") as demo:
    gr.Markdown("# 👣 Footstep Counter\nUpload a workout video. YOLO pose tracks the ankles and counts a step "
                "every time a foot touches the floor, then shows the original and annotated video side by side.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="1. Upload video", sources=["upload"])
            with gr.Accordion("Detection settings (optional)", open=False):
                up = gr.Slider(0.04, 0.3, value=0.10, step=0.01, label="Lift threshold",
                               info="How high (in torso lengths) a foot must rise to count. Lower = more sensitive.")
                down_frac = gr.Slider(0.2, 0.7, value=0.4, step=0.05, label="Touchdown level",
                                      info="Foot is 'on the floor' when it drops below this fraction of its lift.")
                min_gap = gr.Slider(0.1, 0.6, value=0.2, step=0.05, label="Min seconds between steps (same foot)")
            btn = gr.Button("2. Count steps", variant="primary", size="lg")
        with gr.Column(scale=2):
            out_video = gr.Video(label="Result: original | step detection", interactive=False)
            summary = gr.Markdown()
    with gr.Row():
        chart = gr.BarPlot(x="Window", y="Steps", title="Steps per 10 seconds (speed)", label="Speed",
                           sort=None)
        windows = gr.Dataframe(label="Steps per 10 s", interactive=False)
    with gr.Row():
        files = gr.File(label="⬇️ Downloads (video + CSV)", file_count="multiple", interactive=False)
        steps = gr.Dataframe(label="Every step", interactive=False, max_height=300)

    btn.click(analyze, [inp, up, down_frac, min_gap], [out_video, summary, chart, windows, steps, files])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, show_error=True, allowed_paths=[str(JOBS_DIR.parent)], theme=gr.themes.Soft())
