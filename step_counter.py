"""Footstep counter using YOLO pose.

A step is counted each time an ankle comes back down to the floor after being
lifted. Output: side-by-side video (original | annotated) with total steps and
steps per 10-second window, plus CSV files with the step events and per-window counts.

Usage:
    .venv/bin/python step_counter.py "video/morning-excercise copy.mp4"
    .venv/bin/python app.py      # web UI: upload, progress bar, preview, download
"""
import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import savgol_filter

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
WINDOW_SEC = 10
_models = {}


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(name):
    if name not in _models:
        from ultralytics import YOLO
        _models[name] = YOLO(name)
    return _models[name]


def window_info(i, count, duration):
    """Label and steps/min for 10 s window i; the last window may be shorter."""
    start, end = i * WINDOW_SEC, min((i + 1) * WINDOW_SEC, duration)
    return f"{start:g}-{round(end, 1):g}s", round(count * 60 / max(end - start, 1e-6))


def extract_keypoints(video_path, model_name, cache_path, progress=None):
    """Run YOLO pose on every frame; keep the largest person. Cached to .npz.

    progress(done, total) is called as frames are processed.
    """
    if cache_path.exists():
        data = np.load(cache_path)
        return data["kps"], float(data["fps"])

    model = _load_model(model_name)
    device = _device()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kps = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, imgsz=960, device=device, verbose=False)[0]
        if res.keypoints is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            k = res.keypoints.data.cpu().numpy()[areas.argmax()]  # (17, 3): x, y, conf
        else:
            k = np.zeros((17, 3), np.float32)
        kps.append(k)
        i += 1
        if progress and i % 5 == 0:
            progress(i, total)
    cap.release()
    if not kps:
        raise ValueError(f"No frames could be read from: {video_path}")
    kps = np.stack(kps)
    np.savez(cache_path, kps=kps, fps=fps)
    return kps, fps


def fill_low_conf(series, conf, thr=0.3):
    """Linearly interpolate values where keypoint confidence is low."""
    good = conf >= thr
    if good.sum() < 2:
        return series
    idx = np.arange(len(series))
    return np.interp(idx, idx[good], series[good])


def detect_steps(kps, fps, up_thr, down_thr, min_gap_sec, down_frac=0.4, rise=0.08):
    """Return per-foot step frames and per-frame lift signals."""
    torso = np.hypot(
        (kps[:, L_SHOULDER, 0] + kps[:, R_SHOULDER, 0] - kps[:, L_HIP, 0] - kps[:, R_HIP, 0]) / 2,
        (kps[:, L_SHOULDER, 1] + kps[:, R_SHOULDER, 1] - kps[:, L_HIP, 1] - kps[:, R_HIP, 1]) / 2,
    )
    torso = median_filter(np.maximum(torso, 1), size=int(fps))  # body scale, robust to noise

    ground_win = int(2 * fps)
    min_gap = int(min_gap_sec * fps)
    steps, lifts = {}, {}
    for name, j in (("left", L_ANKLE), ("right", R_ANKLE)):
        y = fill_low_conf(kps[:, j, 1], kps[:, j, 2])
        y = savgol_filter(y, 7, 2)
        ground = maximum_filter1d(y, ground_win)  # lowest ankle position nearby = floor
        ground = median_filter(ground, size=int(fps))
        lift = (ground - y) / torso  # 0 = on floor, >0 = foot raised (in torso lengths)
        lifts[name] = lift

        # Hysteresis: foot is "up" once it rises up_thr above floor and above the last valley;
        # it touches down when it falls below down_frac of that lift's peak (or below down_thr).
        events, up, last = [], False, -min_gap
        peak, valley = 0.0, lift[0]
        for f, v in enumerate(lift):
            if not up:
                valley = min(valley, v)
                if v > up_thr and v - valley > rise:
                    up, peak = True, v
            else:
                peak = max(peak, v)
                if v < max(down_thr, down_frac * peak):
                    up, valley = False, v
                    if f - last >= min_gap:
                        events.append(f)
                        last = f
        steps[name] = events
    return steps, lifts


def draw_pose(img, k, scale, step_flash):
    for a, b in SKELETON:
        if k[a, 2] > 0.3 and k[b, 2] > 0.3:
            pa = (int(k[a, 0] * scale), int(k[a, 1] * scale))
            pb = (int(k[b, 0] * scale), int(k[b, 1] * scale))
            cv2.line(img, pa, pb, (255, 200, 0), 3, cv2.LINE_AA)
    for j in range(17):
        if k[j, 2] > 0.3:
            cv2.circle(img, (int(k[j, 0] * scale), int(k[j, 1] * scale)), 5, (0, 255, 255), -1, cv2.LINE_AA)
    for name, j in (("left", L_ANKLE), ("right", R_ANKLE)):
        if k[j, 2] > 0.3:
            c = (int(k[j, 0] * scale), int(k[j, 1] * scale))
            if step_flash[name]:
                cv2.circle(img, c, 26, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(img, "STEP", (c[0] - 32, c[1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 10, (0, 0, 255), -1, cv2.LINE_AA)


def text(img, s, org, size, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def render(video_path, out_path, kps, fps, step_frames, window_counts, panel_h, progress=None):
    cap = cv2.VideoCapture(str(video_path))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = panel_h / H
    pw = int(round(W * scale / 2) * 2)
    out_w, out_h = pw * 2, panel_h + 220
    out_w += out_w % 2

    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
         "-r", str(fps), "-i", "-", "-i", str(video_path), "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE)

    all_steps = sorted((f, n) for n, fs in step_frames.items() for f in fs)
    flash_len = int(0.25 * fps)
    max_win = max(window_counts) if window_counts else 1
    duration = len(kps) / fps
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= len(kps):
            break
        small = cv2.resize(frame, (pw, panel_h), interpolation=cv2.INTER_AREA)
        left, right = small.copy(), small.copy()
        flash = {n: any(0 <= f - s < flash_len for s in step_frames[n]) for n in step_frames}
        draw_pose(right, kps[f], scale, flash)

        total = sum(1 for s, _ in all_steps if s <= f)
        win = int(f / fps // WINDOW_SEC)
        in_win = sum(1 for s, _ in all_steps if win * WINDOW_SEC * fps <= s <= f)
        elapsed = f / fps - win * WINDOW_SEC
        live_rate = f"{round(in_win * 60 / elapsed)}/min" if elapsed >= 2 else "--/min"
        text(left, "ORIGINAL", (20, 50), 1.2)
        text(right, "YOLO POSE - STEP DETECTION", (20, 50), 1.0)
        text(right, f"Total steps: {total}", (20, 105), 1.3, (0, 255, 0), 3)
        text(right, f"This 10s ({window_info(win, 0, duration)[0]}): {in_win}  ~{live_rate}", (20, 155), 1.0, (0, 255, 255))
        text(right, f"Time: {f / fps:5.1f}s", (20, 200), 0.9)

        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        canvas[:panel_h, :pw] = left
        canvas[:panel_h, pw:pw * 2] = right
        cv2.line(canvas, (pw, 0), (pw, panel_h), (80, 80, 80), 2)

        # bottom: steps per 10 s bar chart (speed), current window highlighted
        text(canvas, "Steps per 10 s (speed)", (20, panel_h + 35), 0.8)
        n = len(window_counts)
        bw = (out_w - 40) // max(n, 1)
        base = out_h - 40
        for i, c in enumerate(window_counts):
            shown = c if i < win else (in_win if i == win else 0)
            bh = int(110 * shown / max_win)
            x0 = 20 + i * bw
            color = (0, 200, 255) if i == win else (0, 180, 0)
            cv2.rectangle(canvas, (x0 + 6, base - bh), (x0 + bw - 6, base), color, -1)
            label, rate = window_info(i, c, duration)
            text(canvas, label, (x0 + 8, base + 28), 0.55)
            if i < win:
                text(canvas, f"{c} ({rate}/min)", (x0 + 8, base - bh - 10), 0.55, (255, 255, 255), 1)
            elif i == win:
                text(canvas, f"{shown} so far", (x0 + 8, base - bh - 10), 0.55, (255, 255, 255), 1)

        ff.stdin.write(canvas.tobytes())
        f += 1
        if progress and f % 5 == 0:
            progress(f, len(kps))
    cap.release()
    ff.stdin.close()
    ff.wait()


def process_video(video, out_dir, model="yolo11m-pose.pt", up=0.10, down=0.04, down_frac=0.4,
                  min_gap=0.2, panel_height=1280, progress=None):
    """Full pipeline. progress(stage, done, total) with stage in {"pose", "steps", "render"}.

    Returns a dict with paths of the output video/CSVs and the step statistics.
    """
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem.replace(" ", "_")
    stage = (lambda name: (lambda d, t: progress(name, d, t))) if progress else (lambda name: None)

    kps, fps = extract_keypoints(video, model, out_dir / f"{stem}_keypoints.npz", stage("pose"))

    if progress:
        progress("steps", 0, 1)
    steps, _ = detect_steps(kps, fps, up, down, min_gap, down_frac)
    all_steps = sorted((f, n) for n, fs in steps.items() for f in fs)
    duration = len(kps) / fps
    n_windows = int(np.ceil(duration / WINDOW_SEC))
    window_counts = [sum(1 for f, _ in all_steps if i * WINDOW_SEC <= f / fps < (i + 1) * WINDOW_SEC)
                     for i in range(n_windows)]
    windows = [(*window_info(i, c, duration), c) for i, c in enumerate(window_counts)]  # (label, rate, count)

    steps_csv = out_dir / f"{stem}_steps.csv"
    with open(steps_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step_no", "time_sec", "frame", "foot"])
        for i, (f, n) in enumerate(all_steps, 1):
            w.writerow([i, round(f / fps, 2), f, n])
    windows_csv = out_dir / f"{stem}_steps_per_10s.csv"
    with open(windows_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "steps", "steps_per_min"])
        for label, rate, c in windows:
            w.writerow([label, c, rate])

    out_video = out_dir / f"{stem}_steps_side_by_side.mp4"
    render(video, out_video, kps, fps, steps, window_counts, panel_height, stage("render"))
    return {
        "video": out_video, "steps_csv": steps_csv, "windows_csv": windows_csv,
        "total": len(all_steps), "left": len(steps["left"]), "right": len(steps["right"]),
        "duration": duration, "fps": fps, "windows": windows,
        "steps": [(round(f / fps, 2), n) for f, n in all_steps],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--up", type=float, default=0.10, help="lift (torso lengths) that counts as foot raised")
    ap.add_argument("--down", type=float, default=0.04, help="lift below which the foot is on the floor")
    ap.add_argument("--down-frac", type=float, default=0.4, help="touchdown when lift falls below this fraction of its peak")
    ap.add_argument("--min-gap", type=float, default=0.2, help="min seconds between steps of the same foot")
    ap.add_argument("--panel-height", type=int, default=1280)
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        found = "\n  ".join(str(p) for p in sorted(Path("video").glob("*")) if p.is_file()) or "(none)"
        raise SystemExit(f"Video not found: {video}\nVideos in ./video:\n  {found}")
    last = {}

    def report(stage, done, total):
        pct = int(100 * done / max(total, 1))
        if last.get(stage) != pct // 10:
            last[stage] = pct // 10
            print(f"  {stage}: {pct}%", flush=True)

    r = process_video(video, args.out_dir, args.model, args.up, args.down, args.down_frac,
                      args.min_gap, args.panel_height, report)
    print(f"   total steps: {r['total']}  (left {r['left']}, right {r['right']})")
    for label, rate, c in r["windows"]:
        print(f"   {label:>10}: {c:3d} steps  ({rate} steps/min)")
    print("done:", r["video"])


if __name__ == "__main__":
    main()
