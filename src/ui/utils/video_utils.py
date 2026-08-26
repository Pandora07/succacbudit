import subprocess
from pathlib import Path
import streamlit as st

RAW_DIR = Path("data/raw")
TEMP_DIR = Path("data/temp_clips")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def find_video_path(vid_id: str) -> Path:
    mp4_files = list(RAW_DIR.rglob(f"{vid_id}.mp4"))
    if mp4_files: return mp4_files[0]
    avi_files = list(RAW_DIR.rglob(f"{vid_id}.avi"))
    if avi_files: return avi_files[0]
    return None

def extract_short_clip(vid_id: str, start_ts: float, end_ts: float, buffer_sec: float = 2.0):
    raw_path = find_video_path(vid_id)
    if not raw_path or not raw_path.exists(): return None

    clip_start = max(0, start_ts - buffer_sec)
    duration = (end_ts - start_ts) + (2 * buffer_sec)
    out_path = TEMP_DIR / f"clip_{vid_id}_{clip_start:.1f}_{duration:.1f}.mp4"
    if out_path.exists(): return str(out_path)

    cmd = [
        "ffmpeg", "-y", "-ss", str(clip_start), "-i", str(raw_path),
        "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out_path)
    except Exception as e:
        st.error(f"Lỗi FFmpeg: {e}")
        return None

def extract_single_frame(vid_id: str, frame_idx: int, fps: float) -> str:
    raw_path = find_video_path(vid_id)
    if not raw_path: return ""
    
    out_path = TEMP_DIR / f"frame_{vid_id}_{frame_idx}.jpg"
    if out_path.exists(): return str(out_path)
        
    timestamp = frame_idx / fps
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(raw_path), 
        "-frames:v", "1", "-q:v", "2", str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out_path)
    except Exception:
        return ""

def maybe_cleanup_temp_clips(max_files: int = 200):
    clip_files = sorted(TEMP_DIR.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
    frame_files = sorted(TEMP_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
    all_files = clip_files + frame_files
    if len(all_files) > max_files:
        for old_file in all_files[:len(all_files) - max_files]:
            try: old_file.unlink()
            except Exception: pass