#!/usr/bin/env python3
"""
SCRIPT KIỂM TRA METADATA — ĐỌC MẪU NHANH
Chỉ đọc một số phần đầu để nắm cấu trúc. Không đọc toàn bộ.

Chạy: python3 inspect_metadata.py
"""

import json
import os
import random
from pathlib import Path

JSON_DIR = Path("data/json")
AIC_META = Path("/Users/dangphuoctienthu/Desktop/AIC-2026/data/keyframes_L26_V200_V299_metadata.json")
AIC_KF_ROOT = Path("/Users/dangphuoctienthu/Desktop/AIC-2026/keyframesL26c")
DEMO3_KF_ROOT = Path("data/processed/keyframes")


def hr(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print('='*60)


def check_json_structure():
    hr("1. CẤU TRÚC FILE JSON")
    rich_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith("_rich_metadata.json"))
    stage_a_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith("_stage_a.json"))
    structure_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith("_structure.json"))

    print(f"  rich_metadata  : {len(rich_files):>5} files")
    print(f"  stage_a        : {len(stage_a_files):>5} files")
    print(f"  structure      : {len(structure_files):>5} files")

    # Video prefix breakdown
    prefixes = {}
    for f in rich_files:
        prefix = f.split("_")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print("\n  Video prefix breakdown:")
    for p, cnt in sorted(prefixes.items()):
        print(f"    {p}: {cnt} videos")
    return rich_files


def check_rich_metadata_schema(rich_files):
    hr("2. SCHEMA rich_metadata (mẫu 10 files, 3 shots/file)")
    sample = random.sample(rich_files, min(10, len(rich_files)))

    top_keys_counter = {}
    vlm_keys_counter = {}
    vlm_none_count = 0
    total_shots = 0

    for fname in sample:
        with open(JSON_DIR / fname) as f:
            d = json.load(f)
        for shot in d.get("shots", [])[:3]:
            total_shots += 1
            ed = shot.get("extracted_data", {})
            for k in ed:
                top_keys_counter[k] = top_keys_counter.get(k, 0) + 1

            vlm = ed.get("vlm_caption")
            if vlm is None:
                vlm_none_count += 1
            elif isinstance(vlm, dict):
                for k in vlm:
                    vlm_keys_counter[k] = vlm_keys_counter.get(k, 0) + 1

    print(f"  Tổng shots kiểm tra: {total_shots}")
    print(f"\n  extracted_data keys:")
    for k, cnt in sorted(top_keys_counter.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total_shots * 20)
        print(f"    {k:<25} {cnt:>3}/{total_shots}  {bar}")

    print(f"\n  vlm_caption keys:")
    if vlm_none_count:
        print(f"    [NONE/MISSING]           {vlm_none_count:>3}/{total_shots}")
    for k, cnt in sorted(vlm_keys_counter.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total_shots * 20)
        print(f"    {k:<25} {cnt:>3}/{total_shots}  {bar}")


def check_keyframe_accessibility(rich_files):
    hr("3. KEYFRAME IMAGES — KHẢ NĂNG TRUY CẬP")
    sample = random.sample(rich_files, min(30, len(rich_files)))

    found_demo3 = 0
    found_aic = 0
    found_nowhere = 0
    total = 0

    for fname in sample:
        video_name = fname.replace("_rich_metadata.json", "")
        with open(JSON_DIR / fname) as f:
            d = json.load(f)
        fps = d.get("fps", 25.0)

        for shot in d.get("shots", [])[:2]:
            for kf in shot.get("keyframes", [])[:1]:  # Chỉ check frame đầu tiên
                total += 1
                if isinstance(kf, dict):
                    frame_idx = kf.get("frame_idx", 0)
                    raw = kf.get("image_path", "")
                else:
                    frame_idx = 0
                    raw = str(kf)

                img_name = os.path.basename(raw)

                # Path 1: Demo3 keyframes
                p1 = DEMO3_KF_ROOT / video_name / img_name
                if p1.exists():
                    found_demo3 += 1
                    continue

                # Path 2: AIC keyframesL26c (map frame_idx → 1fps)
                ts = frame_idx / fps if fps else 0
                aic_fname = f"{max(1, round(ts)):03d}.jpg"
                p2 = AIC_KF_ROOT / video_name / aic_fname
                if p2.exists():
                    found_aic += 1
                    continue

                found_nowhere += 1

    print(f"  Tổng frames kiểm tra : {total}")
    print(f"  Tìm thấy (Demo3 KF)  : {found_demo3:>3} ({100*found_demo3//max(total,1)}%)")
    print(f"  Tìm thấy (AIC L26c)  : {found_aic:>3} ({100*found_aic//max(total,1)}%)")
    print(f"  Không tìm thấy       : {found_nowhere:>3} ({100*found_nowhere//max(total,1)}%)")

    if found_nowhere > 0:
        print("\n  ⚠️  Một số video KHÔNG có keyframe images trên máy này.")
        print("     Stage C sẽ bỏ qua những shots này (quality gate).")


def check_aic_metadata():
    hr("4. AIC FRAME-LEVEL METADATA")
    if not AIC_META.exists():
        print(f"  ❌ Không tìm thấy: {AIC_META}")
        return

    with open(AIC_META) as f:
        data = json.load(f)

    print(f"  File         : {AIC_META.name}")
    print(f"  Tổng frames  : {len(data)}")

    videos = set(x["video_id"] for x in data)
    print(f"  Videos       : {len(videos)} (ví dụ: {sorted(videos)[:3]})")

    # Schema
    sample = data[0]
    print(f"\n  Keys: {list(sample.keys())}")
    print(f"\n  Sample item:")
    for k, v in sample.items():
        val_str = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
        print(f"    {k:<25} : {val_str}")

    # Check AIC keyframe dirs
    aic_dirs = set(os.listdir(AIC_KF_ROOT)) if AIC_KF_ROOT.exists() else set()
    overlap = videos & aic_dirs
    print(f"\n  keyframesL26c dirs   : {len(aic_dirs)} thư mục")
    print(f"  Khớp với metadata    : {len(overlap)} videos")


def check_qdrant():
    hr("5. QDRANT STATUS")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"  Collections: {col_names}")
        for name in col_names:
            info = client.get_collection(name)
            print(f"\n  [{name}]")
            print(f"    Vectors count : {info.vectors_count}")
            print(f"    Points count  : {info.points_count}")
            if info.config.params.vectors:
                for vname, vcfg in info.config.params.vectors.items():
                    if hasattr(vcfg, 'size'):
                        print(f"    Vector '{vname}': dim={vcfg.size}")
    except Exception as e:
        print(f"  ❌ Không kết nối được Qdrant: {e}")
        print("     (Hãy chắc chắn Docker Qdrant đang chạy)")


def check_sample_shot_full():
    hr("6. SHOT ĐẦY ĐỦ — MẪU 1 SHOT")
    rich_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith("_rich_metadata.json")
                        and f.startswith("L26_V2"))
    if not rich_files:
        print("  Không có L26_V2xx file")
        return

    fname = rich_files[0]
    with open(JSON_DIR / fname) as f:
        d = json.load(f)

    shot = d["shots"][0]
    video_name = d["video_id"]
    print(f"  Video  : {video_name}")
    print(f"  Shot   : {shot['shot_id']}")
    print(f"  FPS    : {d.get('fps', '?')}")
    print(f"  TS     : {shot['start_ts']:.2f}s → {shot['end_ts']:.2f}s")
    print(f"  Frames : {len(shot.get('keyframes', []))}")

    ed = shot.get("extracted_data", {})
    print(f"\n  transcript   : {repr(ed.get('transcript',''))[:80]}")
    print(f"  ocr_text     : {repr(ed.get('ocr_text',''))[:80]}")

    vlm = ed.get("vlm_caption", {})
    if isinstance(vlm, dict):
        for k, v in vlm.items():
            val = " ".join(v) if isinstance(v, list) else str(v)
            print(f"  vlm.{k:<22}: {val[:80]}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  METADATA INSPECTOR — DEMO3")
    print("="*60)

    rich_files = check_json_structure()
    check_rich_metadata_schema(rich_files)
    check_keyframe_accessibility(rich_files)
    check_aic_metadata()
    check_qdrant()
    check_sample_shot_full()

    hr("XONG")
    print("  Script chỉ đọc MẪU nhỏ — không đọc toàn bộ dữ liệu.\n")
