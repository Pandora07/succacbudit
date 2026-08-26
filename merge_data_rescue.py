import os
import json
from pathlib import Path

# Cấu hình đường dẫn
BASE_DIR = Path.cwd()
DEMO3_DIR = BASE_DIR / "data" / "json"
DEMO2_DIR = BASE_DIR / "demo2_json"


def pick_demo2_file(video_name: str):
    """
    Chọn đúng file Demo2 chứa dữ liệu caption thật (rich_metadata),
    tránh việc glob() trả về thứ tự ngẫu nhiên khiến có lúc trúng
    file _structure.json (không có extracted_data).
    """
    # 1. Ưu tiên tuyệt đối: file *_rich_metadata.json
    rich_matches = list(DEMO2_DIR.glob(f"{video_name}*_rich_metadata.json"))
    if rich_matches:
        return rich_matches, "rich_metadata"

    # 2. Khớp tên chính xác video_name.json
    exact_match = DEMO2_DIR / f"{video_name}.json"
    if exact_match.exists():
        return [exact_match], "exact"

    # 3. Fallback: glob tổng quát, nhưng loại bỏ *_structure.json
    #    vì file này chỉ chứa shot boundary, không có vlm_caption/yolo_objects
    all_matches = list(DEMO2_DIR.glob(f"{video_name}*.json"))
    filtered = [f for f in all_matches if "_structure" not in f.stem]
    if filtered:
        return filtered, "fallback_filtered"

    # 4. Không còn lựa chọn nào khác -> trả về tất cả (kể cả structure) để không bỏ sót
    return all_matches, "fallback_raw"


def merge_demo2_and_demo3():
    if not DEMO3_DIR.exists() or not DEMO2_DIR.exists():
        print("❌ Lỗi: Không tìm thấy thư mục json_data hoặc demo2_json!")
        return

    stage_a_files = list(DEMO3_DIR.glob("*_stage_a.json"))
    if not stage_a_files:
        print("❌ Không có file Stage A nào!")
        return

    print(f"🔄 Bắt đầu chiến dịch 'Cấy ghép & Ép kiểu' cho {len(stage_a_files)} video...\n")
    success_count = 0
    missing_count = 0
    zero_shot_videos = []       # merge được 0 shots dù có match id -> cần soi tiếp
    empty_content_videos = []   # match id nhưng vlm/yolo rỗng ở nguồn Demo2

    for stage_a_file in stage_a_files:
        video_name = stage_a_file.stem.replace("_stage_a", "")

        demo2_files, match_type = pick_demo2_file(video_name)

        if not demo2_files:
            print(f"⚠️ Bỏ qua {video_name}: Không tìm thấy bên Demo 2.")
            missing_count += 1
            continue

        if len(demo2_files) > 1:
            print(f"⚠️⚠️ {video_name}: khớp {len(demo2_files)} file demo2 ({match_type}) -> "
                  f"{[f.name for f in demo2_files]} (đang dùng file đầu tiên)")

        with open(stage_a_file, 'r', encoding='utf-8') as f:
            demo3_data = json.load(f)
        with open(demo2_files[0], 'r', encoding='utf-8') as f:
            demo2_data = json.load(f)

        demo2_shots_raw = demo2_data.get("shots", [])

        # Chuẩn hoá shot_id về string để tránh lệch kiểu int/str
        demo2_shots_dict = {
            str(shot.get("shot_id")).strip(): shot
            for shot in demo2_shots_raw
            if shot.get("shot_id") is not None
        }

        demo3_shots = demo3_data.get("shots", [])
        merged_shots = 0
        matched_but_empty = 0  # id khớp nhưng nội dung caption rỗng

        for shot_3 in demo3_shots:
            raw_shot_id = shot_3.get("shot_id")
            shot_id = str(raw_shot_id).strip() if raw_shot_id is not None else None
            shot_2 = demo2_shots_dict.get(shot_id) if shot_id else None

            if shot_2:
                ext_2 = shot_2.get("extracted_data", {})

                vlm_str = ext_2.get("vlm_caption", "")
                yolo_str = ext_2.get("yolo_objects", "")

                fake_vlm_dict = {}
                if isinstance(vlm_str, str) and vlm_str.strip():
                    fake_vlm_dict["detailed_description"] = vlm_str.strip()
                elif isinstance(vlm_str, dict):
                    fake_vlm_dict = vlm_str

                if isinstance(yolo_str, str) and yolo_str.strip():
                    fake_vlm_dict["objects_and_counts"] = [yolo_str.strip()]

                if fake_vlm_dict:
                    if "extracted_data" not in shot_3:
                        shot_3["extracted_data"] = {}
                    shot_3["extracted_data"]["vlm_caption"] = fake_vlm_dict
                    merged_shots += 1
                else:
                    # id khớp nhưng bên Demo2 không có nội dung caption thật
                    matched_but_empty += 1

        if matched_but_empty > 0 and merged_shots == 0:
            empty_content_videos.append(video_name)

        if merged_shots == 0 and demo3_shots and matched_but_empty == 0:
            # trường hợp id không khớp gì cả (khác nguyên nhân ở trên)
            zero_shot_videos.append(video_name)
            demo3_ids_sample = [s.get("shot_id") for s in demo3_shots[:3]]
            demo2_ids_sample = list(demo2_shots_dict.keys())[:3]
            print(f"   🔍 {video_name}: demo2 có {len(demo2_shots_raw)} shots "
                  f"(dict hợp lệ: {len(demo2_shots_dict)}) | "
                  f"demo3 shot_id mẫu={demo3_ids_sample} | demo2 shot_id mẫu={demo2_ids_sample}")

        rich_path = DEMO3_DIR / f"{video_name}_rich_metadata.json"
        with open(rich_path, 'w', encoding='utf-8') as f:
            json.dump(demo3_data, f, ensure_ascii=False, indent=4)

        print(f"✅ Đã cấy ghép {video_name}: {merged_shots} shots"
              + (f" (⚠️ {matched_but_empty} shot khớp id nhưng Demo2 rỗng nội dung)" if matched_but_empty else ""))
        success_count += 1

    print("\n🎉 CHIẾN DỊCH HOÀN TẤT!")
    print(f"📊 Tổng kết: {success_count} video xử lý thành công, {missing_count} video bỏ qua (không tìm thấy Demo2).")
    if empty_content_videos:
        print(f"⚠️ {len(empty_content_videos)} video khớp shot_id nhưng Demo2 KHÔNG có nội dung caption/yolo "
              f"(nghi ngờ VLM/YOLO ở Demo2 bị lỗi/rỗng cho các video này): {empty_content_videos}")
    if zero_shot_videos:
        print(f"⚠️ {len(zero_shot_videos)} video hoàn toàn không khớp shot_id nào: {zero_shot_videos}")
    print(f"👉 Hãy chạy ngay lệnh: python -m src.data_pipeline.ingestion_stage_c --force-reset")


if __name__ == "__main__":
    merge_demo2_and_demo3()