"""
GIAI ĐOẠN B: VLM Dense Captioning (Nhồi sọ ngữ nghĩa).
Mục tiêu: Đọc _stage_a.json -> VLM caption từng shot -> Lưu thành _rich_metadata.json

Cải tiến:
- Checkpoint trung gian mỗi VLM_CHECKPOINT_INTERVAL shots (chống mất data khi crash).
- Resume tự động từ checkpoint: bỏ qua shot đã có vlm_caption.
- 5-frame sampling cho shot dài (≥ VLM_LONG_SHOT_FRAME_THRESHOLD frames).
- Prompt thêm temporal_event_tags và text_on_screen hỗ trợ TRAKE search.
- Đường dẫn keyframe tái dựng qua KEYFRAMES_ROOT (không phụ thuộc path tuyệt đối).
- FPS và toàn bộ metadata gốc được bảo toàn qua suốt quá trình.

Vị trí: src/data_pipeline/ingestion_stage_b.py
"""

import os
import json
from pathlib import Path

from src.config.common import get_logger
from src.config.config import (
    JSON_ROOT,
    KEYFRAMES_ROOT,
    VLM_CHECKPOINT_INTERVAL,
    VLM_LONG_SHOT_FRAME_THRESHOLD,
)
from src.vlm_engine.engine import VLMEngine

logger = get_logger(__name__)

VLM_PROMPT = """You are a highly objective and meticulous visual analysis AI. Your task is to examine a temporal sequence of frames from a single video shot and extract precise metadata.

CRITICAL RULES:
1. ZERO HALLUCINATION: Describe ONLY what is explicitly visible in the frames. Never guess or infer context outside the image.
2. SYSTEMATIC COUNTING (BOUNDED): For each DISTINCT object category, output EXACTLY ONE entry in "objects_and_counts" — never repeat the same category twice.
   - If instances ≤ 10 and clearly distinguishable: list their positions (left-to-right, top-to-bottom), then give the Exact Count.
   - If instances > 10, or they are small/repetitive/hard to individuate (crowd, rows of chairs, string of lights/speakers, decorative patterns): DO NOT enumerate each one. Instead give ONE aggregated entry with an approximate range (e.g. "~20-30") and a general area (e.g. "scattered across the stage").
3. SPATIAL AWARENESS: Anchor salient objects to relative positions (foreground, background, top-left, center, behind X).
4. TEMPORAL PROGRESSION: Analyze differences between frames to deduce the sequence of actions. If frames show no meaningful change, leave "action_sequence" as an empty array — do not invent motion.
5. STRICT JSON FORMAT: Return ONLY a valid JSON object. No markdown fences, no preamble, no explanation. Start with '{' and end with '}'.
6. HARD STOP / NO REPETITION: "objects_and_counts" and "micro_details" MUST have AT MOST 6 items each. The moment you are about to write an entry that repeats or closely resembles a previous one, STOP adding entries and close the JSON immediately. Repetition of similar strings is a critical failure.

Output ONLY a valid JSON object strictly following this schema:
{
    "scene_type": "Broad category (e.g., News, Cooking, Wildlife, Traffic, Surveillance, Event).",
    "detailed_description": "1-2 objective sentences summarizing the scene. MUST NOT BE EMPTY.",
    "objects_and_counts": [
        "Max 6 entries. Each entry = ONE distinct object category, never repeated.",
        "For ≤10 distinguishable instances: '[Color/Texture] [Object] - Positions: [pos1, pos2...] - Exact Count: [int]'.",
        "For >10 or hard-to-individuate instances: '[Object] - Area: [general location] - Approx Count: [range, e.g. ~20-30]'."
    ],
    "micro_details": [
        "Max 5 entries. Distinct visual attributes, textures, materials, or exact text/numbers/symbols on surfaces.",
        "Example: 'geometric star-shaped cuts on the orange objects'",
        "Example: 'text reading [XYZ] on the top-left corner'"
    ],
    "action_sequence": [
        "Chronological steps of movement/interaction across frames. Empty array if no meaningful change."
    ],
    "temporal_event_tags": [
        "2-5 short action verb phrases (2-4 words) for temporal/event retrieval.",
        "Example: 'person runs left', 'object falls', 'crowd applauds'"
    ],
    "text_on_screen": "Readable text/numbers/signs/captions visible in ANY frame. Empty string if none."
}"""


class StageBVLM:
    def __init__(self):
        logger.info("🚀 BẮT ĐẦU GIAI ĐOẠN B: VLM DENSE CAPTIONING")
        self.vlm = VLMEngine(mode="ingest")

    def _resolve_keyframe_path(self, kf_obj, video_name: str) -> str | None:
        """
        Tái dựng đường dẫn keyframe thực tế trên máy hiện tại.
        Dùng basename để thoát khỏi path tuyệt đối từ máy khác.
        """
        raw_path = kf_obj.get("image_path") if isinstance(kf_obj, dict) else kf_obj
        if not raw_path:
            return None
        filename = os.path.basename(raw_path)
        resolved = KEYFRAMES_ROOT / video_name / filename
        return str(resolved) if resolved.exists() else None

    def _sample_keyframes(self, valid_images: list) -> list:
        """
        Lấy mẫu frames đại diện cho VLM.
        - Shot ngắn (< VLM_LONG_SHOT_FRAME_THRESHOLD frames): 3 điểm (head/mid/tail)
        - Shot dài (≥ VLM_LONG_SHOT_FRAME_THRESHOLD frames): 5 điểm (Q0/Q1/Q2/Q3/Q4)
        để không bỏ qua hành động quan trọng ở giữa shot dài.
        """
        n = len(valid_images)
        if n <= 1:
            return valid_images
        if n < VLM_LONG_SHOT_FRAME_THRESHOLD:
            return [valid_images[0], valid_images[n // 2], valid_images[-1]]
        else:
            return [
                valid_images[0],
                valid_images[n // 4],
                valid_images[n // 2],
                valid_images[3 * n // 4],
                valid_images[-1],
            ]

    def run(self):
        # 1. Tìm file đầu vào (ưu tiên Stage A, fallback về _structure.json)
        input_files = list(JSON_ROOT.rglob("*_stage_a.json"))
        if not input_files:
            logger.warning("Không tìm thấy *_stage_a.json. Thử fallback *_structure.json...")
            input_files = list(JSON_ROOT.rglob("*_structure.json"))

        if not input_files:
            logger.error("❌ Không có file JSON nào để xử lý!")
            return

        logger.info(f"📁 Tìm thấy {len(input_files)} video cần Dense Captioning.")

        for i, meta_file in enumerate(input_files):
            video_name = meta_file.stem.replace("_stage_a", "").replace("_structure", "")
            rich_path = JSON_ROOT / f"{video_name}_rich_metadata.json"
            partial_path = JSON_ROOT / f"{video_name}_stage_b_partial.json"

            # CHECKPOINT: Bỏ qua nếu đã hoàn thành
            if rich_path.exists():
                logger.info(f"⏭️ [{i+1}/{len(input_files)}] [CACHE] {video_name} đã có Rich Metadata. Bỏ qua.")
                continue

            # Resume từ checkpoint trung gian nếu có (tránh làm lại từ đầu khi crash)
            if partial_path.exists():
                logger.info(f"🔄 [{i+1}/{len(input_files)}] Tiếp tục từ checkpoint: {partial_path.name}")
                with open(partial_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.info(f"⏳ [{i+1}/{len(input_files)}] Bắt đầu mới: {video_name}")
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # Trích xuất fps để log (fps được bảo toàn trong data từ _structure.json)
            video_fps = data.get("fps", "unknown")
            shots = data.get("shots", [])
            logger.info(f" 📹 FPS={video_fps} | {len(shots)} shots")

            processed_count = 0

            for shot in shots:
                # Bỏ qua shot đã có VLM caption (từ lần resume trước)
                if shot.get("extracted_data", {}).get("vlm_caption"):
                    processed_count += 1
                    continue

                keyframes = shot.get("keyframes", [])

                # Tái dựng đường dẫn thực tế từ KEYFRAMES_ROOT
                valid_images = []
                for kf in keyframes:
                    img_path = self._resolve_keyframe_path(kf, video_name)
                    if img_path:
                        valid_images.append(img_path)

                if not valid_images:
                    logger.warning(f" ⚠️ Shot {shot['shot_id']}: Không có ảnh hợp lệ. Bỏ qua.")
                    processed_count += 1
                    continue

                # Lấy mẫu frames đại diện (3 hoặc 5 tùy độ dài shot)
                sampled_images = self._sample_keyframes(valid_images)

                try:
                    parsed_json = self.vlm.generate_json_robust(
                        image_paths=sampled_images,
                        prompt=VLM_PROMPT
                    )
                    if "extracted_data" not in shot:
                        shot["extracted_data"] = {}
                    shot["extracted_data"]["vlm_caption"] = parsed_json

                except Exception as e:
                    logger.error(f" ❌ Lỗi VLM tại Shot {shot.get('shot_id')}: {e}")
                    # KHÔNG raise — bỏ qua shot lỗi, tiếp tục shot tiếp theo.
                    # Nếu OOM: Python tự sập, script wrapper bắt lỗi và restart.

                processed_count += 1

                # CHECKPOINT TRUNG GIAN: Lưu sau mỗi VLM_CHECKPOINT_INTERVAL shots
                if processed_count % VLM_CHECKPOINT_INTERVAL == 0:
                    data["shots"] = shots
                    with open(partial_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    logger.info(f" 💾 [Checkpoint] {processed_count}/{len(shots)} shots...")

            # 4. Ghi file chính thức và xóa checkpoint tạm
            data["shots"] = shots
            with open(rich_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            if partial_path.exists():
                partial_path.unlink()

            logger.info(f"✅ {video_name}: {len(shots)} shots → {rich_path.name}\n")


if __name__ == "__main__":
    StageBVLM().run()