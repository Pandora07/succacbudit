"""
GIAI ĐOẠN C: VECTORIZATION & INGESTION LÊN QDRANT
Vị trí: src/data_pipeline/ingestion_stage_c.py

Luồng xử lý:
  1. Đọc *_rich_metadata.json từ data/json/
  2. Với mỗi shot → lấy ảnh keyframe từ metadata, encode 3 vectors
  3. Upsert lên Qdrant

Về đường dẫn ảnh:
  - Dùng image_path ghi trong metadata (Stage B đã lưu đúng đường dẫn tuyệt đối trên máy chạy)
  - Fallback: KEYFRAMES_ROOT/video_name/basename nếu path gốc không tìm thấy
"""

import os
import json
import uuid
from pathlib import Path

from qdrant_client.http import models

from src.config.common import get_logger
from src.config.config import (
    JSON_ROOT,
    KEYFRAMES_ROOT,
    QDRANT_UPSERT_BATCH_SIZE,
)
from src.data_pipeline.vectorizers import HybridVectorizer
from src.db.qdrant_manager import QdrantManager

logger = get_logger(__name__)


class StageC_Ingester:
    def __init__(self, force_reset: bool = False):
        logger.info("🚀 BẮT ĐẦU GIAI ĐOẠN C: BƠM DỮ LIỆU VECTOR")
        self.vectorizer = HybridVectorizer()
        self.db = QdrantManager()

        if force_reset:
            logger.warning("⚠️  force_reset=True: Xóa sạch và tạo lại Collection Qdrant...")
            self.db.setup_new_database()
        else:
            logger.info("ℹ️  Upsert incremental vào Collection hiện tại.")

    # ==========================================
    # TÌM ẢNH KEYFRAME
    # ==========================================
    def _get_frame_path(self, kf_obj: dict | str, video_name: str) -> tuple[str | None, int]:
        """
        Lấy đường dẫn ảnh và frame_idx từ một keyframe object.

        Thử theo thứ tự:
          1. image_path gốc từ metadata (Stage B đã ghi đúng)
          2. KEYFRAMES_ROOT/video_name/basename (fallback nếu đổi thư mục)

        Trả về (path, frame_idx) hoặc (None, 0).
        """
        if isinstance(kf_obj, dict):
            raw_path  = kf_obj.get("image_path", "")
            frame_idx = kf_obj.get("frame_idx", 0)
        else:
            raw_path  = str(kf_obj)
            frame_idx = 0

        # 1. Dùng path trong metadata
        if raw_path and os.path.isfile(raw_path):
            return raw_path, frame_idx

        # 2. Fallback: rebase sang KEYFRAMES_ROOT
        img_name = os.path.basename(raw_path)
        if img_name:
            fallback = KEYFRAMES_ROOT / video_name / img_name
            if fallback.is_file():
                return str(fallback), frame_idx

        return None, frame_idx

    def _pick_representative_frame(self, keyframes: list, video_name: str) -> tuple[str | None, int]:
        """Chọn ảnh ở giữa danh sách keyframes hợp lệ."""
        valid = []
        for kf in keyframes:
            path, idx = self._get_frame_path(kf, video_name)
            if path:
                valid.append((path, idx))

        if not valid:
            return None, 0

        return valid[len(valid) // 2]

    # ==========================================
    # XÂY DỰNG RICH TEXT
    # ==========================================
    def _build_rich_text(self, extracted_data: dict) -> str:
        """
        Gộp tất cả văn bản từ extracted_data:
          - transcript (ASR)
          - ocr_text (Stage A)
          - vlm_caption (Stage B) — đọc tất cả fields có trong metadata,
            bỏ qua field nào không tồn tại (tương thích cả schema cũ và mới)
        """
        parts = []

        if extracted_data.get("transcript"):
            parts.append(extracted_data["transcript"])

        if extracted_data.get("ocr_text"):
            parts.append(extracted_data["ocr_text"])

        vlm = extracted_data.get("vlm_caption", {})
        if isinstance(vlm, dict):
            # Đọc tất cả text fields — field nào có thì dùng, không có thì bỏ qua
            text_fields = [
                "detailed_description",
                "scene_type",
                "text_on_screen",
            ]
            list_fields = [
                "objects_and_counts",
                "micro_details",
                "action_sequence",
                "temporal_event_tags",
            ]
            for field in text_fields:
                if vlm.get(field):
                    parts.append(vlm[field])
            for field in list_fields:
                val = vlm.get(field)
                if val:
                    parts.append(" ".join(val) if isinstance(val, list) else str(val))

        return "\n".join(filter(None, parts))

    # ==========================================
    # MAIN RUN
    # ==========================================
    def run(self):
        rich_files = sorted(JSON_ROOT.rglob("*_rich_metadata.json"))
        if not rich_files:
            logger.error("❌ Không tìm thấy *_rich_metadata.json. Chạy Stage B trước.")
            return

        logger.info(f"📁 {len(rich_files)} videos cần xử lý.")

        total_upserted = 0
        total_skipped  = 0

        for i, meta_file in enumerate(rich_files):
            video_name = meta_file.stem.replace("_rich_metadata", "")
            logger.info(f"⏳ [{i+1}/{len(rich_files)}] {video_name}")

            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            video_fps    = data.get("fps", 25.0)
            shots        = data.get("shots", [])
            batch        = []
            skip_count   = 0

            for shot in shots:
                keyframes      = shot.get("keyframes", [])
                extracted_data = shot.get("extracted_data", {})

                # Lấy ảnh đại diện
                image_path, frame_idx = self._pick_representative_frame(keyframes, video_name)
                if not image_path:
                    skip_count += 1
                    continue

                # Xây dựng văn bản tổng hợp
                rich_text = self._build_rich_text(extracted_data)

                # Encode 3 vectors
                vectors = self.vectorizer.encode_all(image_path, rich_text)

                # Quality gate: image encode thất bại → zero vector
                if all(v == 0.0 for v in vectors["image_dense"]):
                    logger.warning(f"   ⚠️  {shot['shot_id']}: image encode lỗi. Bỏ qua.")
                    skip_count += 1
                    continue

                # Đóng gói point
                point_id = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS, f"{video_name}_{shot['shot_id']}"
                ))
                batch.append(models.PointStruct(
                    id=point_id,
                    vector={
                        "image_dense": vectors["image_dense"],
                        "text_dense":  vectors["text_dense"],
                        "text_sparse": models.SparseVector(
                            indices=vectors["text_sparse"]["indices"],
                            values=vectors["text_sparse"]["values"],
                        ),
                    },
                    payload={
                        "video_id":  video_name,
                        "shot_id":   shot["shot_id"],
                        "start_ts":  shot.get("start_ts", 0.0),
                        "end_ts":    shot.get("end_ts",   0.0),
                        "fps":       video_fps,
                        "frame_idx": frame_idx,
                        "rich_text": rich_text,
                        "thumbnail": image_path,
                    },
                ))

                if len(batch) >= QDRANT_UPSERT_BATCH_SIZE:
                    self.db.upsert_batch(batch)
                    total_upserted += len(batch)
                    batch = []

            if batch:
                self.db.upsert_batch(batch)
                total_upserted += len(batch)

            total_skipped += skip_count
            logger.info(
                f"   ✅ {len(shots) - skip_count} upserted, {skip_count} bỏ qua"
            )

        logger.info(
            f"\n🎉 HOÀN TẤT!\n"
            f"   Upserted : {total_upserted} points\n"
            f"   Bỏ qua   : {total_skipped} shots"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-reset", action="store_true",
                        help="Xóa sạch Collection rồi ingest lại từ đầu")
    args = parser.parse_args()
    StageC_Ingester(force_reset=args.force_reset).run()