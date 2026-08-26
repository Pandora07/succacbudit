"""
GIAI ĐOẠN C: BƠM DỮ LIỆU LÊN QDRANT HYBRID (VECTORIZATION & INGESTION)
Vị trí: src/data_pipeline/ingestion_stage_c.py

Kiến trúc:
  - Đọc *_rich_metadata.json (shot-level) từ data/json/
  - Tìm keyframe images theo 2 cấp:
      Cấp 1: KEYFRAMES_ROOT (data/processed/keyframes/) — nếu tồn tại trên máy này
      Cấp 2: AIC_KEYFRAMES_ROOT (Desktop/AIC-2026/keyframesL26c/) — folder có sẵn từ ban TC
  - Tổng hợp rich_text từ: transcript + OCR + vlm_caption + AIC frame metadata
  - Encode 3 vectors: image_dense (SigLIP 1024D), text_dense (MiniLM 384D), text_sparse (SPLADE)
  - Upsert batch lên Qdrant

Về metadata thực tế:
  - vlm_caption từ Stage B cũ: chỉ có 2 fields: detailed_description + objects_and_counts
  - Stage B mới (nếu chạy lại): thêm scene_type, micro_details, action_sequence,
    temporal_event_tags, text_on_screen — code đều xử lý được cả hai version
  - AIC metadata: description, objects, scene_type, ocr_text (frame-level)
    chỉ có cho L26_V200–V299, nhưng khi có → chất lượng text encoding tốt hơn
"""

import os
import json
import uuid
from pathlib import Path
from collections import Counter
from qdrant_client.http import models

from src.config.common import get_logger
from src.config.config import (
    JSON_ROOT, KEYFRAMES_ROOT,
    AIC_KEYFRAMES_ROOT, AIC_METADATA_FILE,
    QDRANT_UPSERT_BATCH_SIZE,
)
from src.data_pipeline.vectorizers import HybridVectorizer
from src.db.qdrant_manager import QdrantManager

logger = get_logger(__name__)

# Lấy từ config (fallback nếu config cũ chưa có)
try:
    UPSERT_BATCH_SIZE = QDRANT_UPSERT_BATCH_SIZE
except Exception:
    UPSERT_BATCH_SIZE = 32


class StageC_Ingester:
    def __init__(self, force_reset: bool = False):
        """
        Args:
            force_reset: True → xóa sạch và tạo lại Collection.
                         False (mặc định) → incremental upsert.
        """
        logger.info("🚀 BẮT ĐẦU GIAI ĐOẠN C: BƠM DỮ LIỆU VECTOR")
        self.vectorizer = HybridVectorizer()
        self.db = QdrantManager()

        if force_reset:
            logger.warning("⚠️  force_reset=True: Đang XÓA SẠCH và tạo lại Collection Qdrant...")
            self.db.setup_new_database()
        else:
            logger.info("ℹ️  force_reset=False: Sẽ upsert thêm vào Collection hiện tại (incremental).")

        # Load AIC frame-level metadata một lần duy nhất (lazy, chỉ khi file tồn tại)
        self._aic_index = self._load_aic_metadata()

    # ==========================================
    # LOAD AIC METADATA
    # ==========================================
    def _load_aic_metadata(self) -> dict:
        """
        Load AIC frame-level JSON và build index nhanh:
          { video_id: [ {frame_id, timestamp_sec, description, objects, scene_type, ocr_text, frame_path}, ... ] }

        Chỉ load nếu file tồn tại. Không crash nếu thiếu.
        """
        if not AIC_METADATA_FILE.exists():
            logger.warning(f"⚠️ Không tìm thấy AIC metadata: {AIC_METADATA_FILE}. "
                           f"Stage C vẫn chạy được nhưng sẽ thiếu enrichment cho L26_V200–V299.")
            return {}

        logger.info(f"📖 Đang load AIC metadata từ {AIC_METADATA_FILE.name}...")
        with open(AIC_METADATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        index = {}
        for item in raw:
            vid = item.get("video_id", "")
            if vid not in index:
                index[vid] = []
            index[vid].append({
                "frame_id":     item.get("frame_id", 0),
                "timestamp_sec": item.get("timestamp_sec", 0.0),
                "description":  item.get("description", ""),
                "objects":      item.get("objects", []),
                "scene_type":   item.get("scene_type", ""),
                "ocr_text":     item.get("ocr_text", ""),
                "frame_path":   item.get("frame_path", ""),
            })

        logger.info(f"✅ AIC metadata loaded: {len(index)} videos, "
                    f"{sum(len(v) for v in index.values())} frames tổng cộng.")
        return index

    # ==========================================
    # TÌM KIẾM KEYFRAME IMAGE (2 CẤP)
    # ==========================================
    def _resolve_frame_path(self, kf_obj, video_name: str, fps: float) -> tuple[str | None, int]:
        """
        Tìm đường dẫn thực tế của một keyframe, theo 3 cấp ưu tiên:

          Cấp 0: raw_path gốc từ JSON — nếu file vẫn tồn tại trên máy này (e.g. /Users/trandinhquy/...)
          Cấp 1: KEYFRAMES_ROOT/video_name/basename  — máy chạy full pipeline hiện tại
          Cấp 2: AIC_KEYFRAMES_ROOT/video_name/{frame_id:03d}.jpg — folder ban TC (L26_V200–V299)

        Trả về (image_path_str, frame_idx) hoặc (None, frame_idx).
        """
        if isinstance(kf_obj, dict):
            frame_idx = kf_obj.get("frame_idx", 0)
            raw_path  = kf_obj.get("image_path", "")
        else:
            frame_idx = 0
            raw_path  = str(kf_obj)

        img_name = os.path.basename(raw_path)

        # --- Cấp 0: Đường dẫn gốc trong JSON ---
        # Demo được tải từ máy khác nên raw_path có thể là /Users/trandinhquy/...
        # Nếu folder của user đó vẫn mount hoặc tồn tại trên máy này → dùng luôn
        if raw_path and os.path.isfile(raw_path):
            return raw_path, frame_idx

        # --- Cấp 1: Rebase sang KEYFRAMES_ROOT hiện tại ---
        # Dùng basename để thoát khỏi path tuyệt đối từ máy khác
        p1 = KEYFRAMES_ROOT / video_name / img_name
        if p1.exists():
            return str(p1), frame_idx

        # --- Cấp 2: AIC keyframesL26c (chỉ có cho L26_V200–V299) ---
        # AIC lưu ~1fps: frame_id = round(timestamp_sec), filename = f"{frame_id:03d}.jpg"
        if fps and fps > 0:
            timestamp_sec = frame_idx / fps
        else:
            timestamp_sec = 0.0
        aic_frame_id   = max(1, round(timestamp_sec))
        aic_frame_name = f"{aic_frame_id:03d}.jpg"
        p2 = AIC_KEYFRAMES_ROOT / video_name / aic_frame_name
        if p2.exists():
            return str(p2), frame_idx

        return None, frame_idx



    def _select_representative_frame(self, keyframes: list, video_name: str, fps: float):
        """
        Duyệt qua toàn bộ keyframes của một shot, thu thập các ảnh tìm được.
        Chọn ảnh ở vị trí GIỮA danh sách hợp lệ.
        Trả về (image_path, frame_idx) hoặc (None, None) nếu không có ảnh nào.
        """
        valid_frames = []
        for kf in keyframes:
            img_path, frame_idx = self._resolve_frame_path(kf, video_name, fps)
            if img_path:
                valid_frames.append((img_path, frame_idx))

        if not valid_frames:
            return None, None

        mid_idx = len(valid_frames) // 2
        return valid_frames[mid_idx]

    # ==========================================
    # AIC ENRICHMENT CHO SHOT
    # ==========================================
    def _get_aic_enrichment(self, video_name: str, start_ts: float, end_ts: float) -> dict | None:
        """
        Lấy thông tin AIC frame-level cho tất cả frames trong khoảng [start_ts, end_ts].
        Trả về dict gộp hoặc None nếu video không có trong AIC index.

        Ý nghĩa: AIC metadata có description + objects + scene_type phong phú hơn
        so với vlm_caption cũ (chỉ 2 fields), giúp text encoding tốt hơn.
        """
        if video_name not in self._aic_index:
            return None

        # Lấy các frame AIC trong khoảng thời gian shot (±0.5s margin)
        shot_frames = [
            f for f in self._aic_index[video_name]
            if (start_ts - 0.5) <= f["timestamp_sec"] <= (end_ts + 0.5)
        ]

        if not shot_frames:
            # Nếu shot quá ngắn không có frame nào, lấy frame gần nhất
            mid_ts = (start_ts + end_ts) / 2
            all_frames = self._aic_index[video_name]
            if all_frames:
                closest = min(all_frames, key=lambda f: abs(f["timestamp_sec"] - mid_ts))
                shot_frames = [closest]
            else:
                return None

        # Gộp thông tin: lấy scene_type phổ biến nhất, gộp descriptions/objects/ocr
        scene_types  = [f["scene_type"] for f in shot_frames if f.get("scene_type")]
        descriptions = [f["description"] for f in shot_frames if f.get("description")]
        objects      = list(set(obj for f in shot_frames for obj in f.get("objects", [])))
        ocr_texts    = [f["ocr_text"] for f in shot_frames if f.get("ocr_text")]

        return {
            "scene_type":   Counter(scene_types).most_common(1)[0][0] if scene_types else "",
            "descriptions": descriptions[:2],    # Tối đa 2 descriptions để tránh noise
            "objects":      objects[:12],         # Tối đa 12 objects
            "ocr_texts":    list(set(ocr_texts)), # Dedup OCR
        }

    # ==========================================
    # XÂY DỰNG RICH TEXT ĐỂ ENCODE
    # ==========================================
    def _format_rich_text(
        self,
        extracted_data: dict,
        video_name: str,
        start_ts: float,
        end_ts: float,
    ) -> str:
        """
        Tổng hợp văn bản từ nhiều nguồn theo thứ tự ưu tiên:

        1. Transcript (ASR)     — nội dung lời nói
        2. OCR text (Stage A)   — chữ nhìn thấy trên màn hình
        3. VLM Caption (Stage B) — mô tả visual; hỗ trợ cả schema cũ (2 fields) và mới (6 fields)
        4. AIC enrichment       — nếu là L26_V200–V299: thêm description + objects + scene_type
                                  từ metadata frame-level của ban tổ chức

        Không dùng prefix tag ([AUDIO]/[OCR]/...) vì làm loãng SPLADE keyword matching.
        """
        parts = []

        # --- 1. Transcript ---
        if extracted_data.get("transcript"):
            parts.append(extracted_data["transcript"])

        # --- 2. OCR ---
        if extracted_data.get("ocr_text"):
            parts.append(extracted_data["ocr_text"])

        # --- 3. VLM Caption (hỗ trợ cả schema cũ lẫn mới) ---
        vlm_cap = extracted_data.get("vlm_caption", {})
        if isinstance(vlm_cap, dict):
            # Schema cũ (Stage B máy khác): chỉ 2 keys
            if vlm_cap.get("detailed_description"):
                parts.append(vlm_cap["detailed_description"])
            if vlm_cap.get("objects_and_counts"):
                oc = vlm_cap["objects_and_counts"]
                parts.append(" ".join(oc) if isinstance(oc, list) else str(oc))

            # Schema mới (Stage B prompt cải tiến): 4 keys thêm
            if vlm_cap.get("scene_type"):
                parts.append(vlm_cap["scene_type"])
            if vlm_cap.get("micro_details"):
                parts.append(" ".join(vlm_cap["micro_details"]))
            if vlm_cap.get("action_sequence"):
                parts.append(" ".join(vlm_cap["action_sequence"]))
            if vlm_cap.get("temporal_event_tags"):
                parts.append(" ".join(vlm_cap["temporal_event_tags"]))
            if vlm_cap.get("text_on_screen"):
                parts.append(vlm_cap["text_on_screen"])

        # --- 4. AIC Enrichment (chỉ cho L26_V200–V299) ---
        aic = self._get_aic_enrichment(video_name, start_ts, end_ts)
        if aic:
            if aic.get("scene_type"):
                parts.append(aic["scene_type"])
            for desc in aic.get("descriptions", []):
                if desc and desc not in parts:  # Tránh dup với vlm_caption
                    parts.append(desc)
            if aic.get("objects"):
                parts.append(" ".join(aic["objects"]))
            for ocr in aic.get("ocr_texts", []):
                if ocr and ocr not in parts:
                    parts.append(ocr)

        return "\n".join(filter(None, parts))

    # ==========================================
    # QUALITY GATE
    # ==========================================
    @staticmethod
    def _is_zero_vector(vec: list) -> bool:
        """Kiểm tra vector toàn 0 → dấu hiệu encode thất bại."""
        return all(v == 0.0 for v in vec)

    # ==========================================
    # MAIN RUN
    # ==========================================
    def run(self):
        rich_files = sorted(JSON_ROOT.rglob("*_rich_metadata.json"))
        if not rich_files:
            logger.error("❌ Không tìm thấy file *_rich_metadata.json nào! "
                         "Hãy chạy Giai đoạn B trước (hoặc dùng metadata có sẵn).")
            return

        logger.info(f"📁 Tìm thấy {len(rich_files)} rich_metadata files cần xử lý.")

        total_upserted = 0
        total_skipped  = 0

        for i, meta_file in enumerate(rich_files):
            video_name = meta_file.stem.replace("_rich_metadata", "")
            logger.info(f"⏳ [{i+1}/{len(rich_files)}] {video_name}")

            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            video_fps  = data.get("fps", 25.0)
            shots      = data.get("shots", [])
            points_batch = []
            skipped_video = 0

            for shot in shots:
                extracted_data = shot.get("extracted_data", {})
                start_ts = shot.get("start_ts", 0.0)
                end_ts   = shot.get("end_ts",   0.0)

                # 1. Xây dựng rich text để encode (text_dense + text_sparse)
                rich_text = self._format_rich_text(
                    extracted_data, video_name, start_ts, end_ts
                )

                # 2. Chọn ảnh đại diện ở giữa keyframes (image_dense)
                keyframes = shot.get("keyframes", [])
                target_image_path, representative_frame_idx = self._select_representative_frame(
                    keyframes, video_name, video_fps
                )

                if not target_image_path:
                    logger.debug(f"   ⏩ Shot {shot['shot_id']}: không có ảnh → bỏ qua.")
                    skipped_video += 1
                    continue

                # 3. Encode 3 vectors cùng lúc
                vectors = self.vectorizer.encode_all(target_image_path, rich_text)

                # QUALITY GATE: image vector toàn 0 → encode thất bại
                if self._is_zero_vector(vectors["image_dense"]):
                    logger.warning(f"   ⚠️ Shot {shot['shot_id']}: image encode lỗi (zero-vec). Bỏ qua.")
                    skipped_video += 1
                    continue

                # 4. Đóng gói Point cho Qdrant
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_name}_{shot['shot_id']}"))

                payload = {
                    # --- Định danh ---
                    "video_id":   video_name,
                    "shot_id":    shot["shot_id"],
                    # --- Thời gian ---
                    "start_ts":   start_ts,
                    "end_ts":     end_ts,
                    "fps":        video_fps,           # Để UI tính frame chính xác
                    "frame_idx":  representative_frame_idx,
                    # --- Nội dung (để retriever trả về cho UI) ---
                    "rich_text":  rich_text,
                    "thumbnail":  target_image_path,
                }

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "image_dense": vectors["image_dense"],
                        "text_dense":  vectors["text_dense"],
                        "text_sparse": models.SparseVector(
                            indices=vectors["text_sparse"]["indices"],
                            values=vectors["text_sparse"]["values"],
                        ),
                    },
                    payload=payload,
                )
                points_batch.append(point)

                # Đẩy batch khi đủ UPSERT_BATCH_SIZE
                if len(points_batch) >= UPSERT_BATCH_SIZE:
                    self.db.upsert_batch(points_batch)
                    total_upserted += len(points_batch)
                    points_batch = []

            # Đẩy phần còn sót của video
            if points_batch:
                self.db.upsert_batch(points_batch)
                total_upserted += len(points_batch)

            total_skipped += skipped_video
            logger.info(
                f"   ✅ {video_name}: {len(shots) - skipped_video} shots upserted, "
                f"{skipped_video} bỏ qua (không có ảnh/encode lỗi)"
            )

        logger.info(
            f"\n🎉 HOÀN TẤT GIAI ĐOẠN C!\n"
            f"   Tổng upserted : {total_upserted} points\n"
            f"   Tổng bỏ qua   : {total_skipped} shots\n"
            f"   CƠ SỞ DỮ LIỆU ĐÃ SẴN SÀNG."
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage C: Vectorize & Ingest vào Qdrant")
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Xóa sạch và tạo lại Collection Qdrant từ đầu",
    )
    args = parser.parse_args()
    StageC_Ingester(force_reset=args.force_reset).run()