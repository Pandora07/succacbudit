"""
GIAI ĐOẠN A: Khai thác Bề mặt (Audio & OCR).
Mục tiêu: Đọc _structure.json -> Chạy ASR & OCR -> Lưu thành _stage_a.json

Kiến trúc:
  - Multi-threading: ASR (Metal GPU) + OCR (Apple Neural Engine) song song.
  - ThreadPool tạo 1 lần duy nhất, tái dùng cho toàn bộ pipeline.
  - Đường dẫn keyframe được TÁI DỰNG từ KEYFRAMES_ROOT + video_name + basename
    để tránh phụ thuộc vào path tuyệt đối từ máy khác.
  - FPS và video_id được bảo toàn từ _structure.json qua _stage_a.json.

Vị trí: src/data_pipeline/ingestion_stage_a.py
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.config.common import get_logger
from src.config.config import RAW_ROOT, JSON_ROOT, KEYFRAMES_ROOT, ASR_THREAD_WORKERS
from src.data_pipeline.extractors.audio_asr import AudioTranscriber
from src.data_pipeline.extractors.vision_ocr import TextExtractor

logger = get_logger(__name__)

class StageAExtractor:
    def __init__(self):
        logger.info("🚀 BẮT ĐẦU GIAI ĐOẠN A: KHAI THÁC AUDIO & OCR")
        self.asr = AudioTranscriber()  
        self.ocr = TextExtractor()     
        self._executor = ThreadPoolExecutor(max_workers=ASR_THREAD_WORKERS)
        logger.info(f"⚙️ ThreadPool đã sẵn sàng ({ASR_THREAD_WORKERS} workers: ASR + OCR)")

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def _resolve_keyframe_path(self, kf_obj, video_name: str) -> str | None:
        raw_path = kf_obj.get("image_path") if isinstance(kf_obj, dict) else kf_obj
        if not raw_path:
            return None
        filename = os.path.basename(raw_path)
        resolved = KEYFRAMES_ROOT / video_name / filename
        return str(resolved) if resolved.exists() else None

    def _extract_ocr_for_shot(self, keyframes: list, video_name: str) -> str:
        combined_ocr = []
        for kf_obj in keyframes:
            img_path = self._resolve_keyframe_path(kf_obj, video_name)
            if not img_path:
                continue
            text = self.ocr.extract_text(img_path)
            if text:
                combined_ocr.append(text)
        return " | ".join(list(dict.fromkeys(combined_ocr)))

    def run(self):
        structure_files = list(JSON_ROOT.rglob("*_structure.json"))
        if not structure_files:
            logger.error("❌ Không tìm thấy file *_structure.json nào. Hãy chạy TransNetV2 trước.")
            return

        # ==========================================
        # 1. LỌC RA DANH SÁCH VIDEO CHƯA LÀM
        # ==========================================
        pending_files = []
        for f in structure_files:
            video_name = f.stem.replace("_structure", "")
            stage_a_path = JSON_ROOT / f"{video_name}_stage_a.json"
            if not stage_a_path.exists():
                pending_files.append((f, video_name, stage_a_path))

        if not pending_files:
            logger.info("✅ Toàn bộ Giai đoạn A đã được hoàn tất!")
            return

        logger.info(f"📁 Còn {len(pending_files)} video cần xử lý Stage A (trên tổng số {len(structure_files)}).")

        # ==========================================
        # 2. THIẾT LẬP NGƯỠNG TỰ XẢ RAM (Chống Swap)
        # ==========================================
        MAX_VIDEOS_PER_RUN = 15 # Cứ làm 20 video là trả lại RAM cho hệ điều hành
        processed_count = 0

        for i, (meta_file, video_name, stage_a_path) in enumerate(pending_files):
            found_videos = (
                list(RAW_ROOT.rglob(f"{video_name}.mp4")) +
                list(RAW_ROOT.rglob(f"{video_name}.avi"))
            )
            if not found_videos:
                logger.warning(f"⚠️ Không tìm thấy video gốc cho {video_name}. Bỏ qua.")
                continue
            video_path = str(found_videos[0])

            with open(meta_file, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)

            video_fps = structure_data.get("fps", "unknown")
            shots_data = structure_data.get("shots", [])

            logger.info(
                f"⏳ [{i+1}/{len(pending_files)}] {video_name} | FPS={video_fps} | "
                f"{len(shots_data)} shots | Đang cào Audio & OCR..."
            )

            start_time = time.time()
            stage_a_shots = []

            for shot in shots_data:
                start_ts  = shot["start_ts"]
                end_ts    = shot["end_ts"]
                keyframes = shot.get("keyframes", [])

                future_audio  = self._executor.submit(
                    self.asr.transcribe_shot, video_path, start_ts, end_ts
                )
                future_vision = self._executor.submit(
                    self._extract_ocr_for_shot, keyframes, video_name
                )

                audio_text = future_audio.result()
                ocr_text   = future_vision.result()

                shot["extracted_data"] = {
                    "transcript": audio_text,
                    "ocr_text":   ocr_text
                }
                stage_a_shots.append(shot)

            # Lưu file xuống ổ cứng ngay lập tức
            structure_data["shots"] = stage_a_shots
            with open(stage_a_path, 'w', encoding='utf-8') as f:
                json.dump(structure_data, f, ensure_ascii=False, indent=4)

            process_time = time.time() - start_time
            logger.info(f"✅ Xong {video_name} trong {process_time:.1f}s")

            # ==========================================
            # 3. KÍCH HOẠT DỌN RÁC CƯỠNG CHẾ
            # ==========================================
            del structure_data
            del stage_a_shots
            del shots_data
            gc.collect()

            processed_count += 1
            if processed_count >= MAX_VIDEOS_PER_RUN:
                logger.warning(f"🔄 Đạt giới hạn {MAX_VIDEOS_PER_RUN} video. Tiến hành 'Ve sầu thoát xác' để xả 100% RAM...")
                self._executor.shutdown(wait=False) # Đóng threadpool an toàn
                # Gọi hệ điều hành tạo process mới đè lên process cũ
                os.execv(sys.executable, [sys.executable, "-m", "src.data_pipeline.ingestion_stage_a"])

if __name__ == "__main__":
    StageAExtractor().run()