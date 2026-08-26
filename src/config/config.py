"""
Trung tâm cấu hình toàn hệ thống (Demo 3 Architecture).
Vị trí: src/config/config.py

Nguyên tắc: Mọi hằng số có thể cần điều chỉnh đều phải nằm ở đây.
Các stage file chỉ import từ đây, không được tự định nghĩa magic number.
"""

import os
from pathlib import Path
from src.config.common import PROJECT_ROOT

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU (DATA PATHS)
# ==========================================
DATA_ROOT = PROJECT_ROOT / "data"

RAW_ROOT       = DATA_ROOT / "raw"             # File MP4/AVI gốc
PROCESSED_ROOT = DATA_ROOT / "processed"       # Output của TransNetV2
JSON_ROOT      = DATA_ROOT / "json"            # JSON metadata của từng stage
KEYFRAMES_ROOT = PROCESSED_ROOT / "keyframes"  # Keyframe images (fallback)

# ==========================================
# 1b. ĐƯỜNG DẪN DỮ LIỆU AIC-2026 (FOLDER CÓ SẴN)
# Folder keyframes của ban tổ chức — dùng làm nguồn ảnh chính cho Stage C.
# ==========================================
# Keyframe images của L26_V200–V299 (tổng ~100 videos, ~1fps)
AIC_KEYFRAMES_ROOT = Path("/Users/dangphuoctienthu/Desktop/AIC-2026/keyframesL26c")
# JSON metadata frame-level từ ban tổ chức (description, objects, scene_type, ocr_text)
AIC_METADATA_FILE  = Path("/Users/dangphuoctienthu/Desktop/AIC-2026/data/keyframes_L26_V200_V299_metadata.json")

# Khởi tạo sẵn các thư mục để tránh lỗi Not Found
for _path in [RAW_ROOT, PROCESSED_ROOT, JSON_ROOT, KEYFRAMES_ROOT]:
    _path.mkdir(parents=True, exist_ok=True)


def dataset_path(dataset_name: str, kind: str = "raw") -> Path:
    """Trả về đường dẫn dataset an toàn theo kiểu raw/processed."""
    if kind == "raw":
        return RAW_ROOT / dataset_name
    if kind == "processed":
        return PROCESSED_ROOT / dataset_name
    raise ValueError(f"Unsupported dataset kind: {kind}")


# ==========================================
# 2. CẤU HÌNH CƠ SỞ DỮ LIỆU (QDRANT)
# ==========================================
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "demo3_hybrid_v1"


# ==========================================
# 3. CẤU HÌNH CÁC MÔ HÌNH AI (AI ENCODERS)
# ==========================================
# Model IDs
SIGLIP_MODEL_ID     = "google/siglip-base-patch16-224"
TEXT_DENSE_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_SPARSE_MODEL   = "prithivida/Splade_PP_en_v1"
WHISPER_MODEL_PATH  = "mlx-community/whisper-large-v3-mlx"
VLM_LOCAL_MODEL     = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
VLM_INGEST_MODEL    = "mlx-community/Qwen3-VL-4B-Instruct-8bit"
# Để tăng tốc ingest (giảm chất lượng caption), đổi sang:
# VLM_INGEST_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-8bit"

# Chiều vector
DIM_IMAGE_DENSE = 1024  # SigLIP image/text features
DIM_TEXT_DENSE  = 384   # MiniLM multilingual

# FPS mặc định khi metadata không chứa thông tin fps
DEFAULT_FPS = 25.0


# ==========================================
# 4. CẤU HÌNH INGEST PIPELINE (STAGE A)
# ==========================================
# ASR: Bỏ qua shot ngắn hơn ngưỡng này (WAV rỗng → Whisper ra text rác)
MIN_SHOT_DURATION_SEC = 0.5

# Số luồng song song ASR (Metal GPU) + OCR (Apple Neural Engine)
ASR_THREAD_WORKERS = 2

# OCR: Ngưỡng confidence tối thiểu để chấp nhận text
OCR_CONFIDENCE_THRESHOLD = 0.8


# ==========================================
# 5. CẤU HÌNH INGEST PIPELINE (STAGE B — VLM)
# ==========================================
# Lưu checkpoint trung gian sau mỗi N shots (chống mất data khi crash)
VLM_CHECKPOINT_INTERVAL = 10

# Số frames ≥ ngưỡng này → dùng chế độ lấy mẫu 5 điểm (Q0/Q1/Q2/Q3/Q4)
# Dưới ngưỡng → dùng 3 điểm (head/mid/tail)
VLM_LONG_SHOT_FRAME_THRESHOLD = 8


# ==========================================
# 6. CẤU HÌNH INGEST PIPELINE (STAGE C — QDRANT)
# ==========================================
# Số điểm gom lại trước khi upsert một batch lên Qdrant (cân bằng RAM / latency)
QDRANT_UPSERT_BATCH_SIZE = 32


# ==========================================
# 7. CẤU HÌNH RETRIEVER (HYBRID SEARCH)
# ==========================================
# Reciprocal Rank Fusion k-value
# k nhỏ (20) → phạt nặng rank thấp → precision cao (tốt cho KIS)
# k lớn (100) → dàn đều → recall cao (exploratory search)
RRF_K = 60

# Hệ số mở rộng lưới Prefetch trước khi Fusion
PREFETCH_MULT_TEXT_DENSE  = 3   # Ngữ nghĩa câu → cần recall rộng
PREFETCH_MULT_TEXT_SPARSE = 2   # Từ khóa chính xác → precision cao, không cần mở quá rộng
PREFETCH_MULT_IMAGE_DENSE = 5   # Cross-modal (CLIP-style) — ưu tiên ảnh hơn

# Aliases để tương thích với retriever.py (tên cũ)
PREFETCH_MULTIPLIER_DENSE  = PREFETCH_MULT_TEXT_DENSE
PREFETCH_MULTIPLIER_SPARSE = PREFETCH_MULT_TEXT_SPARSE
PREFETCH_MULTIPLIER_IMAGE  = PREFETCH_MULT_IMAGE_DENSE

# Trọng số image trong Weighted RRF (1.0 = bằng nhau, >1.0 = ưu tiên ảnh)
# 2.0 → image stream đóng góp gấp đôi vào final score so với mỗi text stream
IMAGE_RRF_WEIGHT = 2.0


# ==========================================
# 8. CẤU HÌNH UI
# ==========================================
# Số file temp clip tối đa trước khi dọn dẹp
TEMP_CLIPS_MAX_FILES = 200