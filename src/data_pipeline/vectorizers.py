"""
MODULE VECTORIZERS - ĐỘNG CƠ MÃ HÓA ĐỘC LẬP
Vị trí: src/data_pipeline/vectorizers.py
"""

import torch
from PIL import Image
from fastembed import TextEmbedding, SparseTextEmbedding
from transformers import AutoProcessor, AutoModel
from src.config.common import get_logger
from src.config.config import (
    SIGLIP_MODEL_ID,
    TEXT_DENSE_MODEL,
    TEXT_SPARSE_MODEL,
    DIM_IMAGE_DENSE,
    DIM_TEXT_DENSE,
)

logger = get_logger(__name__)


# ==========================================
# 1. BỘ MÃ HÓA HÌNH ẢNH DENSE (SigLIP)
# ==========================================
class ImageDenseVectorizer:
    def __init__(self, model_id: str = SIGLIP_MODEL_ID):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🟢 Khởi tạo Image Dense Vectorizer ({model_id}) trên {self.device.upper()}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, image_path: str) -> list[float]:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # L2 Normalization — cùng chuẩn với text features của SigLIP
            image_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return image_features.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"Lỗi encode hình ảnh {image_path}: {e}")
            return [0.0] * DIM_IMAGE_DENSE


# ==========================================
# 2. BỘ MÃ HÓA VĂN BẢN DENSE (MiniLM Multilingual)
# ==========================================
class TextDenseVectorizer:
    def __init__(self, model_name: str = TEXT_DENSE_MODEL):
        logger.info(f"🔵 Khởi tạo Text Dense Vectorizer ({model_name}) qua FastEmbed")
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * DIM_TEXT_DENSE
        try:
            return list(self.model.embed([text]))[0].tolist()
        except Exception as e:
            logger.error(f"Lỗi encode Text Dense: {e}")
            return [0.0] * DIM_TEXT_DENSE


# ==========================================
# 3. BỘ MÃ HÓA VĂN BẢN SPARSE (SPLADE)
# ==========================================
class TextSparseVectorizer:
    def __init__(self, model_name: str = TEXT_SPARSE_MODEL):
        logger.info(f"🟠 Khởi tạo Text Sparse Vectorizer ({model_name}) qua FastEmbed")
        self.model = SparseTextEmbedding(model_name=model_name)

    def encode(self, text: str) -> dict:
        if not text or not text.strip():
            return {"indices": [], "values": []}
        try:
            sparse_vector = list(self.model.embed([text]))[0]
            return {
                "indices": sparse_vector.indices.tolist(),
                "values": sparse_vector.values.tolist()
            }
        except Exception as e:
            logger.error(f"Lỗi encode Text Sparse: {e}")
            return {"indices": [], "values": []}


# ==========================================
# 4. TRÌNH ĐIỀU PHỐI LAI (HYBRID FACADE)
# ==========================================
class HybridVectorizer:
    """Class bọc ngoài để Pipeline gọi một lần là lấy được cả 3 Vector."""
    def __init__(self):
        self.image_encoder = ImageDenseVectorizer()
        self.text_dense_encoder = TextDenseVectorizer()
        self.text_sparse_encoder = TextSparseVectorizer()
        logger.info("✅ Hybrid Vectorizer đã tập hợp đủ 3 sức mạnh!")

    def encode_all(self, image_path: str, text_content: str) -> dict:
        return {
            "image_dense": self.image_encoder.encode(image_path),
            "text_dense": self.text_dense_encoder.encode(text_content),
            "text_sparse": self.text_sparse_encoder.encode(text_content)
        }