"""
QUẢN LÝ DATABASE QDRANT - HYBRID SEARCH (DEMO 3)
Vị trí: src/db/qdrant_manager.py
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config.common import get_logger
from src.config.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    DIM_IMAGE_DENSE,
    DIM_TEXT_DENSE,
)

logger = get_logger(__name__)

# TẠO DATABASE MỚI TOANH CHO DEMO 3
DEMO3_COLLECTION = "demo3_hybrid_v1"

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = DEMO3_COLLECTION
        logger.info(f"🗄️ Đã kết nối tới Qdrant ({QDRANT_HOST}:{QDRANT_PORT})")

    def setup_new_database(self):
        """Khởi tạo cấu trúc Database Hybrid với 3 loại Vector."""
        # Xóa collection cũ nếu tồn tại (để reset làm lại từ đầu)
        if self.client.collection_exists(self.collection_name):
            logger.warning(f"⚠️ Collection '{self.collection_name}' đã tồn tại. Đang xóa để khởi tạo lại...")
            self.client.delete_collection(self.collection_name)

        logger.info(
            f"🔨 Đang xây dựng cấu trúc Collection Hybrid mới "
            f"(image_dense={DIM_IMAGE_DENSE}, text_dense={DIM_TEXT_DENSE})..."
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            # 1. Cấu hình 2 trục Dense Vector
            # image_dense và text_dense giờ cùng ra từ SigLIP (2 tháp của
            # cùng 1 model) nên PHẢI cùng chiều — đọc từ config, không hardcode
            # để tránh lệch khi đổi model.
            vectors_config={
                "image_dense": models.VectorParams(
                    size=DIM_IMAGE_DENSE,
                    distance=models.Distance.COSINE
                ),
                "text_dense": models.VectorParams(
                    size=DIM_TEXT_DENSE,
                    distance=models.Distance.COSINE
                )
            },
            # 2. Cấu hình trục Sparse Vector (SPLADE)
            # Không dùng modifier=IDF: SPLADE đã tự học trọng số quan trọng
            # cho từng token qua log(1+ReLU(logit)) lúc encode, thêm IDF ở
            # đây sẽ nhân đúp 2 lớp "độ hiếm" và làm lệch phân bố đã học.
            sparse_vectors_config={
                "text_sparse": models.SparseVectorParams()
            }
        )
        logger.info(f"✅ Đã tạo thành công Collection '{self.collection_name}'!")

    def upsert_batch(self, points: list):
        """Bơm một lô dữ liệu lên Qdrant."""
        if not points:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )