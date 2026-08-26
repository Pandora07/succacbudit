"""
QUẢN LÝ DATABASE QDRANT - HYBRID SEARCH (DEMO 3)
Vị trí: src/db/qdrant_manager.py
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config.common import get_logger
from src.config.config import QDRANT_HOST, QDRANT_PORT

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

        logger.info("🔨 Đang xây dựng cấu trúc Collection Hybrid mới...")
        self.client.create_collection(
            collection_name=self.collection_name,
            # 1. Cấu hình 2 trục Dense Vector
            vectors_config={
                "image_dense": models.VectorParams(
                    size=1024, # SigLIP size
                    distance=models.Distance.COSINE
                ),
                "text_dense": models.VectorParams(
                    size=384,  # MiniLM size
                    distance=models.Distance.COSINE
                )
            },
            # 2. Cấu hình trục Sparse Vector (BM25/Splade)
            sparse_vectors_config={
                "text_sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF # Áp dụng thuật toán Inverse Document Frequency
                )
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