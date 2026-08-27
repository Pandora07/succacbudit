"""
BỘ TRUY XUẤT HYBRID SEARCH (DENSE + SPARSE + CROSS-MODAL) + WEIGHTED RRF
Vị trí: src/search_engine/retriever.py

Kiến trúc 3 luồng:
  1. text_dense  (MiniLM 384D)  — Ngữ nghĩa câu văn
  2. text_sparse (SPLADE)       — Từ khóa chính xác, OCR, số lượng
  3. image_dense (SigLIP 768D)  — Cross-modal: text query ↔ image embedding (CLIP-style)

Weighted RRF (ưu tiên ảnh):
  - image_dense nhận trọng số IMAGE_RRF_WEIGHT (mặc định 2.0) → gấp đôi text streams
  - Công thức: score = Σ (weight_i / (RRF_K + rank_i))
  - Fusion thực hiện client-side để kiểm soát trọng số chính xác
"""

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config.common import get_logger
from src.config.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    SIGLIP_MODEL_ID,
    RRF_K, PREFETCH_MULT_TEXT_DENSE, PREFETCH_MULT_TEXT_SPARSE,
    PREFETCH_MULT_IMAGE_DENSE, IMAGE_RRF_WEIGHT,
)
# Nhập đủ 3 Động cơ từ vectorizers.py
from src.data_pipeline.vectorizers import TextDenseVectorizer, TextSparseVectorizer, SiglipEncoder

logger = get_logger(__name__)

# Aliases cho tương thích ngược (retriever.py cũ dùng tên này)
PREFETCH_MULTIPLIER_DENSE  = PREFETCH_MULT_TEXT_DENSE
PREFETCH_MULTIPLIER_SPARSE = PREFETCH_MULT_TEXT_SPARSE
PREFETCH_MULTIPLIER_IMAGE  = PREFETCH_MULT_IMAGE_DENSE


class HybridRetriever:
    def __init__(self):
        logger.info("🔍 Khởi động Bộ máy Tìm kiếm Hybrid (3-Stream Weighted RRF)...")
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        self.text_dense_encoder  = TextDenseVectorizer()
        self.text_sparse_encoder = TextSparseVectorizer()
        self.siglip_encoder      = SiglipEncoder()

        logger.info(
            f"✅ Hybrid Retriever sẵn sàng! "
            f"[image×{IMAGE_RRF_WEIGHT} | text_dense×1.0 | text_sparse×1.0] | RRF_K={RRF_K}"
        )

    # ==========================================
    # WEIGHTED RRF (CLIENT-SIDE)
    # ==========================================
    def _weighted_rrf(
        self,
        stream_results: list,   # List of Qdrant QueryResponse objects
        weights: list[float],   # Trọng số tương ứng với mỗi stream
        top_k: int,
    ) -> list[dict]:
        """
        Thực hiện Weighted Reciprocal Rank Fusion client-side.

        Với mỗi stream i có trọng số w_i:
          score(point) += w_i / (RRF_K + rank_i)

        Ưu điểm so với Qdrant built-in RRF:
          - Kiểm soát trọng số từng stream (ưu tiên image_dense)
          - Kết quả nhất quán và có thể tuỳ chỉnh
        """
        score_map: dict[str, dict] = {}

        for stream_res, weight in zip(stream_results, weights):
            points = stream_res.points if hasattr(stream_res, "points") else stream_res
            for rank, point in enumerate(points):
                pid = str(point.id)
                if pid not in score_map:
                    score_map[pid] = {
                        "rrf_score": 0.0,
                        "payload":   point.payload,
                        "id":        pid,
                    }
                score_map[pid]["rrf_score"] += weight / (RRF_K + rank + 1)

        sorted_points = sorted(
            score_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )
        return sorted_points[:top_k]

    def _format_results(self, fused: list[dict]) -> list[dict]:
        """Chuyển kết quả fused sang format chuẩn cho pipeline."""
        results = []
        for item in fused:
            payload = item.get("payload", {})
            results.append({
                "shot_id":       payload.get("shot_id"),
                "video_id":      payload.get("video_id"),
                "score":         round(item["rrf_score"], 6),
                "thumbnail_url": payload.get("thumbnail"),
                "metadata": {
                    "start_ts":  payload.get("start_ts"),
                    "end_ts":    payload.get("end_ts"),
                    "fps":       payload.get("fps", 25.0),
                    "frame_idx": payload.get("frame_idx", 0),
                    "rich_text": payload.get("rich_text"),
                },
            })
        return results

    # ==========================================
    # SEARCH CHÍNH
    # ==========================================
    def search(self, dense_query: str, sparse_query: str, top_k: int = 20) -> list[dict]:
        if not dense_query or not dense_query.strip():
            return []

        # Sinh 3 loại vector từ 3 mô hình khác nhau
        text_dense_vec = self.text_dense_encoder.encode(dense_query)
        siglip_vec     = self.siglip_encoder.encode_text(dense_query)
        
        safe_sparse = sparse_query if sparse_query and sparse_query.strip() else dense_query
        sparse_dict = self.text_sparse_encoder.encode(safe_sparse)

        try:
            # Stream 1: image_dense (Tìm ảnh bằng SigLIP)
            image_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=siglip_vec, # Dùng vector của SigLIP
                using="image_dense",
                limit=top_k * PREFETCH_MULT_IMAGE_DENSE,
                with_payload=True,
            )

            # Stream 2: text_dense (Tìm text bằng MiniLM)
            text_dense_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=text_dense_vec, # Dùng vector của MiniLM
                using="text_dense",
                limit=top_k * PREFETCH_MULT_TEXT_DENSE,
                with_payload=True,
            )

            # Stream 3: text_sparse (Tìm từ khóa bằng SPLADE)
            text_sparse_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=models.SparseVector(
                    indices=sparse_dict["indices"],
                    values=sparse_dict["values"],
                ),
                using="text_sparse",
                limit=top_k * PREFETCH_MULT_TEXT_SPARSE,
                with_payload=True,
            )

        except Exception as e:
            logger.error(f"❌ Lỗi truy vấn Qdrant: {e}")
            return []

        fused = self._weighted_rrf(
            stream_results=[image_res, text_dense_res, text_sparse_res],
            weights=[IMAGE_RRF_WEIGHT, 1.0, 1.0],
            top_k=top_k,
        )
        return self._format_results(fused)

    # ==========================================
    # SIMILAR IMAGE SEARCH (cho tính năng visual anchor)
    # ==========================================
    def search_similar(self, shot_id: str, top_k: int = 100) -> list[dict]:
        """
        Tìm các shot có hình ảnh tương đồng nhất với mỏ neo đã chọn.
        Dùng cho chức năng 'liked_shot' / visual anchor trong UI.
        """
        try:
            response = self.client.recommend(
                collection_name=COLLECTION_NAME,
                positive=[shot_id],
                using="image_dense",
                limit=top_k,
                with_payload=True,
            )

            results = []
            for point in response:
                payload = point.payload
                results.append({
                    "shot_id":       payload.get("shot_id"),
                    "video_id":      payload.get("video_id"),
                    "score":         round(point.score, 4),
                    "thumbnail_url": payload.get("thumbnail"),
                    "metadata": {
                        "start_ts":  payload.get("start_ts"),
                        "end_ts":    payload.get("end_ts"),
                        "fps":       payload.get("fps", 25.0),
                        "frame_idx": payload.get("frame_idx", 0),
                    },
                })
            return results

        except Exception as e:
            logger.error(f"❌ Lỗi tìm kiếm tương đồng (Recommendation): {e}")
            return []