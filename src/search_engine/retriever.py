"""
BỘ TRUY XUẤT HYBRID SEARCH (DENSE + SPARSE + CROSS-MODAL) + WEIGHTED RRF
Vị trí: src/search_engine/retriever.py

Kiến trúc 3 luồng:
  1. text_dense  (MiniLM 384D)  — Ngữ nghĩa câu văn
  2. text_sparse (SPLADE)       — Từ khóa chính xác, OCR, số lượng
  3. image_dense (SigLIP 1024D) — Cross-modal: text query ↔ image embedding (CLIP-style)

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
    RRF_K,
    PREFETCH_MULT_TEXT_DENSE,
    PREFETCH_MULT_TEXT_SPARSE,
    PREFETCH_MULT_IMAGE_DENSE,
    IMAGE_RRF_WEIGHT,
    SIGLIP_MODEL_ID,
)
from src.data_pipeline.vectorizers import TextDenseVectorizer, TextSparseVectorizer

logger = get_logger(__name__)

# Aliases cho tương thích ngược (retriever.py cũ dùng tên này)
PREFETCH_MULTIPLIER_DENSE  = PREFETCH_MULT_TEXT_DENSE
PREFETCH_MULTIPLIER_SPARSE = PREFETCH_MULT_TEXT_SPARSE
PREFETCH_MULTIPLIER_IMAGE  = PREFETCH_MULT_IMAGE_DENSE


class SigLIPTextEncoder:
    """Encode text query bằng SigLIP text encoder để match vào image_dense space."""

    def __init__(self, model_id: str = SIGLIP_MODEL_ID):
        from transformers import AutoProcessor, AutoModel
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🟣 Khởi tạo SigLIP Text Encoder ({model_id}) trên {self.device.upper()}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        """Encode text thành vector 1024D trong cùng không gian với image_dense."""
        try:
            inputs = self.processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # Cùng cách xử lý với ImageDenseVectorizer — unwrap nếu là Output object
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif hasattr(text_features, "last_hidden_state"):
                text_features = text_features.last_hidden_state[:, 0]
            # else: đã là tensor thuần

            # L2 Normalize — cùng chuẩn với image features đã ingest
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"❌ Lỗi SigLIP text encode: {e}")
            return [0.0] * 1024




class HybridRetriever:
    def __init__(self):
        logger.info("🔍 Khởi động Bộ máy Tìm kiếm Hybrid (3-Stream Weighted RRF)...")
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Luồng 1 & 2: Text encoders
        self.text_dense_encoder  = TextDenseVectorizer()
        self.text_sparse_encoder = TextSparseVectorizer()

        # Luồng 3: SigLIP text encoder — cross-modal search vào image_dense space
        self.siglip_text_encoder = SigLIPTextEncoder()

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
        """
        Hybrid search 3-stream với Weighted RRF.

        Args:
            dense_query:  Query đã dịch sang tiếng Anh (cho SigLIP + MiniLM)
            sparse_query: Query bản gốc/từ khóa (cho SPLADE — bắt OCR, tên riêng)
            top_k:        Số kết quả trả về

        Returns:
            List kết quả đã sort theo score giảm dần.
        """
        if not dense_query or not dense_query.strip():
            return []

        # --- Encode 3 query vectors ---
        dense_vec  = self.text_dense_encoder.encode(dense_query)
        siglip_vec = self.siglip_text_encoder.encode(dense_query)
        safe_sparse = sparse_query if sparse_query and sparse_query.strip() else dense_query
        sparse_dict = self.text_sparse_encoder.encode(safe_sparse)

        # --- 3 lần query Qdrant độc lập (để weighted RRF client-side) ---
        try:
            # Stream 1: image_dense (cross-modal, CLIP-style)
            image_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=siglip_vec,
                using="image_dense",
                limit=top_k * PREFETCH_MULT_IMAGE_DENSE,
                with_payload=True,
            )

            # Stream 2: text_dense (ngữ nghĩa câu văn)
            text_dense_res = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                using="text_dense",
                limit=top_k * PREFETCH_MULT_TEXT_DENSE,
                with_payload=True,
            )

            # Stream 3: text_sparse (từ khóa SPLADE)
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

        # --- Weighted RRF: image × IMAGE_RRF_WEIGHT, text × 1.0 ---
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