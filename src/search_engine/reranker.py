"""
MODULE 3: BỘ CHẤM ĐIỂM LẠI BẰNG THỊ GIÁC (POST-RETRIEVAL)
Vị trí: src/search_engine/reranker.py
"""
import os
from src.config.common import get_logger

logger = get_logger(__name__)

class VLMReranker:
    def __init__(self, vlm_engine):
        self.vlm = vlm_engine

    def rerank(self, search_results: list, target_scenario: str) -> list:
        if not search_results: return []
        
        logger.info(f"👁️ [Reranker] VLM đang soi lại Top {len(search_results)} khung hình...")
        reranked_results = []
        
        # BẢN VÁ: Lấy điểm max của Qdrant để chuẩn hóa (vì search_results đã sort giảm dần)
        max_qdrant_score = search_results[0]["score"] if search_results else 1.0
        if max_qdrant_score == 0: max_qdrant_score = 1.0 

        for i, res in enumerate(search_results):
            img_path = res.get("thumbnail_url")
            if not img_path or not os.path.exists(img_path):
                reranked_results.append(res)
                continue
                
            prompt = f"""Act as a strict video judge. Does this image clearly show the following scenario?
Scenario: "{target_scenario}"

Respond ONLY with a JSON object:
{{
    "match": true or false,
    "confidence_score": 0 to 100,
    "reason": "1 short sentence explaining why"
}}"""
            try:
                evaluation = self.vlm.generate_json_robust(image_paths=[img_path], prompt=prompt)
                
                # BẢN VÁ: Chuẩn hóa Qdrant score
                normalized_qdrant = res["score"] / max_qdrant_score
                vlm_score = evaluation.get("confidence_score", 0) / 100.0
                is_match = evaluation.get("match", False)
                
                penalty = 1.0 if is_match else 0.1
                # Blend công bằng: 40% Qdrant (đã scale lên 1) + 60% VLM
                res["score"] = round((normalized_qdrant * 0.4 + vlm_score * 0.6) * penalty, 4)
                res["vlm_reason"] = evaluation.get("reason", "")
                
            except Exception as e:
                logger.error(f"Lỗi VLM Re-rank tại ảnh {i}: {e}")
                
            reranked_results.append(res)

        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results