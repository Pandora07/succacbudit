"""
TRẠM ĐIỀU PHỐI TÌM KIẾM (SEARCH PIPELINE)
Vị trí: src/search_engine/pipeline.py
"""
from src.config.common import get_logger
from src.search_engine.retriever import HybridRetriever
from src.search_engine.query_parser import QueryParser
from src.search_engine.reranker import VLMReranker
from src.search_engine.trake_matcher import TrakeMatcher

logger = get_logger(__name__)

class SearchPipeline:
    def __init__(self, vlm_engine):
        self.retriever = HybridRetriever()
        self.parser = QueryParser(vlm_engine)
        self.reranker = VLMReranker(vlm_engine)
        self.trake_matcher = TrakeMatcher(self.retriever)

    def run(self, raw_query: str, top_k: int = 60, top_n: int = 20, skip_vlm: bool = False, target_video_id: str = None, task_type: str = "kis", liked_shot_id: str = None) -> list:
        # Bước 1: Phân tách Query (Pre-retrieval)
        # ==========================================
        # ĐỊNH TUYẾN KHẨN CẤP: SEMANTIC BACKUP
        # ==========================================
        if liked_shot_id:
            logger.info(f"🎯 Kích hoạt lưới dự phòng: Tìm ảnh giống với shot {liked_shot_id}")
            return self.retriever.search_similar(liked_shot_id, top_k=top_k)
        
        if not skip_vlm:
            parsed = self.parser.parse(raw_query, task_type=task_type)
        else:
            # Fallback nếu bypass VLM
            parsed = {
                "task_type": task_type,
                "dense_query": raw_query,
                "sparse_query": raw_query,
                "qa_target": "",
                "temporal_sequence": [raw_query] if task_type == "trake" else []
            }

        # ==========================================
        # LUỒNG ĐẶC NHIỆM TRAKE
        # ==========================================
        if task_type == "trake" and parsed.get("temporal_sequence"):
            logger.info("🛤️ Định tuyến sang nhánh xử lý TRAKE...")
            
            sequence = parsed["temporal_sequence"]
            # Nếu bypass VLM (hoặc VLM gãy), nó chỉ có 1 phần tử là raw_query
            if len(sequence) == 1:
                # Nếu chỉ có 1 hành động, tìm kiếm bình thường
                raw_results = self.retriever.search(parsed["dense_query"], parsed["sparse_query"], top_k=top_k)
            else:
                # Nếu có >= 2 hành động, kích hoạt Thuật toán Ghép Chuỗi
                raw_results = self.trake_matcher.match_sequence(sequence, top_k=top_k)
                
        # ==========================================
        # LUỒNG TIÊU CHUẨN (KIS & QA)
        # ==========================================
        else:
            raw_results = self.retriever.search(
                dense_query=parsed["dense_query"], 
                sparse_query=parsed["sparse_query"], 
                top_k=top_k
            )
        
        # Bước 2.5: Lọc theo Video ID (Dành cho chức năng Mini-search trong UI)
        if target_video_id:
            raw_results = [r for r in raw_results if r["video_id"] == target_video_id]

        candidates = raw_results[:top_n]

        # Bước 3: Hậu xử lý Re-ranking
        if not skip_vlm and candidates:
            # Nếu là TRAKE, không cần Rerank lại nguyên video (vì TrakeMatcher đã chọn lọc rất kỹ rồi)
            if task_type == "trake":
                final_results = candidates
            else:
                # Nếu là QA, nhúng qa_target vào target_scenario để VLM soi đúng chỗ
                target_scen = f"{parsed['dense_query']}. Focus on: {parsed['qa_target']}" if task_type == "qa" else parsed['dense_query']
                final_results = self.reranker.rerank(candidates, target_scenario=target_scen)
        else:
            final_results = candidates

        return final_results