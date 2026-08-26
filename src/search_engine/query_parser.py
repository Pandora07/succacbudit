"""
MODULE 1: BỘ PHÂN TÍCH VÀ ĐỊNH TUYẾN TRUY VẤN (UNIFIED QUERY PARSER)
Vị trí: src/search_engine/query_parser.py
Chức năng: "Bộ não" đầu vào, dùng VLM (Text) để thấu hiểu, dịch và phân tách
           truy vấn thô thành các trường dữ liệu sạch cho KIS, QA, và TRAKE.
"""
from src.config.common import get_logger

logger = get_logger(__name__)

class QueryParser:
    def __init__(self, vlm_engine):
        self.vlm = vlm_engine

    def parse(self, raw_query: str, task_type: str = "kis") -> dict:
        """
        Phân tách câu truy vấn thô (raw_query) dựa trên ngữ cảnh bài toán (task_type).
        Trả về một Dictionary chuẩn hóa để các module phía sau dễ dàng xử lý.
        """
        logger.info(f"🧠 [Parser] Bắt đầu mổ xẻ truy vấn ({task_type.upper()})...")
        
        # Mặc định cấu trúc trả về an toàn nếu VLM lỗi
        parsed_intent = {
            "task_type": task_type,
            "dense_query": raw_query,
            "sparse_query": raw_query,
            "qa_target": "",
            "temporal_sequence": []
        }

        # Tạo prompt linh hoạt (Dynamic Prompting) dựa trên task_type
        if task_type == "kis":
            prompt = self._build_kis_prompt(raw_query)
        elif task_type == "qa":
            prompt = self._build_qa_prompt(raw_query)
        elif task_type == "trake":
            prompt = self._build_trake_prompt(raw_query)
        else:
            logger.warning(f"⚠️ [Parser] task_type không hợp lệ: {task_type}. Fallback về KIS.")
            prompt = self._build_kis_prompt(raw_query)

        try:
            # Gọi VLM không truyền ảnh (hoạt động như LLM thuần)
            # Nhiệt độ 0.0 để đảm bảo tính logic và độ chính xác tuyệt đối
            extracted = self.vlm.generate_json_robust(
                image_paths=[], 
                prompt=prompt,
                temperature=0.0,
                is_text_only = True
            )

            # --- ÁNH XẠ DỮ LIỆU SẠCH (DATA MAPPING) ---
            parsed_intent["dense_query"] = extracted.get("english_translation", raw_query)
            
            # Gom OCR và Từ khóa hình ảnh làm đạn cho luồng SPLADE (Sparse)
            ocr = extracted.get("ocr_keywords", "")
            visual = extracted.get("visual_keywords", "")
            sparse = f"{ocr} {visual}".strip()
            parsed_intent["sparse_query"] = sparse if sparse else parsed_intent["dense_query"]

            # Đặc nhiệm QA
            if task_type == "qa":
                parsed_intent["qa_target"] = extracted.get("question_target", "")
            
            # Đặc nhiệm TRAKE (Đảm bảo luôn là mảng)
            if task_type == "trake":
                sequence = extracted.get("temporal_sequence", [])
                parsed_intent["temporal_sequence"] = sequence if isinstance(sequence, list) else [sequence]

            logger.info(f"   -> Dịch (Dense): {parsed_intent['dense_query']}")
            logger.info(f"   -> Từ khóa (Sparse): {parsed_intent['sparse_query']}")
            if task_type == 'trake':
                logger.info(f"   -> Mảng Chuỗi Sự kiện: {parsed_intent['temporal_sequence']}")

        except Exception as e:
            logger.error(f"❌ [Parser] VLM phân tách lỗi: {e}. Dùng truy vấn thô gốc.")

        return parsed_intent

    # ==========================================
    # CÁC HÀM XÂY DỰNG PROMPT CHUYÊN BIỆT
    # ==========================================
    def _build_kis_prompt(self, query: str) -> str:
        return f"""You are a query analysis AI for a video search engine. 
Analyze this user query (which may be in Vietnamese) and extract the exact details.

User Query: "{query}"

Return EXACTLY this JSON schema:
{{
    "english_translation": "Direct, highly accurate English translation of the core scene.",
    "ocr_keywords": "Specific numbers, letters, names, or words that MUST appear on screen (e.g., '59X1', 'HTV'). Empty if none.",
    "visual_keywords": "Key visual nouns or distinct colors extracted."
}}"""

    def _build_qa_prompt(self, query: str) -> str:
        return f"""You are an AI preparing a query for a Visual Question Answering system.
Analyze this user question (which may be in Vietnamese) and extract the details.

User Question: "{query}"

Return EXACTLY this JSON schema:
{{
    "english_translation": "Direct English translation of the question.",
    "ocr_keywords": "Specific text or numbers mentioned in the question.",
    "visual_keywords": "Key objects mentioned to help find the correct image.",
    "question_target": "What exact detail is the question asking for? (e.g., 'color of the shirt', 'number of people', 'the text on the sign'). Keep it extremely brief."
}}"""

    def _build_trake_prompt(self, query: str) -> str:
        return f"""You are an AI expert in video temporal event detection.
Analyze this complex chronological query (which may be in Vietnamese) and break it down into sequential steps.

User Query: "{query}"

Return EXACTLY this JSON schema:
{{
    "english_translation": "Summary of the entire sequence in English.",
    "ocr_keywords": "Any text/numbers that must appear.",
    "visual_keywords": "Key actors/objects across the sequence.",
    "temporal_sequence": [
        "Event 1 in English (e.g., 'Lion starts spinning')",
        "Event 2 in English (e.g., 'Lion lands on the pillar')",
        "Event 3 in English (e.g., 'Person hits the cymbal')"
    ]
}}"""