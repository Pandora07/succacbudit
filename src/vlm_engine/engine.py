"""
VLM Engine Local (MLX/Metal GPU)
Vị trí: src/vlm_engine/engine.py
"""

import re
import json
from typing import List, Dict, Any

# Yêu cầu thư viện: pip install mlx-vlm
import mlx_vlm
from mlx_vlm import load, generate

from src.config.common import get_logger
from src.config.config import VLM_INGEST_MODEL, VLM_LOCAL_MODEL

logger = get_logger(__name__)

class VLMEngine:
    def __init__(self, mode: str = "ingest"):
        """
        Khởi tạo VLM Engine chạy hoàn toàn Local qua MLX.
        - mode="ingest": Dùng bản 3B (nhẹ, cực nhanh) để quét dữ liệu hàng loạt.
        - mode="online": Dùng bản 7B (thông minh) để trả lời Q&A hoặc Rerank.
        """
        self.mode = mode
        # Lựa chọn model dựa trên config hệ thống
        self.model_path = VLM_INGEST_MODEL if mode == "ingest" else VLM_LOCAL_MODEL
        
        logger.info(f"🧠 Đang nạp VLM ({self.model_path}) lên Metal GPU. Vui lòng đợi...")
        try:
            self.model, self.processor = load(self.model_path)
            logger.info("✅ Khởi động VLM Engine thành công!")
        except Exception as e:
            logger.error(f"❌ Lỗi nạp mô hình VLM: {e}")
            raise e

    def _clean_json(self, text: str) -> dict:
        import json
        import re

        if not text or not isinstance(text, str):
            return {}

        text = text.strip()

        # 1. Rạch bỏ các thẻ markdown code block (Nếu AI có thói quen bọc code)
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

        # 2. Tìm điểm bắt đầu của JSON
        start_idx = text.find('{')
        if start_idx == -1:
            logger.error(f"❌ VLM không tạo ra JSON (Không có dấu {{): {text[:100]}...")
            return {}

        # Bỏ qua mọi câu chào hỏi đầu dòng, lấy từ dấu { trở đi
        json_str = text[start_idx:]

        # Tìm điểm kết thúc
        end_idx = json_str.rfind('}')
        
        if end_idx != -1:
            # Trường hợp hoàn hảo: Có ngoặc đóng. Cắt bỏ chữ rác ở đuôi (nếu có)
            json_str = json_str[:end_idx + 1]
        else:
            # Trường hợp bị cụt đuôi (Do hết token hoặc AI dừng đột ngột)
            logger.warning(f"⚠️ VLM bị ngắt cụt đuôi (Chiều dài: {len(text)}). Đang kích hoạt tự động vá lỗi...")
            
            # Vá 1: Nếu AI đang viết dở 1 chuỗi string (số lượng ngoặc kép bị lẻ), đóng chuỗi lại
            if json_str.count('"') % 2 != 0:
                json_str += '"'
            
            # Vá 2: Ép đóng JSON
            json_str += '}'

        # 3. Hàm nội bộ dùng để Cố gắng Parse và Xử lý lỗi cú pháp nhỏ
        def attempt_parse(s: str) -> dict:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                # Chữa lỗi 1: Cắt dấu phẩy thừa ở phần tử cuối cùng (VD: {"a": 1,})
                s_fixed = re.sub(r',\s*([\]}])', r'\1', s)
                try:
                    return json.loads(s_fixed)
                except json.JSONDecodeError:
                    # Chữa lỗi 2: Dùng AST của Python đọc như một Dictionary
                    try:
                        import ast
                        py_dict_str = s_fixed.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                        return ast.literal_eval(py_dict_str)
                    except Exception:
                        return None

        # 4. Thực thi Parse
        result = attempt_parse(json_str)
        if result is not None:
            return result
            
        # 5. Cứu hộ cấp độ cao: Nếu bị ngắt ở giữa một Array, ngoặc } chưa đủ đô, phải vá bằng ]}
        json_str_hard_fix = json_str.rstrip('}') + ']}' 
        result_hard = attempt_parse(json_str_hard_fix)
        if result_hard is not None:
            return result_hard

        # Nếu mọi nỗ lực đều thất bại, in ra phần đuôi để con người kiểm tra
        tail_text = json_str[-150:] if len(json_str) > 150 else json_str
        logger.error(f"❌ JSON hỏng nặng không thể cứu chữa. Đuôi text: ...{tail_text}")
        return {}
        
    def generate(self, image_paths: List[str], prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """
        Hàm generate cốt lõi hỗ trợ đa ảnh (Multi-image) và văn bản thuần (Text-only) cho Qwen-VL.
        """
        # ĐÃ XÓA KHỐI GUARD LỖI: `if not image_paths: return ""`

        # 1. Định dạng cấu trúc hội thoại chuẩn cho Qwen-VL
        content = []
        
        # Chỉ thêm tag ảnh nếu mảng image_paths thực sự có phần tử
        if image_paths:
            for _ in image_paths:
                content.append({"type": "image"})
        
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        
        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            # 3. Kích hoạt suy luận trên GPU
            # Truyền None cho tham số image nếu là text-only mode
            response = generate(
                self.model,
                self.processor,
                prompt=formatted_prompt,
                image=image_paths if image_paths else None, 
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False
            )
            
            # ==========================================
            # [BẢN VÁ] ÉP KIỂU TRẢ VỀ THÀNH STRING THUẦN
            # ==========================================
            if isinstance(response, str):
                return response
            elif hasattr(response, 'text'):
                return response.text
            elif isinstance(response, dict) and "text" in response:
                return response["text"]
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"❌ Lỗi sinh text từ MLX VLM: {e}")
            return ""

    def generate_json_robust(self, image_paths: List[str], prompt: str, retries: int = 2, temperature: float = 0.1, is_text_only: bool = False, max_tokens: int = 2048) -> dict:
        for attempt in range(retries):
            try:
                # Truyền max_tokens xuống hàm sinh text
                raw_response = self.generate(image_paths, prompt, temperature=temperature, max_tokens=max_tokens)
                
                parsed_dict = self._clean_json(raw_response)
                
                # Tách riêng Validation cho 2 Mode
                if is_text_only:
                    if parsed_dict.get("english_translation") or parsed_dict.get("temporal_sequence"):
                        return parsed_dict
                else:
                    if parsed_dict.get("detailed_description") or parsed_dict.get("objects_and_counts") or parsed_dict.get("match") is not None:
                        return parsed_dict
                    
                logger.warning(f"Thử nghiệm {attempt+1}: JSON thiếu trường trọng yếu, thử lại...")
            except Exception as e:
                logger.error(f"Lỗi Robust JSON: {e}")
                
        return {"error": "Phân tích thất bại sau nhiều lần thử."}