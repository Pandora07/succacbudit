"""
ĐỘNG CƠ VLM DỰA TRÊN GOOGLE GEMINI CLOUD API
Hỗ trợ: JSON Mode, Tự động bắt lỗi Rate Limit, Đa luồng
"""
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions

logger = logging.getLogger("gemini_engine")
logging.basicConfig(level=logging.INFO)

class GeminiVLMEngine:
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Không tìm thấy GEMINI_API_KEY. Vui lòng export GEMINI_API_KEY hoặc truyền trực tiếp.")
            
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        
        # Cấu hình Ép kiểu JSON 100% từ Server của Google
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        logger.info(f"✅ Khởi tạo thành công Gemini VLM Engine: {self.model_name}")

    def generate_json_robust(
        self, 
        image_paths: List[str], 
        prompt: str, 
        retries: int = 3, 
        is_text_only: bool = False
    ) -> Dict[str, Any]:
        """
        Gửi ảnh và Prompt lên Gemini API, tự động thử lại khi gặp nghẽn mạng hoặc Rate Limit.
        """
        # 1. Nạp và kiểm tra ảnh hợp lệ
        pil_images = []
        if not is_text_only and image_paths:
            for path in image_paths:
                try:
                    if os.path.exists(path):
                        img = Image.open(path)
                        pil_images.append(img)
                except Exception as e:
                    logger.warning(f"⚠️ Không đọc được ảnh {path}: {e}")

        if not is_text_only and not pil_images:
            return {"error": "Không có ảnh hợp lệ để phân tích."}

        # 2. Chuẩn bị Payload gửi lên API
        contents = pil_images + [prompt] if pil_images else [prompt]

        # 3. Vòng lặp gọi API với Exponential Backoff
        for attempt in range(1, retries + 1):
            try:
                response = self.model.generate_content(contents)
                raw_text = response.text.strip()
                
                # Parse JSON trực tiếp (Google đã đảm bảo chuẩn JSON)
                parsed_data = json.loads(raw_text)
                return parsed_data

            except exceptions.ResourceExhausted:
                wait_time = attempt * 5
                logger.warning(f"⏳ Bị Rate Limit (HTTP 429). Tạm dừng {wait_time}s rồi thử lại (Lần {attempt}/{retries})...")
                time.sleep(wait_time)

            except json.JSONDecodeError as e:
                logger.error(f"❌ Lỗi Parse JSON: {e} | Raw text: {raw_text[:100]}...")
                if attempt == retries:
                    return {"error": f"JSON decode failed: {str(e)}"}
                time.sleep(1)

            except Exception as e:
                logger.error(f"❌ Lỗi Gemini API (Lần {attempt}/{retries}): {e}")
                if attempt == retries:
                    return {"error": str(e)}
                time.sleep(2)

        return {"error": "Thất bại sau nhiều lần thử kết nối."}