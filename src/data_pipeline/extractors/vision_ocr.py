import os
from ocrmac import ocrmac
from src.config.common import get_logger
from src.config.config import OCR_CONFIDENCE_THRESHOLD

logger = get_logger(__name__)


class TextExtractor:
    def __init__(self, confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD):
        """Khởi tạo Apple Native OCR chạy trên Neural Engine (0 tốn GPU)."""
        self.conf_thresh = confidence_threshold
        logger.info(f"👁️ Đã khởi tạo OCR Engine (Confidence ≥ {self.conf_thresh})")

    def extract_text(self, image_path: str) -> str:
        """Đọc chữ trên ảnh, lọc rác và sắp xếp theo chiều đọc của con người."""
        if not os.path.exists(image_path):
            return ""

        try:
            # Gọi API Native của Mac (Cực nhanh, chạy trên Apple Neural Engine)
            annotations = ocrmac.OCR(image_path).recognize()
            if not annotations:
                return ""

            valid_texts = []
            for text, confidence, bbox in annotations:
                # MÀNG LỌC: Bỏ qua chữ quá mờ hoặc rác
                if confidence < self.conf_thresh:
                    continue

                # ocrmac bbox format: [x, y, w, h] (tọa độ gốc ở góc dưới bên trái)
                x, y = bbox[0], bbox[1]
                valid_texts.append({
                    "text": text.strip(),
                    "x": x,
                    "y": y
                })

            if not valid_texts:
                return ""

            # SẮP XẾP KHÔNG GIAN: Đọc từ trên xuống dưới (y giảm dần), trái sang phải (x tăng dần)
            # Lưu ý: Hệ tọa độ của Vision framework có trục y hướng lên.
            valid_texts.sort(key=lambda item: (-item["y"], item["x"]))

            # Nối các chữ lại thành câu hoàn chỉnh
            return " ".join([item["text"] for item in valid_texts])

        except Exception as e:
            logger.error(f"Lỗi ocrmac tại ảnh {os.path.basename(image_path)}: {e}")
            return ""