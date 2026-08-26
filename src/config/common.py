"""
Module tiện ích dùng chung (Đã sửa lỗi định tuyến đường dẫn gốc).
Vị trí: src/config/common.py
"""

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

# 🎯 ĐỊNH TUYẾN CHUẨN XÁC: 
# Vì file này nằm ở src/config/common.py, ta cần lùi 2 cấp (.parent.parent) 
# để về đúng thư mục gốc của toàn bộ repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Thư mục lưu log sẽ nằm gọn gàng ở <repo>/logs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Tạo logger chuẩn với định dạng nhất quán cho toàn bộ dự án Demo 2."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def get_env(name: str, default: str | None = None) -> str | None:
    """Đọc biến môi trường với fallback mặc định."""
    return os.getenv(name, default)