"""
TRẠM ĐIỀU KHIỂN TRUNG TÂM (SYSTEM RUNNER)
Chức năng: Quản lý toàn bộ tiến trình Ingest và Serve của hệ thống Demo 3.
Vị trí: Đặt tại thư mục gốc của dự án (ngang hàng src/)
"""

import argparse
import subprocess
import sys
import time
from src.config.common import get_logger

logger = get_logger("SystemRunner")

def run_ingest(force_reset: bool = False):
    """Chạy toàn bộ dây chuyền Ingest từ A đến C."""
    logger.info("🚀 BẮT ĐẦU DÂY CHUYỀN INGESTION TOÀN DIỆN")
    if force_reset:
        logger.warning("⚠️  force_reset=True: Stage C sẽ XÓA SẠCH và tạo lại Collection Qdrant.")
    
    stages = [
        ("GIAI ĐOẠN A (Audio & OCR)", "src.data_pipeline.ingestion_stage_a", []),
        ("GIAI ĐOẠN B (VLM Dense Captioning)", "src.data_pipeline.ingestion_stage_b", []),
        ("GIAI ĐOẠN C (Mã hóa Vector & Bơm Qdrant)", "src.data_pipeline.ingestion_stage_c",
         ["--force-reset"] if force_reset else []),
    ]

    for name, module, extra_args in stages:
        logger.info("=" * 60)
        logger.info(f"⏳ ĐANG KHỞI CHẠY: {name}")
        logger.info("=" * 60)
        
        # Dùng subprocess để chạy như một process độc lập, giúp giải phóng sạch RAM sau khi xong
        result = subprocess.run([sys.executable, "-m", module] + extra_args)
        
        if result.returncode != 0:
            logger.error(f"❌ SỰ CỐ TẠI {name}! Tiến trình buộc phải dừng lại.")
            sys.exit(1)
            
        logger.info(f"✅ ĐÃ HOÀN THÀNH {name}!\n")
        time.sleep(2) # Nghỉ 2 giây để hệ điều hành dọn dẹp RAM

    logger.info("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH INGESTION! HỆ THỐNG ĐÃ SẴN SÀNG ĐỂ TÌM KIẾM.")

def run_serve():
    """Khởi động đồng thời cả Backend API và Frontend Streamlit."""
    logger.info("🚀 KHỞI ĐỘNG HỆ THỐNG PHỤC VỤ (ONLINE SERVING)")
    
    # 1. Bật FastAPI (Chạy ngầm)
    logger.info("⚙️ Đang khởi động Backend API Server (Port 8000)...")
    api_process = subprocess.Popen([sys.executable, "-m", "src.api.server"])
    
    # Cho API 10 giây để nạp model (Qwen-VL, SigLIP, MiniLM) lên RAM/VRAM
    logger.info("⏳ Vui lòng đợi khoảng 10 giây để nạp các mô hình AI lên GPU...")
    time.sleep(10) 
    
    # 2. Bật Streamlit (Hiển thị UI)
    logger.info("🖥️ Đang khởi động Giao diện Streamlit...")
    ui_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src.ui/app.py"])
    
    try:
        # Giữ cho script chạy liên tục để giám sát 2 process kia
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        logger.info("🛑 Nhận lệnh tắt. Đang dọn dẹp hệ thống...")
        api_process.terminate()
        ui_process.terminate()
        logger.info("✅ Đã tắt an toàn.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trạm Điều Khiển Hệ Thống AI Video Search (Datathon)")
    parser.add_argument(
        "--mode", 
        choices=["ingest", "serve"], 
        required=True, 
        help="Chọn chế độ: 'ingest' (Quét dữ liệu) hoặc 'serve' (Mở giao diện UI)"
    )
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="(Chỉ dùng với --mode ingest) Xóa sạch và tạo lại Qdrant Collection từ đầu."
    )
    
    args = parser.parse_args()

    if args.mode == "ingest":
        # CHÚ Ý: Đảm bảo Docker Qdrant đã bật trước khi chạy cái này (cho Stage C)
        run_ingest(force_reset=args.force_reset)
    elif args.mode == "serve":
        # CHÚ Ý: Đảm bảo Docker Qdrant đang bật
        run_serve()