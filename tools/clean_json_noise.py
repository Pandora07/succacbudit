"""
CÔNG CỤ DỌN RÁC JSON (BẢN BẢO MẬT TUYỆT ĐỐI - BACKUP & LOGGING)
"""
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

# Cấu hình đường dẫn
JSON_ROOT = Path("data/json")
BACKUP_ROOT = Path("data/json_backup")
LOG_ROOT = Path("logs")

# Danh sách cụm từ hệ thống
SYSTEM_PHRASES = [
    "we'll be right back", "we will be right back", 
    "thanks for watching", "thank you for watching",
    "subtitles by", "amara.org", "nhạc nền"
]

def clean_ocr_text(text: str) -> str:
    if not text: return ""
    clean_str = re.sub(r'[^\w\s\.,\-\'\"/:;#@&%+]', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', clean_str).strip()

def clean_transcript(text: str, prev_text: str) -> str:
    if not text: return ""
    text_lower = text.lower().strip()
    
    words = text_lower.split()
    if len(words) <= 10 and text_lower == prev_text.lower().strip(): return ""
    if text_lower.startswith("[") and text_lower.endswith("]"): return ""
    if text_lower.startswith("(") and text_lower.endswith(")"): return ""
    for phrase in SYSTEM_PHRASES:
        if phrase in text_lower: return ""
    if len(words) <= 2 and ("music" in text_lower or "nhạc" in text_lower): return ""

    return text

def run_scrubber():
    target_files = list(JSON_ROOT.rglob("*_stage_a.json")) + list(JSON_ROOT.rglob("*_rich_metadata.json"))
    if not target_files:
        print("❌ Không tìm thấy file JSON nào.")
        return

    # 1. KHỞI TẠO THƯ MỤC BACKUP & LOG BẰNG THỜI GIAN THỰC
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_backup_dir = BACKUP_ROOT / timestamp
    current_backup_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    archive_log = []
    total_asr = 0
    total_ocr = 0

    print(f"🧹 Bắt đầu rà soát {len(target_files)} file JSON...")

    for file_path in target_files:
        # 2. COPY BẢN GỐC SANG THƯ MỤC BACKUP TRƯỚC KHI MỞ
        backup_path = current_backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shots = data.get("shots", [])
        prev_transcript = ""
        file_changed = False

        for shot in shots:
            shot_id = shot.get("shot_id", "unknown")
            extracted = shot.get("extracted_data", {})
            
            # Xử lý ASR
            original_asr = extracted.get("transcript", "")
            if original_asr:
                cleaned_asr = clean_transcript(original_asr, prev_transcript)
                if cleaned_asr != original_asr:
                    # Đưa vào sổ tay báo cáo
                    archive_log.append({
                        "file": file_path.name, "shot_id": shot_id,
                        "type": "ASR", "original": original_asr, "cleaned": cleaned_asr
                    })
                    extracted["transcript"] = cleaned_asr
                    total_asr += 1
                    file_changed = True
                prev_transcript = original_asr 

            # Xử lý OCR
            original_ocr = extracted.get("ocr_text", "")
            if original_ocr:
                cleaned_ocr = clean_ocr_text(original_ocr)
                if cleaned_ocr != original_ocr:
                    archive_log.append({
                        "file": file_path.name, "shot_id": shot_id,
                        "type": "OCR", "original": original_ocr, "cleaned": cleaned_ocr
                    })
                    extracted["ocr_text"] = cleaned_ocr
                    total_ocr += 1
                    file_changed = True

        # 3. CHỈ GHI ĐÈ FILE NẾU THỰC SỰ CÓ RÁC BỊ XÓA
        if file_changed:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            # Nếu file sạch bong từ đầu, xóa bản backup của nó đi cho nhẹ ổ cứng
            backup_path.unlink()

    # 4. XUẤT NHẬT KÝ RA FILE JSON
    if archive_log:
        report_file = LOG_ROOT / f"noise_removal_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(archive_log, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Hoàn tất! Đã triệt tiêu {total_asr} dòng ASR & {total_ocr} dòng OCR.")
        print(f"📁 Bản sao lưu file gốc (chỉ các file bị đổi): {current_backup_dir}")
        print(f"📝 Nhật ký kiểm tra chi tiết: {report_file}")
    else:
        print("✅ Dữ liệu đã sạch bong, không có rác nào được tìm thấy. Đã hủy thư mục backup tạm.")
        current_backup_dir.rmdir()

if __name__ == "__main__":
    run_scrubber()