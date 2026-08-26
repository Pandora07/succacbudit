import os
from pathlib import Path

# Cấu hình đường dẫn tới thư mục chứa JSON
BASE_DIR = Path.cwd()
JSON_ROOT = BASE_DIR / "data" / "json"

def clean_unfinished_files():
    if not JSON_ROOT.exists():
        print("❌ Lỗi: Không tìm thấy thư mục 'json_data'!")
        return

    # Quét tìm các file rich_metadata (có thể bị rỗng/lỗi) và file partial (chạy dở)
    rich_files = list(JSON_ROOT.glob("*_rich_metadata.json"))
    partial_files = list(JSON_ROOT.glob("*_stage_b_partial.json"))
    
    all_garbage = rich_files + partial_files
    
    if not all_garbage:
        print("✨ Thư mục json_data đã sạch sẽ! Không có file dở dang nào.")
        return

    print(f"🧹 Phát hiện {len(all_garbage)} file rác/dở dang. Đang tiến hành tiêu hủy...")
    
    deleted_count = 0
    for file_path in all_garbage:
        try:
            file_path.unlink() # Xóa file
            deleted_count += 1
            print(f"   🗑️ Đã xóa: {file_path.name}")
        except Exception as e:
            print(f"   ⚠️ Lỗi không thể xóa {file_path.name}: {e}")

    print(f"\n✅ Dọn dẹp hoàn tất! Đã xóa thành công {deleted_count}/{len(all_garbage)} file.")
    print("👉 Bây giờ bạn có thể yên tâm chạy file 'merge_data_rescue.py' để cấy ghép data rồi!")

if __name__ == "__main__":
    clean_unfinished_files()