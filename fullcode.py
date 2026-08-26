import os
from pathlib import Path

# Thư mục chứa mã nguồn của bạn
SOURCE_DIR = Path("src")
OUTPUT_FILE = "toan_bo_du_an.txt"

# Các thư mục hệ thống không cần gộp
IGNORE_DIRS = {".git", "__pycache__", "venv", "env", ".idea", ".vscode",".pytest_cache", "node_modules",".pt"}
# Các định dạng file cần lấy
ALLOWED_EXTENSIONS = (".py", ".json", ".yaml", ".sh")

def generate_codebase_txt():
    if not SOURCE_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {SOURCE_DIR}!")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(SOURCE_DIR):
            # Lọc bỏ các thư mục ignore
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file.endswith(ALLOWED_EXTENSIONS):
                    file_path = Path(root) / file
                    
                    # Tạo header phân cách rõ ràng cho từng file
                    outfile.write(f"\n{'='*60}\n")
                    outfile.write(f"--- File: {file_path} ---\n")
                    outfile.write(f"{'='*60}\n\n")
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                            outfile.write("\n")
                    except Exception as e:
                        outfile.write(f"// Lỗi không thể đọc file: {e}\n")

    print(f"✅ Đã gộp thành công! Hãy kiểm tra file: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_codebase_txt()