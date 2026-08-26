import json
from pathlib import Path

BASE_DIR = Path.cwd()
DEMO3_DIR = BASE_DIR / "data" / "json"
DEMO2_DIR = BASE_DIR / "demo2_json"

# Chọn 1 video bị 0 shots từ log của bạn để khám nghiệm
TARGET_VIDEO = "L26_V060"

def inspect_ghost_video():
    demo2_files = list(DEMO2_DIR.glob(f"{TARGET_VIDEO}*.json"))
    demo3_files = list(DEMO3_DIR.glob(f"{TARGET_VIDEO}*_stage_a.json"))

    if not demo2_files or not demo3_files:
        print(f"❌ Không tìm thấy file JSON của {TARGET_VIDEO} ở cả 2 bên để so sánh.")
        return

    print(f"🔍 BÁO CÁO GIÁM ĐỊNH VIDEO: {TARGET_VIDEO}\n")

    # Đọc file Demo 2
    with open(demo2_files[0], 'r', encoding='utf-8') as f:
        d2_data = json.load(f)
        d2_shots = d2_data.get("shots", [])

    # Đọc file Demo 3
    with open(demo3_files[0], 'r', encoding='utf-8') as f:
        d3_data = json.load(f)
        d3_shots = d3_data.get("shots", [])

    print(f"📊 SỐ LƯỢNG CẮT CẢNH (SHOTS):")
    print(f"   - Demo 2 đếm được: {len(d2_shots)} shots")
    print(f"   - Demo 3 đếm được: {len(d3_shots)} shots")
    print("-" * 50)

    if not d2_shots or not d3_shots:
        print("⚠️ Kết luận: Một trong hai file bị rỗng (không có mảng 'shots').")
        return

    # Lấy thử Shot đầu tiên của cả 2 bên ra so sánh
    s2 = d2_shots[0]
    s3 = d3_shots[0]

    print(f"🆔 ĐỊNH DẠNG SHOT ID (Có bị lệch pha không?):")
    print(f"   - Demo 2 ID : '{s2.get('shot_id')}'")
    print(f"   - Demo 3 ID : '{s3.get('shot_id')}'")
    
    if s2.get('shot_id') != s3.get('shot_id'):
        print("   👉 KẾT LUẬN 1: MÃ SHOT BỊ LỆCH! (Demo 2 và Demo 3 cắt cảnh khác nhau, nên tool ghép không khớp được).")
    print("-" * 50)

    print(f"📦 DỮ LIỆU BÊN TRONG DEMO 2 (Shot đầu tiên):")
    extracted = s2.get("extracted_data", {})
    if not extracted:
        print("   - Block 'extracted_data': KHÔNG TỒN TẠI (Hoàn toàn rỗng)")
        print("   👉 KẾT LUẬN 2: DỮ LIỆU DEMO 2 VỐN DĨ CHƯA CHẠY XONG! Máy tính đợt trước đã bỏ qua video này.")
    else:
        vlm = extracted.get("vlm_caption")
        print(f"   - Block 'extracted_data': Tồn tại")
        print(f"   - Trường 'vlm_caption'  : {vlm if vlm else 'RỖNG/KHÔNG CÓ'}")
        
        if not vlm:
            print("   👉 KẾT LUẬN 3: CÓ BLOCK DATA NHƯNG VLM CAPTION BỊ TRỐNG. (Model AI đợt trước bị lỗi ngắt quãng ở video này).")
        elif s2.get('shot_id') == s3.get('shot_id'):
            print("   👉 KẾT LUẬN 4: DỮ LIỆU CÓ ĐỦ, MÃ SHOT KHỚP, NHƯNG DO CẤU TRÚC KEY JSON LẠ ĐỜI. (Cần chụp ảnh đoạn log này cho tôi xem).")

if __name__ == "__main__":
    inspect_ghost_video()