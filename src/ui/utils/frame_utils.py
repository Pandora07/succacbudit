"""
MODULE TIỆN ÍCH FRAME (UI UTILS)
Vị trí: src/ui/utils/frame_utils.py
Chức năng: Quản lý logic tính toán frame_idx và thuật toán Padding 100 frames.
"""

def get_exact_frame_idx(meta: dict) -> int:
    """Lấy frame_idx CHÍNH XÁC của một shot/kết quả từ Qdrant."""
    if not meta:
        return 0

    if meta.get("frame_idx") is not None:
        return int(meta["frame_idx"])

    keyframes = meta.get("keyframes") or []
    if keyframes:
        return int(keyframes[0].get("frame_idx", 0))

    fps = meta.get("fps", 25.0)
    mid_ts = (meta.get("start_ts", 0.0) + meta.get("end_ts", 0.0)) / 2
    return round(mid_ts * fps)


def generate_padded_frames(anchor_vid: str, anchor_frame_idx: int, semantic_results: list, answer: str = None, max_frames: int = 100) -> list:
    cart_items = []
    added_signatures = set()

    def add_to_cart(vid: str, f_idx: int):
        # Chống frame âm
        if f_idx < 0: return
        
        sig = f"{vid}_{f_idx}"
        if sig not in added_signatures and len(cart_items) < max_frames:
            item = {"video_id": vid, "frame_idx": int(f_idx)}
            if answer is not None:
                item["answer"] = answer
                
            cart_items.append(item)
            added_signatures.add(sig)

    # ====================================================
    # CHIẾN LƯỢC 4 TIERS (FRONT-LOADING TỐI ƯU R@k)
    # ====================================================
    
    # TIER 1 (1 slot): Anchor tuyệt đối chính xác
    add_to_cart(anchor_vid, anchor_frame_idx)

    # TIER 2 (50 slots): Dense Cận chiến (±25 frames, step=1)
    # Bắt mọi cử động nhỏ xung quanh mỏ neo (±1 giây)
    for offset in range(1, 26):
        add_to_cart(anchor_vid, anchor_frame_idx - offset)
        add_to_cart(anchor_vid, anchor_frame_idx + offset)

    # TIER 3 (20 slots): Sparse Văng lưới xa (±26 đến 75 frames, step=5)
    # Dành cho trường hợp AI cắt shot bị trễ nhịp (±3 giây)
    for offset in range(26, 76, 5):
        add_to_cart(anchor_vid, anchor_frame_idx - offset)
        add_to_cart(anchor_vid, anchor_frame_idx + offset)

    # TIER 4 (~29 slots + Slot dư từ Edge Case): Semantic Backup
    # Lấy các góc máy khác, video khác từ Qdrant để tạo lưới an toàn chống sai bối cảnh
    for res in semantic_results:
        if len(cart_items) >= max_frames:
            break 
            
        a_meta = res.get("metadata", {})
        a_vid = res.get("video_id")
        a_frame = get_exact_frame_idx(a_meta)
        add_to_cart(a_vid, a_frame)

    return cart_items