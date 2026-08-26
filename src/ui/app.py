"""
Giao diện Người dùng (Frontend) - Phiên bản V7 (Modularized & Padded Edition)
Tích hợp: Giỏ hàng 100 Frames (Padding), Module hóa UI, Auto-fill Rocchio.
Vị trí: src/ui/app.py
"""

import os
import time
import streamlit as st
import pandas as pd

# 📦 Import Modules Tiện ích (UI Utils)
from src.ui.utils.frame_utils import get_exact_frame_idx, generate_padded_frames
from src.ui.utils.api_client import call_search_api, call_qa_api
from src.ui.utils.video_utils import extract_single_frame, find_video_path, maybe_cleanup_temp_clips
from src.ui.utils.submission_utils import parse_btc_uploaded_files, create_zip_in_memory

# ==========================================
# CẤU HÌNH TRANG & KHỞI TẠO BỘ NHỚ ẢO
# ==========================================
st.set_page_config(page_title="AI Video Search - Datathon", page_icon="🏆", layout="wide")

# Khởi chạy dọn dẹp rác nền (Xóa bớt temp video/ảnh cũ)
maybe_cleanup_temp_clips()

if "btc_queries" not in st.session_state:
    st.session_state.btc_queries = []
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "liked_shot_id" not in st.session_state:
    st.session_state.liked_shot_id = None

# ==========================================
# THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 1. Quản lý Đề thi")
    uploaded_files = st.file_uploader("Kéo thả các file đề của BTC (.txt)", type=["txt"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("🔄 Xử lý danh sách đề thi"):
            st.session_state.btc_queries = parse_btc_uploaded_files(uploaded_files)
            st.success(f"Đã nạp thành công {len(st.session_state.btc_queries)} câu hỏi!")

    if st.session_state.btc_queries:
        st.markdown("---")
        st.subheader("📊 Tiến độ làm bài")
        total_q = len(st.session_state.btc_queries)
        done_q = len(st.session_state.cart.keys())
        progress = done_q / total_q if total_q > 0 else 0

        st.progress(progress)
        st.write(f"Đã chốt đáp án: **{done_q}/{total_q}** câu")

        if st.button("🗑️ Xóa toàn bộ Giỏ hàng (Làm lại từ đầu)"):
            st.session_state.cart = {}
            st.session_state.liked_shot_id = None
            st.rerun()

# ==========================================
# GIAO DIỆN CHÍNH: 4 TABS CHIẾN LƯỢC
# ==========================================
st.title("🏆 Trạm Chỉ Huy Datathon (AI Multimodal)")
st.markdown("---")

tab_kis, tab_qa, tab_trake, tab_submit = st.tabs([
    "🔍 1. KIS (Truy vấn Hình ảnh)",
    "💬 2. Q&A (Hỏi Đáp)",
    "⏱️ 3. TRAKE (Truy Vết)",
    "📦 4. QUẢN LÝ & XUẤT FILE"
])

# ==========================================
# PHASE 2.1: KIS (TRẠM NỘI SOI & RẢI THẢM 100 FRAMES)
# ==========================================
with tab_kis:
    st.header("🔍 Giải quyết KIS (Truy vấn Hình ảnh)")
    kis_queries = [q for q in st.session_state.btc_queries if q["type"] == "kis"]

    if not kis_queries:
        st.warning("⚠️ Vui lòng Upload file đề thi của BTC ở thanh Sidebar bên trái trước!")
    else:
        query_options = {q["id"]: f"{q['id']}: {q['text']}" for q in kis_queries}
        selected_q_id = st.selectbox("📌 Chọn câu hỏi KIS:", options=list(query_options.keys()), format_func=lambda x: query_options[x])
        selected_q_text = [q["text"] for q in kis_queries if q["id"] == selected_q_id][0]

        # Quản lý State cho KIS
        if "kis_search_results" not in st.session_state:
            st.session_state.kis_search_results = []
        if "kis_active_shot" not in st.session_state:
            st.session_state.kis_active_shot = None

        col1, col2 = st.columns([1, 5])
        with col1:
            kis_fast_mode = st.checkbox("⚡ Siêu tốc (bỏ qua VLM)", key="kis_fast_mode")
            if st.button("🚀 Tìm Top 20", use_container_width=True):
                spinner_msg = "Quét nhanh từ Qdrant..." if kis_fast_mode else "VLM đang chấm điểm... (~15s)"
                with st.spinner(spinner_msg):
                    st.session_state.kis_search_results = call_search_api(
                        query=selected_q_text, top_k=80, top_n=20, skip_vlm=kis_fast_mode
                    )
                st.session_state.kis_active_shot = None

        if selected_q_id in st.session_state.cart:
            st.success(f"🛒 Câu này đã chốt {len(st.session_state.cart[selected_q_id])} frames. Có thể chọn ảnh khác để đè lại.")

        # ==========================================
        # CHẶNG 1: LƯỚI TÌM KIẾM
        # ==========================================
        if st.session_state.kis_search_results and not st.session_state.kis_active_shot:
            st.markdown("---")
            st.markdown("### Lướt và Chọn 1 ảnh để Nội soi")
            cols = st.columns(4)

            for i, res in enumerate(st.session_state.kis_search_results):
                with cols[i % 4]:
                    meta = res["metadata"]
                    vid_id = res["video_id"]
                    est_frame_idx = get_exact_frame_idx(meta)

                    img_url = res.get("thumbnail_url", "")
                    if img_url and os.path.exists(img_url):
                        st.image(img_url, use_container_width=True)
                    else:
                        st.warning("Ảnh không khả dụng")

                    st.caption(f"**{vid_id}** | F: {est_frame_idx} | Điểm: {res['score']}")

                    if st.button("🔬 Nội soi cảnh này", key=f"inspect_{res['shot_id']}_{i}", use_container_width=True):
                        st.session_state.kis_active_shot = res
                        st.rerun()

        # ==========================================
        # CHẶNG 2: TRẠM NỘI SOI KIS (Phát Video & Kéo Frame)
        # ==========================================
        if st.session_state.kis_active_shot:
            active_res = st.session_state.kis_active_shot
            vid_id = active_res["video_id"]
            meta = active_res["metadata"]
            fps = meta.get("fps", 25.0)

            st.markdown("---")
            st.subheader(f"🔬 Trạm Nội Soi: {vid_id}")
            if st.button("⬅️ Quay lại danh sách tìm kiếm"):
                st.session_state.kis_active_shot = None
                st.rerun()

            raw_video_path = find_video_path(vid_id)
            total_duration = 1000.0
            if raw_video_path and raw_video_path.exists():
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(raw_video_path))
                    if cap.isOpened():
                        total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                    cap.release()
                except Exception:
                    pass

            col_vid, col_ctrl = st.columns([2, 1])
            with col_vid:
                if raw_video_path and raw_video_path.exists():
                    st.video(str(raw_video_path))
                else:
                    st.error("Không tìm thấy file video gốc.")
                    if os.path.exists(active_res.get("thumbnail_url", "")):
                        st.image(active_res.get("thumbnail_url"))

            with col_ctrl:
                st.info("Sử dụng thanh trượt và băng chuyền để chốt chính xác frame mỏ neo.")
                
                # Đồng bộ thời gian
                default_ts = float(meta.get("start_ts", 0.0))
                current_slider_val = st.session_state.get(f"kis_slider_{active_res['shot_id']}", default_ts)
                
                selected_ts = st.slider("Kéo nhanh (Định vị thô):", min_value=0.0, max_value=float(total_duration), value=float(current_slider_val), step=0.1)
                fine_ts = st.number_input("Nhập số (giây):", min_value=0.0, max_value=float(total_duration), value=float(selected_ts), step=round(1.0/fps, 3), format="%.3f")

                exact_frame = round(fine_ts * fps)
                st.markdown(f"### Frame đang trỏ: **<span style='color:#ff4b4b'>{exact_frame}</span>**", unsafe_allow_html=True)

                # Băng chuyền Frame
                gap_sec_kis = st.slider("Độ giãn cách băng chuyền (giây):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                step_frames = max(1, int(fps * gap_sec_kis))
                offsets = [-step_frames*2, -step_frames, 0, step_frames, step_frames*2]
                carousel_cols = st.columns(5)
                
                final_selected_frame = exact_frame
                
                for i, offset in enumerate(offsets):
                    target_f = max(0, exact_frame + offset)
                    with carousel_cols[i]:
                        img_path = extract_single_frame(vid_id, target_f, fps)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.caption(f"F: **{target_f}**")
                        if st.button("📍 Chọn", key=f"kis_mark_{target_f}_{exact_frame}"):
                            st.session_state[f"kis_final_frame_{vid_id}"] = target_f
                            st.rerun()

                # Ghi nhận frame được chọn từ băng chuyền (nếu có)
                if f"kis_final_frame_{vid_id}" in st.session_state:
                    final_selected_frame = st.session_state[f"kis_final_frame_{vid_id}"]
                    st.success(f"Đã ghim mỏ neo tại Frame: {final_selected_frame}")

                st.markdown("---")
                if st.button("🛒 CHỐT RẢI THẢM ĐÁP ÁN NÀY", type="primary", use_container_width=True):
                    with st.spinner("⚡ Đang vét 100 frames (Cận chiến + Lưới dự phòng)..."):
                        # Lấy lưới dự phòng
                        auto_fill_results = call_search_api(
                            query=selected_q_text, top_k=100, 
                            top_n=100,
                            liked_shot_id=active_res["shot_id"], 
                            skip_vlm=True
                        )
                        # Kích hoạt module cào
                        padded_cart = generate_padded_frames(
                            anchor_vid=vid_id,
                            anchor_frame_idx=final_selected_frame,
                            semantic_results=auto_fill_results,
                            max_frames=100
                        )
                        
                        st.session_state.cart[selected_q_id] = padded_cart
                        st.success(f"🎉 Đã chốt {len(padded_cart)} Frames an toàn vào giỏ hàng!")
                        time.sleep(1.5)
                        st.session_state.kis_active_shot = None
                        st.rerun()

# ==========================================
# PHASE 2.2: Q&A (CHẾ ĐỘ RẢI THẢM TEXT)
# ==========================================
with tab_qa:
    st.header("💬 Giải quyết Q&A (Phủ Text 100 Frames)")
    qa_queries = [q for q in st.session_state.btc_queries if q["type"] == "qa"]

    if not qa_queries:
        st.warning("⚠️ Vui lòng Upload file đề thi của BTC!")
    else:
        qa_options = {q["id"]: f"{q['id']}: {q['text']}" for q in qa_queries}
        sel_qa_id = st.selectbox("📌 Chọn câu hỏi Q&A:", options=list(qa_options.keys()), format_func=lambda x: qa_options[x])
        sel_qa_text = [q["text"] for q in qa_queries if q["id"] == sel_qa_id][0]

        edited_search_query = st.text_area("🔑 Từ khóa tìm kiếm (Sửa gọn lại nếu cần):", value=sel_qa_text, height=100, key=f"search_input_{sel_qa_id}")

        if f"qa_frames_{sel_qa_id}" not in st.session_state: st.session_state[f"qa_frames_{sel_qa_id}"] = []
        if f"qa_active_shot_{sel_qa_id}" not in st.session_state: st.session_state[f"qa_active_shot_{sel_qa_id}"] = None
        if f"qa_final_res_{sel_qa_id}" not in st.session_state: st.session_state[f"qa_final_res_{sel_qa_id}"] = None

        # [CHẶNG 1] TÌM KIẾM MỎ NEO
        qa_fast_mode = st.checkbox("⚡ Siêu tốc (bỏ qua VLM)", key=f"qa_fast_{sel_qa_id}")
        if st.button("🔎 1. Tìm 10 khung hình ứng viên", key=f"btn_suggest_{sel_qa_id}", use_container_width=True):
            with st.spinner("Đang quét..."):
                st.session_state[f"qa_frames_{sel_qa_id}"] = call_search_api(
                    query=edited_search_query.strip(), 
                    top_k=60, 
                    top_n=10, 
                    skip_vlm=qa_fast_mode,
                    task_type="qa"
                )
                st.session_state[f"qa_final_res_{sel_qa_id}"] = None 
                st.session_state[f"qa_active_shot_{sel_qa_id}"] = None

        suggested_frames = st.session_state.get(f"qa_frames_{sel_qa_id}", [])
        
        if suggested_frames and not st.session_state[f"qa_active_shot_{sel_qa_id}"] and not st.session_state[f"qa_final_res_{sel_qa_id}"]:
            st.markdown("### 🎯 Chọn 1 cảnh chứa manh mối (Mỏ neo):")
            q_cols = st.columns(5)
            for idx, s_res in enumerate(suggested_frames):
                with q_cols[idx % 5]:
                    if os.path.exists(s_res.get("thumbnail_url", "")):
                        st.image(s_res["thumbnail_url"], use_container_width=True)
                    st.caption(f"**{s_res['video_id']}** | Score: {s_res['score']}")
                    if st.button("📌 Chọn mỏ neo", key=f"sel_anchor_{sel_qa_id}_{idx}"):
                        st.session_state[f"qa_active_shot_{sel_qa_id}"] = s_res
                        st.rerun()

        # [CHẶNG 2] BĂNG CHUYỀN VÀ QWEN-VL SOI ẢNH
        if st.session_state[f"qa_active_shot_{sel_qa_id}"] and not st.session_state[f"qa_final_res_{sel_qa_id}"]:
            active_shot = st.session_state[f"qa_active_shot_{sel_qa_id}"]
            vid_id = active_shot["video_id"]
            meta = active_shot["metadata"]
            fps = meta.get("fps", 25.0)
            anchor_frame = get_exact_frame_idx(meta)

            st.markdown("---")
            col_b1, col_b2, col_b3 = st.columns([2, 2, 1])
            with col_b1: st.subheader("🎞️ Băng chuyền lân cận")
            with col_b2: gap_sec = st.slider("Độ giãn cách (giây):", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            with col_b3:
                if st.button("⬅️ Quay lại", use_container_width=True):
                    st.session_state[f"qa_active_shot_{sel_qa_id}"] = None
                    st.rerun()

            step_frames = max(1, int(fps * gap_sec))
            offsets = [-step_frames*2, -step_frames, 0, step_frames, step_frames*2]
            carousel_cols = st.columns(5)
            
            for i, offset in enumerate(offsets):
                target_frame = max(0, anchor_frame + offset)
                with carousel_cols[i]:
                    img_path = extract_single_frame(vid_id, target_frame, fps)
                    if img_path and os.path.exists(img_path): st.image(img_path, use_container_width=True)
                    st.caption(f"F: {target_frame} ({(offset/fps):+.1f}s)")
                    
                    if st.button("🧠 Gửi VLM ảnh này", key=f"gen_{sel_qa_id}_{target_frame}", use_container_width=True):
                        with st.spinner("Qwen-VL đang đọc ảnh..."):
                            qa_result = call_qa_api(question=sel_qa_text, shot_id=active_shot["shot_id"], custom_image_path=img_path)
                            if qa_result:
                                qa_result["frame_idx"] = target_frame
                                st.session_state[f"qa_final_res_{sel_qa_id}"] = qa_result
                            st.rerun()

        # [CHẶNG 3] DUYỆT VÀ CHỐT 100 FRAMES
        if sel_qa_id in st.session_state.cart: st.success(f"🛒 Câu hỏi này đã chốt đáp án trong Giỏ hàng!")

        final_res = st.session_state[f"qa_final_res_{sel_qa_id}"]
        if final_res:
            st.markdown("---")
            col_img, col_txt = st.columns([1, 1])
            with col_img:
                if os.path.exists(final_res.get("thumbnail_url", "")): st.image(final_res["thumbnail_url"])
            with col_txt:
                user_edited_ans = st.text_area("Đáp án nộp BTC:", value=final_res.get("answer", ""), max_chars=100)
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("🛒 Chốt nộp đáp án này", type="primary", use_container_width=True):
                        padded_cart = generate_padded_frames(
                            anchor_vid=final_res.get("video_id"),
                            anchor_frame_idx=get_exact_frame_idx(final_res),
                            semantic_results=st.session_state.get(f"qa_frames_{sel_qa_id}", []),
                            answer=user_edited_ans.strip(),
                            max_frames=100
                        )
                        st.session_state.cart[sel_qa_id] = padded_cart
                        st.success(f"🎉 Đã chốt {len(padded_cart)} Frames! Câu trả lời đã phủ kín giỏ hàng.")
                        time.sleep(1.5)
                        st.rerun()
                with cc2:
                    if st.button("🔄 Chọn lại", use_container_width=True):
                        st.session_state[f"qa_final_res_{sel_qa_id}"] = None
                        st.rerun()

# ==========================================
# PHASE 2.3: TRAKE (TRUY VẾT BẰNG THANH TRƯỢT ĐỘNG)
# ==========================================
with tab_trake:
    st.header("⏱️ Giải quyết TRAKE (Đồ thị Thời gian)")
    trake_queries = [q for q in st.session_state.btc_queries if q["type"] == "trake"]

    if not trake_queries:
        st.warning("⚠️ Không có câu hỏi TRAKE nào trong bộ đề.")
    else:
        tr_options = {q["id"]: f"{q['id']}: {q['text']}" for q in trake_queries}
        sel_tr_id = st.selectbox("📌 Chọn câu hỏi TRAKE:", options=list(tr_options.keys()), format_func=lambda x: tr_options[x])
        sel_tr_text = [q["text"] for q in trake_queries if q["id"] == sel_tr_id][0]

        if "trake_results" not in st.session_state: st.session_state.trake_results = []
        if "trake_active_shot" not in st.session_state: st.session_state.trake_active_shot = None

        trake_fast_mode = st.checkbox("⚡ Siêu tốc (bỏ qua VLM - Chỉ dùng để test)", key="trake_fast_mode")
        if st.button("🔎 Phân tích & Dò chuỗi", use_container_width=True):
            with st.spinner("VLM đang bóc tách hành động và dò đồ thị thời gian..."):
                st.session_state.trake_results = call_search_api(
                    query=sel_tr_text, top_k=60, top_n=12, skip_vlm=trake_fast_mode, task_type="trake"
                )
                st.session_state.trake_active_shot = None 

        # HIỂN THỊ LƯỚI KẾT QUẢ ĐÃ LỌC
        if st.session_state.trake_results and not st.session_state.trake_active_shot:
            st.success(f"Đã tìm thấy {len(st.session_state.trake_results)} Video chứa đủ chuỗi sự kiện!")
            tr_cols = st.columns(4)
            for i, res in enumerate(st.session_state.trake_results):
                with tr_cols[i % 4]:
                    if os.path.exists(res.get("thumbnail_url", "")) : st.image(res.get("thumbnail_url"))
                    
                    # Trích xuất số lượng hành động AI tìm được
                    seq_len = len(res.get("suggested_timestamps", []))
                    st.caption(f"**{res['video_id']}** | Chứa {seq_len} Hành động")
                    
                    if st.button("🔬 Nội soi Video này", key=f"inspect_tr_{res['shot_id']}_{i}", use_container_width=True):
                        st.session_state.trake_active_shot = res
                        st.rerun()

        # TRẠM NỘI SOI ĐA SỰ KIỆN
        if st.session_state.trake_active_shot:
            active_res = st.session_state.trake_active_shot
            vid_id = active_res["video_id"]
            fps = active_res["metadata"].get("fps", 25.0)
            
            # Lấy mảng thời gian đề xuất từ Backend
            suggested_ts = active_res.get("suggested_timestamps", [float(active_res["metadata"].get("start_ts", 0.0))])
            num_events = len(suggested_ts)

            st.markdown("---")
            st.subheader(f"🔬 Trạm Nội Soi Đa Sự Kiện: {vid_id}")
            if st.button("⬅️ Quay lại danh sách tìm kiếm"):
                st.session_state.trake_active_shot = None
                st.rerun()

            raw_video_path = find_video_path(vid_id)
            total_duration = 1000.0
            if raw_video_path and raw_video_path.exists():
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(raw_video_path))
                    if cap.isOpened(): total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                    cap.release()
                except Exception: pass

            col_vid, col_ctrl = st.columns([2, 1])
            with col_vid:
                if raw_video_path and raw_video_path.exists(): st.video(str(raw_video_path))
                else: st.error("Lỗi Video")
                
                st.info(f"💡 AI đã tự động đặt sẵn **{num_events}** thanh trượt ở các mốc thời gian hợp lệ. Hãy kiểm tra băng chuyền bên phải và chốt Frame!")

            with col_ctrl:
                # Quản lý mảng Frame đã chốt cho từng hành động
                state_key = f"trake_frames_{vid_id}"
                if state_key not in st.session_state or len(st.session_state[state_key]) != num_events:
                    st.session_state[state_key] = [None] * num_events

                # ==========================================
                # VÒNG LẶP SINH RA N KHỐI UI (Dựa theo số hành động)
                # ==========================================
                for ev_idx, base_ts in enumerate(suggested_ts):
                    with st.expander(f"🎬 Sự kiện {ev_idx + 1} (AI Đề xuất: {base_ts:.1f}s)", expanded=True):
                        current_slider_val = st.session_state.get(f"slider_{vid_id}_{ev_idx}", float(base_ts))
                        
                        # Thanh trượt và Ô nhập số liệu độc lập cho mỗi sự kiện
                        selected_ts = st.slider(f"Định vị thô (giây):", min_value=0.0, max_value=float(total_duration), value=float(current_slider_val), step=0.1, key=f"sl_raw_{vid_id}_{ev_idx}")
                        fine_ts = st.number_input(f"Nhập số (giây):", min_value=0.0, max_value=float(total_duration), value=float(selected_ts), step=round(1.0/fps, 3), format="%.3f", key=f"sl_fine_{vid_id}_{ev_idx}")

                        exact_frame = round(fine_ts * fps)
                        st.markdown(f"Frame đang trỏ: **<span style='color:#ff4b4b'>{exact_frame}</span>**", unsafe_allow_html=True)

                        # Băng chuyền độc lập
                        gap_sec_tr = st.slider("Độ giãn cách:", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key=f"gap_{vid_id}_{ev_idx}")
                        step_frames = max(1, int(fps * gap_sec_tr))
                        offsets = [-step_frames*2, -step_frames, 0, step_frames, step_frames*2]
                        carousel_cols = st.columns(5)
                        
                        for c_i, offset in enumerate(offsets):
                            target_f = max(0, exact_frame + offset)
                            with carousel_cols[c_i]:
                                img_path = extract_single_frame(vid_id, target_f, fps)
                                if img_path and os.path.exists(img_path): st.image(img_path)
                                st.caption(f"F: **{target_f}**")
                                if st.button("📍 Chốt", key=f"tr_mark_{vid_id}_{ev_idx}_{target_f}"):
                                    st.session_state[state_key][ev_idx] = target_f
                                    st.rerun()

                st.markdown("---")
                st.write("**Chuỗi sự kiện đang chọn:**")
                
                # Hiển thị trạng thái Array (VD: 1253 -> [...] -> [...])
                seq_display = [str(f) if f is not None else "[...]" for f in st.session_state[state_key]]
                st.code(" ➡️ ".join(seq_display))
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("⏪ Xóa làm lại", use_container_width=True):
                        st.session_state[state_key] = [None] * num_events
                        st.rerun()
                with c2:
                    is_complete = all(f is not None for f in st.session_state[state_key])
                    
                    # LOGIC KIỂM ĐỊNH (Gatekeeper): Không cho nộp nếu Frame sau < Frame trước
                    is_valid_order = True
                    if is_complete:
                        for i in range(num_events - 1):
                            if st.session_state[state_key][i] > st.session_state[state_key][i+1]:
                                is_valid_order = False
                                break

                    if not is_complete:
                        st.button("🛒 Chốt (Chưa đủ)", disabled=True, use_container_width=True)
                    elif not is_valid_order:
                        st.error("Lỗi trình tự thời gian!")
                    else:
                        if st.button("🛒 Chốt Chuỗi Này", type="primary", use_container_width=True):
                            st.session_state.cart[sel_tr_id] = [{"video_id": vid_id, "frames": list(st.session_state[state_key])}]
                            st.success("Tuyệt vời! Đã nạp vào Giỏ hàng.")
                            time.sleep(1)
                            st.session_state.trake_active_shot = None
                            st.rerun()

# ==========================================
# PHASE 3: SUBMISSION HUB (TRẠM NỘP BÀI)
# ==========================================
with tab_submit:
    st.header("📦 Trạm Quản lý & Nộp Bài")

    if not st.session_state.cart:
        st.info("Giỏ hàng đang trống. Hãy làm bài ở các Tab trước nhé!")
    else:
        preview_data = []
        is_valid = True

        for q_id, items in st.session_state.cart.items():
            row_count = len(items)
            ans_preview = items[0].get("answer", "-") if items else "-"

            status = "✅ Hợp lệ"
            if row_count > 100:
                status = "❌ Lỗi: > 100 dòng"
                is_valid = False
            elif ans_preview != "-" and len(ans_preview) > 100:
                status = "❌ Lỗi: > 100 ký tự"
                is_valid = False

            preview_data.append({
                "Mã Câu Hỏi": q_id, "Số Dòng": row_count,
                "Đáp Án Kèm Theo": ans_preview, "Trạng Thái": status
            })

        st.table(pd.DataFrame(preview_data))

        if is_valid:
            st.success("Tất cả các file đều hợp lệ. Đã sẵn sàng đóng gói!")
            zip_data = create_zip_in_memory(st.session_state.cart)

            st.download_button(
                label="📥 TẢI XUỐNG FILE ZIP", data=zip_data,
                file_name="TEAM_SUBMISSION_ROUND1.zip", mime="application/zip",
                type="primary", use_container_width=True
            )
        else:
            st.error("Dữ liệu vi phạm quy định của BTC!")