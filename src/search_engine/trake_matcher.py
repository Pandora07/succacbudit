"""
MODULE 4: BỘ GHÉP NỐI CHUỖI SỰ KIỆN (TEMPORAL SEQUENCE MATCHER)
Vị trí: src/search_engine/trake_matcher.py
Chức năng: Nhận danh sách các hành động từ QueryParser, tìm kiếm song song
           và giao cắt kết quả để đảm bảo chuỗi thời gian hợp lý (T1 < T2 < T3).
"""
import copy
from typing import List, Dict, Any
from src.config.common import get_logger

logger = get_logger(__name__)

class TrakeMatcher:
    def __init__(self, retriever):
        self.retriever = retriever

    def match_sequence(self, temporal_sequence: List[str], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các video thỏa mãn MỘT CHUỖI CÁC SỰ KIỆN theo đúng trình tự thời gian.
        """
        if not temporal_sequence:
            return []

        logger.info(f"⏱️ [TrakeMatcher] Bắt đầu truy vết chuỗi {len(temporal_sequence)} hành động.")
        
        # 1. Tìm kiếm từng hành động độc lập
        event_results = []
        for i, event_query in enumerate(temporal_sequence):
            logger.info(f"   -> Tìm kiếm sự kiện {i+1}: '{event_query}'")
            # Sử dụng luồng lai Hybrid cho từng hành động lẻ (Mặc định dense = sparse = event_query)
            raw_shots = self.retriever.search(
                dense_query=event_query, 
                sparse_query=event_query, 
                top_k=top_k
            )
            # Gom nhóm kết quả theo video_id để dễ truy xuất
            video_map = {}
            for shot in raw_shots:
                vid = shot["video_id"]
                if vid not in video_map:
                    video_map[vid] = []
                video_map[vid].append(shot)
            
            event_results.append(video_map)

        # 2. Lấy giao của các Video xuất hiện trong TẤT CẢ các hành động
        # (Chỉ những video có chứa đủ cả N hành động mới được xét tiếp)
        common_videos = set(event_results[0].keys())
        for v_map in event_results[1:]:
            common_videos.intersection_update(v_map.keys())

        if not common_videos:
            logger.warning("⚠️ Không tìm thấy Video nào chứa đủ toàn bộ chuỗi sự kiện.")
            return []

        logger.info(f"   -> Tìm thấy {len(common_videos)} Video tiềm năng. Đang nội soi mốc thời gian...")

        # 3. Thuật toán Đồ thị (Dò tìm chuỗi thời gian hợp lệ: t1 < t2 < t3)
        valid_sequences = []
        
        for vid in common_videos:
            # Lấy list các shot chứa hành động 1, 2, 3... trong video này
            shots_per_event = [event_results[i][vid] for i in range(len(temporal_sequence))]
            
            # Sắp xếp các shot của mỗi hành động theo thời gian (start_ts)
            for shots in shots_per_event:
                shots.sort(key=lambda x: x["metadata"]["start_ts"])

            # Hàm đệ quy tìm mảng đường đi hợp lệ
            def find_path(event_index: int, current_path: list, last_ts: float):
                if event_index == len(temporal_sequence):
                    return current_path
                
                for shot in shots_per_event[event_index]:
                    shot_start = shot["metadata"]["start_ts"]
                    # Điều kiện sinh tồn: Hành động sau phải xảy ra bằng hoặc sau hành động trước
                    # (Lưu ý: Có những hành động diễn ra trong cùng 1 shot, nên dùng >=)
                    if shot_start >= last_ts:
                        new_path = current_path + [shot]
                        result = find_path(event_index + 1, new_path, shot_start)
                        if result:
                            return result
                return None

            # Bắt đầu dò từ hành động đầu tiên
            valid_path = find_path(0, [], -1.0)
            
            if valid_path:
                # Đóng gói kết quả: Video này hợp lệ, lưu lại mảng các shot tương ứng
                # Điểm số của video = Trung bình cộng điểm RRF của các shot tạo nên chuỗi
                avg_score = sum(shot["score"] for shot in valid_path) / len(valid_path)
                
                # Trích xuất mảng thời gian đề xuất để trả về cho UI
                suggested_ts = [(shot["metadata"]["start_ts"] + shot["metadata"]["end_ts"]) / 2 for shot in valid_path]
                
                # Tạo đối tượng đại diện (Lấy thông tin của hành động đầu tiên để làm thumbnail)
                representative_obj = copy.deepcopy(valid_path[0])
                representative_obj["score"] = round(avg_score, 4)
                representative_obj["trake_sequence"] = valid_path
                representative_obj["suggested_timestamps"] = suggested_ts
                
                valid_sequences.append(representative_obj)

        # 4. Sắp xếp các Video thỏa mãn theo Điểm số tổng hợp (Giảm dần)
        valid_sequences.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"✅ Đã chốt {len(valid_sequences)} Video thỏa mãn chuẩn thời gian.")
        return valid_sequences