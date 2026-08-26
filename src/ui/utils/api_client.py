import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1/search"
QA_API_URL = "http://localhost:8000/api/v1/qa"

def call_search_api(query: str, top_k: int = 20, top_n: int = 5, liked_shot_id: str = None, skip_vlm: bool = False, target_video_id: str = None, task_type: str = "kis"):
    try:
        payload = {
            "query": query, "top_k": top_k, "top_n": top_n,
            "skip_vlm": skip_vlm, "task_type": task_type
        }
        if target_video_id: payload["target_video_id"] = target_video_id
        if liked_shot_id: payload["liked_shot_id"] = liked_shot_id
        
        # Thêm timeout 120s
        response = requests.post(API_URL, json=payload, timeout=120)
        return response.json().get("results", []) if response.status_code == 200 else []
    except Exception as e:
        st.error(f"Lỗi kết nối Backend (Timeout hoặc Server sập): {e}")
        return []

def call_qa_api(question: str, shot_id: str = None, custom_image_path: str = None):
    try:
        payload = {"question": question}
        if shot_id: payload["shot_id"] = shot_id
        if custom_image_path: payload["custom_image_path"] = custom_image_path
            
        response = requests.post(QA_API_URL, json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return None