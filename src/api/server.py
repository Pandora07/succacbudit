"""
TRẠM TRUNG CHUYỂN API (FASTAPI)
Vị trí: src/api/server.py
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.config.common import get_logger
from src.vlm_engine.engine import VLMEngine
from src.search_engine.pipeline import SearchPipeline

logger = get_logger(__name__)

app = FastAPI(title="Datathon AI Search API", version="3.1")

logger.info("⏳ Đang nạp các Động cơ AI vào bộ nhớ...")
vlm_online = VLMEngine(mode="online")
search_pipeline = SearchPipeline(vlm_online)
logger.info("✅ Backend API đã sẵn sàng!")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 60
    top_n: int = 20
    skip_vlm: bool = False
    task_type: str = "kis"
    target_video_id: str | None = None
    liked_shot_id: str | None = None

class QARequest(BaseModel):
    question: str
    shot_id: str | None = None
    custom_image_path: str | None = None

@app.post("/api/v1/search")
def api_search(req: SearchRequest):
    logger.info(f"🔎 Nhận truy vấn: '{req.query}' (Task: {req.task_type}, Skip VLM: {req.skip_vlm})")
    
    results = search_pipeline.run(
        raw_query=req.query,
        top_k=req.top_k,
        top_n=req.top_n,
        skip_vlm=req.skip_vlm,
        target_video_id=req.target_video_id,
        task_type=req.task_type,
        liked_shot_id=req.liked_shot_id 
    )
    return {"results": results}

@app.post("/api/v1/qa")
def api_qa(req: QARequest):
    if not req.custom_image_path or not os.path.exists(req.custom_image_path):
        return {"error": "Không tìm thấy ảnh."}
        
    prompt = f"Answer this question based on the image briefly and accurately. Question: {req.question}"
    answer_text = vlm_online.generate(
        image_paths=[req.custom_image_path], 
        prompt=prompt, 
        max_tokens=30, 
        temperature=0.1
    )
    return {"answer": answer_text.strip(), "thumbnail_url": req.custom_image_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")