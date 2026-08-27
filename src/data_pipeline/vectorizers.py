"""
MODULE VECTORIZERS - ĐỘNG CƠ MÃ HÓA ĐỘC LẬP
Vị trí: src/data_pipeline/vectorizers.py

100% local, không dùng fastembed:
  - image_dense: SigLIP vision tower (get_image_features)
  - text_dense : SigLIP text tower (get_text_features) — CÙNG không gian
                 embedding với ảnh, vì cùng 1 model SigLIP
  - text_sparse: SPLADE (AutoModelForMaskedLM), pooling log(1+ReLU(x)),
                 chạy native PyTorch trên MPS (Mac) hoặc CPU

Yêu cầu trong config.py:
    SIGLIP_MODEL_ID     = "google/siglip-base-patch16-224"
    TEXT_SPARSE_MODEL   = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
    TEXT_DENSE_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DIM_IMAGE_DENSE     = 768   # siglip-base hidden size (vision + text towers)
    DIM_TEXT_DENSE      = 384   # MiniLM L12 hidden size — KHÁC không gian với image_dense

    Lưu ý: image_dense và text_dense KHÔNG cùng không gian vector (khác model, khác chiều).
    Cross-modal search (text → image) được thực hiện qua SiglipEncoder.encode_text()
    (SigLIP text tower, 768D) gán vào index "image_dense" của Qdrant.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForMaskedLM
from src.config.common import get_logger
from src.config.config import (
    SIGLIP_MODEL_ID,
    TEXT_SPARSE_MODEL,
    TEXT_DENSE_MODEL,  
    DIM_IMAGE_DENSE,
    DIM_TEXT_DENSE,    
)

logger = get_logger(__name__)


def _get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ==========================================
# 1+2. BỘ MÃ HÓA SIGLIP DÙNG CHUNG (ẢNH + VĂN BẢN DENSE)
# ==========================================
class SiglipEncoder:
    """
    Load 1 lần SigLIP, dùng chung cho cả 2 tháp:
      - encode_image(): tháp vision
      - encode_text() : tháp text
    image_dense và text_dense nằm chung 1 không gian vector (cùng chiều,
    có thể so khớp trực tiếp bằng cosine/dot product trong Qdrant).
    """

    def __init__(self, model_id: str = SIGLIP_MODEL_ID):
        self.device = _get_device()
        logger.info(f"🟢 Khởi tạo SigLIP Encoder ({model_id}) trên {self.device.upper()}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _unwrap(outputs):
        """
        Transformers versions mới (>= 4.40) có thể trả về BaseModelOutputWithPooling
        thay vì tensor thuần — cần unwrap an toàn trước khi norm.
        """
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        return outputs

    def encode_image(self, image_path: str) -> list[float]:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            image_features = self._unwrap(outputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"Lỗi encode hình ảnh {image_path}: {e}")
            return [0.0] * DIM_IMAGE_DENSE

    def encode_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * DIM_TEXT_DENSE
        try:
            # SigLIP train với padding cố định "max_length" — giữ đúng kiểu
            # padding này để embedding text ra đúng phân bố như lúc train.
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)

            text_features = self._unwrap(outputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"Lỗi encode Text Dense (SigLIP): {e}")
            return [0.0] * DIM_TEXT_DENSE


# ==========================================
# 3. BỘ MÃ HÓA VĂN BẢN SPARSE (SPLADE thuần transformers)
# ==========================================
class TextSparseVectorizer:
    """
    SPLADE native trên PyTorch/MPS (không qua fastembed/ONNX).
    Công thức chuẩn: w_j = max_i log(1 + ReLU(logit_ij)) qua các token i,
    cho mỗi chiều j trong vocab của model.
    """

    def __init__(self, model_id: str = TEXT_SPARSE_MODEL):
        self.device = _get_device()
        logger.info(f"🟠 Khởi tạo SPLADE Sparse Vectorizer ({model_id}) trên {self.device.upper()}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> dict:
        if not text or not text.strip():
            return {"indices": [], "values": []}
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits  # [1, seq_len, vocab_size]

            activated = torch.log1p(torch.relu(logits))
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            activated = activated * attention_mask  # loại bỏ ảnh hưởng của padding token
            sparse_vec = activated.max(dim=1).values.squeeze(0)  # [vocab_size]

            nonzero = torch.nonzero(sparse_vec, as_tuple=True)[0]
            return {
                "indices": nonzero.cpu().tolist(),
                "values": sparse_vec[nonzero].cpu().tolist(),
            }
        except Exception as e:
            logger.error(f"Lỗi encode Text Sparse (SPLADE): {e}")
            return {"indices": [], "values": []}

class TextDenseVectorizer:
    """Mã hóa Text Dense thuần PyTorch (Local 100%)"""
    def __init__(self, model_id: str = TEXT_DENSE_MODEL):
        self.device = _get_device()
        logger.info(f"🔵 Khởi tạo Text Dense Vectorizer ({model_id}) bằng PyTorch thuần trên {self.device.upper()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * DIM_TEXT_DENSE
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Thực hiện Mean Pooling chuẩn của MiniLM
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # Chuẩn hóa L2 (L2 Normalization)
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            
            return normalized[0].cpu().tolist()
        except Exception as e:
            logger.error(f"Lỗi encode Text Dense (PyTorch): {e}")
            return [0.0] * DIM_TEXT_DENSE
# ==========================================
# 4. TRÌNH ĐIỀU PHỐI LAI (HYBRID FACADE)
# ==========================================
class HybridVectorizer:
    def __init__(self):
        self.siglip = SiglipEncoder()
        self.text_dense_encoder = TextDenseVectorizer()
        self.text_sparse_encoder = TextSparseVectorizer()
        logger.info("✅ Hybrid Vectorizer đã tập hợp đủ 3 sức mạnh độc lập!")

    def encode_all(self, image_path: str, text_content: str) -> dict:
        return {
            "image_dense": self.siglip.encode_image(image_path),
            "text_dense": self.text_dense_encoder.encode(text_content),
            "text_sparse": self.text_sparse_encoder.encode(text_content),
        }