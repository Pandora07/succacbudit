import os
import tempfile
import subprocess
import mlx_whisper
from src.config.common import get_logger
from src.config.config import MIN_SHOT_DURATION_SEC, WHISPER_MODEL_PATH

logger = get_logger(__name__)


class AudioTranscriber:
    def __init__(self, model_path: str = WHISPER_MODEL_PATH):
        """Sử dụng MLX Whisper để tận dụng tối đa Metal GPU của Mac."""
        self.model_path = model_path
        logger.info(f"🎙️ Đã khởi tạo Audio Engine (Model: {self.model_path})")

    def _extract_audio_slice(self, video_path: str, start_ts: float, end_ts: float) -> str:
        """Dùng FFmpeg cắt audio siêu tốc, không cần render lại video."""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        duration = end_ts - start_ts

        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start_ts),
            "-t", str(duration),
            "-vn",              # Bỏ hình, chỉ lấy tiếng
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            temp_audio
        ]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return temp_audio
        except subprocess.CalledProcessError:
            return None

    def transcribe_shot(self, video_path: str, start_ts: float, end_ts: float) -> str:
        """Cắt và transcribe một đoạn Shot ngắn."""
        duration = end_ts - start_ts

        # Guard: Bỏ qua shot quá ngắn → WAV rỗng → Whisper ra text rác
        if duration < MIN_SHOT_DURATION_SEC:
            return ""

        temp_audio_path = self._extract_audio_slice(video_path, start_ts, end_ts)
        if not temp_audio_path or not os.path.exists(temp_audio_path):
            return ""

        try:
            # language=None: Whisper tự detect ngôn ngữ.
            # KHÔNG ép "vi" cứng — video có thể đa ngữ hoặc tiếng Anh.
            # Ép "vi" với audio tiếng Anh → transcript rác nhiễm vector nghiêm trọng.
            result = mlx_whisper.transcribe(
                temp_audio_path,
                path_or_hf_repo=self.model_path,
                language=None,  # Auto-detect
                fp16=True
            )
            text = result.get("text", "").strip()
            return text
        except Exception as e:
            logger.error(f"Lỗi Whisper ASR: {e}")
            return ""
        finally:
            # Dọn rác ngay sau khi dùng xong
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)