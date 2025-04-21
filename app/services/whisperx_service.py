import logging
import whisperx
import torch

logger = logging.getLogger(__name__)

class WhisperXService:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisperx.load_model("large-v2", self.device)

    def transcribe_and_diarize(self, audio_path: str):
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return result