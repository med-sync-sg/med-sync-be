from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.services.transcription_service import TranscriptionService
import tempfile

router = APIRouter()

@router.post("/transcribe-with-diarization")
async def transcribe_with_diarization(
    user_id: int = Form(...),
    note_id: int = Form(...),
    audio: UploadFile = File(...),
    hf_token: str = Form(...)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())

        transcription_service = TranscriptionService(hf_token=hf_token)
        segments = transcription_service.process_audio_segment_with_diarization(user_id, note_id)
        return JSONResponse(content={"segments": segments})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
