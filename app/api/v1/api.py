from .diarization import api_router as diarization_router

# Main router that aggregates all versioned (v1) endpoints.
# Now all routes from diarization.py 
# (including /transcription/transcribe-with-diarization) are served under /api/v1.

api_router = APIRouter()
api_router.include_router(diarization_router)
