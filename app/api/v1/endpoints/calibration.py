from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Any, Optional
from app.models.models import User, CalibrationPhrase, CalibrationRecording
from app.db.local_session import DatabaseManager
import logging
from app.services.voice_calibration_service import VoiceCalibrationService
from app.schemas.calibration import CalibrationStatus, CalibrationPhraseBase, RecordingStatus, RecordingBase
# Configure logger
logger = logging.getLogger(__name__)
router = APIRouter()
get_session = DatabaseManager().get_session

calibration_service = VoiceCalibrationService()
    
@router.get("/phrases", response_model=List[CalibrationPhraseBase])
async def get_calibration_phrases(db: Session = Depends(get_session)):
    """Get calibration phrases for voice training"""
    return calibration_service.get_calibration_phrases(db)

@router.get("/status/{user_id}", response_model=CalibrationStatus)
async def get_calibration_status(user_id: int, db: Session = Depends(get_session)):
    """Get calibration status for a user"""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return calibration_service.get_calibration_status(user_id, db)

@router.post("/record/{user_id}/{phrase_id}")
async def record_calibration_phrase(
    user_id: int, 
    phrase_id: int, 
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_session)
):
    """Record a calibration phrase for a user"""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Verify phrase exists
    phrase = db.query(CalibrationPhrase).filter(CalibrationPhrase.id == phrase_id).first()
    if not phrase:
        raise HTTPException(status_code=404, detail="Phrase not found")
    
    # Process the recording
    success = calibration_service.process_calibration_recording(user_id, phrase_id, audio_file.file, db)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process calibration recording")
        
    # Get updated status
    status = calibration_service.get_calibration_status(user_id, db)
    
    return {
        "success": True,
        "message": f"Recorded phrase {phrase_id} for user {user_id}",
        "status": status.dict()
    }

@router.post("/create-profile/{user_id}")
async def create_speaker_profile(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """Create a speaker profile from recorded calibration phrases"""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check status
    status = calibration_service.get_calibration_status(user_id, db)
    if status.phrases_recorded < 3:
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough calibration data. Recorded {status.phrases_recorded}/3 required phrases."
        )
    
    # Create profile
    profile_id = calibration_service.create_speaker_profile(user_id, db)
    
    if not profile_id:
        raise HTTPException(status_code=500, detail="Failed to create speaker profile")
    
    return {
        "success": True,
        "message": "Speaker profile created successfully",
        "user_id": user_id,
        "profile_id": profile_id
    }

@router.delete("/reset/{user_id}")
async def reset_calibration(user_id: int, db: Session = Depends(get_session)):
    """Reset all calibration data for a user"""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    success = calibration_service.delete_calibration_data(user_id, db)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset calibration data")
        
    return {
        "success": True,
        "message": f"Reset calibration data for user {user_id}"
    }
    
@router.get("/recordings/{user_id}", response_model=List[RecordingStatus])
async def get_user_recordings(user_id: int, db: Session = Depends(get_session)):
    """
    Get all calibration recordings for a user
    
    Args:
        user_id: User ID
        db: Database session
        
    Returns:
        List of recording statuses
    """
    try:
        # Verify user exists
        from app.models.models import User
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User {user_id} not found when getting recordings")
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get all recordings for this user
        recordings = db.query(CalibrationRecording).filter(
            CalibrationRecording.user_id == user_id
        ).all()
        
        # Convert to recording status
        results = []
        for recording in recordings:
            # Get phrase if available
            phrase = db.query(CalibrationPhrase).filter(
                CalibrationPhrase.id == recording.phrase_id
            ).first()
            
            # Determine recording status
            status = "completed"
            
            # Create response object
            results.append({
                "phrase_id": recording.phrase_id,
                "status": status,
                "timestamp": recording.created_at.isoformat() if recording.created_at else None,
                "phrase_text": phrase.text if phrase else None,
                "duration_ms": recording.duration_ms
            })
        
        return results
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting user recordings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recordings: {str(e)}"
        )

@router.get("/recording/{user_id}/{phrase_id}", response_model=RecordingStatus)
async def get_phrase_recording(user_id: int, phrase_id: int, db: Session = Depends(get_session)):
    """
    Get calibration recording status for a specific phrase
    
    Args:
        user_id: User ID
        phrase_id: Phrase ID
        db: Database session
        
    Returns:
        Recording status
    """
    try:
        # Verify phrase exists
        phrase = db.query(CalibrationPhrase).filter(CalibrationPhrase.id == phrase_id).first()
        if not phrase:
            logger.warning(f"Phrase {phrase_id} not found")
            raise HTTPException(status_code=404, detail="Phrase not found")
        
        # Get the recording
        recording = db.query(CalibrationRecording).filter(
            CalibrationRecording.user_id == user_id,
            CalibrationRecording.phrase_id == phrase_id
        ).first()
        
        if not recording:
            # No recording found
            return {
                "phrase_id": phrase_id,
                "status": "pending",
                "timestamp": None,
                "phrase_text": phrase.text,
                "duration_ms": None
            }
        
        # Return recording status
        return {
            "phrase_id": phrase_id,
            "status": "completed",
            "timestamp": recording.created_at.isoformat() if recording.created_at else None,
            "phrase_text": phrase.text,
            "duration_ms": recording.duration_ms
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting phrase recording: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recording: {str(e)}"
        )