from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.base import BaseAuthModel

class CalibrationPhraseBase(BaseModel):
    """Base schema for calibration phrases"""
    id: int
    text: str
    category: str
    description: Optional[str] = ""
    medical_terms: List[str] = []
    
    class Config:
        orm_mode = True

class SpeakerProfileBase(BaseModel):
    """Base schema for speaker profiles"""
    user_id: int
    feature_dimension: int
    training_phrases_count: int
    is_active: bool = True
    description: Optional[str] = None

class SpeakerProfileCreate(SpeakerProfileBase):
    """Schema for creating a speaker profile"""
    profile_data: bytes  # Binary profile data

class SpeakerProfileRead(SpeakerProfileBase):
    """Schema for reading a speaker profile"""
    id: int
    created_at: datetime
    updated_at: datetime
    stats: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True

class CalibrationStatus(BaseModel):
    """Schema for calibration status"""
    user_id: int
    calibration_complete: bool
    phrases_recorded: int
    phrases_total: int
    profile_id: Optional[int] = None
    last_updated: Optional[str] = None

class CalibrationRecordingBase(BaseModel):
    """Base schema for calibration recordings"""
    user_id: int
    phrase_id: int
    duration_ms: Optional[float] = None
    sample_rate: Optional[int] = None
    feature_type: str = "mfcc"

class CalibrationRecordingCreate(CalibrationRecordingBase):
    """Schema for creating a calibration recording"""
    features: bytes  # Binary feature data

class CalibrationRecordingRead(CalibrationRecordingBase):
    """Schema for reading a calibration recording"""
    id: int
    created_at: datetime
    speaker_profile_id: Optional[int] = None

    class Config:
        orm_mode = True

class CalibrationRequest(BaseAuthModel):
    """Schema for calibration requests"""
    phrase_id: int

class CalibrationResponse(BaseModel):
    """Schema for calibration responses"""
    success: bool
    message: str
    status: Optional[CalibrationStatus] = None

class ProfileCreationResponse(BaseModel):
    """Schema for profile creation responses"""
    success: bool
    message: str
    user_id: int
    profile_id: Optional[int] = None