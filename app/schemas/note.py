from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, field_serializer, ConfigDict
from .section import SectionCreate, SectionRead, SectionUpdate
from datetime import datetime

class NoteCreate(BaseModel):
    """Schema for creating a new note"""
    patient_id: Optional[int] = None
    user_id: int
    title: str
    encounter_date: datetime
    sections: List[SectionCreate] = []

    @field_serializer('encounter_date')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
    )
    
class NoteRead(BaseModel):
    """Schema for reading a note"""
    id: int
    title: str
    patient_id: Optional[int]
    user_id: int
    encounter_date: datetime
    sections: List[SectionRead] = []
    
    @field_serializer('encounter_date')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
    )
    
class NoteUpdate(BaseModel):
    """Schema for updating a note"""
    title: Optional[str] = None
    patient_id: Optional[int] = None
    user_id: int
    encounter_date: Optional[datetime] = None
    sections: List[SectionUpdate] = []
    
    @field_serializer('encounter_date')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
    )