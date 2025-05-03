from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, field_validator, ConfigDict
from .section import SectionCreate, SectionRead, SectionUpdate
from datetime import datetime

class NoteCreate(BaseModel):
    """Schema for creating a new note"""
    patient_id: Optional[int] = None
    user_id: int
    title: str
    encounter_date: datetime
    sections: List[Union[SectionCreate, Dict[str, Any]]] = []

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )
    
class NoteRead(BaseModel):
    """Schema for reading a note"""
    id: int
    title: str
    patient_id: Optional[int]
    user_id: int
    encounter_date: datetime
    sections: List[SectionRead] = []
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )
    
class NoteUpdate(BaseModel):
    """Schema for updating a note"""
    title: Optional[str] = None
    patient_id: Optional[int] = None
    user_id: int
    encounter_date: Optional[datetime] = None
    sections: List[SectionUpdate] = []
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )