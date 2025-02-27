from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Annotated
from datetime import date, datetime
from .section import BaseSectionCreate, BaseSectionRead, BaseSectionUpdate
from .base import BaseAuthModel

class BaseNote(BaseModel):
    schema_version: int = 1
    consultation_id: int
    note_id: int
    patient_id: int
    encounter_date: str
    sections: List = Field(default_factory=list)
    class Config:
        orm_mode = True
        
class NoteCreate(BaseAuthModel):
    schema_version: int = 1
    patient_id: Optional[int]
    title: str
    encounter_date: date
    sections: List[BaseSectionCreate] = []
    
    class Config:
        orm_mode = True

class NoteRead(BaseAuthModel):
    id: int
    title: str
    patient_id: Optional[int]
    sections: List[BaseSectionRead]
    
    class Config:
        orm_mode = True
class NoteUpdate(BaseAuthModel):
    title: str
    patient_id: Optional[int]
    sections: List[BaseSectionUpdate] = []
    
    class Config:
        orm_mode = True