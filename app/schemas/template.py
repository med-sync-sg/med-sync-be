from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Optional
from .section import SectionCreate, SectionRead, SectionUpdate
from .base import BaseAuthModel

class NoteTemplateCreate(BaseAuthModel):
    user_id: int
    consultation_id: int
    note_id: int
    patient_id: int
    encounter_date: date
    title: str
    sections: List[SectionCreate] = []

    class Config:
        orm_mode = True

class NoteTemplateRead(BaseAuthModel):
    id: int
    user_id: int
    consultation_id: int
    note_id: int
    patient_id: int
    encounter_date: date
    title: str
    sections: List[SectionRead] = []
    created_at: datetime

    class Config:
        orm_mode = True

class NoteTemplateUpdate(BaseAuthModel):
    title: Optional[str] = None
    sections: Optional[List[SectionUpdate]] = None

    class Config:
        orm_mode = True

class SectionTemplateCreate(BaseAuthModel):
    pass

class SectionTemplateRead(BaseAuthModel):
    id: int
    title: str
    description: str
    section_type: str = Field(default="OTHERS")
    metadata_keys: List[str] = []
    content_keys: List[str] = []

    class Config:
        orm_mode = True

class SectionTemplateUpdate(BaseAuthModel):
    title: Optional[str] = None
    description: Optional[str] = None
    section_type: Optional[str] = None
    metadata_keys: Optional[List[str]] = None
    content_keys: Optional[List[str]] = None

    class Config:
        orm_mode = True