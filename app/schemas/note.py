from typing import List, Optional
from datetime import date
from .base import BaseAuthModel
from .section import SectionCreate, SectionRead, SectionUpdate

class NoteCreate(BaseAuthModel):
    patient_id: Optional[int]
    title: str
    encounter_date: date
    sections: List[SectionCreate] = []

    class Config:
        orm_mode = True

class NoteRead(BaseAuthModel):
    id: int
    title: str
    patient_id: Optional[int]
    encounter_date: date
    sections: List[SectionRead] = []

    class Config:
        orm_mode = True

class NoteUpdate(BaseAuthModel):
    title: Optional[str] = None
    patient_id: Optional[int] = None
    encounter_date: Optional[date] = None
    sections: Optional[List[SectionUpdate]] = None

    class Config:
        orm_mode = True