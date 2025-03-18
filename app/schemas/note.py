from typing import List, Optional
from datetime import date
from pydantic import BaseModel
from .section import SectionCreate, SectionRead, SectionUpdate

class NoteCreate(BaseModel):
    patient_id: Optional[int]
    user_id: int
    title: str
    encounter_date: date
    sections: List[SectionCreate]

    class Config:
        orm_mode = True

class NoteRead(BaseModel):
    id: int
    title: str
    patient_id: Optional[int]
    encounter_date: date
    sections: List[SectionRead] = []

    class Config:
        orm_mode = True

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    patient_id: Optional[int] = None
    encounter_date: Optional[date] = None
    sections: Optional[List[SectionUpdate]] = None

    class Config:
        orm_mode = True