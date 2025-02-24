from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Annotated
from datetime import date, datetime
from .section import BaseSectionCreate, BaseSectionRead, BaseSectionUpdate

class BaseNote(BaseModel):
    schema_version: int = 1
    consultation_id: int
    note_id: int
    patient_id: int
    encounter_date: date
    sections: List = Field(default_factory=list)

class NoteCreate(BaseNote):
    title: str
    sections: List[BaseSectionCreate] = []

class NoteRead(BaseNote):
    title: str
    sections: List[BaseSectionRead]
    created_at: datetime

class NoteUpdate(BaseNote):
    title: str
    sections: List[BaseSectionUpdate] = []