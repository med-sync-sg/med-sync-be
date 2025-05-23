from typing import List, Optional, Union, Dict, Any
from datetime import date
from pydantic import BaseModel, field_validator
from .section import SectionCreate, SectionRead, SectionUpdate


class NoteCreate(BaseModel):
    """Schema for creating a new note"""
    patient_id: Optional[int] = None
    user_id: int
    title: str
    encounter_date: Union[date, str]
    sections: List[Union[SectionCreate, Dict[str, Any]]] = []

    # Validate and convert string dates to date objects
    @field_validator('encounter_date')
    def parse_encounter_date(cls, value):
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                from datetime import datetime
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError('encounter_date must be in format YYYY-MM-DD')
        raise ValueError('encounter_date must be a date or string in format YYYY-MM-DD')

    # Convert dict sections to NoteSectionCreate
    @field_validator('sections')
    def validate_sections(cls, value):
        if not value:
            return []
        
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(SectionCreate(**item))
            else:
                result.append(item)
        return result
    class Config:
        orm_mode = True
        
class NoteRead(BaseModel):
    """Schema for reading a note"""
    id: int
    title: str
    patient_id: Optional[int]
    user_id: int
    encounter_date: date
    sections: List[Union[SectionRead, Dict[str, Any]]] = []
    class Config:
        orm_mode = True

class NoteUpdate(BaseModel):
    """Schema for updating a note"""
    title: Optional[str] = None
    patient_id: Optional[int] = None
    user_id: int
    encounter_date: Optional[date] = None
    sections: List[Union[SectionUpdate, Dict[str, Any]]] = []
    class Config:
        orm_mode = True