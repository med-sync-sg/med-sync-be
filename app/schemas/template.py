from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Annotated
from datetime import date, datetime
from .section import BaseSectionCreate, BaseSectionRead, BaseSectionUpdate
from app.schemas.section import TextCategoryEnum 

class BaseSectionTemplate(BaseModel):
    section_id: int
    title: str
    description: str
    metadata: List[str] = []
    content: List[str] = []
    order: int = 0
    section_type: str = Field(default=TextCategoryEnum.OTHERS.name)

class BaseSectionTemplateCreate(BaseSectionTemplate):
    title: str
    description: str
    order: int
    content: List[str] = []
    order: int = 0
    section_type: Optional[str] = Field(default=TextCategoryEnum.OTHERS.name)

class BaseSectionTemplateUpdate(BaseSectionTemplate):
    pass # Most fields are optional by deafult

class BaseSectionTemplateRead(BaseSectionTemplate):
    pass

class BaseNoteTemplate(BaseModel):
    schema_version: int = 1
    consultation_id: int
    note_id: int
    patient_id: int
    encounter_date: date
    sections: List = Field(default_factory=list)

class NoteTemplateCreate(BaseNoteTemplate):
    title: str
    sections: List[BaseSectionCreate] = []

class NoteTemplateRead(BaseNoteTemplate):
    title: str
    sections: List[BaseSectionRead]
    created_at: datetime

class NoteTemplateUpdate(BaseNoteTemplate):
    title: str
    sections: List[BaseSectionUpdate] = []