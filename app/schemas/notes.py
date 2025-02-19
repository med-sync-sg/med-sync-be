from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Annotated
from datetime import date, datetime
from .section import BaseSectionCreate, BaseSectionRead, BaseSectionUpdate, ChiefComplaintSectionCreate, ChiefComplaintSectionRead, ChiefComplaintSectionUpdate, PatientInformationSectionCreate, PatientInformationSectionRead, PatientInformationSectionUpdate

class BaseNote(BaseModel):
    schema_version: int = 1
    consultation_id: int
    note_id: int
    patient_id: int
    encounter_date: date
    sections: List = Field(default_factory=list)

SectionCreateUnion = Annotated[
    Union[BaseSectionCreate, ChiefComplaintSectionCreate, PatientInformationSectionCreate],
    Field(discriminator="section_type")
]
SectionReadUnion = Annotated[
    Union[BaseSectionRead, ChiefComplaintSectionRead, PatientInformationSectionRead],
    Field(discriminator="section_type")
]
SectionUpdateUnion = Annotated[
    Union[BaseSectionUpdate, ChiefComplaintSectionUpdate, PatientInformationSectionUpdate],
    Field(discriminator="section_type")
]

class NoteCreate(BaseNote):
    title: str
    sections: List[SectionCreateUnion] = []

class NoteRead(BaseNote):
    title: str
    sections: List[SectionReadUnion]
    created_at: datetime

class NoteUpdate(BaseNote):
    title: str
    sections: List[SectionCreateUnion] = []