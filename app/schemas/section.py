from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import date, datetime
from .schemas import TextCategoryEnum
import json
from sqlalchemy import Engine

class Diagnosis(BaseModel):
    description: str
    icd_10: Optional[str] = None
    snomed: Optional[str] = None

class BaseSection(BaseModel):
    section_id: int
    title: str
    metadata: Optional[Dict[str, Any]] = None
    content: Dict[str, Any] = {}
    order: int = 1
    section_type: TextCategoryEnum.OTHERS.name = Field(default=TextCategoryEnum.OTHERS.name)
    section_description: str = TextCategoryEnum.OTHERS.value

class BaseSectionCreate(BaseSection):
    title: str
    order: int
    content: Dict[str, Any]

class BaseSectionUpdate(BaseSection):
    pass # Most fields are optional by deafult; add things into here!

class BaseSectionRead(BaseSection):
    pass

class ChiefComplaintSection(BaseSection):
    """
    Specialized section. 
    We'll override the content with a stricter type,
    but from the DB perspective, it's still the same row structure.
    """
    content: List[Dict[str, Any]] = [{"Symptom": "", "Onset": "", "Duration": "", "Severity": ""}]
    section_type: TextCategoryEnum.CHIEF_COMPLAINT.name = Field(default=TextCategoryEnum.CHIEF_COMPLAINT.name)
    section_description: str = TextCategoryEnum.CHIEF_COMPLAINT.value

class ChiefComplaintSectionCreate(BaseSectionCreate, ChiefComplaintSection):
    title: str
    order: int
    content: Dict[str, Any]

class ChiefComplaintSectionUpdate(BaseSectionUpdate, ChiefComplaintSection):
    pass

class ChiefComplaintSectionRead(BaseSectionRead, ChiefComplaintSection):
    pass


    
class PatientInformationSection(BaseSection):
    section_type: str = "patient_information"
    content: Dict[str, Any] = {"age": 0, "name": ""}
    section_type: TextCategoryEnum.PATIENT_INFORMATION.name = Field(default=TextCategoryEnum.CHIEF_COMPLAINT.name)
    section_description: str = TextCategoryEnum.PATIENT_INFORMATION.value

class PatientInformationSectionCreate(BaseSectionCreate, PatientInformationSection):
    pass

class PatientInformationSectionUpdate(BaseSectionUpdate, PatientInformationSection):
    pass

class PatientInformationSectionRead(BaseSectionRead, PatientInformationSection):
    pass