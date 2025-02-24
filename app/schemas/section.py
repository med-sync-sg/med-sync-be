from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import date, datetime
import json
from sqlalchemy import Engine

from enum import Enum

class TextCategoryEnum(str, Enum):
    """
    An Enum class representing the categories of text chunks (usually sentence-length or shorter)
    The name acts as the category name, and the value is the decsription of that category.
    """
    CHIEF_COMPLAINT = "This text describes the patient’s primary symptoms or issues that prompted the visit, such as pain, discomfort, or concern, usually stated at the beginning of a consultation. Examples: 'I have had severe headaches for 2 days...', 'I'm having some real bad diarrhea...'"
    PATIENT_INFORMATION="This text describes the demographic and personal details such as age, occupation, address, occupation, and many other details."
    PATIENT_MEDICAL_HISTORY ="This text describes the patient's medical history. This part can be very different to the ChiefComplaint part."
    OTHERS ="This text refers to all other contents not classified as the CHIEF_COMPLAINT, PATIENT_INFORMATION, PATIENT_MEDICAL_HISTORY categories."

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
    pass # Most fields are optional by deafult

class BaseSectionRead(BaseSection):
    pass