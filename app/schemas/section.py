from app.schemas.base import BaseAuthModel
from pydantic import BaseModel, Field
from typing import Dict, Any, Union
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

class SectionCreate(BaseAuthModel):
    note_id: int
    title: str
    content: Dict[str, Any] = {}
    section_type: str = Field(default=TextCategoryEnum.OTHERS.value)
    section_description: str = Field(default=TextCategoryEnum.OTHERS.value)

    class Config:
        orm_mode = True

class SectionRead(BaseAuthModel):
    id: int
    note_id: int
    title: str
    content: Union[Dict[str, Any], None]
    section_type: str = Field(default=TextCategoryEnum.OTHERS.value)
    section_description: str = Field(default=TextCategoryEnum.OTHERS.value)

    class Config:
        orm_mode = True

class SectionUpdate(BaseAuthModel):
    note_id: int
    title: Union[str, None] = None
    content: Union[Dict[str, Any], None] = None
    section_type: Union[str, None] = None
    section_description: Union[str, None] = None

    class Config:
        orm_mode = True