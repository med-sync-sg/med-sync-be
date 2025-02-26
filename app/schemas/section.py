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

CHIEF_COMPLAINT_EXAMPLE = {
  "Main Symptom": {
    "name": "diarrhea",
    "duration": "3 days",
    "severity": "moderate",
    "additional_details": "e.g., frequency, triggers, associated symptoms"
  }
}

PATIENT_INFORMATION_EXAMPLE = {
    "Demographics": {
        "name": "John Doe",
        "age": 45,
        "gender": "Male"
    },
    "Contact Information": {
        "address": "123 Main St, City, Country",
        "phone": "N/A",
        "email": "N/A"
    },
    "Occupation": "Pharmaceutical Manager",
    "Additional Details": "e.g., marital status, living conditions"
}

PATIENT_MEDICAL_HISTORY_EXAMPLE = {
  "Chronic Conditions": [
    {
      "name": "Eczema",
      "diagnosed_date": "YYYY-MM-DD",
      "severity": "mild",
      "notes": "e.g., treatment response, frequency of flare-ups"
    }
  ],
  "Past Diagnoses": [
    {
      "name": "Asthma",
      "diagnosed_date": "YYYY-MM-DD",
      "notes": "e.g., management with inhalers, exacerbation history"
    }
  ],
  "Medications": [
    {
      "name": "Ibuprofen",
      "dosage": "200 mg",
      "frequency": "as needed",
      "notes": "e.g., response, side effects"
    }
  ],
  "Allergies": [
    {
      "substance": "None Known",
      "notes": "NKDA if applicable"
    }
  ]
}

OTHER_EXAMPLE = {
  "Other Observations": [
    {
      "observation": "Lives with flatmates",
      "notes": "Non-smoker, social alcohol consumption"
    },
    {
      "observation": "Additional non-clinical details",
      "notes": "Any other observations not covered in the above sections"
    }
  ]
}
class Diagnosis(BaseModel):
    description: str
    icd_10: Optional[str] = None
    snomed: Optional[str] = None

class BaseSection(BaseModel):
    id: int
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