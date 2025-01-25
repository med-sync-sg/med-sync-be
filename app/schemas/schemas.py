from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Diagnosis(BaseModel):
    description: str
    icd_10: Optional[str] = None
    snomed: Optional[str] = None

class Section(BaseModel):
    title: str
    content: str
    # e.g., store numeric or coded data here
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    # store diagnoses if relevant to this section
    diagnoses: List[Diagnosis] = Field(default_factory=list)
    
class ConsultationNote(BaseModel):
    schema_version: int = 1
    consultation_id: str
    patient_id: str
    encounter_date: str
    sections: List[Section] = Field(default_factory=list)