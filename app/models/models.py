import json
from pydantic import BaseModel, Field
from typing import Union, Optional, List, Any

class Concept(BaseModel):
    id : str
    cui : str
    name : str
    definition : str


class MedicalEntry(BaseModel):
    topic: str = Field(default="")
    entry: Optional[Union[str, int, float, List, 'MedicalEntry', List['MedicalEntry']]] = None
    parent: Optional['MedicalEntry'] = None

    class Config:
        json_encoders = {
            'MedicalEntry': lambda v: v.to_dict(exclude={'parent'})
        }

class ConsultationModel(BaseModel):
    entries: List[MedicalEntry]
    def __init__(self):
        self.entries = []

class SOAPModel(ConsultationModel):
    subjective_entries: List[MedicalEntry]
    objective_entries: List[MedicalEntry]
    assessment_entries: List[MedicalEntry]
    plan_entries: List[MedicalEntry]
    
    def __init__(self):
        super(self)
        self.subjective_entries = []
        self.objective_entries = []
        self.assessment_entries = []
        self.plan_entries = []