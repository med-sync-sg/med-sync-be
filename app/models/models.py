import json
from pydantic import BaseModel, Field
from typing import Union, Optional, List, Any

def generate_primary_concern_template():
    root = Entry(key="Primary Concern", value="Main concern of the patient for this visit", root=None, parent_entry=None, child_entries=[])
    symptoms = Entry(key="Symptoms", value="N/A", parent_entry=root, root=root, child_entries=[]) # The direct child Entry should have the symptom name as the key and other details as its values or child_entries
    recent_medications = Entry(key="Recent Medications", value="N/A", parent_entry=root, root=root, child_entries=[])
    vitals = Entry(key="Vitals", value="N/A", parent_entry=root, root=root, child_entries=[])
    others = Entry(key="others", value="", parent_entry=root, root=root, child_entries=[])
    root.add_child(symptoms)
    root.add_child(recent_medications)
    root.add_child(vitals)
    root.add_child(others)
    return root


class Entry(BaseModel):
    key: str = Field(default="")
    value: str | List[str] = Field(default="")
    root: 'Entry' | None = Field(default=None)
    parent_entry: 'Entry' | None = Field(default=None)
    child_entries: List['Entry'] | 'Entry' | None = Field(default=None)
    
    def add_child(self, child: 'Entry'):
        if child == None:
            raise ValueError("The child cannot be None.")
        self.child_entries.append(child)

class Section(BaseModel):
    section_name: str = Field(default="")
    data: Entry = Field(default=None)
    
class PrimaryConcernSection(Section):
    section_name: str = Field(default="Primary Concern")
    data: Entry = Field(default_factory=generate_primary_concern_template)
    
    
    