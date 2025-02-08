from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import json
from sqlalchemy import Engine
from enum import Enum

class TextCategoryEnum(Enum):
    """
    An Enum class representing the categories of text chunks (usually sentence-length or shorter)
    The name acts as the category name, and the value is the decsription of that category.
    """
    CHIEF_COMPLAINT:str="This text describes the patient’s primary symptoms or issues that prompted the visit, such as pain, discomfort, or concern, usually stated at the beginning of a consultation. Examples: 'I have had severe headaches for 2 days...', 'I'm having some real bad diarrhea...'"
    PATIENT_INFORMATION:str="This text describes the demographic and personal details such as age, occupation, address, occupation, and many other details."
    PATIENT_MEDICAL_HISTORY:str="This text describes the patient's medical history. This part can be very different to the ChiefComplaint part."
    OTHERS:str="This text refers to all other contents not classified as the CHIEF_COMPLAINT, PATIENT_INFORMATION, PATIENT_MEDICAL_HISTORY categories."

class User(BaseModel):
    user_id: str
    first_name: str
    middle_name: Optional[str]
    last_name: str

class Diagnosis(BaseModel):
    description: str
    icd_10: Optional[str] = None
    snomed: Optional[str] = None

class Section(BaseModel):
    """
    Base Section model for generic or untyped sections.
    Specialized sections will subclass this,
    but the DB representation is the same (one row in `sections`).
    """
    section_id: str
    title: str
    metadata: Optional[Dict[str, Any]] = None
    # The content field will be stored in the DB as JSONB
    # We keep it generic here (Dict[str, Any]) for base sections.
    content: Dict[str, Any] = {}
    
    # Optional: a "type" discriminator field
    # to identify specialized sections in the DB.
    section_type: str = TextCategoryEnum.OTHERS.name
    section_description: str = TextCategoryEnum[section_type]

class ChiefComplaintContent(BaseModel):
    """Structure for the 'content' of a Chief Complaint."""
    symptoms: List[Dict[str, Any]] = [{"Symptom": "", "Onset": "", "Duration": "", "Severity": ""}]
    main_concern: str = ""

class ChiefComplaintSection(Section):
    """
    Specialized section. 
    We'll override the content with a stricter type,
    but from the DB perspective, it's still the same row structure.
    """
    content: ChiefComplaintContent
    section_type: str = TextCategoryEnum.CHIEF_COMPLAINT.name
    section_description: str = TextCategoryEnum[section_type]


class PatientInformationContent(BaseModel):
    """Structure for the 'content' of a PatientInformationSection."""
    information: Dict[str, Any] = {"age": 0, "name": ""}

    
class PatientInformationSection(Section):
    section_type: str = "patient_information"
    content: PatientInformationContent
    section_type: str = TextCategoryEnum.PATIENT_INFORMATION.name
    section_description: str = TextCategoryEnum[section_type]


class ConsultationNote(BaseModel):
    schema_version: int = 1
    consultation_id: str
    note_id: int
    patient_id: str
    encounter_date: date
    sections: List[Section] = Field(default_factory=list)
    
def insert_section(conn: Engine, section: Section, note_id: str):
    """
    Insert a generic or specialized Section (including ChiefComplaintSection)
    into the sections table.
    """
    with conn.cursor() as cur:
        insert_sql = """
            INSERT INTO sections (
                note_id, section_id, title, section_type, metadata, content
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        cur.execute(
            insert_sql,
            (
                note_id,
                section.section_id,
                section.title,
                section.section_type,
                json.dumps(section.metadata) if section.metadata else None,
                section.content.json() if hasattr(section.content, 'json') else json.dumps(section.content)
            )
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id
    
def fetch_section(conn, section_db_id: int) -> Section:
    with conn.cursor() as cur:
        cur.execute("SELECT note_id, section_id, title, section_type, metadata, content FROM sections WHERE id = %s", (section_db_id,))
        row = cur.fetchone()
        if not row:
            return None
        
        note_id, sec_id, title, sec_type, metadata, content_json = row
        
        # Convert from JSON
        metadata_dict = metadata if metadata is not None else {}
        content_dict = content_json
        
        # Decide which Pydantic model to use based on section_type
        if sec_type == "chief_complaint":
            # Parse the content with ChiefComplaintContent
            content = ChiefComplaintContent(**content_dict)
            return ChiefComplaintSection(
                section_id=sec_id,
                title=title,
                section_type=sec_type,
                metadata=metadata_dict,
                content=content
            )
        elif sec_type == "patient_information":
            content = PatientInformationContent(**content_dict)
            return PatientInformationSection(
                section_id=sec_id,
                title=title,
                section_type=sec_type,
                metadata=metadata_dict,
                content=content
            )
        else:
            # Fallback to a generic Section
            return Section(
                section_id=sec_id,
                title=title,
                section_type=sec_type,
                metadata=metadata_dict,
                content=content_dict
            )