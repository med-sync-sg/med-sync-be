from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import json

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
    section_type: Optional[str] = None


class ChiefComplaintContent(BaseModel):
    """Structure for the 'content' of a Chief Complaint."""
    symptoms: List[str]
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    modifying_factors: Optional[List[str]] = None


class ChiefComplaintSection(Section):
    """
    Specialized section. 
    We'll override the content with a stricter type,
    but from the DB perspective, it's still the same row structure.
    """
    section_type: str = "chief_complaint"
    content: ChiefComplaintContent

class ConsultationNote(BaseModel):
    schema_version: int = 1
    consultation_id: str
    patient_id: str
    encounter_date: date
    sections: List[Section] = Field(default_factory=list)
    
def insert_section(conn, section: Section, note_id: str):
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
            cc_content = ChiefComplaintContent(**content_dict)
            return ChiefComplaintSection(
                section_id=sec_id,
                title=title,
                section_type=sec_type,
                metadata=metadata_dict,
                content=cc_content
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