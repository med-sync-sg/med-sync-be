from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from app.schemas import ConsultationNote

Base = declarative_base()

class ConsultationNoteRecord(Base):
    __tablename__ = "consultation_notes"

    consultation_id = Column(String, primary_key=True)
    patient_id = Column(String)
    # The entire Pydantic-defined note structure stored as JSON
    note_data = Column(JSONB)
    
    
def save_consultation_note(session, note: ConsultationNote):
    record = ConsultationNoteRecord(
        consultation_id=note.consultation_id,
        patient_id=note.patient_id,
        note_data=note.dict()  # serialize to a Python dict for storage
    )
    session.add(record)
    session.commit()
    
def get_consultation_note(session, consultation_id: str) -> ConsultationNote:
    record = session.query(ConsultationNoteRecord).filter_by(consultation_id=consultation_id).first()
    if record is None:
        raise ValueError("Not found")
    # Re-construct the Pydantic model from the stored dict
    return ConsultationNote(**record.note_data)