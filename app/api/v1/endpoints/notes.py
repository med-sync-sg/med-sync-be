from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from typing import List

from app.models.models import Note
from app.schemas.note import NoteCreate, NoteRead, NoteUpdate
from app.db.local_session import get_db

router = APIRouter()

@router.post("/", response_model=NoteRead, status_code=201)
def create_note(note_in: NoteCreate, db: Session = Depends(get_db)):
    # Convert Pydantic sections to a JSON-serializable list of dicts
    db_note = Note(
        title=note_in.title,
        sections=note_in.sections,
        patient_id=note_in.patient_id,
        user_id=note_in.user_id,
        encounter_date=note_in.encounter_date
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    
    return db_note  # FastAPI auto-converts to NoteRead


@router.get("/", response_model=List[NoteRead], status_code=200)
def list_notes(db: Session = Depends(get_db)):
    notes = db.query(Note).options(joinedload(Note.sections)).all()
    return notes

@router.get("/{note_id}", response_model=NoteRead)
def get_note(note_id: int, db: Session = Depends(get_db)):
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")
    return db_note


@router.put("/{note_id}", response_model=NoteUpdate)
def update_note(note_id: int, note_in: NoteCreate, db: Session = Depends(get_db)):
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Overwrite fields
    db_note.title = note_in.title
    db_note.sections = [section.model_dump() for section in note_in.sections]
    db.commit()
    db.refresh(db_note)
    return db_note


@router.delete("/{note_id}", status_code=204)
def delete_note(note_id: int, db: Session = Depends(get_db)):
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(db_note)
    db.commit()
    return None