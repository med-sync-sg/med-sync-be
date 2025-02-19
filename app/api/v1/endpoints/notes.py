from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import DataStore
from app.models.models import Note
from app.schemas.notes import NoteCreate, NoteRead, NoteUpdate

router = APIRouter()
data_store = DataStore()

@router.post("/notes", response_model=NoteRead, status_code=201)
def create_note(note_in: NoteCreate, db: Session = Depends(DataStore.get_db())):
    # Convert Pydantic sections to a JSON-serializable list of dicts
    sections_data = [section.model_dump() for section in note_in.sections]

    db_note = Note(
        title=note_in.title,
        sections=sections_data
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note  # FastAPI auto-converts to NoteRead


@router.get("/notes", response_model=List[NoteRead])
def list_notes(db: Session = Depends(data_store)):
    notes = db.query(Note).all()
    return notes

@router.get("/notes/{note_id}", response_model=NoteRead)
def get_note(note_id: int, db: Session = Depends(data_store.get_db())):
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")
    return db_note


@router.put("/notes/{note_id}", response_model=NoteUpdate)
def update_note(note_id: int, note_in: NoteCreate, db: Session = Depends(data_store.get_db())):
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Overwrite fields
    db_note.title = note_in.title
    db_note.sections = [section.model_dump() for section in note_in.sections]
    db.commit()
    db.refresh(db_note)
    return db_note


@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int, db: Session = Depends(data_store.get_db())):
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(db_note)
    db.commit()
    return None