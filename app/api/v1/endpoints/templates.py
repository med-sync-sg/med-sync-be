from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.models.models import NoteTemplate
from app.schemas.template import NoteTemplateCreate, NoteTemplateRead, NoteTemplateUpdate
from app.db.umls_data import SessionMaker

router = APIRouter()


@router.post("/notes", response_model=NoteTemplateRead, status_code=201)
def create_note_template(note_in: NoteTemplateCreate, db: Session = Depends(lambda x: SessionMaker())):
    # Convert Pydantic sections to a JSON-serializable list of dicts
    sections_data = [section.model_dump() for section in note_in.sections]

    db_note = NoteTemplate(
        title=note_in.title,
        sections=sections_data
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note  # FastAPI auto-converts to NoteTemplateRead


@router.get("/notes", response_model=List[NoteTemplateRead])
def list_notes(db: Session = Depends(lambda x: SessionMaker())):
    notes = db.query(NoteTemplate).all()
    return notes

@router.get("/notes/{note_id}", response_model=NoteTemplateRead)
def get_note(note_id: int, db: Session = Depends(lambda x: SessionMaker())):
    db_note = db.query(NoteTemplate).filter(NoteTemplate.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")
    return db_note


@router.put("/notes/{note_id}", response_model=NoteTemplateUpdate)
def update_note(note_id: int, note_in: NoteTemplateCreate, db: Session = Depends(lambda x: SessionMaker())):
    db_note = db.query(NoteTemplate).filter(NoteTemplate.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Overwrite fields
    db_note.title = note_in.title
    db_note.sections = [section.model_dump() for section in note_in.sections]
    db.commit()
    db.refresh(db_note)
    return db_note


@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int, db: Session = Depends(lambda x: SessionMaker())):
    db_note = db.query(NoteTemplate).filter(NoteTemplate.id == note_id).first()
    if not db_note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(db_note)
    db.commit()
    return None