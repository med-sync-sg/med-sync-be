from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models.models import Note
from app.schemas.note import NoteCreate, NoteRead, NoteUpdate
from app.db.local_session import DatabaseManager
from app.services.note_service import NoteService

router = APIRouter()
get_session = DatabaseManager.get_session

@router.post("/", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
def create_note(note_in: NoteCreate, db: Session = Depends(get_session)):
    """
    Create a new note with sections
    """
    # Initialize the note service with DB session
    note_service = NoteService(db)
    
    # Use service to create note
    db_note = note_service.create_note(note_in)
    
    if not db_note:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create note"
        )
    
    return db_note  # FastAPI auto-converts to NoteRead


@router.get("/", response_model=List[NoteRead])
def list_notes(user_id: int = None, patient_id: int = None, db: Session = Depends(get_session)):
    """
    List notes with optional filtering by user_id or patient_id
    """
    note_service = NoteService(db)
    
    if user_id:
        notes = note_service.get_notes_by_user(user_id)
    elif patient_id:
        notes = note_service.get_notes_by_patient(patient_id)
    else:
        # This would be handled by a repository in a full implementation
        notes = db.query(Note).all()
    
    return notes


@router.get("/{note_id}", response_model=NoteRead)
def get_note(note_id: int, db: Session = Depends(get_session)):
    """
    Get a specific note by ID
    """
    note_service = NoteService(db)
    db_note = note_service.get_note_by_id(note_id)
    
    if not db_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    return db_note


@router.put("/{note_id}", response_model=NoteRead)
def update_note(note_id: int, note_in: NoteUpdate, db: Session = Depends(get_session)):
    """
    Update an existing note
    """
    note_service = NoteService(db)
    updated_note = note_service.update_note(note_id, note_in)
    
    if not updated_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    return updated_note


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: int, db: Session = Depends(get_session)):
    """
    Delete a note
    """
    note_service = NoteService(db)
    success = note_service.delete_note(note_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    return None