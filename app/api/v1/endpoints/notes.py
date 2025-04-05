from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Any

from app.models.models import Note, Section
from app.schemas.note import NoteCreate, NoteRead, NoteUpdate
from app.db.local_session import DatabaseManager
import logging
from datetime import date, datetime

# Configure logger
logger = logging.getLogger(__name__)
router = APIRouter()
get_session = DatabaseManager().get_session

# Helper function to parse date strings
def parse_date(date_str: str) -> date:
    """Parse date string to date object"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

@router.post("/create", status_code=status.HTTP_201_CREATED)
def create_note(note_in: NoteCreate = Body(...), db: Session = Depends(get_session)):
    """
    Create a new note with sections
    
    Args:
        note_in: Note creation data
        db: Database session
        
    Returns:
        Created note
    """
    try:
        logger.info(f"Creating note: {note_in.title} for user {note_in.user_id}")
        
        # First create the note without sections
        db_note = Note(title=note_in.title, encounter_date=note_in.encounter_date, patient_id=note_in.patient_id, user_id=note_in.user_id)

        # Add note to session and flush to get the ID
        db.add(db_note)
        db.flush()  # This gets the ID without committing
        
        # Add sections if provided
        if note_in.sections:
            for section_data in note_in.sections:
                # Create section with the new note_id
                db_section = Section(
                    note_id=db_note.id,
                    user_id=db_note.user_id,
                    title=section_data.title,
                    content=section_data.content if hasattr(section_data, 'content') else {},
                    section_type=getattr(section_data, 'section_type', 'OTHERS'),
                    section_description=getattr(section_data, 'section_description', '')
                )
                db.add(db_section)
        
        # Commit the transaction
        db.commit()
        db.refresh(db_note)
        
        logger.info(f"Created note with ID: {db_note.id}")
        return db_note
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating note: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create note: {str(e)}"
        )

# For debugging validation issues
@router.post("/debug", status_code=status.HTTP_201_CREATED)
def debug_note_creation(data: Dict[str, Any] = Body(...)):
    """
    Debug endpoint to validate note creation data
    """
    from pydantic import ValidationError
    try:
        # Try to validate the data against NoteCreate schema
        note_in = NoteCreate(**data)
        return {
            "success": True,
            "validated_data": {
                "title": note_in.title,
                "user_id": note_in.user_id,
                "patient_id": note_in.patient_id,
                "encounter_date": str(note_in.encounter_date),
                "sections_count": len(note_in.sections)
            }
        }
    except ValidationError as e:
        return {
            "success": False,
            "errors": e.errors()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/", response_model=List[NoteRead])
def list_notes(db: Session = Depends(get_session)):
    """
    List all notes
    """
    notes = db.query(Note).options(joinedload(Note.sections)).all()
    return notes

@router.get("/{note_id}", response_model=NoteRead)
def get_note(note_id: int, db: Session = Depends(get_session)):
    """
    Get a specific note by ID
    """
    db_note = db.query(Note).filter(Note.id == note_id).first()
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
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )

    # Update basic note attributes
    if note_in.title is not None:
        db_note.title = note_in.title
    if note_in.patient_id is not None:
        db_note.patient_id = note_in.patient_id
    if note_in.encounter_date is not None:
        db_note.encounter_date = note_in.encounter_date
    
    # Update sections if provided
    if note_in.sections:
        # This would require more complex logic to update/delete/add sections
        # For simplicity, we'll just log that sections are being ignored
        logger.warning("Note section updates are not implemented in this endpoint")
    
    db.commit()
    db.refresh(db_note)
    return db_note

@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: int, db: Session = Depends(get_session)):
    """
    Delete a note
    """
    db_note = db.query(Note).filter(Note.id == note_id).first()
    if not db_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )

    db.delete(db_note)
    db.commit()
    return None