from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.local_session import DatabaseManager
from app.models.models import User, Note
from app.schemas.note import NoteCreate, NoteRead, NoteUpdate
from app.schemas.section import SectionCreate
from app.services.note_service import NoteService
from app.utils.auth_utils import get_current_user

router = APIRouter()
get_session = DatabaseManager().get_session

@router.get("/", response_model=List[NoteRead])
async def get_all_notes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all notes for the authenticated user.
    """
    note_service = NoteService(db)
    note_models = note_service.get_notes_by_user(current_user.id)
    
    # Convert each Note model to a NoteRead schema
    note_read_objects = []
    for note in note_models:
        # Use Pydantic's from_orm to convert the model to a schema
        # This handles the datetime conversion automatically
        note_read = NoteRead.model_validate(note)
        note_read_objects.append(note_read)
    
    return note_read_objects

@router.get("/{note_id}", response_model=NoteRead)
async def get_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get a specific note by ID, ensuring the user owns it.
    """
    note_service = NoteService(db)
    note = note_service.get_note_by_id(note_id)
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    # Check if user owns the note
    if note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this note"
        )
    
    return note

@router.post("/", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
async def create_note(
    note_data: NoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Create a new note, ensuring user_id matches the authenticated user.
    """
    # Verify user_id in the request matches the authenticated user
    if note_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create notes for yourself"
        )
    
    note_service = NoteService(db)
    note = note_service.create_note(note_data)
    
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create note"
        )
    
    return note

@router.put("/{note_id}", response_model=NoteRead)
async def update_note(
    note_id: int,
    note_data: NoteUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update an existing note, ensuring user ownership.
    """
    # Verify user_id in the request matches the authenticated user
    if note_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own notes"
        )
    
    note_service = NoteService(db)
    
    # First verify note exists and belongs to user
    existing_note = note_service.get_note_by_id(note_id)
    if not existing_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    if existing_note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this note"
        )
    
    # Update the note
    updated_note = note_service.update_note(note_id, note_data)
    
    if updated_note is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update note"
        )
    
    return updated_note

@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    note_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Delete a note, ensuring user ownership.
    """
    note_service = NoteService(db)
    
    # First verify note exists and belongs to user
    existing_note = note_service.get_note_by_id(note_id)
    if not existing_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    if existing_note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this note"
        )
    
    # Delete the note
    success = note_service.delete_note(note_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete note"
        )
    
    return None

@router.get("/{user_id}/", response_model=List[NoteRead])
async def get_user_notes(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all notes for a specific user, ensuring the requestor is the same user.
    """
    # Check if the requested user_id matches the authenticated user
    if user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own notes"
        )
    
    note_service = NoteService(db)
    notes = note_service.get_notes_by_user(user_id)
    return notes

@router.post("/{note_id}/sections", response_model=SectionCreate)
async def add_section_to_note(
    note_id: int,
    section_data: SectionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Add a new section to an existing note, ensuring user ownership.
    """
    # Verify user_id in the request matches the authenticated user
    if section_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only add sections to your own notes"
        )
    
    note_service = NoteService(db)
    
    # First verify note exists and belongs to user
    existing_note = note_service.get_note_by_id(note_id)
    if not existing_note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    if existing_note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this note"
        )
    
    # Add section to note
    section = note_service.add_section_to_note(note_id, section_data)
    
    if section is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add section to note"
        )
    
    return section

@router.get("/templates", response_model=List[NoteRead])
async def get_note_templates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all note templates for the authenticated user.
    """
    # In a real implementation, this would use a separate NoteTemplateService
    # For simplicity, we're using the NoteService here
    note_service = NoteService(db)
    
    # This is a placeholder - ideally you would have a dedicated method for templates
    templates = note_service.get_notes_by_user(current_user.id)
    return templates

@router.post("/from-template", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
async def create_note_from_template(
    template_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Create a new note from a template.
    """
    # This is a placeholder implementation
    # In a real app, you would have a dedicated method for creating from template
    
    template_id = template_data.get("template_id")
    patient_id = template_data.get("patient_id")
    
    if not template_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="template_id is required"
        )
    
    note_service = NoteService(db)
    
    # First verify template exists and belongs to user
    template = note_service.get_note_by_id(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    
    if template.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to use this template"
        )
    
    # Create note from template (placeholder implementation)
    # In a real app, you would have a more sophisticated template system
    note_data = NoteCreate(
        user_id=current_user.id,
        title=f"Note from template {template_id}",
        patient_id=patient_id,
        encounter_date=template.encounter_date,
        sections=[]  # In a real app, you would copy sections from the template
    )
    
    note = note_service.create_note(note_data)
    
    if note is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create note from template"
        )
    
    return note