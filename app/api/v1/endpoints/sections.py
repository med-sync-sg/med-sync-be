from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from app.db.local_session import DatabaseManager
from app.db.neo4j_session import get_neo4j_session, Neo4jSession
from app.models.models import User, Section
from app.schemas.section import SectionCreate, SectionRead, SectionUpdate, SectionFieldUpdate, FieldValueUpdate
from app.services.note_service import NoteService
from app.api.v1.endpoints.auth import get_current_user

router = APIRouter()
get_session = DatabaseManager().get_session

@router.get("/{section_id}", response_model=SectionRead)
async def get_section(
    section_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get a specific section by ID
    """
    note_service = NoteService(db)
    section = db.query(Section).filter(Section.id == section_id).first()
    
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Section not found"
        )
    
    # Check if user owns the section
    if section.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this section"
        )
    
    return section

@router.put("/{section_id}", response_model=SectionRead)
async def update_section(
    section_id: int,
    section_data: SectionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update a section
    """
    # First verify section exists and belongs to user
    section = db.query(Section).filter(Section.id == section_id).first()
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Section not found"
        )
    
    if section.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this section"
        )
    
    # Update the section
    note_service = NoteService(db)
    updated_section = note_service.update_section(section_id, section_data)
    
    if updated_section is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update section"
        )
    
    return updated_section

@router.delete("/{section_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_section(
    section_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Delete a section
    """
    # First verify section exists and belongs to user
    section = db.query(Section).filter(Section.id == section_id).first()
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Section not found"
        )
    
    if section.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this section"
        )
    
    # Delete the section
    note_service = NoteService(db)
    success = note_service.delete_section(section_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete section"
        )
    
    return None
