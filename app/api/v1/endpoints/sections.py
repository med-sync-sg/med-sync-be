from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from app.db.local_session import DatabaseManager
from app.db.neo4j_session import get_neo4j_session, Neo4jSession
from app.models.models import User, Section
from app.schemas.section import SectionCreate, SectionRead, SectionUpdate, SectionFieldUpdate, FieldValueUpdate
from app.services.note_service import NoteService
from app.services.section_template_integration import TemplateSectionIntegration
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

@router.post("/{section_id}/field-values", response_model=SectionRead)
async def update_section_field_values(
    section_id: int,
    field_updates: List[FieldValueUpdate],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update field values for a section
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
    
    # Update field values
    note_service = NoteService(db)
    updated_section = note_service.update_section_field_values(section_id, field_updates)
    
    if updated_section is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update field values"
        )
    
    return updated_section

@router.get("/{section_id}/template-data", response_model=Dict[str, Any])
async def get_section_with_template(
    section_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """
    Get a section with its template and field values
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
            detail="Not authorized to access this section"
        )
    
    # Get template data
    template_integration = TemplateSectionIntegration(db)
    result = template_integration.get_template_with_field_values(section_id)
    
    if "error" in result and result.get("template") is None:
        if "not found" in result.get("error", ""):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
    
    return result

@router.post("/{section_id}/apply-template/{template_id}", response_model=Dict[str, Any])
async def apply_template_to_section(
    section_id: int,
    template_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """
    Apply a template to an existing section
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
    
    # Apply template
    template_integration = TemplateSectionIntegration(db)
    success = template_integration.apply_template_to_section(section_id, template_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to apply template to section"
        )
    
    # Get updated section with template
    result = template_integration.get_template_with_field_values(section_id)
    return result

@router.post("/create-from-template", response_model=Dict[str, Any])
async def create_section_from_template(
    note_id: int = Query(..., description="Note ID"),
    template_id: str = Query(..., description="Template ID"),
    title: Optional[str] = Query(None, description="Section title"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """
    Create a new section from a template
    """
    # Create section from template
    template_integration = TemplateSectionIntegration(db)
    section_id = template_integration.create_section_from_template(
        note_id=note_id,
        user_id=current_user.id,
        template_id=template_id,
        title=title
    )
    
    if not section_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create section from template"
        )
    
    # Get created section with template
    result = template_integration.get_template_with_field_values(section_id)
    result["created"] = True
    result["section_id"] = section_id
    
    return result

@router.post("/{section_id}/update-content", response_model=SectionRead)
async def update_content_from_fields(
    section_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """
    Update section content based on field values
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
    
    # Update content
    template_integration = TemplateSectionIntegration(db)
    success = template_integration.update_content_from_field_values(section_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update content from field values"
        )
    
    # Get updated section
    db.refresh(section)
    return section