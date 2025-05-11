from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.local_session import DatabaseManager
from app.models.models import User
from app.schemas.report.report_template import (
    ReportTemplateRead,
    ReportTemplateCreate,
    ReportTemplateUpdate,
    ReportTemplateSectionConfigRead
)
from app.services.report_generation.report_template_service import ReportTemplateService
from app.utils.auth_utils import get_current_user

router = APIRouter()
get_session = DatabaseManager().get_session

@router.get("/", response_model=List[ReportTemplateRead])
async def get_templates(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    template_type: Optional[str] = Query(None, description="Filter by template type"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all available report templates.
    
    Optionally filter by user_id or template_type.
    Users can see their own templates and default system templates.
    """
    template_service = ReportTemplateService(db)
    
    # Get templates
    templates = template_service.get_all_templates(user_id)
    
    # Filter by template type if specified
    if template_type:
        templates = [t for t in templates if t.template_type == template_type]
    
    # Filter for access control (users can see their own and default templates)
    accessible_templates = []
    for template in templates:
        if template.user_id == current_user.id or template.is_default:
            accessible_templates.append(template)
    
    return accessible_templates

@router.get("/{template_id}", response_model=ReportTemplateRead)
async def get_template(
    template_id: int = Path(..., description="Template ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get a specific report template by ID.
    
    Users can access their own templates and default system templates.
    """
    template_service = ReportTemplateService(db)
    
    # Get template
    template = template_service.get_template_by_id(template_id)
    
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template with ID {template_id} not found"
        )
    
    # Check access (users can access their own and default templates)
    if template.user_id != current_user.id and not template.is_default:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this template"
        )
    
    return template

@router.post("/", response_model=ReportTemplateRead, status_code=status.HTTP_201_CREATED)
async def create_template(
    template_data: ReportTemplateCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Create a new report template.
    """
    # Ensure user_id in the request matches the authenticated user
    if template_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create templates for yourself"
        )
    
    template_service = ReportTemplateService(db)
    
    # Create template
    new_template = template_service.create_template(template_data)
    
    return new_template

@router.put("/{template_id}", response_model=ReportTemplateRead)
async def update_template(
    template_id: int,
    template_data: ReportTemplateUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update an existing report template.
    
    Users can only update their own templates.
    """
    # Ensure user_id in the request matches the authenticated user
    if template_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own templates"
        )
    
    template_service = ReportTemplateService(db)
    
    # First verify template exists and belongs to user
    existing_template = template_service.get_template_by_id(template_id)
    if not existing_template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    
    if existing_template.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this template"
        )
    
    # Update the template
    updated_template = template_service.update_template(template_id, template_data)
    
    return updated_template

@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Delete a report template.
    
    Users can only delete their own templates.
    """
    template_service = ReportTemplateService(db)
    
    # First verify template exists and belongs to user
    existing_template = template_service.get_template_by_id(template_id)
    if not existing_template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    
    if existing_template.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this template"
        )
    
    # Prevent deletion of default templates if not an admin
    if existing_template.is_default and current_user.id != 1:  # Assuming admin has ID 1
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Default templates cannot be deleted"
        )
    
    # Delete the template
    success = template_service.delete_template(template_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete template"
        )
    
    return None

@router.get("/defaults/{template_type}", response_model=ReportTemplateRead)
async def get_default_template(
    template_type: str = Path(..., description="Template type (e.g., 'doctor', 'patient')"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get the default template for a specific type.
    
    Available to all authenticated users.
    """
    template_service = ReportTemplateService(db)
    
    # Get default template for the specified type
    template = template_service.get_default_template(template_type)
    
    if not template:
        # If no default template exists, create one
        template_service.create_default_templates()
        template = template_service.get_default_template(template_type)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No default template found for type '{template_type}'"
            )
    
    return template

@router.get("/{template_id}/sections", response_model=List[ReportTemplateSectionConfigRead])
async def get_template_sections(
    template_id: int = Path(..., description="Template ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all section configurations for a template.
    
    Users can access sections from their own templates and default templates.
    """
    template_service = ReportTemplateService(db)
    
    # First verify template exists and user has access
    template = template_service.get_template_by_id(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    
    # Check access (users can access their own and default templates)
    if template.user_id != current_user.id and not template.is_default:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this template"
        )
    
    # Get section configurations
    sections = template_service.get_section_configs(template_id)
    
    return sections