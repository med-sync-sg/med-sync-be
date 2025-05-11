from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from fastapi.responses import StreamingResponse
import io

from app.db.local_session import DatabaseManager
from app.models.models import User, Note, ReportInstance, ReportSection, ReportField
from app.schemas.report.report_instance import (
    ReportInstanceRead,
    ReportInstanceCreate,
    ReportInstanceUpdate,
    ReportSectionRead,
    ReportFieldRead
)
from app.services.report_generation.report_instance_service import ReportInstanceService
from app.services.report_generation.report_template_service import ReportTemplateService
from app.utils.auth_utils import get_current_user

router = APIRouter()
get_session = DatabaseManager().get_session

@router.get("/", response_model=List[ReportInstanceRead])
async def get_report_instances(
    note_id: Optional[int] = Query(None, description="Filter by note ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all report instances for the current user.
    
    Optionally filter by note_id.
    """
    report_service = ReportInstanceService(db)
    
    # Get reports for the current user, optionally filtered by note_id
    reports = report_service.get_report_instances(user_id=current_user.id, note_id=note_id)
    
    return reports

@router.get("/{report_id}", response_model=ReportInstanceRead)
async def get_report_instance(
    report_id: int = Path(..., description="Report instance ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get a specific report instance by ID.
    
    Users can only access their own reports.
    """
    report_service = ReportInstanceService(db)
    
    # Get report instance
    report = report_service.get_report_instance_by_id(report_id)
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report instance with ID {report_id} not found"
        )
    
    # Check access (users can only access their own reports)
    if report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this report"
        )
    
    return report

@router.post("/", response_model=ReportInstanceRead, status_code=status.HTTP_201_CREATED)
async def create_report_instance(
    note_id: int = Body(..., embed=True),
    template_id: int = Body(..., embed=True),
    name: str = Body(..., embed=True),
    description: Optional[str] = Body(None, embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Create a new report instance from a note using a template.
    """
    report_service = ReportInstanceService(db)
    
    # Create report instance
    new_report = report_service.create_report_instance_from_note(
        note_id=note_id,
        template_id=template_id,
        user_id=current_user.id,
        name=name,
        description=description
    )
    
    return new_report

@router.put("/{report_id}", response_model=ReportInstanceRead)
async def update_report_instance(
    report_id: int,
    update_data: ReportInstanceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update an existing report instance.
    
    Users can only update their own reports.
    """
    # Ensure user_id in the request matches the authenticated user
    if update_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own reports"
        )
    
    report_service = ReportInstanceService(db)
    
    # First verify report exists and belongs to user
    existing_report = report_service.get_report_instance_by_id(report_id)
    if not existing_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if existing_report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this report"
        )
    
    # Update the report
    updated_report = report_service.update_report_instance(report_id, update_data)
    
    return updated_report

@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report_instance(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Delete a report instance.
    
    Users can only delete their own reports.
    """
    report_service = ReportInstanceService(db)
    
    # First verify report exists and belongs to user
    existing_report = report_service.get_report_instance_by_id(report_id)
    if not existing_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if existing_report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this report"
        )
    
    # Delete the report
    success = report_service.delete_report_instance(report_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete report"
        )
    
    return None

@router.get("/{report_id}/sections", response_model=List[ReportSectionRead])
async def get_report_sections(
    report_id: int = Path(..., description="Report instance ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all sections for a report instance.
    
    Users can only access sections from their own reports.
    """
    report_service = ReportInstanceService(db)
    
    # First verify report exists and user has access
    report = report_service.get_report_instance_by_id(report_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    # Check access (users can only access their own reports)
    if report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this report"
        )
    
    # Get sections
    sections = report_service.get_report_sections(report_id)
    
    return sections

@router.get("/sections/{section_id}/fields", response_model=List[ReportFieldRead])
async def get_section_fields(
    section_id: int = Path(..., description="Report section ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get all fields for a report section.
    
    Users can only access fields from their own reports.
    """
    report_service = ReportInstanceService(db)
    
    # First verify section exists and get its report instance
    section = db.query(ReportSection).filter(ReportSection.id == section_id).first()
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Section not found"
        )
    
    # Get report instance
    report = db.query(ReportInstance).filter(ReportInstance.id == section.report_instance_id).first()
    
    # Check access (users can only access their own reports)
    if report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this section"
        )
    
    # Get fields
    fields = report_service.get_report_fields(section_id)
    
    return fields

@router.put("/{report_id}/section-order")
async def update_section_order(
    report_id: int,
    section_orders: List[Dict[str, Any]] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update the order of sections in a report instance.
    
    This enables drag-and-drop rearrangement of sections.
    """
    report_service = ReportInstanceService(db)
    
    # Update section order
    success = report_service.update_section_order(
        report_id=report_id,
        user_id=current_user.id,
        section_orders=section_orders
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update section order"
        )
    
    return {"success": True, "message": "Section order updated successfully"}

@router.put("/sections/{section_id}/field-order")
async def update_field_order(
    section_id: int,
    field_orders: List[Dict[str, Any]] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update the order of fields in a report section.
    
    This enables drag-and-drop rearrangement of fields within a section.
    """
    report_service = ReportInstanceService(db)
    
    # Update field order
    success = report_service.update_field_order(
        section_id=section_id,
        user_id=current_user.id,
        field_orders=field_orders
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update field order"
        )
    
    return {"success": True, "message": "Field order updated successfully"}

@router.put("/sections/{section_id}/visibility")
async def toggle_section_visibility(
    section_id: int,
    is_visible: bool = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Toggle visibility of a report section.
    """
    report_service = ReportInstanceService(db)
    
    # Toggle section visibility
    section = report_service.toggle_section_visibility(
        section_id=section_id,
        user_id=current_user.id,
        is_visible=is_visible
    )
    
    return {"success": True, "message": f"Section visibility set to {is_visible}", "section": section}

@router.put("/fields/{field_id}/visibility")
async def toggle_field_visibility(
    field_id: int,
    is_visible: bool = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Toggle visibility of a report field.
    """
    report_service = ReportInstanceService(db)
    
    # Toggle field visibility
    field = report_service.toggle_field_visibility(
        field_id=field_id,
        user_id=current_user.id,
        is_visible=is_visible
    )
    
    return {"success": True, "message": f"Field visibility set to {is_visible}", "field": field}

@router.put("/sections/{section_id}/title")
async def update_section_title(
    section_id: int,
    new_title: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update the title of a report section.
    """
    report_service = ReportInstanceService(db)
    
    # Update section title
    section = report_service.update_section_title(
        section_id=section_id,
        user_id=current_user.id,
        new_title=new_title
    )
    
    return {"success": True, "message": "Section title updated successfully", "section": section}

@router.put("/fields/{field_id}/name")
async def update_field_name(
    field_id: int,
    new_name: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update the display name of a report field.
    """
    report_service = ReportInstanceService(db)
    
    # Update field name
    field = report_service.update_field_name(
        field_id=field_id,
        user_id=current_user.id,
        new_name=new_name
    )
    
    return {"success": True, "message": "Field name updated successfully", "field": field}

@router.put("/fields/{field_id}/value")
async def update_field_value(
    field_id: int,
    new_value: Any = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Update the value of a report field.
    """
    report_service = ReportInstanceService(db)
    
    # Update field value
    field = report_service.update_field_value(
        field_id=field_id,
        user_id=current_user.id,
        new_value=new_value
    )
    
    return {"success": True, "message": "Field value updated successfully", "field": field}

@router.post("/{report_id}/finalize", response_model=ReportInstanceRead)
async def finalize_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Finalize a report instance (mark as complete and ready for PDF generation).
    
    Finalized reports cannot be modified.
    """
    report_service = ReportInstanceService(db)
    
    # Finalize report
    report = report_service.finalize_report(
        report_id=report_id,
        user_id=current_user.id
    )
    
    return report

@router.post("/notes/{note_id}/create-report", response_model=ReportInstanceRead, status_code=status.HTTP_201_CREATED)
async def create_report_from_note(
    note_id: int,
    template_id: int = Body(..., embed=True),
    name: str = Body(..., embed=True),
    description: Optional[str] = Body(None, embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Convenience endpoint to create a report directly from a note.
    """
    # First verify note exists and belongs to user
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    if note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this note"
        )
    
    # Create report using the service
    report_service = ReportInstanceService(db)
    report = report_service.create_report_instance_from_note(
        note_id=note_id,
        template_id=template_id,
        user_id=current_user.id,
        name=name,
        description=description
    )
    
    return report

@router.post("/{report_id}/clone", response_model=ReportInstanceRead, status_code=status.HTTP_201_CREATED)
async def clone_report(
    report_id: int,
    new_name: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Create a copy of an existing report instance.
    
    Useful for creating templates or revisions.
    """
    report_service = ReportInstanceService(db)
    
    # First verify report exists and belongs to user
    existing_report = report_service.get_report_instance_by_id(report_id)
    if not existing_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if existing_report.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to clone this report"
        )
    
    # Create a new report based on the existing one
    new_report = report_service.create_report_instance_from_note(
        note_id=existing_report.note_id,
        template_id=existing_report.template_id,
        user_id=current_user.id,
        name=new_name,
        description=f"Clone of report {report_id}: {existing_report.name}"
    )
    
    # Note: This is a simple clone implementation. For a full clone with 
    # copied customizations, we would need to add a dedicated clone method 
    # to the service that copies all custom layouts, field values, etc.
    
    return new_report