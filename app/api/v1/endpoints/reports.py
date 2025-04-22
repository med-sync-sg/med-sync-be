from fastapi import APIRouter, HTTPException, Depends, Session
from typing import Dict, List

from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse

from app.services.report_service import ReportService
from app.db.local_session import DatabaseManager
from app.schemas.report import ReportTemplateCreate, ReportTemplateRead, ReportTemplateUpdate

router = APIRouter()
get_session = DatabaseManager().get_session

@router.get("/templates/{user_id}", response_model=List[ReportTemplateRead])
def get_report_templates(user_id: int, db: Session = Depends(get_session)):
    """Get all report templates for a user"""
    report_service = ReportService(db)
    return report_service.get_templates_by_user(user_id)

@router.get("/templates/{template_id}/detail", response_model=ReportTemplateRead)
def get_report_template_detail(template_id: int, user_id: int, db: Session = Depends(get_session)):
    """Get detailed information about a report template"""
    report_service = ReportService(db)
    template = report_service.get_template_by_id(template_id, user_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
        
    return template

@router.post("/templates", response_model=ReportTemplateRead)
def create_report_template(template: ReportTemplateCreate, db: Session = Depends(get_session)):
    """Create a new report template"""
    report_service = ReportService(db)
    result = report_service.create_template(template.user_id, template.dict())
    
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create template")
        
    return result

@router.put("/templates/{template_id}", response_model=ReportTemplateRead)
def update_report_template(
    template_id: int, 
    template: ReportTemplateUpdate, 
    db: Session = Depends(get_session)
):
    """Update an existing report template"""
    report_service = ReportService(db)
    result = report_service.update_template(template_id, template.user_id, template.dict(exclude_unset=True))
    
    if not result:
        raise HTTPException(status_code=404, detail="Template not found or update failed")
        
    return result

@router.delete("/templates/{template_id}", response_model=Dict[str, bool])
def delete_report_template(template_id: int, user_id: int, db: Session = Depends(get_session)):
    """Delete a report template"""
    report_service = ReportService(db)
    success = report_service.delete_template(template_id, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Template not found or delete failed")
        
    return {"success": success}

@router.get("/{note_id}/custom/{template_id}")
def generate_custom_report(note_id: int, template_id: int, db: Session = Depends(get_session)):
    """Generate a custom report using a specific template"""
    report_service = ReportService(db)
    report_html = report_service.generate_custom_report(note_id, template_id)
    
    if not report_html:
        raise HTTPException(status_code=404, detail="Failed to generate report")
    
    return HTMLResponse(content=report_html, media_type="text/html")