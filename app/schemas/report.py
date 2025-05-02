from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.schemas.base import BaseAuthModel

class ReportSectionConfig(BaseModel):
    """Configuration for a section in a report template"""
    section_type: str  # Matches section_type in Section model
    include: bool = True
    order: int  # Display order in the report
    title_override: Optional[str] = None
    format_options: Dict[str, Any] = {}  # Formatting options 
    
class ReportTemplateBase(BaseModel):
    """Base schema for report templates"""
    name: str
    description: Optional[str] = None
    report_type: str  # "doctor", "patient", "custom" 
    is_default: bool = False
    
class ReportTemplateCreate(ReportTemplateBase, BaseAuthModel):
    """Schema for creating a report template"""
    html_template: Optional[str] = None
    template_data: Dict[str, Any]  # Should contain 'sections' key with section configs
    
class ReportTemplateRead(ReportTemplateBase):
    """Schema for reading a report template"""
    id: int
    user_id: int
    html_template: Optional[str] = None
    template_data: Dict[str, Any]
    created_at: str
    updated_at: str
    
    class Config:
        orm_mode = True
        
class ReportTemplateUpdate(BaseAuthModel):
    """Schema for updating a report template"""
    name: Optional[str] = None
    description: Optional[str] = None
    report_type: Optional[str] = None
    is_default: Optional[bool] = None
    html_template: Optional[str] = None
    template_data: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True