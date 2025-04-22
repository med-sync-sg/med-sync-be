# app/schemas/report.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.schemas.base import BaseAuthModel

class ReportSectionConfig(BaseModel):
    """Configuration for a section in a report template"""
    section_type: str  # Matches section_type in Section model
    include: bool = True
    order: int  # Display order in the report
    title_override: Optional[str] = None
    format_options: Dict[str, Any] = {}  # Formatting options (font, color, etc.)
    
class ReportTemplateBase(BaseModel):
    """Base schema for report templates"""
    name: str
    description: Optional[str] = None
    report_type: str  # "doctor", "patient", "custom"
    
class ReportTemplateCreate(ReportTemplateBase, BaseAuthModel):
    """Schema for creating a report template"""
    sections: Dict[str, ReportSectionConfig]  # Section configurations keyed by type
    
class ReportTemplateRead(ReportTemplateBase):
    """Schema for reading a report template"""
    id: int
    user_id: int
    sections: Dict[str, ReportSectionConfig]
    created_at: str
    updated_at: str
    
    class Config:
        orm_mode = True
        
class ReportTemplateUpdate(BaseAuthModel):
    """Schema for updating a report template"""
    name: Optional[str] = None
    description: Optional[str] = None
    report_type: Optional[str] = None
    sections: Optional[Dict[str, ReportSectionConfig]] = None
    
    class Config:
        orm_mode = True