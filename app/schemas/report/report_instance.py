from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.base import BaseAuthModel
from datetime import datetime

class ReportInstanceBase(BaseModel):
    name: str
    description: Optional[str] = None
    note_id: int
    template_id: int
    custom_layout: Optional[Dict[str, Any]] = None
    is_finalized: bool = False
    
class ReportInstanceCreate(ReportInstanceBase, BaseAuthModel):
    # Keep the base fields
    pass
    
class ReportInstanceRead(ReportInstanceBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    sections: List["ReportSectionRead"] = []
    
    model_config = ConfigDict(
        from_attributes=True
    )
        
class ReportInstanceUpdate(BaseAuthModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template_id: Optional[int] = None
    custom_layout: Optional[Dict[str, Any]] = None
    is_finalized: Optional[bool] = None
    
class ReportSectionBase(BaseModel):
    soap_category: str
    title: str
    display_order: int = 0
    is_visible: bool = True
    
class ReportSectionCreate(ReportSectionBase):
    original_section_id: Optional[int] = None
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportSectionRead(ReportSectionBase):
    id: int
    report_instance_id: int
    original_section_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    fields: List["ReportFieldRead"] = []
    
    model_config = ConfigDict(
        from_attributes=True
    )
        
class ReportFieldBase(BaseModel):
    field_id: str
    display_name: str
    field_type: str
    display_order: int = 0
    value: Any = None
    original_value: Any = None
    is_visible: bool = True
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportFieldCreate(ReportFieldBase):
    pass
    
class ReportFieldRead(ReportFieldBase):
    id: int
    report_section_id: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(
        from_attributes=True
    )
        
class ReportFieldUpdate(BaseModel):
    display_name: Optional[str] = None
    display_order: Optional[int] = None
    value: Optional[Any] = None
    is_visible: Optional[bool] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )