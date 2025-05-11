from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.base import BaseAuthModel
from datetime import datetime

class ReportTemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    template_type: str  # "doctor", "patient", etc.
    is_default: bool = False
    layout_config: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportTemplateCreate(ReportTemplateBase, BaseAuthModel):
    html_template: Optional[str] = None
    section_configs: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportTemplateRead(ReportTemplateBase):
    id: int
    user_id: int
    html_template: Optional[str] = None
    version: str
    created_at: datetime
    updated_at: datetime
    section_configs: List["ReportTemplateSectionConfigRead"] = []
    
    model_config = ConfigDict(
        from_attributes=True
    )

class ReportTemplateUpdate(BaseAuthModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template_type: Optional[str] = None
    is_default: Optional[bool] = None
    layout_config: Optional[Dict[str, Any]] = None
    html_template: Optional[str] = None
    version: Optional[str] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportTemplateSectionConfigBase(BaseModel):
    soap_category: str
    display_order: int
    title: str
    is_visible: bool = True
    field_mappings: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportTemplateSectionConfigCreate(ReportTemplateSectionConfigBase):
    field_configs: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportTemplateSectionConfigRead(ReportTemplateSectionConfigBase):
    id: int
    template_id: int
    created_at: datetime
    updated_at: datetime
    field_configs: List["ReportTemplateFieldConfigRead"] = []
    
    model_config = ConfigDict(
        from_attributes=True
    )
        
class ReportTemplateSectionConfigUpdate(BaseModel):
    soap_category: Optional[str] = None
    display_order: Optional[int] = None
    title: Optional[str] = None
    is_visible: Optional[bool] = None
    field_mappings: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )
    
class ReportTemplateFieldConfigBase(BaseModel):
    field_id: str
    display_name: str
    field_type: str
    display_order: int = 0
    is_visible: bool = True

    model_config = ConfigDict(
        from_attributes=True
    )    
    
class ReportTemplateFieldConfigCreate(ReportTemplateFieldConfigBase):
    pass
    
class ReportTemplateFieldConfigRead(ReportTemplateFieldConfigBase):
    id: int
    section_config_id: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(
        from_attributes=True
    )
        
class ReportTemplateFieldConfigUpdate(BaseModel):
    display_name: Optional[str] = None
    field_type: Optional[str] = None
    display_order: Optional[int] = None
    is_visible: Optional[bool] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )