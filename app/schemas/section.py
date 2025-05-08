from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Template Field Schema
class TemplateFieldContent(BaseModel):
    """Schema for a template field with content"""
    id: str
    name: str
    data_type: str = "string"
    value: Any = None
    description: Optional[str] = None
    required: bool = False
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

# Section Schemas
class SectionBase(BaseModel):
    """Base schema with common fields for all Section operations"""
    title: str = ""
    template_id: Optional[str] = None
    soap_category: str = "OTHER"
    content: Dict[str, Union[TemplateFieldContent, List[TemplateFieldContent]]] = Field(default_factory=dict)
    is_visible_to_patient: bool = True
    display_order: int = 100
    parent_id: Optional[int] = None
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

class SectionCreate(SectionBase):
    """Schema for creating a new section"""
    note_id: int
    user_id: int
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )
class SectionRead(SectionBase):
    """Schema for reading a section"""
    id: int
    note_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    last_modified_by_id: Optional[int] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

class SectionUpdate(BaseModel):
    """Schema for updating a section"""
    title: Optional[str] = None
    template_id: Optional[str] = None
    soap_category: Optional[str] = None
    content: Optional[Dict[str, Union[TemplateFieldContent, List[TemplateFieldContent]]]] = None
    is_visible_to_patient: Optional[bool] = None
    display_order: Optional[int] = None
    parent_id: Optional[int] = None
    last_modified_by_id: Optional[int] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )