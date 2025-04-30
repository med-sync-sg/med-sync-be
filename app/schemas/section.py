from app.schemas.base import BaseAuthModel
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional, List

class FieldValue(BaseModel):
    """Schema representing a template field value"""
    field_id: str
    name: str
    value: Any
    
class SectionCreate(BaseAuthModel):
    title: str
    template_id: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    soap_category: Optional[str] = None
    field_values: Optional[Dict[str, Any]] = None
    parent_id: Optional[int] = None
    display_order: Optional[int] = None
    is_visible_to_patient: Optional[bool] = True

class SectionRead(BaseAuthModel):
    id: int
    note_id: int
    user_id: int
    title: str
    template_id: Optional[str] = None
    soap_category: str
    content: Optional[Dict[str, Any]] = None
    field_values: Dict[str, Any] = {}
    is_visible_to_patient: bool = True
    display_order: int = 100
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        orm_mode = True

class SectionUpdate(BaseAuthModel):
    id: int
    note_id: int
    title: Optional[str] = None
    template_id: Optional[str] = None
    soap_category: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    field_values: Optional[Dict[str, Any]] = None
    is_visible_to_patient: Optional[bool] = None
    display_order: Optional[int] = None

    class Config:
        orm_mode = True

class FieldValueUpdate(BaseModel):
    field_id: str
    field_name: str
    value: Any
    
class SectionFieldUpdate(BaseModel):
    section_id: int
    field_values: List[FieldValueUpdate]