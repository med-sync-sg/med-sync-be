from app.schemas.base import BaseAuthModel
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Union, Optional, List
from datetime import datetime

### FOR DATA VALIDATION
class FieldValueCreate(BaseModel):
    """Schema representing a template field value"""
    field_id: str
    name: str
    value: Any
    data_type: str

class FieldValueRead(BaseModel):
    """Schema representing a template field value"""
    field_id: str
    name: str
    value: Any
    data_type: str

class FieldValueUpdate(BaseModel):
    field_id: str
    field_name: str
    value: Any
    data_type: str
    
class SectionFieldUpdate(BaseModel):
    section_id: int
    field_values: List[FieldValueUpdate]

# Section Models

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
    field_values: Optional[List[FieldValueRead]] = None
    is_visible_to_patient: bool = True
    display_order: int = 100
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

class SectionUpdate(BaseAuthModel):
    id: int
    note_id: int
    title: Optional[str] = None
    template_id: Optional[str] = None
    soap_category: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    field_values: Optional[List[FieldValueUpdate]] = None
    is_visible_to_patient: Optional[bool] = None
    display_order: Optional[int] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

