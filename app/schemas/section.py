from pydantic import BaseModel, ConfigDict, Field, field_serializer
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

    @field_serializer('updated_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
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
    )

class SectionCreate(SectionBase):
    """Schema for creating a new section"""
    note_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    @field_serializer('created_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    @field_serializer('updated_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
    )

class SectionRead(SectionBase):
    """Schema for reading a section"""
    id: int
    note_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    last_modified_by_id: Optional[int] = None
    
    @field_serializer('created_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    @field_serializer('updated_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
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
    created_at: datetime
    updated_at: datetime
    last_modified_by_id: Optional[int] = None
    
    @field_serializer('created_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    @field_serializer('updated_at')
    def serialize_datetime(self, datetime_object: datetime, _info):
        return datetime_object.isoformat()
    
    model_config = ConfigDict(
        from_attributes=True,
    )