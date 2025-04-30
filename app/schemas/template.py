from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Pydantic models for request/response validation
class TemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    
class TemplateCreate(TemplateBase):
    system_defined: bool = False
    created_by: Optional[str] = "user"
    version: Optional[str] = "1.0"
    
class TemplateRead(TemplateBase):
    id: str
    system_defined: bool = False
    version: str = "1.0"
    
class FieldBase(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str = "text"
    
class FieldCreate(FieldBase):
    required: bool = False
    system_defined: bool = False
    
class FieldRead(FieldBase):
    id: str
    required: bool = False
    
class TemplateWithFields(TemplateRead):
    fields: List[FieldRead] = []
    
class FieldAssignment(BaseModel):
    field_id: str
    field_name: str
    required: bool = False
    order: int = 0
    
class SearchResult(BaseModel):
    templates: List[Dict[str, Any]] = []
    fields: List[Dict[str, Any]] = []
