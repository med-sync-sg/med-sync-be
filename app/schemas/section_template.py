from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Pydantic models for request/response validation
class SectionTemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    
class SectionTemplateCreate(SectionTemplateBase):
    system_defined: bool = False
    created_by: Optional[str] = "user"
    version: Optional[str] = "1.0"
    
class SectionTemplateRead(SectionTemplateBase):
    id: str
    system_defined: bool = False
    version: str = "1.0"
    
class TemplateFieldBase(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str = "text"
    
class TemplateFieldCreate(TemplateFieldBase):
    required: bool = False
    system_defined: bool = False
    
class TemplateFieldRead(TemplateFieldBase):
    id: str
    required: bool = False
    
class TemplateFieldUpdate(TemplateFieldBase):
    id: str
    required: bool = False
    name: Optional[str]
    description: Optional[str]
    
    
class SectionTemplateWithTemplateFields(SectionTemplateRead):
    fields: List[TemplateFieldRead] = []
    
class FieldValueAssignment(BaseModel):
    field_id: str
    field_name: str
    required: bool = False
    order: int = 0
    
class SearchResult(BaseModel):
    templates: List[Dict[str, Any]] = []
    fields: List[Dict[str, Any]] = []
