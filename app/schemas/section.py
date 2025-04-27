from app.schemas.base import BaseAuthModel
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional

class SectionCreate(BaseAuthModel):
    title: str
    content: Dict[str, Any] = {}
    section_type_id: int
    parent_id: Optional[int] = None
    display_order: Optional[int] = None
    is_visible_to_patient: Optional[bool] = True

class SectionRead(BaseAuthModel):
    id: int
    note_id: int
    user_id: int
    title: str
    content: Union[Dict[str, Any], None]
    section_type_id: int
    section_type_code: str
    soap_category: str

    class Config:
        orm_mode = True

class SectionUpdate(BaseAuthModel):
    id: int
    note_id: int
    title: Union[str, None] = None
    content: Union[Dict[str, Any], None] = None
    section_type_id: Union[int, None] = None

    class Config:
        orm_mode = True