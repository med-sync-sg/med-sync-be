from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime

class BaseUser(BaseModel):
    id: int
    username: str
    first_name: str
    middle_name: Optional[str]
    last_name: str
    email: str
    age: int
    
class BaseUserCreate(BaseModel):
    username: str
    password: str
    first_name: str
    middle_name: Optional[str]
    last_name: str
    email: str
    age: int
    
class BaseUserRead(BaseUser):
    pass

class BaseUserUpdate(BaseModel):
    first_name: str
    middle_name: Optional[str]
    last_name: str
    email: str
    age: int