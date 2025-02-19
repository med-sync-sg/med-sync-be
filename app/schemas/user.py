from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime

class User(BaseModel):
    id: int
    username: str
    first_name: str
    middle_name: Optional[str]
    last_name: str
    email: str
    age: int
    
class UserCreate(User):
    pass

class UserRead(User):
    pass

class UserUpdate(User):
    pass