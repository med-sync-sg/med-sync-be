from pydantic import BaseModel
from typing import Optional

class UserBase(BaseModel):
    username: str
    first_name: str
    middle_name: Optional[str] = None
    last_name: str
    email: str
    age: int

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id: int

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None