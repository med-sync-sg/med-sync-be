from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import DataStore
from app.models.models import User
from app.schemas.user import BaseUser, BaseUserRead, BaseUserCreate, BaseUserUpdate

router = APIRouter()
data_store = DataStore()

@router.get("/users", response_model=List[BaseUserRead])
def list_users(db: Session = Depends(data_store.get_db)):
    return db.query(BaseUser).all()

@router.get("/users/{user_id}", response_model=BaseUserRead)
def get_user(user_id: str, db: Session = Depends(data_store.get_db)):
    db_user = db.query(BaseUser).filter(BaseUser.id == user_id).first()
    if not db_user:
        raise HTTPException(404, detail="User not found")
    return db_user

@router.put("/users/{user_id}", response_model=BaseUserRead)
def update_user(user_id: str, user_in: BaseUserUpdate, db: Session = Depends(data_store.get_db)):
    db_user = db.query(BaseUser).filter(BaseUser.id == user_id).first()
    if not db_user:
        raise HTTPException(404, detail="User not found")

    if user_in.username is not None:
        db_user.username = user_in.username
    if user_in.email is not None:
        db_user.email = user_in.email

    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: str, db: Session = Depends(data_store.get_db)):
    db_user = db.query(BaseUser).filter(BaseUser.id == user_id).first()
    if not db_user:
        raise HTTPException(404, detail="User not found")

    db.delete(db_user)
    db.commit()
    return None