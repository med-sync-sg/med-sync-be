from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import DataStore
from models import User
from app.schemas.user import User, UserRead, UserCreate, UserUpdate

router = APIRouter()
data_store = DataStore()

@router.post("/users", response_model=UserRead, status_code=201)
def create_user(user_in: UserCreate, db: Session = Depends(data_store.get_db())):
    db_user = User(username=user_in.username, email=user_in.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/users", response_model=List[UserRead])
def list_users(db: Session = Depends(data_store.get_db())):
    return db.query(User).all()

@router.get("/users/{user_id}", response_model=UserRead)
def get_user(user_id: str, db: Session = Depends(data_store.get_db())):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(404, detail="User not found")
    return db_user

@router.put("/users/{user_id}", response_model=UserRead)
def update_user(user_id: str, user_in: UserUpdate, db: Session = Depends(data_store.get_db())):
    db_user = db.query(User).filter(User.id == user_id).first()
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
def delete_user(user_id: str, db: Session = Depends(data_store.get_db())):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(404, detail="User not found")

    db.delete(db_user)
    db.commit()
    return None