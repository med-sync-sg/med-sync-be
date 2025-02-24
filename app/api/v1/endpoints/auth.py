from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session
from typing import List
from app.db.session import DataStore
from app.models.models import User  # SQLAlchemy user model
from app.schemas.user import BaseUserCreate
from app.utils.auth_utils import create_access_token, verify_password, hash_password

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

router = APIRouter()
data_store = DataStore()

@router.post("/login", response_model=TokenResponse)
def login(login_req: LoginRequest, db: Session = Depends(data_store.get_db)):
    db_user = db.query(User).filter(User.username == login_req.username).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(login_req.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # If OK, create JWT
    token = create_access_token({"sub": db_user.username})
    return TokenResponse(access_token=token)

@router.post("/sign-up")
def sign_up(user_in: BaseUserCreate, db: Session = Depends(data_store.get_db)):
    # Check if a user with the same username already exists
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already taken."
        )
    
    # Hash the plaintext password
    hashed_pw = hash_password(user_in.password)
    
    # Create a new user instance
    new_user = User(
        username=user_in.username,
        age=user_in.age,
        hashed_password=hashed_pw,
        email=user_in.email
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    token = create_access_token({"sub": new_user.username})
    return { "id": new_user.id, "token": token }