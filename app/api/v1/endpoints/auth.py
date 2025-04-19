from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

from app.db.local_session import DatabaseManager
from app.models.models import User
import os
# Configure JWT settings
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")  # Change this to a secure key in production
ALGORITHM = os.environ.get("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter()

get_session = DatabaseManager().get_session
DEV_MODE = os.environ.get("DEV_MODE", "True").lower() == "true"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=not DEV_MODE)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """
    Verify a JWT token
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme), 
    db: Session = Depends(get_session)
):
    """
    Get the current authenticated user based on the JWT token.
    In development mode, will use the user ID from the X-User-ID header.
    """
    # Development mode - check for user ID in headers
    if DEV_MODE and (token is None or token == "dev_mode_dummy_token"):
        try:
            # Check for user ID in header
            user_id_header = request.headers.get("X-User-ID")
            if user_id_header:
                user_id = int(user_id_header)
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    return user
                
                print(f"WARNING: User ID {user_id} from header not found in database")
            else:
                print("WARNING: No X-User-ID header provided in development mode")
                
            # Fallback to first user in database if header is missing or invalid
            user = db.query(User).first()
            if user:
                print(f"Using fallback user: ID={user.id}, Username={user.username}")
                return user
                
            # If no users exist, raise an exception
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No users found in the database"
            )
                
        except ValueError:
            print(f"WARNING: Invalid user ID format in X-User-ID header: {user_id_header}")
    
    # Normal authentication flow for production
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    return user