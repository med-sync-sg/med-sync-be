from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from app.models.models import User  # SQLAlchemy user model
from app.schemas.user import UserCreate
from app.utils.auth_utils import create_access_token, verify_password, hash_password
from app.db.local_session import DatabaseManager
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import logging

# Configure logger
logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Get database session dependency
get_session = DatabaseManager().get_session

# Define response schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int

class SignupResponse(BaseModel):
    user_id: int
    access_token: str
    token_type: str = "bearer"

router = APIRouter()

# Define request and response schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int

class SignupResponse(BaseModel):
    user_id: int
    access_token: str
    token_type: str = "bearer"

router = APIRouter()

@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: LoginRequest,
    db: Session = Depends(get_session)
):
    """
    Authenticate a user using JSON and return an access token
    
    This endpoint accepts JSON with username and password fields
    """
    return await authenticate_user(credentials.username, credentials.password, db)


@router.post("/sign-up", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate, db: Session = Depends(get_session)):
    """
    Register a new user
    
    Args:
        user_data: User creation data
        db: Database session
        
    Returns:
        New user information and access token
    """
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        logger.warning(f"Signup attempt with existing username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Check if email already exists (if provided)
    if user_data.email:
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            logger.warning(f"Signup attempt with existing email: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    try:
        # Hash the password
        hashed_password = hash_password(user_data.password)
        
        # Create user object
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            first_name=user_data.first_name,
            middle_name=user_data.middle_name,
            last_name=user_data.last_name,
            age=user_data.age
        )
        
        # Save to database
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Generate JWT token
        token_data = {"sub": str(new_user.id), "username": new_user.username}
        access_token = create_access_token(token_data)
        
        logger.info(f"New user registered: ID {new_user.id}, username {new_user.username}")
        return {
            "user_id": new_user.id,
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error during user registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again later."
        )

async def authenticate_user(username: str, password: str, db: Session):
    """
    Authenticate a user and generate access token
    
    Args:
        username: User's username
        password: User's password
        db: Database session
        
    Returns:
        Token response with access token and user ID
    """
    # Find the user by username
    user = db.query(User).filter(User.username == username).first()
    
    # Verify user exists and password is correct
    if not user or not verify_password(password, user.hashed_password):
        logger.warning(f"Failed login attempt for username: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT token
    token_data = {"sub": str(user.id), "username": user.username}
    access_token = create_access_token(token_data)
    
    logger.info(f"User {user.id} ({user.username}) logged in successfully")
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id}

@router.post("/validate-token")
async def validate_token(token: str = Depends(oauth2_scheme), db: Session = Depends(get_session)):
    """
    Validate an access token and return user information
    
    Args:
        token: JWT access token
        db: Database session
        
    Returns:
        User information if token is valid
    """
    try:
        # Get current user from token
        current_user = get_current_user(token, db)
        
        # Return user info (excluding sensitive fields)
        return {
            "valid": True,
            "user_id": current_user.id,
            "username": current_user.username
        }
    except HTTPException:
        return {"valid": False}

@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    old_password: str,
    new_password: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_session)
):
    """
    Change user password
    
    Args:
        old_password: Current password
        new_password: New password
        token: JWT access token
        db: Database session
        
    Returns:
        Success message
    """
    # Get current user from token
    current_user = get_current_user(token, db)
    
    # Verify old password
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    try:
        # Hash and set new password
        current_user.hashed_password = hash_password(new_password)
        db.commit()
        
        logger.info(f"Password changed for user {current_user.id}")
        return {"message": "Password changed successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error changing password for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

# Helper function to get current user from token
def get_current_user(token: str, db: Session) -> User:
    """
    Get the current user from a JWT token
    
    Args:
        token: JWT access token
        db: Database session
        
    Returns:
        User object
    """
    from app.utils.auth_utils import decode_access_token
    
    try:
        # Decode token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return user
        
    except Exception as e:
        logger.error(f"Error authenticating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )