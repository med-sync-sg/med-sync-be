from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.utils.auth_utils import hash_password
from app.models.models import User
from app.schemas.user import UserBase, UserRead, UserCreate, UserUpdate
from app.db.local_session import DatabaseManager
import logging

router = APIRouter()

get_session = DatabaseManager().get_session
logger = logging.getLogger(__name__)

@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user_in: UserCreate, db: Session = Depends(get_session)):
    """
    Create a new user
    
    Args:
        user_in: User creation data
        db: Database session
        
    Returns:
        Created user
    """
    # Check if username is already taken
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Check if email is already taken
    if user_in.email:
        existing_email = db.query(User).filter(User.email == user_in.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
    
    try:
        # Hash the password
        hashed_password = hash_password(user_in.password)
        
        # Create new user
        db_user = User(
            username=user_in.username,
            email=user_in.email,
            hashed_password=hashed_password,
            first_name=user_in.first_name,
            middle_name=user_in.middle_name,
            last_name=user_in.last_name,
            age=user_in.age
        )
        
        # Add to database
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Created user with ID {db_user.id}")
        return db_user
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/", response_model=List[UserRead])
def list_users(db: Session = Depends(get_session)):
    """
    List all users
    
    Args:
        db: Database session
        
    Returns:
        List of users
    """
    try:
        users = db.query(User).all()
        logger.info(f"Retrieved {len(users)} users")
        return users
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: int, db: Session = Depends(get_session)):
    """
    Get a specific user by ID
    
    Args:
        user_id: ID of the user
        db: Database session
        
    Returns:
        User details
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.warning(f"User {user_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )
    
    logger.info(f"Retrieved user {user_id}")
    return db_user

@router.put("/{user_id}", response_model=UserRead)
def update_user(user_id: int, user_in: UserUpdate, db: Session = Depends(get_session)):
    """
    Update a user
    
    Args:
        user_id: ID of the user to update
        user_in: User update data
        db: Database session
        
    Returns:
        Updated user
    """
    # Get the user
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.warning(f"User {user_id} not found for update")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )

    try:
        # Update fields if provided
        if user_in.first_name is not None:
            db_user.first_name = user_in.first_name
        if user_in.middle_name is not None:
            db_user.middle_name = user_in.middle_name
        if user_in.last_name is not None:
            db_user.last_name = user_in.last_name
        if user_in.email is not None:
            # Check if email is taken by another user
            existing_email = db.query(User).filter(
                User.email == user_in.email, 
                User.id != user_id
            ).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use by another user"
                )
            db_user.email = user_in.email
        if user_in.age is not None:
            db_user.age = user_in.age

        # Commit changes
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Updated user {user_id}")
        return db_user
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_session)):
    """
    Delete a user
    
    Args:
        user_id: ID of the user to delete
        db: Database session
        
    Returns:
        None
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        logger.warning(f"User {user_id} not found for deletion")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )

    try:
        # Delete the user
        db.delete(db_user)
        db.commit()
        
        logger.info(f"Deleted user {user_id}")
        return None
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )