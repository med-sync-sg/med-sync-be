from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

router = APIRouter()


@router.get("/users/{user_id}")
def get_user(user_id: int):
    pass

@router.post("/users")
def create_user(user):
    pass