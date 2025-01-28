from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

router = APIRouter()

@router.get("/note/{note_id}")
def get_note(note_id: int):
    pass

@router.post("/note")
def create_note(user: User):
    pass