from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

router = APIRouter()


@router.get("/reports/{user_id}/{report_id}")
def get_user(user_id: int, report_id: int):
    pass

@router.post("/reports")
def create_report(report_data):
    pass