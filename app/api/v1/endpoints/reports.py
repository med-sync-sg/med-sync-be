from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.db.local_session import DatabaseManager
from app.models.models import Note, Section, SOAPCategory, User
from app.schemas.note import NoteRead
from app.services.note_service import NoteService

router = APIRouter()
get_session = DatabaseManager().get_session

@router.get("/dart-format/{note_id}")
async def get_note_dart_format(
    note_id: int = Path(..., description="ID of the note to format"),
    include_section_content: bool = Query(True, description="Whether to include full section content"),
    db: Session = Depends(get_session)
):
    """
    Get a note formatted specifically for Dart clients to generate PDF reports
    
    This endpoint provides a standardized format that's easily consumable by Dart
    applications for rendering widgets and generating PDFs.
    """
    # Get the note with sections
    note_service = NoteService(db)
    note = note_service.get_note_by_id(note_id)
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    # Format the note for Dart consumption
    result = format_note_for_dart(note, include_section_content)
    
    return result

def format_note_for_dart(note: Note, include_section_content: bool = True) -> Dict[str, Any]:
    """
    Format a note and its sections in a standardized way for Dart clients
    """
    # Basic note information
    formatted_note = {
        "id": note.id,
        "title": note.title,
        "patientId": note.patient_id,
        "userId": note.user_id,
        "encounterDate": note.encounter_date.isoformat() if note.encounter_date else None,
        "metadata": {
            "createdAt": datetime.now().isoformat(),
            "noteType": "medical",
            "version": "1.0"
        },
        "soapCategories": {
            "subjective": [],
            "objective": [],
            "assessment": [],
            "plan": [],
            "other": []
        }
    }
    
    # Format sections and organize by SOAP category
    if note.sections:
        for section in note.sections:
            # Format section
            formatted_section = {
                "id": section.id,
                "title": section.title,
                "templateId": section.template_id,
                "displayOrder": section.display_order,
                "isVisibleToPatient": section.is_visible_to_patient,
            }
            
            # Add content if requested
            if include_section_content:
                # Format content for easier use in Dart
                if section.content:
                    formatted_fields = []
                    
                    for field_id, field_data in section.content.items():
                        if isinstance(field_data, list):
                            # Handle list of fields
                            for item in field_data:
                                formatted_fields.append({
                                    "id": field_id,
                                    "name": item.get("name", ""),
                                    "type": item.get("data_type", "string"),
                                    "value": item.get("value", ""),
                                    "required": item.get("required", False),
                                    "description": item.get("description", "")
                                })
                        else:
                            # Handle single field
                            formatted_fields.append({
                                "id": field_id,
                                "name": field_data.get("name", ""),
                                "type": field_data.get("data_type", "string"),
                                "value": field_data.get("value", ""),
                                "required": field_data.get("required", False),
                                "description": field_data.get("description", "")
                            })
                    
                    formatted_section["fields"] = formatted_fields
                else:
                    formatted_section["fields"] = []
            
            # Add to appropriate SOAP category
            soap_category = section.soap_category.lower() if section.soap_category else "other"
            
            if soap_category == "subjective":
                formatted_note["soapCategories"]["subjective"].append(formatted_section)
            elif soap_category == "objective":
                formatted_note["soapCategories"]["objective"].append(formatted_section)
            elif soap_category == "assessment":
                formatted_note["soapCategories"]["assessment"].append(formatted_section)
            elif soap_category == "plan":
                formatted_note["soapCategories"]["plan"].append(formatted_section)
            else:
                formatted_note["soapCategories"]["other"].append(formatted_section)
    
    # Sort sections by display order within each category
    for category in formatted_note["soapCategories"]:
        formatted_note["soapCategories"][category].sort(key=lambda s: s.get("displayOrder", 100))
    
    return formatted_note

@router.get("/pdf-data/{note_id}")
async def get_pdf_data(
    note_id: int = Path(..., description="ID of the note"),
    report_type: str = Query("doctor", description="Type of report (doctor or patient)"),
    db: Session = Depends(get_session)
):
    """
    Get complete data needed for PDF generation, combining note content with template
    """
    # Get note data
    note_service = NoteService(db)
    note = note_service.get_note_by_id(note_id)
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    # Format note for Dart
    formatted_note = format_note_for_dart(note, include_section_content=True)
    
    # Get template - this is a simplified version
    # In a real app, you'd use a proper ReportTemplate model
    template = {
        "id": 1,
        "name": f"{report_type.capitalize()} Report Template",
        "reportType": report_type,
        "version": "1.0",
        "sections": {
            "subjective": {
                "include": True,
                "order": 0,
                "titleOverride": "SUBJECTIVE",
                "formatOptions": {}
            },
            "objective": {
                "include": True,
                "order": 1,
                "titleOverride": "OBJECTIVE",
                "formatOptions": {}
            },
            "assessment": {
                "include": True,
                "order": 2,
                "titleOverride": "ASSESSMENT",
                "formatOptions": {}
            },
            "plan": {
                "include": True,
                "order": 3,
                "titleOverride": "PLAN",
                "formatOptions": {}
            },
            "other": {
                "include": True,
                "order": 4,
                "titleOverride": "OTHER",
                "formatOptions": {}
            }
        },
        "settings": {
            "showHeader": True,
            "showFooter": True
        }
    }
    
    # Get patient info - this is a placeholder
    # In a real app, you'd query your patient model
    patient_info = get_patient_info(note.patient_id, db)
    
    # Get provider info
    provider_info = get_provider_info(note.user_id, db)
    
    # Create the complete PDF data structure
    pdf_data = {
        "note": formatted_note,
        "template": template,
        "metadata": {
            "generatedAt": datetime.now().isoformat(),
            "reportType": report_type,
        },
        "patient": patient_info,
        "provider": provider_info
    }
    
    return pdf_data

def get_patient_info(patient_id: Optional[int], db: Session) -> Dict[str, Any]:
    """
    Get patient information - placeholder implementation
    
    In a real application, you would query your patient model here
    """
    if patient_id:
        return {
            "id": patient_id,
            "name": "Patient Name",  # Placeholder
            "dob": "YYYY-MM-DD",     # Placeholder
            "gender": "Unknown",     # Placeholder
            "mrn": f"MRN{patient_id}"  # Placeholder
        }
    else:
        return {
            "id": None,
            "name": "Unknown Patient",
            "dob": "Unknown",
            "gender": "Unknown",
            "mrn": "Unknown"
        }

def get_provider_info(user_id: int, db: Session) -> Dict[str, Any]:
    """
    Get provider information based on user_id
    """
    # Try to get user from database
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if user:
            return {
                "id": user.id,
                "name": f"{user.first_name} {user.last_name}",
                "credentials": "MD",  # Placeholder
                "specialty": "Specialty",  # Placeholder
                "contact": user.email
            }
    except:
        # If there's an error or no user found, return placeholder
        pass
        
    # Default placeholder 
    return {
        "id": user_id,
        "name": "Unknown Provider",
        "credentials": "",
        "specialty": "",
        "contact": ""
    }