from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.orm import Session
import logging
import json

from app.db.local_session import DatabaseManager
from app.models.models import SOAPCategory, SectionType
from app.api.v1.endpoints import auth, notes, users, reports, tests, calibration
from app.utils.websocket_handler import websocket_endpoint

# Configure logger
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    logger.info("Creating FastAPI application")
    
    app = FastAPI(
        title="MedSync API",
        description="Backend API for medical transcription and analysis",
        version="1.0.0"
    )
    
    # Register routers
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(notes.router, prefix="/notes", tags=["note"])
    app.include_router(users.router, prefix="/users", tags=["user"])
    app.include_router(reports.router, prefix="/reports", tags=["report"])
    app.include_router(tests.router, prefix="/tests", tags=["test"])
    app.include_router(calibration.router, prefix="/calibration", tags=["calibration"])

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure with proper origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add WebSocket endpoint
    app.add_api_websocket_route("/ws", websocket_endpoint)
    
    return app

# Create FastAPI application
app = create_app()

def seed_section_types(db: Session):
        """Seed initial section types for SOAP documentation"""
        
        # Check if section types already exist
        count = db.query(SectionType).count()
        if count > 0:
            print(f"{count} section types already exist, skipping seed")
            return
        
        # Define section types by SOAP category
        section_types = [
            # Subjective
            {
                "code": "CHIEF_COMPLAINT",
                "name": "Chief Complaint",
                "description": "The patient's stated reason for the medical encounter",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 10,
                "is_required": True,
                "content_schema": {
                    "type": "object",
                    "properties": {
                        "complaint_text": {"type": "string"},
                        "duration": {"type": "string"},
                        "severity": {"type": "integer", "minimum": 1, "maximum": 10}
                    }
                }
            },
            {
                "code": "HPI",
                "name": "History of Present Illness",
                "description": "Detailed chronological description of the development of the patient's illness",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 20,
                "is_required": True
            },
            {
                "code": "PMH",
                "name": "Past Medical History",
                "description": "List of patient's significant past illnesses, surgeries, and medical conditions",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 30
            },
            {
                "code": "MEDS",
                "name": "Current Medications",
                "description": "List of medications the patient is currently taking",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 40
            },
            {
                "code": "ALLERGIES",
                "name": "Allergies",
                "description": "Patient's allergies and reactions",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 50
            },
            {
                "code": "ROS",
                "name": "Review of Systems",
                "description": "Systematic review of body systems to identify symptoms",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 60
            },
            {
                "code": "FAM_HX",
                "name": "Family History",
                "description": "Health status of immediate family members",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 70
            },
            {
                "code": "SOC_HX",
                "name": "Social History",
                "description": "Patient's lifestyle, habits, and social factors",
                "soap_category": SOAPCategory.SUBJECTIVE,
                "default_order": 80
            },
            
            # Objective
            {
                "code": "VITALS",
                "name": "Vital Signs",
                "description": "Patient's vital signs: BP, HR, RR, Temp, SpO2, etc.",
                "soap_category": SOAPCategory.OBJECTIVE,
                "default_order": 100,
                "is_required": True
            },
            {
                "code": "PHYSICAL_EXAM",
                "name": "Physical Examination",
                "description": "Findings from the physical examination",
                "soap_category": SOAPCategory.OBJECTIVE,
                "default_order": 110,
                "is_required": True
            },
            {
                "code": "LAB_RESULTS",
                "name": "Laboratory Results",
                "description": "Results from laboratory tests",
                "soap_category": SOAPCategory.OBJECTIVE,
                "default_order": 120
            },
            {
                "code": "IMAGING",
                "name": "Imaging Results",
                "description": "Results from imaging studies",
                "soap_category": SOAPCategory.OBJECTIVE,
                "default_order": 130
            },
            {
                "code": "DIAGNOSTIC_TESTS",
                "name": "Diagnostic Test Results",
                "description": "Results from other diagnostic tests",
                "soap_category": SOAPCategory.OBJECTIVE,
                "default_order": 140
            },
            
            # Assessment
            {
                "code": "ASSESSMENT",
                "name": "Assessment",
                "description": "Clinician's assessment and diagnosis",
                "soap_category": SOAPCategory.ASSESSMENT,
                "default_order": 200,
                "is_required": True
            },
            {
                "code": "DIFFERENTIAL",
                "name": "Differential Diagnosis",
                "description": "List of possible diagnoses with rationale",
                "soap_category": SOAPCategory.ASSESSMENT,
                "default_order": 210
            },
            {
                "code": "PROBLEM_LIST",
                "name": "Problem List",
                "description": "List of active problems",
                "soap_category": SOAPCategory.ASSESSMENT,
                "default_order": 220
            },
            
            # Plan
            {
                "code": "TREATMENT_PLAN",
                "name": "Treatment Plan",
                "description": "Plan for treating the identified problems",
                "soap_category": SOAPCategory.PLAN,
                "default_order": 300,
                "is_required": True
            },
            {
                "code": "MEDICATIONS_PLAN",
                "name": "Medications",
                "description": "Medications prescribed or adjusted",
                "soap_category": SOAPCategory.PLAN,
                "default_order": 310
            },
            {
                "code": "PROCEDURES",
                "name": "Procedures",
                "description": "Procedures to be performed",
                "soap_category": SOAPCategory.PLAN,
                "default_order": 320
            },
            {
                "code": "REFERRALS",
                "name": "Referrals/Consultations",
                "description": "Referrals to specialists or consultations",
                "soap_category": SOAPCategory.PLAN,
                "default_order": 330
            },
            {
                "code": "FOLLOW_UP",
                "name": "Follow-up",
                "description": "Follow-up instructions and timeline",
                "soap_category": SOAPCategory.PLAN,
                "default_order": 340
            },
            {
                "code": "PATIENT_EDUCATION",
                "name": "Patient Education",
                "description": "Education provided to the patient",
                "soap_category": SOAPCategory.PLAN,
                "default_order": 350,
                "is_visible_to_patient": True
            },
            
            # Other
            {
                "code": "ENCOUNTER_INFO",
                "name": "Encounter Information",
                "description": "Administrative details about the encounter",
                "soap_category": SOAPCategory.OTHER,
                "default_order": 5
            },
            {
                "code": "BILLING",
                "name": "Billing Information",
                "description": "Codes and information related to billing",
                "soap_category": SOAPCategory.OTHER,
                "default_order": 900,
                "is_visible_to_patient": False
            }
        ]
        
        # Create section types
        for type_data in section_types:
            section_type = SectionType(**type_data)
            db.add(section_type)
        
        # Create hierarchical relationships if needed
        # (This would be more complex, linking child types to parents)
        
        db.commit()
        print(f"Created {len(section_types)} section types")
        
seed_section_types(next(DatabaseManager().get_session()))