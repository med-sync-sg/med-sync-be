import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from app.models.models import Note, Section, User, ReportTemplate
from app.utils.nlp.nlp_utils import get_semantic_section_type
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

class DocumentTransformer:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def transform_note(self, note_id, template_type):
        """Transform note into SOAP-structured document"""
        # Get note with all sections
        note = self.db.query(Note).options(
            joinedload(Note.sections)
        ).filter(Note.id == note_id).first()
        
        if not note:
            return None
        
        # Apply template-specific transformations
        # if template_type == "doctor":
        return self._prepare_doctor_soap_document(note)
        # elif template_type == "patient":
        #     return self._prepare_doctor_soap_document(note)
        # else:
        #     return self._prepare_standard_soap_document(note)
    
    def _prepare_doctor_soap_document(self, note: Note):
        """Prepare SOAP-structured document for medical professionals"""
        # Extract patient info
        patient_info = self._get_patient_info(note.patient_id)
        
        # Extract provider info
        provider_info = self._get_provider_info(note.user_id)
        
        # Structure document according to SOAP
        document = {
            "metadata": {
                "note_id": note.id,
                "title": note.title,
                "patient": patient_info,
                "provider": provider_info,
                "encounter_date": note.encounter_date,
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "document_type": "Clinical Documentation - SOAP Note",
                "confidentiality": "Medical - Confidential"
            },
            "soap": {
                "subjective": self._extract_subjective_sections(note.sections),
                "objective": self._extract_objective_sections(note.sections),
                "assessment": self._extract_assessment_sections(note.sections),
                "plan": self._extract_plan_sections(note.sections)
            },
            # Keep original sections for reference if needed
            # "original_sections": self._categorize_original_sections(note.sections)
        }
        
        return document
    
    def _prepare_patient_soap_document(self, note):
        """Prepare patient-friendly version of SOAP document"""
        # Extract patient info
        patient_info = self._get_patient_info(note.patient_id)
        
        # Extract provider info
        provider_info = self._get_provider_info(note.user_id)
        
        # Structure document with patient-friendly categories
        document = {
            "metadata": {
                "note_id": note.id,
                "title": "Your Health Summary",
                "patient": patient_info,
                "provider": provider_info,
                "encounter_date": note.encounter_date,
                "friendly_date": self._format_friendly_date(note.encounter_date),
                "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "soap": {
                "subjective": {
                    "title": "What You Told Us",
                    "sections": self._extract_subjective_sections(note.sections)
                },
                "objective": {
                    "title": "What We Found",
                    "sections": self._extract_objective_sections(note.sections)
                },
                "assessment": {
                    "title": "Our Assessment",
                    "sections": self._extract_assessment_sections(note.sections)
                },
                "plan": {
                    "title": "Your Care Plan",
                    "sections": self._extract_plan_sections(note.sections)
                }
            }
        }
        
        return document
    
    def _prepare_standard_soap_document(self, note):
        """Prepare standard SOAP document with minimal processing"""
        document = {
            "metadata": {
                "note_id": note.id,
                "title": note.title,
                "patient_id": note.patient_id,
                "provider_id": note.user_id,
                "encounter_date": note.encounter_date,
            },
            "soap": {
                "subjective": self._extract_subjective_sections(note.sections),
                "objective": self._extract_objective_sections(note.sections),
                "assessment": self._extract_assessment_sections(note.sections),
                "plan": self._extract_plan_sections(note.sections)
            }
        }
        
        return document
    
    def _get_patient_info(self, patient_id: int):
        """Get patient information from database"""
        # This would connect to your patient database
        # For now, returning placeholder data
        return {
            "id": patient_id,
            "name": "Patient Name",  # Replace with actual patient info
            "age": 45,
            "gender": "Female",
            "mrn": f"MRN{patient_id}",
            "dob": "1977-05-15"
        }
    
    def _get_provider_info(self, user_id: int):
        """Get provider information from database"""
        # Query your User model for provider info
        provider = self.db.query(User).filter(User.id == user_id).first()
        
        if provider:
            return {
                "id": provider.id,
                "name": f"{provider.first_name} {provider.last_name}",
                "credentials": "MD",  # You might need to add this field to your User model
                "department": "Internal Medicine"  # You might need to add this field
            }
        
        return {
            "id": user_id,
            "name": "Provider Name",
            "credentials": "MD",
            "department": "Unknown"
        }
    
    def _format_friendly_date(self, date_str):
        """Format date in a patient-friendly way"""
        try:
            # Parse the date string based on your format
            if isinstance(date_str, str):
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_obj.strftime("%B %d, %Y")  # e.g., "April 15, 2025"
        except:
            pass
        
        return date_str
    
    def _extract_subjective_sections(self, sections: List[Section]):
        """Extract sections that belong in the Subjective part of SOAP"""
        subjective_sections = []
        
        for section in sections:
            # Parse section content
            content = self._parse_section_content(section)
            
            # Check if section belongs in Subjective
            if self._is_subjective_section(section):
                subjective_sections.append({
                    "id": section.id,
                    "title": section.title,
                    "type": section.section_type,
                    "content": content
                })
        
        return subjective_sections
    
    def _extract_objective_sections(self, sections: List[Section]):
        """Extract sections that belong in the Objective part of SOAP"""
        objective_sections = []
        
        for section in sections:
            # Parse section content
            content = self._parse_section_content(section)
            
            # Check if section belongs in Objective
            if self._is_objective_section(section):
                objective_sections.append({
                    "id": section.id,
                    "title": section.title,
                    "type": section.section_type,
                    "content": content
                })
        
        return objective_sections
    
    def _extract_assessment_sections(self, sections: List[Section]):
        """Extract sections that belong in the Assessment part of SOAP"""
        assessment_sections = []
        
        for section in sections:
            # Parse section content
            content = self._parse_section_content(section)
            
            # Check if section belongs in Assessment
            if self._is_assessment_section(section):
                assessment_sections.append({
                    "id": section.id,
                    "title": section.title,
                    "type": section.section_type,
                    "content": content
                })
        
        return assessment_sections
    
    def _extract_plan_sections(self, sections: List[Section]):
        """Extract sections that belong in the Plan part of SOAP"""
        plan_sections = []
        
        for section in sections:
            # Parse section content
            content = self._parse_section_content(section)
            
            # Check if section belongs in Plan
            if self._is_plan_section(section):
                plan_sections.append({
                    "id": section.id,
                    "title": section.title,
                    "type": section.section_type,
                    "content": content
                })
        
        return plan_sections
    
    def _parse_section_content(self, section: List[Section]):
        """Parse section content from JSON or string"""
        content = section.content
        
        if isinstance(content, str):
            try:
                return json.loads(content)
            except:
                return {"text": content}
        
        return content
    
    def _is_subjective_section(self, section: List[Section]):
        """Determine if a section belongs in the Subjective part of SOAP"""
        # Get section type (semantic or explicit)
        section_type = self._determine_section_type(section)
        
        # Sections that typically go in Subjective
        subjective_types = [
            "CHIEF_COMPLAINT",
            "PATIENT_MEDICAL_HISTORY",
            "HISTORY_OF_PRESENT_ILLNESS",
            "REVIEW_OF_SYSTEMS",
            "SOCIAL_HISTORY",
            "FAMILY_HISTORY",
            "ALLERGIES"
        ]
        
        # Check if section type is in the subjective list
        if section_type in subjective_types:
            return True
            
        # Look for keywords in the title
        subjective_keywords = [
            "chief complaint", "history", "symptoms", "reported", 
            "subjective", "patient states", "hpi"
        ]
        
        if any(keyword in section.title.lower() for keyword in subjective_keywords):
            return True
            
        return False
    
    def _is_objective_section(self, section):
        """Determine if a section belongs in the Objective part of SOAP"""
        # Get section type (semantic or explicit)
        section_type = self._determine_section_type(section)
        
        # Sections that typically go in Objective
        objective_types = [
            "VITAL_SIGNS",
            "PHYSICAL_EXAM",
            "LABORATORY_RESULTS",
            "IMAGING_RESULTS",
            "DIAGNOSTIC_RESULTS"
        ]
        
        # Check if section type is in the objective list
        if section_type in objective_types:
            return True
            
        # Look for keywords in the title
        objective_keywords = [
            "exam", "physical", "vital", "lab", "test", "diagnostic",
            "objective", "observation", "finding"
        ]
        
        if any(keyword in section.title.lower() for keyword in objective_keywords):
            return True
            
        return False
    
    def _is_assessment_section(self, section):
        """Determine if a section belongs in the Assessment part of SOAP"""
        # Get section type (semantic or explicit)
        section_type = self._determine_section_type(section)
        
        # Sections that typically go in Assessment
        assessment_types = [
            "ASSESSMENT",
            "DIAGNOSIS",
            "IMPRESSION",
            "DIFFERENTIAL_DIAGNOSIS"
        ]
        
        # Check if section type is in the assessment list
        if section_type in assessment_types:
            return True
            
        # Look for keywords in the title
        assessment_keywords = [
            "assessment", "diagnosis", "impression", "differential",
            "conclusion", "problem"
        ]
        
        if any(keyword in section.title.lower() for keyword in assessment_keywords):
            return True
            
        return False
    
    def _is_plan_section(self, section):
        """Determine if a section belongs in the Plan part of SOAP"""
        # Get section type (semantic or explicit)
        section_type = self._determine_section_type(section)
        
        # Sections that typically go in Plan
        plan_types = [
            "PLAN",
            "TREATMENT_PLAN",
            "MEDICATIONS",
            "PROCEDURES",
            "FOLLOW_UP",
            "REFERRALS",
            "EDUCATION"
        ]
        
        # Check if section type is in the plan list
        if section_type in plan_types:
            return True
            
        # Look for keywords in the title
        plan_keywords = [
            "plan", "treatment", "medication", "prescription", "therapy",
            "follow-up", "referral", "education", "recommendation"
        ]
        
        if any(keyword in section.title.lower() for keyword in plan_keywords):
            return True
            
        return False
    
    def _determine_section_type(self, section):
        """Determine section type, using semantic analysis if needed"""
        section_type = section.section_type
        
        if section_type == "OTHERS" or not section_type:
            content = section.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    content = {"text": content}
            
            # Use semantic analysis from your existing utility
            section_type = get_semantic_section_type(section.title, content)
            
        return section_type
    
    def _categorize_original_sections(self, sections):
        """Categorize original sections by type"""
        categorized = {}
        
        for section in sections:
            section_type = self._determine_section_type(section)
            
            if section_type not in categorized:
                categorized[section_type] = []
                
            content = self._parse_section_content(section)
            
            categorized[section_type].append({
                "id": section.id,
                "title": section.title,
                "content": content
            })
        
        return categorized