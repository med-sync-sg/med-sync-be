import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from app.models.models import Note, Section, User
from app.utils.nlp.summarizer import generate_summary

# Configure logger
logger = logging.getLogger(__name__)

class ReportService:
    """
    Service for generating medical reports from notes and sections.
    Handles template rendering, data formatting, and report generation.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize report service
        
        Args:
            db_session: SQLAlchemy session for database operations
        """
        self.db = db_session
        self.templates_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils", "nlp", "report_templates"
        )
        self.env = Environment(loader=FileSystemLoader(self.templates_path))
        
        logger.info(f"ReportService initialized with templates from {self.templates_path}")
    
    def generate_doctor_report(self, note_id: int) -> Optional[str]:
        """
        Generate a doctor-focused report from a note
        
        Args:
            note_id: ID of the note to generate report from
            
        Returns:
            HTML report as a string, or None if generation failed
        """
        try:
            # Get note with sections
            note = self._get_note_with_sections(note_id)
            if not note:
                return None
                
            # Get patient info (from user table for now)
            patient_info = self._get_patient_info(note.patient_id)
            
            # Format sections for report
            report_sections = self._format_sections_for_doctor_report(note.sections)
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "patient_info": patient_info,
                "sections": report_sections
            }
            
            # Render template
            template = self.env.get_template("default_doctor_report.html")
            rendered_report = template.render(report_data)
            
            logger.info(f"Generated doctor report for note {note_id}")
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating doctor report for note {note_id}: {str(e)}")
            return None
    
    def generate_patient_report(self, note_id: int) -> Optional[str]:
        """
        Generate a patient-friendly report from a note
        
        Args:
            note_id: ID of the note to generate report from
            
        Returns:
            HTML report as a string, or None if generation failed
        """
        try:
            # Get note with sections
            note = self._get_note_with_sections(note_id)
            if not note:
                return None
                
            # Get patient info
            patient_info = self._get_patient_info(note.patient_id)
            
            # Format sections for patient report (simplified language)
            report_sections = self._format_sections_for_patient_report(note.sections)
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "patient_info": patient_info,
                "sections": report_sections
            }
            
            # Render template
            template = self.env.get_template("default_patient_report.html")
            rendered_report = template.render(report_data)
            
            logger.info(f"Generated patient report for note {note_id}")
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating patient report for note {note_id}: {str(e)}")
            return None
    
    def save_report_to_file(self, report_html: str, file_path: str) -> bool:
        """
        Save a generated report to a file
        
        Args:
            report_html: HTML report content
            file_path: Path where to save the report
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_html)
                
            logger.info(f"Report saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report to {file_path}: {str(e)}")
            return False
    
    def _get_note_with_sections(self, note_id: int) -> Optional[Note]:
        """
        Get a note with all its sections
        
        Args:
            note_id: ID of the note
            
        Returns:
            Note object with sections or None if not found
        """
        try:
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note {note_id} not found for report generation")
                return None
                
            # Load sections
            sections = self.db.query(Section).filter(Section.note_id == note_id).all()
            note.sections = sections
            
            return note
            
        except Exception as e:
            logger.error(f"Error retrieving note {note_id} with sections: {str(e)}")
            return None
    
    def _get_patient_info(self, patient_id: int) -> Dict[str, Any]:
        """
        Get patient information
        
        Args:
            patient_id: ID of the patient
            
        Returns:
            Dictionary with patient information
        """
        # In a real implementation, this would query a Patient model
        # For now, we'll return placeholder data
        return {
            "id": patient_id,
            "name": "Patient Name",  # Placeholder
            "age": 45,               # Placeholder
            "gender": "Unknown"      # Placeholder
        }
    
    def _format_sections_for_doctor_report(self, sections: List[Section]) -> List[Dict[str, Any]]:
        """
        Format sections for a doctor report
        
        Args:
            sections: List of Section objects
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        
        for section in sections:
            # Skip empty sections
            if not section.content:
                continue
                
            try:
                # Parse content from JSON if it's a string
                content = section.content
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Extract evidence from content
                evidence = self._extract_evidence_from_content(content)
                
                # Format section
                formatted_section = {
                    "title": section.title,
                    "summary": self._get_section_summary(content),
                    "evidence": evidence,
                    "type": section.section_type
                }
                
                formatted_sections.append(formatted_section)
                
            except Exception as e:
                logger.error(f"Error formatting section {section.id}: {str(e)}")
        
        return formatted_sections
    
    def _format_sections_for_patient_report(self, sections: List[Section]) -> List[Dict[str, Any]]:
        """
        Format sections for a patient-friendly report
        
        Args:
            sections: List of Section objects
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        
        for section in sections:
            # Skip empty sections
            if not section.content:
                continue
                
            try:
                # Parse content from JSON if it's a string
                content = section.content
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Format section (simpler version for patients)
                formatted_section = {
                    "title": self._simplify_title(section.title),
                    "summary": self._get_patient_friendly_summary(content),
                }
                
                formatted_sections.append(formatted_section)
                
            except Exception as e:
                logger.error(f"Error formatting patient section {section.id}: {str(e)}")
        
        return formatted_sections
    
    def _extract_evidence_from_content(self, content: Dict[str, Any]) -> List[str]:
        """
        Extract evidence items from section content
        
        Args:
            content: Section content dictionary
            
        Returns:
            List of evidence statements
        """
        evidence = []
        
        # The exact implementation depends on your content structure
        # This is a generic approach that looks for relevant fields
        
        # Check for Main Symptom
        if "Main Symptom" in content:
            symptom = content["Main Symptom"]
            if isinstance(symptom, dict):
                # Extract name, duration, severity, etc.
                if "name" in symptom and symptom["name"]:
                    evidence.append(f"Patient reports {symptom['name']}")
                    
                if "duration" in symptom and symptom["duration"]:
                    evidence.append(f"Duration: {symptom['duration']}")
                    
                if "severity" in symptom and symptom["severity"]:
                    evidence.append(f"Severity: {symptom['severity']}")
        
        # Add more content extraction logic based on your data model
        
        return evidence
    
    def _get_section_summary(self, content: Dict[str, Any]) -> str:
        """
        Generate a summary for a section
        
        Args:
            content: Section content dictionary
            
        Returns:
            Summary text
        """
        # Convert content to text for summarization
        content_text = json.dumps(content)
        
        # Generate summary (from your existing summarizer)
        summary = generate_summary(content_text, top_n=2)
        
        # If the summary is too short or failed, create a basic summary
        if len(summary) < 20:
            summary = self._create_basic_summary(content)
            
        return summary
    
    def _create_basic_summary(self, content: Dict[str, Any]) -> str:
        """
        Create a basic summary from content structure
        
        Args:
            content: Section content dictionary
            
        Returns:
            Basic summary text
        """
        # The implementation depends on your content structure
        # This is a simplified approach
        
        summary_parts = []
        
        if "Main Symptom" in content and isinstance(content["Main Symptom"], dict):
            symptom = content["Main Symptom"]
            
            # Build summary from available fields
            if "name" in symptom and symptom["name"]:
                summary_parts.append(f"Patient reports {symptom['name']}")
                
                if "duration" in symptom and symptom["duration"]:
                    summary_parts.append(f"for {symptom['duration']}")
                    
                if "severity" in symptom and symptom["severity"]:
                    summary_parts.append(f"with {symptom['severity']} severity")
        
        # Join parts
        if summary_parts:
            return " ".join(summary_parts)
        else:
            return "No detailed information available."
    
    def _get_patient_friendly_summary(self, content: Dict[str, Any]) -> str:
        """
        Create a patient-friendly summary from content
        
        Args:
            content: Section content dictionary
            
        Returns:
            Patient-friendly summary
        """
        # Similar to _create_basic_summary but with simplified language
        # This would typically use a medical terminology simplifier
        
        # Simplified implementation
        summary_parts = []
        
        if "Main Symptom" in content and isinstance(content["Main Symptom"], dict):
            symptom = content["Main Symptom"]
            
            if "name" in symptom and symptom["name"]:
                summary_parts.append(f"You mentioned having {symptom['name']}")
                
                if "duration" in symptom and symptom["duration"]:
                    summary_parts.append(f"for {symptom['duration']}")
                    
                if "severity" in symptom and symptom["severity"]:
                    severity = symptom["severity"].lower()
                    if severity in ["severe", "intense", "extreme"]:
                        summary_parts.append("that was causing significant discomfort")
                    elif severity in ["moderate", "medium"]:
                        summary_parts.append("that was causing moderate discomfort")
                    else:
                        summary_parts.append("that was causing mild discomfort")
        
        # Join parts
        if summary_parts:
            return " ".join(summary_parts)
        else:
            return "We discussed your health concerns during the appointment."
    
    def _simplify_title(self, title: str) -> str:
        """
        Simplify a medical title for patient-friendly reports
        
        Args:
            title: Original title
            
        Returns:
            Simplified title
        """
        # Map of medical terms to simplified versions
        simplifications = {
            "Chief Complaint": "Your Main Concern",
            "Vital Signs": "Your Health Measurements",
            "Assessment": "What We Found",
            "Diagnosis": "What We Found",
            "Plan": "Next Steps",
            "Medications": "Your Medications",
            "Physical Examination": "Physical Checkup Results",
        }
        
        # Try direct mapping
        if title in simplifications:
            return simplifications[title]
            
        # Otherwise return original with first letter capitalized
        return title.capitalize()