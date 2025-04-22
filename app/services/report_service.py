# app/services/report_service.py

import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from app.models.models import Note, Section, User, ReportTemplate
from app.utils.nlp.summarizer import generate_summary
from sqlalchemy import func

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
    
    def generate_custom_report(self, note_id: int, template_id: int) -> Optional[str]:
        """
        Generate a custom report using a specific template
        
        Args:
            note_id: ID of the note to generate report from
            template_id: ID of the report template to use
            
        Returns:
            HTML report as a string, or None if generation failed
        """
        try:
            # Get note with sections
            note = self._get_note_with_sections(note_id)
            if not note:
                return None
                
            # Get the template
            template = self.db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
            if not template:
                logger.error(f"Report template {template_id} not found")
                return None
                
            # Get patient info
            patient_info = self._get_patient_info(note.patient_id)
            
            # Format sections according to template
            report_sections = self._format_sections_with_template(note.sections, template)
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "template_name": template.name,
                "patient_info": patient_info,
                "sections": report_sections
            }
            
            # Use the custom template if specified, or fall back to default
            template_name = f"custom_{template.id}.html"
            try:
                template_obj = self.env.get_template(template_name)
            except:
                # If custom template doesn't exist, use default based on report type
                template_name = f"default_{template.report_type}_report.html"
                template_obj = self.env.get_template(template_name)
                
            rendered_report = template_obj.render(report_data)
            
            logger.info(f"Generated custom report for note {note_id} using template {template_id}")
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating custom report: {str(e)}")
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
    
    def create_template(self, user_id: int, template_data: Dict[str, Any]) -> Optional[ReportTemplate]:
        """
        Create a new report template
        
        Args:
            user_id: User ID for the template owner
            template_data: Template data dictionary
            
        Returns:
            Created ReportTemplate or None if creation failed
        """
        try:
            # Ensure all required fields are present
            if 'name' not in template_data or 'report_type' not in template_data:
                logger.error("Missing required fields for template creation")
                return None
                
            # Create template object
            template = ReportTemplate(
                user_id=user_id,
                name=template_data['name'],
                description=template_data.get('description', ''),
                report_type=template_data['report_type'],
                template_data={
                    'sections': template_data.get('sections', {})
                }
            )
            
            self.db.add(template)
            self.db.commit()
            self.db.refresh(template)
            
            logger.info(f"Created report template {template.id} for user {user_id}")
            return template
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating template: {str(e)}")
            return None
        
    def update_template(self, template_id: int, user_id: int, template_data: Dict[str, Any]) -> Optional[ReportTemplate]:
        """
        Update an existing report template
        
        Args:
            template_id: ID of template to update
            user_id: User ID for authorization check
            template_data: Template data to update
            
        Returns:
            Updated ReportTemplate or None if update failed
        """
        try:
            # Get the template and verify ownership
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id,
                ReportTemplate.user_id == user_id
            ).first()
            
            if not template:
                logger.error(f"Template {template_id} not found or not owned by user {user_id}")
                return None
                
            # Update fields if provided
            if 'name' in template_data:
                template.name = template_data['name']
                
            if 'description' in template_data:
                template.description = template_data['description']
                
            if 'report_type' in template_data:
                template.report_type = template_data['report_type']
                
            # Update template data if provided
            if 'sections' in template_data:
                current_data = template.template_data
                current_data['sections'] = template_data['sections']
                template.template_data = current_data
                
            self.db.commit()
            self.db.refresh(template)
            
            logger.info(f"Updated template {template_id}")
            return template
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating template: {str(e)}")
            return None
        
    def delete_template(self, template_id: int, user_id: int) -> bool:
        """
        Delete a report template
        
        Args:
            template_id: ID of template to delete
            user_id: User ID for authorization check
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Get the template and verify ownership
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id,
                ReportTemplate.user_id == user_id
            ).first()
            
            if not template:
                logger.error(f"Template {template_id} not found or not owned by user {user_id}")
                return False
                
            # Delete the template
            self.db.delete(template)
            self.db.commit()
            
            logger.info(f"Deleted template {template_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting template: {str(e)}")
            return False
        
    def get_templates_by_user(self, user_id: int) -> List[ReportTemplate]:
        """
        Get all report templates for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of ReportTemplate objects
        """
        try:
            templates = self.db.query(ReportTemplate).filter(
                ReportTemplate.user_id == user_id
            ).all()
            
            return templates
        except Exception as e:
            logger.error(f"Error getting templates for user {user_id}: {str(e)}")
            return []

    def get_template_by_id(self, template_id: int, user_id: int) -> Optional[ReportTemplate]:
        """
        Get a report template by ID with ownership check
        
        Args:
            template_id: Template ID
            user_id: User ID for authorization
            
        Returns:
            ReportTemplate or None if not found or not owned by user
        """
        try:
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id,
                ReportTemplate.user_id == user_id
            ).first()
            
            return template
        except Exception as e:
            logger.error(f"Error getting template {template_id}: {str(e)}")
            return None
    
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
    
    def _format_sections_with_template(self, sections: List[Section], template: ReportTemplate) -> List[Dict[str, Any]]:
        """
        Format sections according to a custom template
        
        Args:
            sections: List of Section objects
            template: ReportTemplate object with formatting rules
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        section_configs = template.template_data.get('sections', {})
        
        # Group sections by type for easier processing
        section_by_type = {}
        for section in sections:
            if section.section_type not in section_by_type:
                section_by_type[section.section_type] = []
            section_by_type[section.section_type].append(section)
        
        # Process each section type according to template
        for section_type, config in section_configs.items():
            if not config.get('include', True):
                continue  # Skip sections that shouldn't be included
                
            if section_type not in section_by_type:
                continue  # Skip if no sections of this type exist
                
            for section in section_by_type[section_type]:
                # Skip empty sections
                if not section.content:
                    continue
                    
                try:
                    # Parse content from JSON if it's a string
                    content = section.content
                    if isinstance(content, str):
                        content = json.loads(content)
                    
                    # Extract evidence and summary based on section type and report type
                    evidence = self._extract_evidence_from_content(content)
                    summary = self._get_section_summary(content)
                    
                    # Format section according to template
                    formatted_section = {
                        "title": config.get('title_override') or section.title,
                        "summary": summary,
                        "evidence": evidence,
                        "content": content,
                        "type": section.section_type,
                        "format_options": config.get('format_options', {}),
                        "order": config.get('order', 999)  # Default to end of report
                    }
                    
                    formatted_sections.append(formatted_section)
                    
                except Exception as e:
                    logger.error(f"Error formatting section {section.id}: {str(e)}")
        
        # Sort sections by order
        formatted_sections.sort(key=lambda s: s.get('order', 999))
        
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