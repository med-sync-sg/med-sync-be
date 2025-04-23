import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from app.models.models import Note, Section, User, ReportTemplate
from app.services.report_generation.section_management_service import SectionManagementService
from sqlalchemy import func
from sqlalchemy.orm import joinedload

doctor_report_data = {
    "name": "My Custom Doctor Report",
    "description": "Specialized report for cardiology",
    "report_type": "doctor",
    "template_data": {
        "sections": {
            "CHIEF_COMPLAINT": {
                "section_type": "CHIEF_COMPLAINT",
                "include": True,
                "order": 0,
                "title_override": "Presenting Symptoms",
                "format_options": {
                    "show_evidence": True,
                    "detailed_view": True,
                    "highlight_cardiac_terms": True
                }
            },
            "PATIENT_MEDICAL_HISTORY": {
                "section_type": "PATIENT_MEDICAL_HISTORY",
                "include": True,
                "order": 1,
                "format_options": {
                    "show_evidence": True,
                    "detailed_view": True,
                    "filter_by_relevance": "cardiology"
                }
            },
            "OTHERS": {
                "section_type": "OTHERS",
                "include": False  # Exclude other sections
            }
        }
    }
}

patient_report_data = {
    "name": "Default Patient Report",
    "description": "Patient-friendly summary with simplified language",
    "report_type": "patient",
    "is_default": True,
    "template_data": {
        "sections": {
            "CHIEF_COMPLAINT": {
                "section_type": "CHIEF_COMPLAINT",
                "include": True,
                "order": 0,
                "title_override": "Your Main Concerns",
                "format_options": {
                    "show_actions": True,
                    "patient_friendly": True
                }
            },
            "PATIENT_MEDICAL_HISTORY": {
                "section_type": "PATIENT_MEDICAL_HISTORY",
                "include": True,
                "order": 1,
                "title_override": "Your Medical Background",
                "format_options": {
                    "show_actions": True,
                    "patient_friendly": True
                }
            },
            "OTHERS": {
                "section_type": "OTHERS",
                "include": True,
                "order": 2,
                "title_override": "Other Information",
                "format_options": {
                    "show_actions": True,
                    "patient_friendly": True
                }
            }
        }
    }
}

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
        self.section_management_service = SectionManagementService(self.db)
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
            note = self.db.query(Note).options(joinedload(Note.sections)).filter(Note.id == note_id).first()
            if not note:
                return None
                
            # Get patient info
            patient_info = {
                "id": note.patient_id,
                "name": "Hello TEst",
                "age": 5,
                "gender": "MALE"
            }            
            report_sections = self._format_sections_with_template(note.sections, template=doctor_report_data)
            
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "patient_info": patient_info,
                "sections": report_sections
            }
            
            # Render template
            template = self.env.get_template("default_doctor_report.html")
            rendered_report = template.render(report_data)
            print(rendered_report)
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
            note = self.db.query(Note).options(joinedload(Note.sections)).filter(Note.id == note_id).first()
            if not note:
                return None
                
            # Get patient info
            patient_info = {
                "id": note.patient_id,
                "name": "Hello TEst",
                "age": 5,
                "gender": "MALE"
            }
            # Format sections for patient report (simplified language)
            report_sections = self._format_sections_with_template(note.sections, patient_report_data)
            
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

    def _format_sections_with_template(self, sections: List[Section], template: ReportTemplate) -> List[Dict[str, Any]]:
        """
        Format sections according to a template
        
        Args:
            sections: List of Section objects
            template: ReportTemplate object with formatting rules
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        section_configs = template.template_data.get('sections', {})
        report_type = template.report_type
        
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
                    
                    # Format section based on template type and section type
                    if report_type == "doctor":
                        # Format for doctor report
                        if section_type == "CHIEF_COMPLAINT":
                            formatted_section = self._format_chief_complaint_for_doctor(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                        elif section_type == "PATIENT_MEDICAL_HISTORY":
                            formatted_section = self._format_medical_history_for_doctor(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                        elif section_type == "PATIENT_INFORMATION":
                            formatted_section = self._format_patient_info_for_doctor(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                        else:
                            formatted_section = self._format_generic_section_for_doctor(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                    elif report_type == "patient":
                        # Format for patient report
                        if section_type == "CHIEF_COMPLAINT":
                            formatted_section = self._format_chief_complaint_for_patient(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                        elif section_type == "PATIENT_MEDICAL_HISTORY":
                            formatted_section = self._format_medical_history_for_patient(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                        elif section_type == "PATIENT_INFORMATION":
                            formatted_section = self._format_patient_info_for_patient(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                        else:
                            formatted_section = self._format_generic_section_for_patient(
                                config.get('title_override') or section.title, 
                                content,
                                config.get('format_options', {})
                            )
                    else:
                        # Format for custom report type - use doctor formatting by default
                        formatted_section = self._format_generic_section_for_doctor(
                            config.get('title_override') or section.title, 
                            content,
                            config.get('format_options', {})
                        )
                    
                    # Add order from template
                    formatted_section['order'] = config.get('order', 999)
                    
                    formatted_sections.append(formatted_section)
                    
                except Exception as e:
                    logger.error(f"Error formatting section {section.id}: {str(e)}")
        
        # Sort sections by order
        formatted_sections.sort(key=lambda s: s.get('order', 999))
        
        return formatted_sections
    
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
    
    def _format_sections_with_template(self, sections: List[Section], template: Dict) -> List[Dict[str, Any]]:
        """
        Format sections according to a template with semantic matching for flexible section handling
        
        Args:
            sections: List of Section objects
            template: ReportTemplate object with formatting rules
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        section_configs = template['template_data'].get('sections', {})
        report_type = template['report_type']
        
        # Process each section individually
        for section in sections:
            # Skip empty sections
            if not section.content:
                continue
                
            try:
                # Parse content from JSON if it's a string
                content = section.content
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Determine section type semantically if needed
                section_type = section.section_type
                if section_type == "OTHERS" or not section_type:
                    section_type = self.section_management_service.get_semantic_section_type(section.title, content)
                
                # Check if this section type is configured in the template
                if section_type in section_configs:
                    config = section_configs[section_type]
                    
                    # Skip if configured not to include
                    if not config.get('include', True):
                        continue
                    
                    # Extract format options
                    format_options = config.get('format_options', {})
                    
                    # Use title override if specified
                    section_title = config.get('title_override') or section.title
                    
                    # Set order from template
                    order = config.get('order', 999)
                else:
                    # Use default settings for unknown section types
                    format_options = {}
                    section_title = section.title
                    order = 999  # Place at end by default
                
                # Format section based on type and report type
                formatted_section = None
                
                if report_type == "doctor":
                    # Format for doctor report
                    if section_type == "CHIEF_COMPLAINT":
                        formatted_section = self._format_chief_complaint_for_doctor(
                            section_title, content, format_options
                        )
                    elif section_type == "PATIENT_MEDICAL_HISTORY":
                        formatted_section = self._format_medical_history_for_doctor(
                            section_title, content, format_options
                        )
                    elif section_type == "PATIENT_INFORMATION":
                        formatted_section = self._format_patient_info_for_doctor(
                            section_title, content, format_options
                        )
                    elif section_type == "ASSESSMENT":
                        formatted_section = self._format_assessment_for_doctor(
                            section_title, content, format_options
                        )
                    elif section_type == "PLAN":
                        formatted_section = self._format_plan_for_doctor(
                            section_title, content, format_options
                        )
                    elif section_type == "PHYSICAL_EXAM":
                        formatted_section = self._format_physical_exam_for_doctor(
                            section_title, content, format_options
                        )
                    else:
                        formatted_section = self._format_generic_section_for_doctor(
                            section_title, content, format_options
                        )
                elif report_type == "patient":
                    # Format for patient report
                    if section_type == "CHIEF_COMPLAINT":
                        formatted_section = self._format_chief_complaint_for_patient(
                            section_title, content, format_options
                        )
                    elif section_type == "PATIENT_MEDICAL_HISTORY":
                        formatted_section = self._format_medical_history_for_patient(
                            section_title, content, format_options
                        )
                    elif section_type == "PATIENT_INFORMATION":
                        formatted_section = self._format_patient_info_for_patient(
                            section_title, content, format_options
                        )
                    elif section_type == "ASSESSMENT":
                        formatted_section = self._format_assessment_for_patient(
                            section_title, content, format_options
                        )
                    elif section_type == "PLAN":
                        formatted_section = self._format_plan_for_patient(
                            section_title, content, format_options
                        )
                    elif section_type == "PHYSICAL_EXAM":
                        formatted_section = self._format_physical_exam_for_patient(
                            section_title, content, format_options
                        )
                    else:
                        formatted_section = self._format_generic_section_for_patient(
                            section_title, content, format_options
                        )
                else:
                    # Format for custom report type - use doctor formatting by default
                    formatted_section = self._format_generic_section_for_doctor(
                        section_title, content, format_options
                    )
                
                if formatted_section:
                    # Add order and original section ID for reference
                    formatted_section['order'] = order
                    formatted_section['original_section_id'] = section.id
                    
                    formatted_sections.append(formatted_section)
                    
            except Exception as e:
                logger.error(f"Error formatting section {section.id}: {str(e)}")
        
        # Sort sections by order
        formatted_sections.sort(key=lambda s: s.get('order', 999))
        
        return formatted_sections