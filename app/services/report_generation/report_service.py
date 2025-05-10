import logging
import os
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_

from app.models.models import Note, Section, User, ReportTemplate

# Configure logger
logger = logging.getLogger(__name__)

# Dictionary of medical jargon replacements for patient-friendly reports
MEDICAL_JARGON_REPLACEMENTS = {
    "myocardial infarction": "heart attack",
    "cerebrovascular accident": "stroke",
    "hypertension": "high blood pressure",
    "hyperlipidemia": "high cholesterol",
    "dyspnea": "shortness of breath",
    "pyrexia": "fever",
    "edema": "swelling",
    "emesis": "vomiting",
    "syncope": "fainting",
    "tachycardia": "rapid heart rate",
    "bradycardia": "slow heart rate",
    "pruritus": "itching",
    "erythema": "redness of skin",
    "contusion": "bruise",
    "laceration": "cut",
    "cephalgia": "headache",
    "angina pectoris": "chest pain",
    "dysphagia": "difficulty swallowing",
    "dysuria": "painful urination",
    "nausea": "feeling sick",
    "vertigo": "dizziness",
    "alopecia": "hair loss",
    "hypoesthesia": "reduced sensation",
    "paresthesia": "tingling sensation",
    "tinnitus": "ringing in ears",
    "rhinorrhea": "runny nose",
    "hyperhidrosis": "excessive sweating",
    "arrhythmia": "irregular heartbeat",
    "arthralgia": "joint pain",
    "myalgia": "muscle pain",
    "dyspepsia": "indigestion",
    "epistaxis": "nosebleed",
    "anorexia": "loss of appetite",
    "hematuria": "blood in urine",
    "melena": "blood in stool",
    "petechiae": "small red spots on skin"
}

class ReportService:
    """
    Service for generating medical reports from notes and sections.
    Handles template lookup, data mapping, content formatting, and report generation.
    """
    
    def __init__(self, db: Session):
        """
        Initialize report service
        
        Args:
            db: SQLAlchemy session for database operations
        """
        self.db = db
        self.templates_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils", "nlp", "report_templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(self.templates_path),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Register custom filters
        self.env.filters['simplify_medical_terms'] = self._simplify_medical_terms
        
        logger.info(f"ReportService initialized with templates from {self.templates_path}")
    
    #
    # TEMPLATE MANAGEMENT METHODS
    #
    
    def get_templates_by_user(self, user_id: int) -> List[ReportTemplate]:
        """
        Get all report templates created by or available to a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            List of templates
        """
        try:
            # Get user's custom templates and default templates
            templates = self.db.query(ReportTemplate).filter(
                or_(
                    ReportTemplate.user_id == user_id,
                    ReportTemplate.is_default == True
                )
            ).all()
            
            return templates
            
        except Exception as e:
            logger.error(f"Error getting templates for user {user_id}: {str(e)}")
            return []
    
    def get_template_by_id(self, template_id: int, user_id: int) -> Optional[ReportTemplate]:
        """
        Get a specific template by ID
        
        Args:
            template_id: Template ID
            user_id: User ID for authorization
            
        Returns:
            Template or None if not found
        """
        try:
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id,
                or_(
                    ReportTemplate.user_id == user_id,
                    ReportTemplate.is_default == True
                )
            ).first()
            
            return template
            
        except Exception as e:
            logger.error(f"Error getting template {template_id}: {str(e)}")
            return None
    
    def get_default_template(self, report_type: str = "doctor") -> Optional[ReportTemplate]:
        """
        Get the default template for a specific report type
        
        Args:
            report_type: Report type ("doctor" or "patient")
            
        Returns:
            Default template or None if not found
        """
        try:
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.report_type == report_type,
                ReportTemplate.is_default == True
            ).first()
            
            return template
            
        except Exception as e:
            logger.error(f"Error getting default template for {report_type}: {str(e)}")
            return None
    
    def create_template(self, user_id: int, template_data: Dict[str, Any]) -> Optional[ReportTemplate]:
        """
        Create a new report template
        
        Args:
            user_id: User ID creating the template
            template_data: Template data
            
        Returns:
            Created template or None if failed
        """
        try:
            # Create template
            template = ReportTemplate(
                user_id=user_id,
                name=template_data.get("name"),
                description=template_data.get("description"),
                report_type=template_data.get("report_type", "doctor"),
                is_default=template_data.get("is_default", False),
                html_template=template_data.get("html_template"),
                template_data=template_data.get("template_data", {})
            )
            
            self.db.add(template)
            self.db.commit()
            self.db.refresh(template)
            
            return template
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating template: {str(e)}")
            return None
    
    def update_template(self, template_id: int, user_id: int, template_data: Dict[str, Any]) -> Optional[ReportTemplate]:
        """
        Update an existing report template
        
        Args:
            template_id: Template ID
            user_id: User ID for authorization
            template_data: Template data
            
        Returns:
            Updated template or None if failed
        """
        try:
            # Get template
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id,
                ReportTemplate.user_id == user_id
            ).first()
            
            if not template:
                logger.error(f"Template {template_id} not found or not owned by user {user_id}")
                return None
            
            # Update fields
            if "name" in template_data:
                template.name = template_data["name"]
            if "description" in template_data:
                template.description = template_data["description"]
            if "report_type" in template_data:
                template.report_type = template_data["report_type"]
            if "is_default" in template_data:
                template.is_default = template_data["is_default"]
            if "html_template" in template_data:
                template.html_template = template_data["html_template"]
            if "template_data" in template_data:
                template.template_data = template_data["template_data"]
            
            self.db.commit()
            self.db.refresh(template)
            
            return template
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating template {template_id}: {str(e)}")
            return None
    
    def delete_template(self, template_id: int, user_id: int) -> bool:
        """
        Delete a report template
        
        Args:
            template_id: Template ID
            user_id: User ID for authorization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get template
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id,
                ReportTemplate.user_id == user_id,
                ReportTemplate.is_default == False  # Can't delete default templates
            ).first()
            
            if not template:
                logger.error(f"Template {template_id} not found, is default, or not owned by user {user_id}")
                return False
            
            self.db.delete(template)
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting template {template_id}: {str(e)}")
            return False
    
    #
    # DOCTOR REPORT GENERATION METHODS
    #
    
    def generate_doctor_report(self, note_id: int, template_id: Optional[int] = None) -> Optional[str]:
        """
        Generate a doctor-focused report from a note
        
        Args:
            note_id: ID of the note to generate report from
            template_id: Optional ID of a specific template to use
            
        Returns:
            HTML report as a string, or None if generation failed
        """
        try:
            # Get note with sections
            note = self.db.query(Note).options(joinedload(Note.sections)).filter(Note.id == note_id).first()
            if not note:
                logger.error(f"Note {note_id} not found")
                return None
            
            # Get template to use (custom or default)
            template = None
            if template_id:
                template = self.db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
            
            # Use default template if no custom template specified or found
            if not template:
                template = self.get_default_template("doctor")
                if not template:
                    logger.error(f"No default doctor template found")
                    return None
            
            # Get patient info if available
            patient_info = self._get_patient_info(note.patient_id)
            
            # Get note creator info
            creator_info = self._get_user_info(note.user_id)
            
            # Format sections for doctor report
            formatted_sections = self._format_sections_for_doctor(note.sections, template)
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "encounter_date": note.encounter_date.strftime("%Y-%m-%d") if note.encounter_date else "Unknown",
                "patient_info": patient_info,
                "provider_info": creator_info,
                "note_title": note.title,
                "sections": formatted_sections,
                "report_type": "doctor"
            }
            
            # Render template
            rendered_report = self._render_report(template, report_data)
            
            logger.info(f"Generated doctor report for note {note_id}")
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating doctor report for note {note_id}: {str(e)}")
            return None
    
    def _format_sections_for_doctor(self, sections: List[Section], template: ReportTemplate) -> List[Dict[str, Any]]:
        """
        Format sections for a doctor's report with SOAP categorization
        
        Args:
            sections: List of Section objects
            template: ReportTemplate object with formatting rules
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        
        # Get template configuration
        template_data = template.template_data
        section_configs = template_data.get('sections', {})
        
        # Process each section
        for section in sections:
            # Skip empty sections
            if not section.content:
                continue
            
            # Get section category and config
            soap_category = section.soap_category
            config = section_configs.get(soap_category, section_configs.get("OTHER", {}))
            
            # Skip if this category should not be included
            if not config.get('include', True):
                continue
                
            # Get format options from template
            format_options = config.get('format_options', {})
            
            # Format section title
            title = config.get('title_override', section.title)
            
            # Format section content
            formatted_section = self._format_section_for_doctor(
                section, 
                title,
                format_options
            )
            
            # Set SOAP category for template grouping
            formatted_section['soap_category'] = soap_category
            
            # Set display order from template if available
            formatted_section['order'] = config.get('order', 999)
            
            # Add to list
            formatted_sections.append(formatted_section)
        
        # Sort sections by order within each SOAP category
        formatted_sections.sort(key=lambda s: s.get('order', 999))
        
        return formatted_sections
    
    def _format_section_for_doctor(self, section: Section, title: str, format_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a section for doctor report
        
        Args:
            section: Section object
            title: Section title to use
            format_options: Formatting options from template
            
        Returns:
            Formatted section dictionary
        """
        # Initialize formatted section
        formatted_section = {
            'id': section.id,
            'title': title,
            'soap_category': section.soap_category,
            'content_html': ''
        }
        
        # Format content based on its type
        content = section.content
        
        # Process content if available
        if content:
            # For doctor report, use detailed technical content
            if isinstance(content, dict):
                # Format each part of the content
                content_html = self._format_dictionary_content(content, format_options, is_doctor=True)
                formatted_section['content_html'] = content_html
            elif isinstance(content, str):
                # Format text content
                formatted_section['content_html'] = content
            else:
                # Unknown content type
                formatted_section['content_html'] = str(content)
        
        return formatted_section
    
    def _format_dictionary_content(self, content: Dict[str, Any], format_options: Dict[str, Any], is_doctor: bool) -> str:
        """
        Format dictionary content into HTML with specialized formatting for doctor reports
        
        Args:
            content: Dictionary content
            format_options: Formatting options
            is_doctor: Whether formatting for doctor report
            
        Returns:
            HTML string
        """
        html_parts = []
        
        for key, value in content.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, dict):
                # Nested dictionary - use section-based formatting
                nested_parts = []
                for sub_key, sub_value in value.items():
                    formatted_sub_key = sub_key.replace('_', ' ').title()
                    formatted_sub_value = str(sub_value)
                    
                    # Apply special formatting for doctor reports
                    if is_doctor:
                        # Highlight abnormal values
                        highlight_abnormal = format_options.get('highlight_abnormal', False)
                        if highlight_abnormal and self._is_abnormal_value(sub_key, sub_value):
                            formatted_sub_value = f'<span class="abnormal">{formatted_sub_value}</span>'
                        
                        # Format diagnostic codes
                        if sub_key.lower() in ["code", "codes", "diagnosis_code", "icd", "cpt", "snomed"]:
                            formatted_sub_value = f'<span class="diagnosis-code">{formatted_sub_value}</span>'
                    
                    nested_parts.append(f'<div class="field-item"><span class="field-name">{formatted_sub_key}:</span> <span class="field-value">{formatted_sub_value}</span></div>')
                
                nested_html = "\n".join(nested_parts)
                html_parts.append(f'<div class="content-section"><h4>{formatted_key}</h4><div class="nested-fields">{nested_html}</div></div>')
            else:
                # Simple key-value
                formatted_value = str(value)
                
                # Apply special formatting for doctor reports
                if is_doctor:
                    # Highlight abnormal values
                    if format_options.get('highlight_abnormal', False) and self._is_abnormal_value(key, value):
                        formatted_value = f'<span class="abnormal">{formatted_value}</span>'
                    
                    # Format diagnostic codes
                    if key.lower() in ["code", "codes", "diagnosis_code", "icd", "cpt", "snomed"]:
                        formatted_value = f'<span class="diagnosis-code">{formatted_value}</span>'
                
                html_parts.append(f'<div class="field-item"><span class="field-name">{formatted_key}:</span> <span class="field-value">{formatted_value}</span></div>')
        
        return "\n".join(html_parts)
    
    def _is_abnormal_value(self, key: str, value: Any) -> bool:
        """
        Check if a value should be highlighted as abnormal based on key and value
        
        Args:
            key: Field key
            value: Field value
            
        Returns:
            True if value should be highlighted as abnormal
        """
        # Convert value to string for checking
        str_value = str(value).lower()
        
        # Check for explicit abnormal indicators
        abnormal_indicators = ["abnormal", "elevated", "high", "low", "positive", "critical", "severe", "irregular"]
        if any(indicator in str_value for indicator in abnormal_indicators):
            return True
        
        # Check for visual indicators
        if "*" in str_value or "!" in str_value:
            return True
        
        # Check specific vital signs or lab values (could be expanded with actual ranges)
        if "blood pressure" in key.lower() and any(str_value.startswith(prefix) for prefix in ["14", "15", "16", "17", "18", "19"]):
            return True
        if "temperature" in key.lower() and any(num in str_value for num in ["99.", "100", "101", "102", "103", "104"]):
            return True
        if "heart rate" in key.lower() or "pulse" in key.lower():
            # Extract numeric part
            import re
            nums = re.findall(r'\d+', str_value)
            if nums and (int(nums[0]) > 100 or int(nums[0]) < 50):
                return True
        
        return False
    
    #
    # PATIENT REPORT GENERATION METHODS
    #
    
    def generate_patient_report(self, note_id: int, template_id: Optional[int] = None) -> Optional[str]:
        """
        Generate a patient-friendly report from a note
        
        Args:
            note_id: ID of the note to generate report from
            template_id: Optional ID of a specific template to use
            
        Returns:
            HTML report as a string, or None if generation failed
        """
        try:
            # Get note with sections
            note = self.db.query(Note).options(joinedload(Note.sections)).filter(Note.id == note_id).first()
            if not note:
                logger.error(f"Note {note_id} not found")
                return None
            
            # Get template to use (custom or default)
            template = None
            if template_id:
                template = self.db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
            
            # Use default template if no custom template specified or found
            if not template:
                template = self.get_default_template("patient")
                if not template:
                    logger.error(f"No default patient template found")
                    return None
            
            # Get patient info if available
            patient_info = self._get_patient_info(note.patient_id)
            
            # Get note creator info
            creator_info = self._get_user_info(note.user_id)
            
            # Format sections for patient report
            formatted_sections = self._format_sections_for_patient(note.sections, template)
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "encounter_date": note.encounter_date.strftime("%Y-%m-%d") if note.encounter_date else "Unknown",
                "patient_info": patient_info,
                "provider_info": creator_info,
                "note_title": note.title,
                "sections": formatted_sections,
                "report_type": "patient"
            }
            
            # Render template
            rendered_report = self._render_report(template, report_data)
            
            logger.info(f"Generated patient report for note {note_id}")
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating patient report for note {note_id}: {str(e)}")
            return None
    
    def _format_sections_for_patient(self, sections: List[Section], template: ReportTemplate) -> List[Dict[str, Any]]:
        """
        Format sections for a patient-friendly report with SOAP categorization
        
        Args:
            sections: List of Section objects
            template: ReportTemplate object with formatting rules
            
        Returns:
            List of formatted section dictionaries
        """
        formatted_sections = []
        
        # Get template configuration
        template_data = template.template_data
        section_configs = template_data.get('sections', {})
        
        # Process each section
        for section in sections:
            # Skip empty sections
            if not section.content:
                continue
            
            # Get section category and config
            soap_category = section.soap_category
            config = section_configs.get(soap_category, section_configs.get("OTHER", {}))
            
            # Skip if this category should not be included
            if not config.get('include', True):
                continue
                
            # Get format options from template
            format_options = config.get('format_options', {})
            
            # Format section title
            title = config.get('title_override', section.title)
            
            # Format section content
            formatted_section = self._format_section_for_patient(
                section, 
                title,
                format_options
            )
            
            # Set SOAP category for template grouping
            formatted_section['soap_category'] = soap_category
            
            # Set display order from template if available
            formatted_section['order'] = config.get('order', 999)
            
            # Add to list
            formatted_sections.append(formatted_section)
        
        # Sort sections by order within each SOAP category
        formatted_sections.sort(key=lambda s: s.get('order', 999))
        
        return formatted_sections
    
    def _format_section_for_patient(self, section: Section, title: str, format_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a section for patient report with simplified language
        
        Args:
            section: Section object
            title: Section title to use
            format_options: Formatting options from template
            
        Returns:
            Formatted section dictionary
        """
        # Initialize formatted section
        formatted_section = {
            'id': section.id,
            'title': title,
            'soap_category': section.soap_category,
            'content_html': ''
        }
        
        # Format content based on its type
        content = section.content
        
        # Process content if available
        if content:
            # For patient report, simplify language
            if isinstance(content, dict):
                # Format each part of the content
                content_html = self._format_dictionary_content(content, format_options, is_doctor=False)
                # Simplify medical terms in content
                content_html = self._simplify_medical_terms(content_html)
                formatted_section['content_html'] = content_html
            elif isinstance(content, str):
                # Format text content and simplify medical terms
                formatted_section['content_html'] = self._simplify_medical_terms(content)
            else:
                # Unknown content type
                formatted_section['content_html'] = self._simplify_medical_terms(str(content))
        
        return formatted_section
    
    #
    # HELPER METHODS
    #
    
    def _get_patient_info(self, patient_id: Optional[int]) -> Dict[str, Any]:
        """
        Get patient information for the report
        
        Args:
            patient_id: Patient ID or None
            
        Returns:
            Dictionary with patient information
        """
        # Note: In a real implementation, you would query a Patient model
        # For now, return placeholder data if patient_id is provided
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
    
    def _get_user_info(self, user_id: int) -> Dict[str, Any]:
        """
        Get user (provider) information for the report
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user information
        """
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                return {
                    "id": user.id,
                    "name": f"{user.first_name} {user.last_name}",
                    "credentials": "MD",  # Placeholder
                    "specialty": "Specialty",  # Placeholder
                    "contact": user.email
                }
            else:
                return {
                    "id": user_id,
                    "name": "Unknown Provider",
                    "credentials": "",
                    "specialty": "",
                    "contact": ""
                }
        except Exception as e:
            logger.error(f"Error getting user info for {user_id}: {str(e)}")
            return {
                "id": user_id,
                "name": "Unknown Provider",
                "credentials": "",
                "specialty": "",
                "contact": ""
            }
    
    def _simplify_medical_terms(self, text: str) -> str:
        """
        Replace medical jargon with patient-friendly terms
        
        Args:
            text: Text to simplify
            
        Returns:
            Simplified text
        """
        # Simple implementation - replace known terms
        result = text
        for medical_term, simple_term in MEDICAL_JARGON_REPLACEMENTS.items():
            # Case insensitive replacement
            result = result.replace(medical_term, simple_term)
            result = result.replace(medical_term.capitalize(), simple_term.capitalize())
        
        return result
    
    def _render_report(self, template: ReportTemplate, report_data: Dict[str, Any]) -> str:
        """
        Render a report using template
        
        Args:
            template: ReportTemplate object
            report_data: Data for rendering
            
        Returns:
            Rendered HTML
        """
        # Check if template has custom HTML
        if template.html_template:
            # Use custom HTML template
            template_obj = Template(template.html_template)
            return template_obj.render(**report_data)
        else:
            # Use default template based on report type
            template_file = "default_patient.html" if template.report_type == "patient" else "default_doctor.html"
            return self._render_template(template_file, report_data)
    
    def _render_template(self, template_file: str, report_data: Dict[str, Any]) -> str:
        """
        Render a template with data
        
        Args:
            template_file: Template file name
            report_data: Data for rendering
            
        Returns:
            Rendered HTML
        """
        try:
            template = self.env.get_template(template_file)
            return template.render(**report_data)
        except Exception as e:
            logger.error(f"Error rendering template {template_file}: {str(e)}")
            # Fallback to simple HTML
            return self._generate_fallback_html(report_data)
    
    def _generate_fallback_html(self, report_data: Dict[str, Any]) -> str:
        """
        Generate fallback HTML when template rendering fails
        
        Args:
            report_data: Report data
            
        Returns:
            Simple HTML report
        """
        # Create a simple HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data.get('note_title', 'Medical Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; margin-top: 20px; }}
                .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>{report_data.get('note_title', 'Medical Report')}</h1>
            <p>Date: {report_data.get('report_date', '')}</p>
            <p>Encounter Date: {report_data.get('encounter_date', '')}</p>
            
            <div class="patient-info">
                <h2>Patient Information</h2>
                <p>Name: {report_data.get('patient_info', {}).get('name', 'Unknown')}</p>
                <p>DOB: {report_data.get('patient_info', {}).get('dob', 'Unknown')}</p>
                <p>MRN: {report_data.get('patient_info', {}).get('mrn', 'Unknown')}</p>
            </div>
            
            <div class="provider-info">
                <h2>Provider Information</h2>
                <p>Name: {report_data.get('provider_info', {}).get('name', 'Unknown')}</p>
                <p>Contact: {report_data.get('provider_info', {}).get('contact', 'Unknown')}</p>
            </div>
        """
        
        # Add sections
        for section in report_data.get('sections', []):
            html += f"""
            <div class="section">
                <h2>{section.get('title', 'Untitled Section')}</h2>
                <div class="content">
                    {section.get('content_html', '')}
                </div>
            </div>
            """
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        return html
    
    #
    # DIRECT DATA REPORT GENERATION
    #
    
    def generate_report_from_data(self, note_data: Dict[str, Any], report_type: str, template_id: Optional[int] = None) -> Optional[str]:
        """
        Generate a report directly from note data without requiring a database record
        
        Args:
            note_data: Dictionary with note data including sections
            report_type: Type of report ("doctor" or "patient")
            template_id: Optional template ID to use
            
        Returns:
            HTML report as a string, or None if generation failed
        """
        try:
            # Get template to use (custom or default)
            template = None
            if template_id:
                template = self.db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
            
            # Use default template if no custom template specified or found
            if not template:
                template = self.get_default_template(report_type)
                if not template:
                    logger.error(f"No default {report_type} template found")
                    return None
            
            # Extract note information
            title = note_data.get("title", "Medical Note")
            patient_id = note_data.get("patient_id")
            user_id = note_data.get("user_id", 1)  # Default to system user if not provided
            
            # Handle encounter date
            encounter_date_str = note_data.get("encounter_date")
            encounter_date = datetime.now()
            if encounter_date_str:
                try:
                    if isinstance(encounter_date_str, str):
                        # Try different date formats
                        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"]:
                            try:
                                encounter_date = datetime.strptime(encounter_date_str, fmt)
                                break
                            except ValueError:
                                continue
                except Exception as e:
                    logger.warning(f"Could not parse encounter date '{encounter_date_str}': {str(e)}")
            
            # Extract sections
            sections_data = note_data.get("sections", [])
            
            # Create Section objects from the data
            sections = []
            for section_data in sections_data:
                section = self._create_section_from_data(section_data)
                sections.append(section)
            
            # Get patient info if available
            patient_info = self._get_patient_info(patient_id)
            
            # Get note creator info
            creator_info = self._get_user_info(user_id)
            
            # Format sections based on report type
            formatted_sections = []
            if report_type.lower() == "doctor":
                formatted_sections = self._format_sections_for_doctor(sections, template)
            else:
                formatted_sections = self._format_sections_for_patient(sections, template)
            
            # Prepare report data
            report_data = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "encounter_date": encounter_date.strftime("%Y-%m-%d"),
                "patient_info": patient_info,
                "provider_info": creator_info,
                "note_title": title,
                "sections": formatted_sections,
                "report_type": report_type
            }
            
            # Render template
            rendered_report = self._render_report(template, report_data)
            
            logger.info(f"Generated {report_type} report from direct data")
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating report from data: {str(e)}")
            return None
    
    def _create_section_from_data(self, section_data: Dict[str, Any]) -> Section:
        """
        Create a Section object from dictionary data
        
        Args:
            section_data: Dictionary with section data
            
        Returns:
            Section object
        """
        # Create a temporary Section object (not saved to database)
        section = Section(
            id=section_data.get("id", 0),
            note_id=section_data.get("note_id", 0),
            user_id=section_data.get("user_id", 1),
            title=section_data.get("title", "Untitled Section"),
            template_id=section_data.get("template_id"),
            soap_category=section_data.get("soap_category", "OTHER"),
            content=section_data.get("content", {})
        )
        
        return section