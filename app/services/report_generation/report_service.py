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


DEBUG_PATH = os.path.join("test_client_results")
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
        
        # Update the templates path to correctly point to the default_html directory
        self.templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "default_html"
        )
        
        # Check if the templates directory exists
        if not os.path.exists(self.templates_path):
            logger.warning(f"Templates directory not found: {self.templates_path}")
            # Try alternate path - fallback
            alternate_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "utils", "nlp", "report_templates"
            )
            if os.path.exists(alternate_path):
                logger.info(f"Using alternate templates path: {alternate_path}")
                self.templates_path = alternate_path
            else:
                logger.warning(f"Alternate templates directory not found: {alternate_path}")
        
        try:
            self.env = Environment(
                loader=FileSystemLoader(self.templates_path),
                autoescape=select_autoescape(['html', 'xml'])
            )
            
            # Register custom filters
            self.env.filters['simplify_medical_terms'] = self._simplify_medical_terms
            
            # Check if required templates exist
            required_templates = ["default_doctor.html", "default_patient.html"]
            for template_name in required_templates:
                if not os.path.exists(os.path.join(self.templates_path, template_name)):
                    logger.warning(f"Required template not found: {template_name}")
                else:
                    logger.info(f"Found template: {template_name}")
                    
            logger.info(f"ReportService initialized with templates from {self.templates_path}")
            
        except Exception as e:
            logger.error(f"Error initializing Jinja2 environment: {str(e)}")
            # Initialize with a dummy environment as fallback
            self.env = Environment()
    
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
            
            logger.info(f"Retrieved {len(templates)} templates for user {user_id}")
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
            
            if template:
                logger.info(f"Retrieved template {template_id} for user {user_id}")
            else:
                logger.warning(f"Template {template_id} not found for user {user_id}")
                
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
            ).order_by(ReportTemplate.updated_at.desc()).first()
            
            if template:
                logger.info(f"Retrieved default {report_type} template: {template.name}")
            else:
                logger.warning(f"No default template found for {report_type}")
                
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
            
            logger.info(f"Created template: {template.name} (ID: {template.id})")
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
            
            logger.info(f"Updated template {template_id}")
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
            logger.info(f"Deleted template {template_id}")
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
                if template:
                    logger.info(f"Using custom template: {template.name}")
                else:
                    logger.warning(f"Custom template {template_id} not found, falling back to default")
            
            # Use default template if no custom template specified or found
            if not template:
                template = self.get_default_template("doctor")
                if not template:
                    logger.error(f"No default doctor template found")
                    return None
                logger.info(f"Using default doctor template: {template.name}")
            
            # Get patient info if available
            patient_info = self._get_patient_info(note.patient_id)
            
            # Get note creator info
            creator_info = self._get_user_info(note.user_id)
            
            # Format sections for doctor report
            formatted_sections = self._format_sections_for_doctor(note.sections, template)
            logger.info(f"Formatted {len(formatted_sections)} sections for doctor report")
            
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
            
            if rendered_report:
                logger.info(f"Generated doctor report for note {note_id} ({len(rendered_report)} bytes)")
            else:
                logger.error(f"Failed to render doctor report for note {note_id}")
                
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating doctor report for note {note_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        template_data = template.template_data or {}
        section_configs = template_data.get('sections', {})
        
        # Debug info
        logger.info(f"Formatting {len(sections)} sections for doctor report")
        logger.info(f"Template has section configs: {list(section_configs.keys())}")
        
        # Process each section
        for section in sections:
            # Skip empty sections
            if not section.content:
                logger.debug(f"Skipping empty section (no content)")
                continue
            
            # Get section category and config
            soap_category = section.soap_category or "OTHER"
            config = section_configs.get(soap_category, section_configs.get("OTHER", {}))
            
            # Skip if this category should not be included
            if not config.get('include', True):
                logger.debug(f"Skipping section {section.id} (category {soap_category} not included)")
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
            logger.debug(f"Added formatted section '{title}' ({soap_category})")
        
        # Sort sections by order within each SOAP category
        formatted_sections.sort(key=lambda s: s.get('order', 999))
        
        # Log the formatted sections
        logger.info(f"Formatted {len(formatted_sections)} sections for doctor report")
        for i, section in enumerate(formatted_sections):
            logger.info(f"  Section {i+1}: {section.get('title')} ({section.get('soap_category')})")
            logger.info(f"    Content keys: {list(section.get('content', {}).keys())}")
            logger.info(f"    Has content_html: {'Yes' if section.get('content_html') else 'No'}")
        
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
            'soap_category': section.soap_category or "OTHER",
            'content_html': '',
            'content': {} # Add the actual content for template rendering
        }
        
        # Format content based on its type
        content = section.content
        
        # Process content if available
        if content:
            logger.debug(f"Processing content for section {section.id}: {type(content)}")
            
            # For doctor report, use detailed technical content
            if isinstance(content, dict):
                # Copy the content to the formatted section
                formatted_section['content'] = content
                
                # Also generate HTML representation
                content_html = self._format_dictionary_content(content, format_options, is_doctor=True)
                formatted_section['content_html'] = content_html
            elif isinstance(content, str):
                # Format text content
                formatted_section['content_html'] = content
                formatted_section['content'] = {"text": {"name": "Text", "data_type": "string", "value": content}}
            else:
                # Unknown content type
                logger.warning(f"Unknown content type for section {section.id}: {type(content)}")
                formatted_section['content_html'] = str(content)
                formatted_section['content'] = {"text": {"name": "Text", "data_type": "string", "value": str(content)}}
        
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
            
            # Handle single FieldTemplate object
            if isinstance(value, dict) and all(k in value for k in ['name', 'data_type']):
                # This is a field object, format accordingly
                field_name = value.get('name', formatted_key)
                field_value = value.get('value', '')
                field_type = value.get('data_type', 'string')
                
                # Apply special formatting for doctor reports
                if is_doctor:
                    # Highlight abnormal values
                    highlight_abnormal = format_options.get('highlight_abnormal', False)
                    if highlight_abnormal and self._is_abnormal_value(key, field_value):
                        field_value = f'<span class="abnormal">{field_value}</span>'
                    
                    # Format diagnostic codes
                    if field_type.lower() in ["code"] or key.lower() in ["code", "codes", "diagnosis_code", "icd", "cpt", "snomed"]:
                        field_value = f'<span class="diagnosis-code">{field_value}</span>'
                
                html_parts.append(f'<div class="field-item"><span class="field-name">{field_name}:</span> <span class="field-value">{field_value}</span></div>')
            
            # Handle list of FieldTemplate objects
            elif isinstance(value, list):
                list_items = []
                
                for i, item in enumerate(value):
                    if isinstance(item, dict) and all(k in item for k in ['name', 'data_type']):
                        # This is a field object in a list
                        field_name = item.get('name', f"{formatted_key} {i+1}")
                        field_value = item.get('value', '')
                        field_type = item.get('data_type', 'string')
                        
                        # Apply special formatting for doctor reports
                        if is_doctor:
                            # Highlight abnormal values
                            highlight_abnormal = format_options.get('highlight_abnormal', False)
                            if highlight_abnormal and self._is_abnormal_value(key, field_value):
                                field_value = f'<span class="abnormal">{field_value}</span>'
                            
                            # Format diagnostic codes
                            if field_type.lower() in ["code"] or key.lower() in ["code", "codes", "diagnosis_code", "icd", "cpt", "snomed"]:
                                field_value = f'<span class="diagnosis-code">{field_value}</span>'
                        
                        list_items.append(f'<div class="list-item"><span class="field-name">{field_name}:</span> <span class="field-value">{field_value}</span></div>')
                    else:
                        # Regular list item
                        list_items.append(f'<div class="list-item">{item}</div>')
                
                if list_items:
                    list_html = "\n".join(list_items)
                    html_parts.append(f'<div class="content-section"><h4>{formatted_key}</h4><div class="list-content">{list_html}</div></div>')
            
            # Handle nested dictionary
            elif isinstance(value, dict):
                # Nested dictionary - use section-based formatting
                nested_parts = []
                for sub_key, sub_value in value.items():
                    formatted_sub_key = sub_key.replace('_', ' ').title()
                    
                    # Handle different value types
                    if isinstance(sub_value, dict):
                        # Nested dictionary, format recursively
                        formatted_sub_value = self._format_dictionary_content({sub_key: sub_value}, format_options, is_doctor)
                    else:
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
            
            # Handle simple key-value
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
        
        # Check specific vital signs or lab values
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
                if template:
                    logger.info(f"Using custom template: {template.name}")
                else:
                    logger.warning(f"Custom template {template_id} not found, falling back to default")
            
            # Use default template if no custom template specified or found
            if not template:
                template = self.get_default_template("patient")
                if not template:
                    logger.error(f"No default patient template found")
                    return None
                logger.info(f"Using default patient template: {template.name}")
            
            # Get patient info if available
            patient_info = self._get_patient_info(note.patient_id)
            
            # Get note creator info
            creator_info = self._get_user_info(note.user_id)
            
            # Format sections for patient report
            formatted_sections = self._format_sections_for_patient(note.sections, template)
            logger.info(f"Formatted {len(formatted_sections)} sections for patient report")
            
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
            
            if rendered_report:
                logger.info(f"Generated patient report for note {note_id} ({len(rendered_report)} bytes)")
            else:
                logger.error(f"Failed to render patient report for note {note_id}")
                
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating patient report for note {note_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        template_data = template.template_data or {}
        section_configs = template_data.get('sections', {})
        
        # Process each section
        for section in sections:
            # Skip empty sections
            if not section.content:
                continue
            
            # Skip sections not visible to patient
            if hasattr(section, 'is_visible_to_patient') and not section.is_visible_to_patient:
                logger.debug(f"Skipping section {section.id} (not visible to patient)")
                continue
            
            # Get section category and config
            soap_category = section.soap_category or "OTHER"
            config = section_configs.get(soap_category, section_configs.get("OTHER", {}))
            
            # Skip if this category should not be included
            if not config.get('include', True):
                logger.debug(f"Skipping section {section.id} (category {soap_category} not included)")
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
            'soap_category': section.soap_category or "OTHER",
            'content_html': '',
            'content': {} # Add the actual content for template rendering
        }
        
        # Format content based on its type
        content = section.content
        
        # Process content if available
        if content:
            # For patient report, simplify language
            if isinstance(content, dict):
                # Copy the content to the formatted section, but possibly simplify field names
                simplified_content = {}
                for key, value in content.items():
                    if isinstance(value, dict) and 'name' in value:
                        # Simplify the field name
                        value['name'] = self._simplify_medical_terms(value['name'])
                        # Simplify the value if it's a string
                        if isinstance(value.get('value'), str):
                            value['value'] = self._simplify_medical_terms(value['value'])
                    simplified_content[key] = value
                
                formatted_section['content'] = simplified_content
                
                # Format each part of the content
                content_html = self._format_dictionary_content(simplified_content, format_options, is_doctor=False)
                formatted_section['content_html'] = content_html
            elif isinstance(content, str):
                # Format text content and simplify medical terms
                simplified_text = self._simplify_medical_terms(content)
                formatted_section['content_html'] = simplified_text
                formatted_section['content'] = {"text": {"name": "Text", "data_type": "string", "value": simplified_text}}
            else:
                # Unknown content type
                logger.warning(f"Unknown content type for section {section.id}: {type(content)}")
                formatted_section['content_html'] = str(content)
                formatted_section['content'] = {"text": {"name": "Text", "data_type": "string", "value": str(content)}}
        
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
                logger.warning(f"User {user_id} not found")
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
        if not text:
            return ""
            
        # Simple implementation - replace known terms
        result = text
        try:
            for medical_term, simple_term in MEDICAL_JARGON_REPLACEMENTS.items():
                # Case insensitive replacement
                result = result.replace(medical_term, simple_term)
                result = result.replace(medical_term.capitalize(), simple_term.capitalize())
        except (AttributeError, TypeError):
            # Handle case where text is not a string
            logger.warning(f"Cannot simplify non-string content: {type(text)}")
            return str(text)
        
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
        try:
            # Debug the report data to ensure sections are properly formatted
            logger.info(f"Rendering report with data: note_title='{report_data.get('note_title')}'")
            logger.info(f"Number of sections in report data: {len(report_data.get('sections', []))}")
            
            # Debug section data structure more deeply
            for i, section in enumerate(report_data.get('sections', [])):
                logger.info(f"Section {i+1}: {section.get('title')} ({section.get('soap_category')})")
                logger.info(f"  Content keys: {list(section.get('content', {}).keys())}")
                
                # Debug the first few content fields
                for j, (key, value) in enumerate(section.get('content', {}).items()):
                    if j < 3:  # Only show first 3 fields to avoid log flooding
                        if isinstance(value, dict):
                            logger.info(f"    Field '{key}': name='{value.get('name')}', type='{value.get('data_type')}', value={value.get('value')}")
                        else:
                            logger.info(f"    Field '{key}': {type(value).__name__}={value}")
            
            # Check if template has custom HTML
            if template.html_template:
                # Use custom HTML template
                logger.info(f"Using custom HTML template from database ({len(template.html_template)} bytes)")
                try:
                    # Create Jinja2 environment for the template string
                    env = Environment(autoescape=True)
                    template_obj = env.from_string(template.html_template)
                    
                    # Debugging the template variables
                    template_source = template.html_template[:500] + "..." if len(template.html_template) > 500 else template.html_template
                    logger.info(f"Template source preview: {template_source}")
                    
                    # Add a debug filter to the environment
                    def debug_filter(value):
                        logger.info(f"DEBUG: {type(value).__name__} = {value}")
                        return value
                    env.filters['debug'] = debug_filter
                    
                    # Render the template with our report data
                    html = template_obj.render(**report_data)
                    return html
                except Exception as e:
                    logger.error(f"Error rendering custom template: {str(e)}")
                    logger.error(f"Falling back to default template")
            
            # Use default template based on report type
            template_file = "default_patient.html" if template.report_type == "patient" else "default_doctor.html"
            logger.info(f"Using default template file: {template_file}")
            return self._render_template(template_file, report_data)
            
        except Exception as e:
            logger.error(f"Error in _render_report: {str(e)}")
            # Fallback to generate a simple HTML report
            return self._generate_fallback_html(report_data)

    
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
            logger.info(f"Rendering template {template_file}")
            report_data['debug_mode'] = True

            # Add a debug filter to the environment
            def add_debug_filters(jinja_env):
                """Add debugging filters to a Jinja environment"""
                
                def debug_filter(value):
                    """Print the value and its type to the logs"""
                    print(f"DEBUG: {type(value).__name__} = {str(value)[:100]}")
                    return value
                
                def dump_filter(value):
                    """Dump detailed structure of complex objects"""
                    import json
                    try:
                        if isinstance(value, (list, dict)):
                            return json.dumps(value, indent=2, default=str)
                        else:
                            return str(value)
                    except:
                        return f"<{type(value).__name__} - not serializable>"
                
                jinja_env.filters['debug'] = debug_filter
                jinja_env.filters['dump'] = dump_filter
                return jinja_env

            # Add to your environment
            self.env = add_debug_filters(self.env)

            # Log key sections data
            logger.info(f"Rendering with {len(report_data.get('sections', []))} sections")
            for i, section in enumerate(report_data.get('sections', [])[:3]):  # First 3 sections
                soap_cat = section.get('soap_category', 'N/A')
                logger.info(f"Section {i+1}: {section.get('title')} - SOAP Category: {soap_cat}")
                
                # Check if sections will be sorted correctly
                if soap_cat == 'SUBJECTIVE':
                    logger.info(f"  Will be added to subjective_sections")
                elif soap_cat == 'OBJECTIVE':
                    logger.info(f"  Will be added to objective_sections")
                elif soap_cat == 'ASSESSMENT':
                    logger.info(f"  Will be added to assessment_sections")
                elif soap_cat == 'PLAN':
                    logger.info(f"  Will be added to plan_sections")
                else:
                    logger.info(f"  Will be added to other_sections")
            
            # Add a Jinja variable tracer
            def tracer(name, value):
                logger.info(f"TRACE: {name} = {type(value).__name__}")
                return value
                
            self.env.globals['trace'] = tracer
            
            # Dump all variables into debug.json for inspection
            import json
            try:
                debug_data = {
                    "note_title": report_data.get("note_title"),
                    "report_date": report_data.get("report_date"),
                    "encounter_date": report_data.get("encounter_date"),
                    "patient_info": report_data.get("patient_info"),
                    "provider_info": report_data.get("provider_info"),
                    "sections_count": len(report_data.get("sections", [])),
                    "sections_sample": [
                        {
                            "title": section.get("title"),
                            "soap_category": section.get("soap_category"),
                            "content_keys": list(section.get("content", {}).keys()),
                            "has_content_html": "content_html" in section and bool(section.get("content_html")),
                            "content": section.get("content")
                        }
                        for section in report_data.get("sections", [])[:3]  # First 3 sections
                    ]
                }
                
                with open(os.path.join(DEBUG_PATH, 'debug_template_data.json'), 'w') as f:
                    json.dump(debug_data, f, indent=2, default=str)
                    
                logger.info("Debug data written to debug_template_data.json")
            except Exception as debug_error:
                logger.warning(f"Could not write debug data: {str(debug_error)}")
            
            template = self.env.get_template(template_file)
            html = template.render(**report_data)
            logger.info(f"Successfully rendered template ({len(html)} bytes)")
            
            # Save the report data and HTML output for debugging
            try:
                with open(os.path.join(DEBUG_PATH, 'rendered_report_debug.html'), 'w') as f:
                    f.write(html)
                logger.info("Rendered template saved to rendered_report_debug.html")
            except Exception as save_error:
                logger.warning(f"Could not save rendered HTML: {str(save_error)}")
                
            return html
        except Exception as e:
            logger.error(f"Error rendering template {template_file}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        logger.warning("Generating fallback HTML report")
        
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
        
        # Group sections by SOAP category
        section_groups = {}
        for section in report_data.get('sections', []):
            category = section.get('soap_category', 'OTHER')
            if category not in section_groups:
                section_groups[category] = []
            section_groups[category].append(section)
        
        # Add sections grouped by SOAP category
        soap_order = ['SUBJECTIVE', 'OBJECTIVE', 'ASSESSMENT', 'PLAN', 'OTHER']
        
        for category in soap_order:
            if category in section_groups:
                html += f"""
                <div class="soap-category">
                    <h2>{category}</h2>
                """
                
                for section in section_groups[category]:
                    html += f"""
                    <div class="section">
                        <h3>{section.get('title', 'Untitled Section')}</h3>
                        <div class="content">
                            {section.get('content_html', '')}
                        </div>
                    </div>
                    """
                
                html += """
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
                if template:
                    logger.info(f"Using custom template: {template.name}")
                else:
                    logger.warning(f"Custom template {template_id} not found, falling back to default")
            
            # Use default template if no custom template specified or found
            if not template:
                template = self.get_default_template(report_type)
                if not template:
                    logger.error(f"No default {report_type} template found")
                    return None
                logger.info(f"Using default {report_type} template: {template.name}")
            
            logger.info(note_data)
            
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
            logger.info(f"Processing {len(sections_data)} sections from data")
            
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
            
            # Debug information to log what's being sent to the template
            logger.debug(f"Report data note_title: {report_data['note_title']}")
            logger.debug(f"Report data sections count: {len(report_data['sections'])}")
            for i, section in enumerate(report_data['sections']):
                logger.debug(f"Section {i+1} - {section.get('title')}: {section.get('soap_category')}")
                if 'content' in section:
                    logger.debug(f"  Content fields: {list(section['content'].keys())}")
                if 'content_html' in section:
                    logger.debug(f"  Content HTML length: {len(section['content_html'])}")
            
            # Render template
            rendered_report = self._render_report(template, report_data)
            
            if rendered_report:
                logger.info(f"Generated {report_type} report from direct data ({len(rendered_report)} bytes)")
            else:
                logger.error(f"Failed to render {report_type} report from direct data")
                
            return rendered_report
            
        except Exception as e:
            logger.error(f"Error generating report from data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _create_section_from_data(self, section_data: Dict[str, Any]) -> Section:
        """
        Create a Section object from dictionary data
        
        Args:
            section_data: Dictionary with section data
            
        Returns:
            Section object
        """
        try:
            # Extract required fields with defaults
            section_id = section_data.get("id", 0)
            note_id = section_data.get("note_id", 0)
            user_id = section_data.get("user_id", 1)
            title = section_data.get("title", "Untitled Section")
            template_id = section_data.get("template_id")
            soap_category = section_data.get("soap_category", "OTHER")
            
            # Handle content field - ensure it's a dictionary with proper field structure
            content = {}
            raw_content = section_data.get("content", {})
            
            # Log the raw content for debugging
            logger.debug(f"Raw content for section '{title}': {type(raw_content).__name__}")
            if isinstance(raw_content, dict):
                logger.debug(f"  Content keys: {list(raw_content.keys())}")
            
            # Format content properly if it's not already structured correctly
            if isinstance(raw_content, dict):
                # Check if content has the expected structure with fields
                for key, value in raw_content.items():
                    # If value is already a field object with name, data_type, value
                    if isinstance(value, dict) and "name" in value and "data_type" in value and "value" in value:
                        content[key] = value
                        logger.debug(f"  Field '{key}' already properly formatted")
                    
                    # If value is a list of field objects
                    elif isinstance(value, list):
                        # Check if all items have proper structure
                        formatted_list = []
                        for i, item in enumerate(value):
                            if isinstance(item, dict) and "name" in item and "data_type" in item and "value" in item:
                                # Already properly formatted
                                formatted_list.append(item)
                                logger.debug(f"  List item {i} in field '{key}' already properly formatted")
                            else:
                                # Create field structure for this item
                                field_name = f"{key.replace('_', ' ').title()} {i+1}"
                                data_type = "string"
                                
                                # Try to infer data type
                                if isinstance(item, bool):
                                    data_type = "boolean"
                                elif isinstance(item, (int, float)):
                                    data_type = "number"
                                elif isinstance(item, dict):
                                    data_type = "object"
                                
                                formatted_list.append({
                                    "name": field_name,
                                    "data_type": data_type,
                                    "value": item
                                })
                                logger.debug(f"  Formatted list item {i} in field '{key}' as {data_type}")
                        
                        # Add the formatted list to content
                        content[key] = formatted_list
                        logger.debug(f"  Formatted list field '{key}' with {len(formatted_list)} items")
                    
                    # If value is a simple type, convert to a field object
                    else:
                        field_name = key.replace('_', ' ').title()
                        data_type = "string"
                        
                        # Try to infer data type
                        if isinstance(value, bool):
                            data_type = "boolean"
                        elif isinstance(value, (int, float)):
                            data_type = "number"
                        elif isinstance(value, dict):
                            data_type = "object"
                            # For complex objects, consider a string representation
                            if len(value) > 10:  # If it's a large object
                                value = f"Complex object with {len(value)} fields"
                        elif isinstance(value, list):
                            data_type = "array"
                            # For simple lists, convert to comma-separated string
                            if all(isinstance(x, (str, int, float, bool)) for x in value):
                                value = ", ".join(str(x) for x in value)
                        
                        content[key] = {
                            "name": field_name,
                            "data_type": data_type,
                            "value": value
                        }
                        logger.debug(f"  Formatted field '{key}' as {data_type}")
            elif isinstance(raw_content, str):
                # If content is a string, create a text field
                content["text"] = {
                    "name": "Text",
                    "data_type": "string",
                    "value": raw_content
                }
                logger.debug("  Converted string content to text field")
            elif raw_content is None:
                # Empty content
                content = {}
                logger.debug("  Empty content")
            else:
                # Unknown content type, use string representation
                content["data"] = {
                    "name": "Data",
                    "data_type": "string",
                    "value": str(raw_content)
                }
                logger.debug(f"  Unknown content type: {type(raw_content).__name__}")
            
            # Create a temporary Section object (not saved to database)
            section = Section(
                id=section_id,
                note_id=note_id,
                user_id=user_id,
                title=title,
                template_id=template_id,
                soap_category=soap_category,
                content=content
            )
            
            logger.info(f"Created section: {title} ({soap_category}) with {len(content)} content fields")
            return section
            
        except Exception as e:
            logger.error(f"Error creating section from data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a minimal valid Section
            return Section(
                id=0,
                note_id=0,
                user_id=1,
                title="Error",
                soap_category="OTHER",
                content={"error": {"name": "Error", "data_type": "string", "value": str(e)}}
            )