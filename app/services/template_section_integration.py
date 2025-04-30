import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.models import Section
from app.services.template_service import TemplateService
from app.schemas.section import FieldValueUpdate
from app.db.neo4j_session import neo4j_session

logger = logging.getLogger(__name__)

class TemplateSectionIntegration:
    """
    Service for integrating Neo4j templates with PostgreSQL sections
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize with a database session
        
        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session
        self.template_service = TemplateService()
    
    def get_template_with_field_values(self, section_id: int) -> Dict[str, Any]:
        """
        Get combined view of template with field values from a section
        
        Args:
            section_id: Section ID
            
        Returns:
            Dictionary with template data and field values
        """
        try:
            # Get the section from PostgreSQL
            section = self.db.query(Section).filter(Section.id == section_id).first()
            
            if not section or not section.template_id:
                logger.warning(f"Section {section_id} not found or has no template_id")
                return {
                    "section_id": section_id,
                    "template": None,
                    "field_values": {},
                    "error": "Section not found or has no template"
                }
            
            # Get the template from Neo4j
            template = self.template_service.get_template_with_fields(section.template_id)
            
            if not template:
                logger.warning(f"Template {section.template_id} not found in Neo4j")
                return {
                    "section_id": section_id,
                    "template_id": section.template_id,
                    "template": None,
                    "field_values": section.field_values or {},
                    "error": "Template not found in Neo4j"
                }
            
            # Get field values from section
            field_values = section.get_all_field_values()
            
            # Build response with template and values
            result = {
                "section_id": section_id,
                "template": template,
                "field_values": field_values,
                "content": section.content
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting template with field values for section {section_id}: {str(e)}")
            return {
                "section_id": section_id,
                "template": None,
                "field_values": {},
                "error": str(e)
            }
    
    def apply_template_to_section(self, section_id: int, template_id: str) -> bool:
        """
        Apply a template to an existing section
        
        Args:
            section_id: Section ID
            template_id: Template ID from Neo4j
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the section from PostgreSQL
            section = self.db.query(Section).filter(Section.id == section_id).first()
            
            if not section:
                logger.warning(f"Section {section_id} not found")
                return False
            
            # Get the template from Neo4j
            template = self.template_service.get_template_with_fields(template_id)
            
            if not template:
                logger.warning(f"Template {template_id} not found in Neo4j")
                return False
            
            # Update section with template ID
            section.template_id = template_id
            
            # Update SOAP category if available in template
            soap_category = self._get_soap_category_from_template(template)
            if soap_category:
                section.soap_category = soap_category
            
            # Initialize empty field values if needed
            if not section.field_values:
                section.field_values = {}
            
            # Initialize fields with empty values if they don't exist
            for field in template.get("fields", []):
                field_id = field.get("id")
                field_name = field.get("field_name", field.get("name", ""))
                
                # Only add if field doesn't exist yet
                if field_id and field_id not in section.field_values:
                    section.add_field_value(
                        field_id=field_id,
                        field_name=field_name,
                        value=""  # Initialize with empty value
                    )
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error applying template {template_id} to section {section_id}: {str(e)}")
            return False
    
    def create_section_from_template(self, note_id: int, user_id: int, 
                                   template_id: str, title: Optional[str] = None) -> Optional[int]:
        """
        Create a new section based on a template
        
        Args:
            note_id: Note ID
            user_id: User ID
            template_id: Template ID from Neo4j
            title: Optional title (uses template name if not provided)
            
        Returns:
            ID of created section or None if failed
        """
        try:
            # Get the template from Neo4j
            template = self.template_service.get_template_with_fields(template_id)
            
            if not template:
                logger.warning(f"Template {template_id} not found in Neo4j")
                return None
            
            # Use template name as title if not provided
            if not title:
                title = template.get("name", "Untitled Section")
            
            # Create section
            section = Section(
                note_id=note_id,
                user_id=user_id,
                title=title,
                template_id=template_id,
                soap_category=self._get_soap_category_from_template(template),
                field_values={},
                content={}
            )
            
            # Initialize fields with empty values
            for field in template.get("fields", []):
                field_id = field.get("id")
                field_name = field.get("field_name", field.get("name", ""))
                
                if field_id:
                    section.add_field_value(
                        field_id=field_id,
                        field_name=field_name,
                        value=""  # Initialize with empty value
                    )
            
            self.db.add(section)
            self.db.commit()
            self.db.refresh(section)
            
            return section.id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating section from template {template_id}: {str(e)}")
            return None
    
    def update_content_from_field_values(self, section_id: int) -> bool:
        """
        Update the content field based on field values
        
        Args:
            section_id: Section ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the section from PostgreSQL
            section = self.db.query(Section).filter(Section.id == section_id).first()
            
            if not section or not section.template_id:
                logger.warning(f"Section {section_id} not found or has no template_id")
                return False
            
            # Get the template from Neo4j
            template = self.template_service.get_template_with_fields(section.template_id)
            
            if not template:
                logger.warning(f"Template {section.template_id} not found in Neo4j")
                return False
            
            # Create content structure from fields
            content = self._build_content_from_fields(section, template)
            
            # Update section content
            section.content = content
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating content from field values for section {section_id}: {str(e)}")
            return False
    
    def _build_content_from_fields(self, section: Section, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a structured content dictionary from field values
        
        Args:
            section: Section with field values
            template: Template with fields structure
            
        Returns:
            Structured content dictionary
        """
        content = {}
        field_values = section.get_all_field_values()
        
        # Group fields by name pattern
        field_groups = {}
        
        for field in template.get("fields", []):
            field_id = field.get("id")
            field_name = field.get("field_name", field.get("name", ""))
            
            if not field_id or field_id not in field_values:
                continue
                
            # Get value
            value = field_values.get(field_id, {}).get("value", "")
            
            # Check if field name has a group pattern like "Main Symptom name"
            if " " in field_name:
                parts = field_name.split(" ", 1)
                group_name = parts[0]
                sub_field = parts[1]
                
                # Create group if it doesn't exist
                if group_name not in field_groups:
                    field_groups[group_name] = {}
                    
                # Add field to group
                field_groups[group_name][sub_field] = value
            else:
                # Add directly to content
                content[field_name] = value
        
        # Add groups to content
        for group_name, group_fields in field_groups.items():
            content[group_name] = group_fields
            
        return content
    
    def _get_soap_category_from_template(self, template: Dict[str, Any]) -> str:
        """
        Determine SOAP category from template
        
        Args:
            template: Template dictionary
            
        Returns:
            SOAP category
        """
        # Default to OTHER
        soap_category = "OTHER"
        
        # Check template name for SOAP indicators
        template_name = template.get("name", "").upper()
        
        if any(term in template_name for term in ["SUBJECTIVE", "HISTORY", "COMPLAINT"]):
            soap_category = "SUBJECTIVE"
        elif any(term in template_name for term in ["OBJECTIVE", "EXAM", "FINDING"]):
            soap_category = "OBJECTIVE"
        elif any(term in template_name for term in ["ASSESSMENT", "DIAGNOSIS", "IMPRESSION"]):
            soap_category = "ASSESSMENT"
        elif any(term in template_name for term in ["PLAN", "TREATMENT", "RECOMMENDATION"]):
            soap_category = "PLAN"
            
        return soap_category