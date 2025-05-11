import logging
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status

from app.models.models import (
    ReportTemplate, 
    ReportTemplateSectionConfig, 
    ReportTemplateFieldConfig,
    User
)
from app.schemas.report.report_template import (
    ReportTemplateCreate,
    ReportTemplateRead,
    ReportTemplateUpdate,
    ReportTemplateSectionConfigCreate
)

logger = logging.getLogger(__name__)

class ReportTemplateService:
    """
    Service for managing report templates.
    Handles creating, retrieving, updating, and deleting report templates.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize with a database session
        
        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session
    
    def get_all_templates(self, user_id: Optional[int] = None) -> List[ReportTemplate]:
        """
        Get all templates, optionally filtered by user ID
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            List of templates
        """
        try:
            query = self.db.query(ReportTemplate)
            
            if user_id is not None:
                query = query.filter(ReportTemplate.user_id == user_id)
                
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving templates: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while retrieving templates"
            )
    
    def get_template_by_id(self, template_id: int) -> Optional[ReportTemplate]:
        """
        Get a template by ID
        
        Args:
            template_id: Template ID
            
        Returns:
            Template or None if not found
        """
        try:
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id
            ).first()
            
            return template
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving template {template_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while retrieving template {template_id}"
            )
    
    def get_default_template(self, template_type: str) -> Optional[ReportTemplate]:
        """
        Get the default template for a given type
        
        Args:
            template_type: Template type (e.g., "doctor", "patient")
            
        Returns:
            Default template or None if not found
        """
        try:
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.template_type == template_type,
                ReportTemplate.is_default == True
            ).first()
            
            return template
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving default template for {template_type}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while retrieving default template"
            )
    
    def create_template(self, template_data: ReportTemplateCreate) -> ReportTemplate:
        """
        Create a new template
        
        Args:
            template_data: Template data
            
        Returns:
            Created template
        """
        try:
            # Check if user exists
            user = self.db.query(User).filter(User.id == template_data.user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {template_data.user_id} not found"
                )
            
            # If this is set as default, unset other defaults of the same type
            if template_data.is_default:
                self._unset_default_templates(template_data.template_type)
            
            # Create the template record
            template = ReportTemplate(
                name=template_data.name,
                description=template_data.description,
                user_id=template_data.user_id,
                template_type=template_data.template_type,
                html_template=template_data.html_template,
                is_default=template_data.is_default,
                layout_config=template_data.layout_config
            )
            
            self.db.add(template)
            self.db.flush()  # Get the template ID without committing
            
            # Process section configurations if provided
            if hasattr(template_data, 'section_configs') and template_data.section_configs:
                for section_config_data in template_data.section_configs:
                    self._create_section_config(template.id, section_config_data)
            
            self.db.commit()
            self.db.refresh(template)
            
            return template
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating template: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while creating template"
            )
    
    def update_template(self, template_id: int, template_data: ReportTemplateUpdate) -> ReportTemplate:
        """
        Update an existing template
        
        Args:
            template_id: Template ID
            template_data: Template update data
            
        Returns:
            Updated template
        """
        try:
            # Get the template
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id
            ).first()
            
            if not template:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Template with ID {template_id} not found"
                )
            
            # Verify user is allowed to update this template
            if template.user_id != template_data.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this template"
                )
            
            # If setting as default, unset other defaults
            if template_data.is_default and not template.is_default:
                self._unset_default_templates(template.template_type)
            
            # Update fields if provided
            if template_data.name is not None:
                template.name = template_data.name
            
            if template_data.description is not None:
                template.description = template_data.description
                
            if template_data.template_type is not None:
                template.template_type = template_data.template_type
                
            if template_data.is_default is not None:
                template.is_default = template_data.is_default
                
            if template_data.layout_config is not None:
                template.layout_config = template_data.layout_config
                
            if template_data.html_template is not None:
                template.html_template = template_data.html_template
                
            if template_data.version is not None:
                template.version = template_data.version
            
            self.db.commit()
            self.db.refresh(template)
            
            return template
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating template {template_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while updating template {template_id}"
            )
    
    def delete_template(self, template_id: int, user_id: int) -> bool:
        """
        Delete a template
        
        Args:
            template_id: Template ID
            user_id: User ID (for permission check)
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            # Get the template
            template = self.db.query(ReportTemplate).filter(
                ReportTemplate.id == template_id
            ).first()
            
            if not template:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Template with ID {template_id} not found"
                )
            
            # Verify user is allowed to delete this template
            if template.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to delete this template"
                )
            
            # Delete the template (cascade will delete section configs and field configs)
            self.db.delete(template)
            self.db.commit()
            
            return True
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting template {template_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while deleting template {template_id}"
            )
    
    def _unset_default_templates(self, template_type: str) -> None:
        """
        Unset default flag for all templates of a given type
        
        Args:
            template_type: Template type
        """
        try:
            self.db.query(ReportTemplate).filter(
                ReportTemplate.template_type == template_type,
                ReportTemplate.is_default == True
            ).update({ReportTemplate.is_default: False})
            
        except SQLAlchemyError as e:
            logger.error(f"Error unsetting default templates: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating default templates"
            )
    
    def _create_section_config(
        self, 
        template_id: int, 
        section_config_data: Union[Dict[str, Any], ReportTemplateSectionConfigCreate]
    ) -> ReportTemplateSectionConfig:
        """
        Create a section configuration for a template
        
        Args:
            template_id: Template ID
            section_config_data: Section configuration data
            
        Returns:
            Created section configuration
        """
        try:
            # Convert dict to Pydantic model if needed
            if isinstance(section_config_data, dict):
                section_config_data = ReportTemplateSectionConfigCreate(**section_config_data)
            
            # Create section config
            section_config = ReportTemplateSectionConfig(
                template_id=template_id,
                soap_category=section_config_data.soap_category,
                display_order=section_config_data.display_order,
                title=section_config_data.title,
                is_visible=section_config_data.is_visible,
                field_mappings=section_config_data.field_mappings
            )
            
            self.db.add(section_config)
            self.db.flush()  # Get ID without committing
            
            # Process field configurations if provided
            if hasattr(section_config_data, 'field_configs') and section_config_data.field_configs:
                for field_config_data in section_config_data.field_configs:
                    self._create_field_config(section_config.id, field_config_data)
            
            return section_config
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating section config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while creating section configuration"
            )
    
    def _create_field_config(
        self, 
        section_config_id: int, 
        field_config_data: Dict[str, Any]
    ) -> ReportTemplateFieldConfig:
        """
        Create a field configuration for a section
        
        Args:
            section_config_id: Section configuration ID
            field_config_data: Field configuration data
            
        Returns:
            Created field configuration
        """
        try:
            # Create field config
            field_config = ReportTemplateFieldConfig(
                section_config_id=section_config_id,
                field_id=field_config_data.get("field_id"),
                display_name=field_config_data.get("display_name"),
                field_type=field_config_data.get("field_type"),
                display_order=field_config_data.get("display_order", 0),
                is_visible=field_config_data.get("is_visible", True)
            )
            
            self.db.add(field_config)
            
            return field_config
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating field config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while creating field configuration"
            )
    
    def get_section_configs(self, template_id: int) -> List[ReportTemplateSectionConfig]:
        """
        Get all section configurations for a template
        
        Args:
            template_id: Template ID
            
        Returns:
            List of section configurations
        """
        try:
            section_configs = self.db.query(ReportTemplateSectionConfig).filter(
                ReportTemplateSectionConfig.template_id == template_id
            ).order_by(
                ReportTemplateSectionConfig.display_order
            ).all()
            
            return section_configs
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving section configs: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while retrieving section configurations"
            )
    
    def get_field_configs(self, section_config_id: int) -> List[ReportTemplateFieldConfig]:
        """
        Get all field configurations for a section
        
        Args:
            section_config_id: Section configuration ID
            
        Returns:
            List of field configurations
        """
        try:
            field_configs = self.db.query(ReportTemplateFieldConfig).filter(
                ReportTemplateFieldConfig.section_config_id == section_config_id
            ).order_by(
                ReportTemplateFieldConfig.display_order
            ).all()
            
            return field_configs
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving field configs: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while retrieving field configurations"
            )
    
    def create_default_templates(self) -> List[ReportTemplate]:
        """
        Create default templates if they don't exist
        
        Returns:
            List of created default templates
        """
        default_templates = []
        
        try:
            # Check if default doctor template exists
            doctor_template = self.get_default_template("doctor")
            if not doctor_template:
                # Create default doctor template
                doctor_template_data = self._get_default_doctor_template_data()
                doctor_template = self.create_template(doctor_template_data)
                default_templates.append(doctor_template)
            
            # Check if default patient template exists
            patient_template = self.get_default_template("patient")
            if not patient_template:
                # Create default patient template
                patient_template_data = self._get_default_patient_template_data()
                patient_template = self.create_template(patient_template_data)
                default_templates.append(patient_template)
            
            return default_templates
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating default templates: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while creating default templates"
            )
    
    def _get_default_doctor_template_data(self) -> ReportTemplateCreate:
        """
        Get data for default doctor template
        
        Returns:
            Template data
        """
        # This would be populated with appropriate section and field configurations
        # for the default doctor template
        return ReportTemplateCreate(
            name="Standard Doctor Report",
            description="Standard medical report with SOAP structure for healthcare professionals",
            user_id=1,  # System user
            template_type="doctor",
            is_default=True,
            layout_config={
                "page_format": "A4",
                "orientation": "portrait",
                "sections_order": ["header", "patient_info", "soap", "footer"],
                "show_header": True,
                "show_footer": True
            },
            section_configs=[
                # Subjective section config
                {
                    "soap_category": "SUBJECTIVE",
                    "display_order": 0,
                    "title": "Subjective",
                    "is_visible": True,
                },
                # Objective section config
                {
                    "soap_category": "OBJECTIVE",
                    "display_order": 1,
                    "title": "Objective",
                    "is_visible": True,
                },
                # Assessment section config
                {
                    "soap_category": "ASSESSMENT",
                    "display_order": 2,
                    "title": "Assessment",
                    "is_visible": True,
                },
                # Plan section config
                {
                    "soap_category": "PLAN",
                    "display_order": 3,
                    "title": "Plan",
                    "is_visible": True,
                }
            ]
        )
    
    def _get_default_patient_template_data(self) -> ReportTemplateCreate:
        """
        Get data for default patient template
        
        Returns:
            Template data
        """
        # This would be populated with appropriate section and field configurations
        # for the default patient template
        return ReportTemplateCreate(
            name="Patient-Friendly Report",
            description="Simplified medical report for patient understanding",
            user_id=1,  # System user
            template_type="patient",
            is_default=True,
            layout_config={
                "page_format": "A4",
                "orientation": "portrait",
                "sections_order": ["header", "patient_info", "summary", "recommendations", "footer"],
                "show_header": True,
                "show_footer": True
            },
            section_configs=[
                # Summary section config (combining subjective and objective)
                {
                    "soap_category": "SUBJECTIVE",
                    "display_order": 0,
                    "title": "Your Symptoms and Our Findings",
                    "is_visible": True,
                    "field_mappings": {
                        "patient_symptoms": "Your Symptoms"
                    },
                    "field_configs": [
                        {
                            "field_id": "summary_symptoms",
                            "display_name": "Your Symptoms",
                            "field_type": "text",
                            "display_order": 0,
                            "is_visible": True
                        },
                        {
                            "field_id": "summary_findings",
                            "display_name": "Our Findings",
                            "field_type": "text",
                            "display_order": 1,
                            "is_visible": True
                        }
                    ]
                },
                # Assessment section config (simplified for patients)
                {
                    "soap_category": "ASSESSMENT",
                    "display_order": 1,
                    "title": "What We Found",
                    "is_visible": True,
                },
                # Plan section config (renamed for patients)
                {
                    "soap_category": "PLAN",
                    "display_order": 2,
                    "title": "Your Treatment Plan",
                    "is_visible": True,

                }
            ]
        )