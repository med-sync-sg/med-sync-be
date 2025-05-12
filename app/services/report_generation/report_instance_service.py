import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from datetime import datetime

from app.models.models import (
    Note, 
    Section, 
    User,
    ReportTemplate, 
    ReportTemplateSectionConfig,
    ReportInstance,
    ReportSection,
    ReportField,
    SOAPCategory
)
from app.schemas.report.report_instance import (
    ReportInstanceCreate,
    ReportInstanceRead,
    ReportInstanceUpdate,
    ReportSectionCreate,
    ReportFieldCreate
)
from app.services.report_generation.report_template_service import ReportTemplateService

logger = logging.getLogger(__name__)

class ReportInstanceService:
    """
    Service for managing report instances.
    Handles creating, retrieving, updating, and managing reports generated from notes.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize with a database session
        
        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session
        self.template_service = ReportTemplateService(db_session)
    
    def get_report_instances(self, user_id: Optional[int] = None, note_id: Optional[int] = None) -> List[ReportInstance]:
        """
        Get report instances, optionally filtered by user ID or note ID
        
        Args:
            user_id: Optional user ID filter
            note_id: Optional note ID filter
            
        Returns:
            List of report instances
        """
        try:
            query = self.db.query(ReportInstance)
            
            if user_id is not None:
                query = query.filter(ReportInstance.user_id == user_id)
                
            if note_id is not None:
                query = query.filter(ReportInstance.note_id == note_id)
                
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving report instances: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while retrieving report instances"
            )
    
    def get_report_instance_by_id(self, report_id: int) -> Optional[ReportInstance]:
        """
        Get a report instance by ID
        
        Args:
            report_id: Report instance ID
            
        Returns:
            Report instance or None if not found
        """
        try:
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == report_id
            ).first()
            
            return report_instance
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving report instance {report_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while retrieving report instance {report_id}"
            )
    
    def create_report_instance_from_note(
        self, note_id: int, template_id: int, user_id: int, name: str, description: Optional[str] = None
    ) -> ReportInstance:
        """
        Create a new report instance from an existing note using a template
        
        Args:
            note_id: Note ID
            template_id: Template ID
            user_id: User ID
            name: Report name
            description: Optional report description
            
        Returns:
            Created report instance
        """
        try:
            # Verify note exists
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Note with ID {note_id} not found"
                )
            
            # Verify template exists
            template = self.db.query(ReportTemplate).filter(ReportTemplate.id == template_id).first()
            if not template:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Template with ID {template_id} not found"
                )
            
            # Verify user exists and has permission to access note
            if note.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to access this note"
                )
            
            # Create report instance
            report_instance = ReportInstance(
                name=name,
                description=description,
                user_id=user_id,
                note_id=note_id,
                template_id=template_id,
                is_finalized=False
            )
            
            self.db.add(report_instance)
            self.db.flush()  # Get ID without committing
            
            # Get template section configs
            section_configs = self.template_service.get_section_configs(template_id)
            
            # Get note sections
            note_sections = self.db.query(Section).filter(Section.note_id == note_id).all()
            
            # Transform note sections into report sections based on template
            self._transform_note_to_report_sections(
                report_instance.id, 
                note_sections, 
                section_configs
            )
            
            self.db.commit()
            self.db.refresh(report_instance)
            
            return report_instance
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating report instance: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while creating report instance"
            )
    
    def update_report_instance(self, report_id: int, update_data: ReportInstanceUpdate) -> ReportInstance:
        """
        Update a report instance
        
        Args:
            report_id: Report instance ID
            update_data: Update data
            
        Returns:
            Updated report instance
        """
        try:
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == report_id
            ).first()
            
            if not report_instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report instance with ID {report_id} not found"
                )
            
            # Verify user has permission
            if report_instance.user_id != update_data.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report instance"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized and not update_data.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update fields if provided
            if update_data.name is not None:
                report_instance.name = update_data.name
                
            if update_data.description is not None:
                report_instance.description = update_data.description
                
            if update_data.template_id is not None:
                # Verify template exists
                template = self.db.query(ReportTemplate).filter(
                    ReportTemplate.id == update_data.template_id
                ).first()
                
                if not template:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Template with ID {update_data.template_id} not found"
                    )
                
                report_instance.template_id = update_data.template_id
                
            if update_data.custom_layout is not None:
                report_instance.custom_layout = update_data.custom_layout
                
            if update_data.is_finalized is not None:
                report_instance.is_finalized = update_data.is_finalized
            
            self.db.commit()
            self.db.refresh(report_instance)
            
            return report_instance
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating report instance {report_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while updating report instance {report_id}"
            )
    
    def delete_report_instance(self, report_id: int, user_id: int) -> bool:
        """
        Delete a report instance
        
        Args:
            report_id: Report instance ID
            user_id: User ID (for permission check)
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == report_id
            ).first()
            
            if not report_instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report instance with ID {report_id} not found"
                )
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to delete this report instance"
                )
            
            # Delete report instance (cascade will delete sections and fields)
            self.db.delete(report_instance)
            self.db.commit()
            
            return True
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting report instance {report_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while deleting report instance {report_id}"
            )
    
    def _transform_note_to_report_sections(
        self, 
        report_instance_id: int, 
        note_sections: List[Section], 
        section_configs: List[ReportTemplateSectionConfig]
    ) -> List[ReportSection]:
        """
        Transform note sections into report sections based on template configuration
        
        Args:
            report_instance_id: Report instance ID
            note_sections: List of note sections
            section_configs: List of template section configurations
            
        Returns:
            List of created report sections
        """
        report_sections = []
        
        # Group note sections by SOAP category
        soap_sections: Dict[str, List[Section]] = {
            "SUBJECTIVE": [],
            "OBJECTIVE": [],
            "ASSESSMENT": [],
            "PLAN": [],
            "OTHER": []
        }
        
        for section in note_sections:
            category = section.soap_category
            if category in soap_sections:
                soap_sections[category].append(section)
            else:
                soap_sections["OTHER"].append(section)
        
        # Create report sections based on template configuration
        for config in section_configs:
            # Get note sections for this SOAP category
            category_sections = soap_sections.get(config.soap_category, [])
            
            # Skip if no sections and config is not visible
            if not category_sections and not config.is_visible:
                continue
            
            # Create report section
            report_section = ReportSection(
                report_instance_id=report_instance_id,
                soap_category=config.soap_category,
                title=config.title,
                display_order=config.display_order,
                is_visible=config.is_visible
            )
            
            self.db.add(report_section)
            self.db.flush()  # Get ID without committing
            
            # Process note sections for this category
            for note_section in category_sections:
                report_section.original_section_id = note_section.id
                
                # Transform section content to report fields
                self._create_report_fields_from_section(
                    report_section.id, 
                    note_section, 
                    config.field_mappings
                )
            
            report_sections.append(report_section)
        
        return report_sections
    
    def _create_report_fields_from_section(
        self, 
        report_section_id: int, 
        note_section: Section, 
        field_mappings: Dict[str, Any]
    ) -> List[ReportField]:
        """
        Create report fields from a note section's content
        
        Args:
            report_section_id: Report section ID
            note_section: Note section
            field_mappings: Field mappings from template
            
        Returns:
            List of created report fields
        """
        report_fields = []
        display_order = 0
        
        # Process content from note section
        if not note_section.content:
            return report_fields
            
        for field_id, field_data in note_section.content.items():
            # Field data can be a single field or a list of fields
            if isinstance(field_data, list):
                # Process list of fields
                for i, field_item in enumerate(field_data):
                    # Create report field
                    report_field = self._create_single_report_field(
                        report_section_id,
                        field_id,
                        field_item,
                        display_order,
                        field_mappings
                    )
                    
                    if report_field:
                        report_fields.append(report_field)
                        display_order += 1
            else:
                # Process single field
                report_field = self._create_single_report_field(
                    report_section_id,
                    field_id,
                    field_data,
                    display_order,
                    field_mappings
                )
                
                if report_field:
                    report_fields.append(report_field)
                    display_order += 1
        
        return report_fields
    
    def _create_single_report_field(
        self, 
        report_section_id: int,
        field_id: str,
        field_data: Dict[str, Any],
        display_order: int,
        field_mappings: Dict[str, Any]
    ) -> Optional[ReportField]:
        """
        Create a single report field from note field data
        
        Args:
            report_section_id: Report section ID
            field_id: Field ID
            field_data: Field data
            display_order: Display order
            field_mappings: Field mappings from template
            
        Returns:
            Created report field or None if invalid
        """
        try:
            # Get field properties
            display_name = field_data.get("name", "Untitled Field")
            
            # Override display name if mapping exists
            if field_id in field_mappings:
                display_name = field_mappings[field_id]
            
            field_type = field_data.get("data_type", "string")
            value = field_data.get("value")
            
            # Create report field
            report_field = ReportField(
                report_section_id=report_section_id,
                field_id=field_id,
                display_name=display_name,
                field_type=field_type,
                display_order=display_order,
                value=value,
                original_value=value,
                is_visible=True
            )
            
            self.db.add(report_field)
            
            return report_field
            
        except Exception as e:
            logger.warning(f"Error creating report field: {str(e)}")
            return None
    
    def get_report_sections(self, report_id: int) -> List[ReportSection]:
        """
        Get all sections for a report instance
        
        Args:
            report_id: Report instance ID
            
        Returns:
            List of report sections
        """
        try:
            sections = self.db.query(ReportSection).filter(
                ReportSection.report_instance_id == report_id
            ).order_by(
                ReportSection.display_order
            ).all()
            
            return sections
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving report sections: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while retrieving report sections"
            )
    
    def get_report_fields(self, section_id: int) -> List[ReportField]:
        """
        Get all fields for a report section
        
        Args:
            section_id: Report section ID
            
        Returns:
            List of report fields
        """
        try:
            fields = self.db.query(ReportField).filter(
                ReportField.report_section_id == section_id
            ).order_by(
                ReportField.display_order
            ).all()
            
            return fields
            
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving report fields: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while retrieving report fields"
            )
    
    def update_section_order(self, report_id: int, user_id: int, section_orders: List[Dict[str, Any]]) -> bool:
        """
        Update the order of sections in a report instance
        
        Args:
            report_id: Report instance ID
            user_id: User ID (for permission check)
            section_orders: List of section IDs and their new display orders
            
        Returns:
            True if updated, False otherwise
        """
        try:
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == report_id
            ).first()
            
            if not report_instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report instance with ID {report_id} not found"
                )
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report instance"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update section display orders
            for order_data in section_orders:
                section_id = order_data.get("id")
                new_order = order_data.get("display_order")
                
                if section_id is None or new_order is None:
                    continue
                
                # Get section
                section = self.db.query(ReportSection).filter(
                    ReportSection.id == section_id,
                    ReportSection.report_instance_id == report_id
                ).first()
                
                if section:
                    section.display_order = new_order
            
            self.db.commit()
            
            return True
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating section order: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating section order"
            )
    
    def update_field_order(self, section_id: int, user_id: int, field_orders: List[Dict[str, Any]]) -> bool:
        """
        Update the order of fields in a report section
        
        Args:
            section_id: Report section ID
            user_id: User ID (for permission check)
            field_orders: List of field IDs and their new display orders
            
        Returns:
            True if updated, False otherwise
        """
        try:
            # Get report section
            section = self.db.query(ReportSection).filter(
                ReportSection.id == section_id
            ).first()
            
            if not section:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report section with ID {section_id} not found"
                )
            
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == section.report_instance_id
            ).first()
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report section"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update field display orders
            for order_data in field_orders:
                field_id = order_data.get("id")
                new_order = order_data.get("display_order")
                
                if field_id is None or new_order is None:
                    continue
                
                # Get field
                field = self.db.query(ReportField).filter(
                    ReportField.id == field_id,
                    ReportField.report_section_id == section_id
                ).first()
                
                if field:
                    field.display_order = new_order
            
            self.db.commit()
            
            return True
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating field order: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating field order"
            )
    
    def update_field_value(self, field_id: int, user_id: int, new_value: Any) -> ReportField:
        """
        Update the value of a report field
        
        Args:
            field_id: Report field ID
            user_id: User ID (for permission check)
            new_value: New field value
            
        Returns:
            Updated report field
        """
        try:
            # Get report field
            field = self.db.query(ReportField).filter(
                ReportField.id == field_id
            ).first()
            
            if not field:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report field with ID {field_id} not found"
                )
            
            # Get report section
            section = self.db.query(ReportSection).filter(
                ReportSection.id == field.report_section_id
            ).first()
            
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == section.report_instance_id
            ).first()
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report field"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update field value
            field.value = new_value
            
            self.db.commit()
            self.db.refresh(field)
            
            return field
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating field value: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating field value"
            )
    
    def finalize_report(self, report_id: int, user_id: int) -> ReportInstance:
        """
        Finalize a report instance (mark as complete and ready for PDF generation)
        
        Args:
            report_id: Report instance ID
            user_id: User ID (for permission check)
            
        Returns:
            Finalized report instance
        """
        try:
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == report_id
            ).first()
            
            if not report_instance:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report instance with ID {report_id} not found"
                )
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to finalize this report instance"
                )
            
            # Set report as finalized
            report_instance.is_finalized = True
            
            self.db.commit()
            self.db.refresh(report_instance)
            
            return report_instance
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error finalizing report instance {report_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred while finalizing report instance {report_id}"
            )
    
    def toggle_section_visibility(self, section_id: int, user_id: int, is_visible: bool) -> ReportSection:
        """
        Toggle visibility of a report section
        
        Args:
            section_id: Report section ID
            user_id: User ID (for permission check)
            is_visible: New visibility state
            
        Returns:
            Updated report section
        """
        try:
            # Get report section
            section = self.db.query(ReportSection).filter(
                ReportSection.id == section_id
            ).first()
            
            if not section:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report section with ID {section_id} not found"
                )
            
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == section.report_instance_id
            ).first()
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report section"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update section visibility
            section.is_visible = is_visible
            
            self.db.commit()
            self.db.refresh(section)
            
            return section
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating section visibility: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating section visibility"
            )
    
    def toggle_field_visibility(self, field_id: int, user_id: int, is_visible: bool) -> ReportField:
        """
        Toggle visibility of a report field
        
        Args:
            field_id: Report field ID
            user_id: User ID (for permission check)
            is_visible: New visibility state
            
        Returns:
            Updated report field
        """
        try:
            # Get report field
            field = self.db.query(ReportField).filter(
                ReportField.id == field_id
            ).first()
            
            if not field:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report field with ID {field_id} not found"
                )
            
            # Get report section
            section = self.db.query(ReportSection).filter(
                ReportSection.id == field.report_section_id
            ).first()
            
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == section.report_instance_id
            ).first()
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report field"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update field visibility
            field.is_visible = is_visible
            
            self.db.commit()
            self.db.refresh(field)
            
            return field
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating field visibility: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating field visibility"
            )
    
    def update_section_title(self, section_id: int, user_id: int, new_title: str) -> ReportSection:
        """
        Update the title of a report section
        
        Args:
            section_id: Report section ID
            user_id: User ID (for permission check)
            new_title: New section title
            
        Returns:
            Updated report section
        """
        try:
            # Get report section
            section = self.db.query(ReportSection).filter(
                ReportSection.id == section_id
            ).first()
            
            if not section:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report section with ID {section_id} not found"
                )
            
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == section.report_instance_id
            ).first()
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report section"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update section title
            section.title = new_title
            
            self.db.commit()
            self.db.refresh(section)
            
            return section
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating section title: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating section title"
            )
    
    def update_field_name(self, field_id: int, user_id: int, new_name: str) -> ReportField:
        """
        Update the display name of a report field
        
        Args:
            field_id: Report field ID
            user_id: User ID (for permission check)
            new_name: New field display name
            
        Returns:
            Updated report field
        """
        try:
            # Get report field
            field = self.db.query(ReportField).filter(
                ReportField.id == field_id
            ).first()
            
            if not field:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Report field with ID {field_id} not found"
                )
            
            # Get report section
            section = self.db.query(ReportSection).filter(
                ReportSection.id == field.report_section_id
            ).first()
            
            # Get report instance
            report_instance = self.db.query(ReportInstance).filter(
                ReportInstance.id == section.report_instance_id
            ).first()
            
            # Verify user has permission
            if report_instance.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to update this report field"
                )
            
            # Prevent updating finalized reports
            if report_instance.is_finalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot modify a finalized report"
                )
            
            # Update field display name
            field.display_name = new_name
            
            self.db.commit()
            self.db.refresh(field)
            
            return field
            
        except HTTPException:
            self.db.rollback()
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating field name: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred while updating field name"
            )