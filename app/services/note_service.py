import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from app.models.models import Note, Section, User
from app.schemas.note import NoteCreate, NoteUpdate
from app.schemas.section import SectionCreate, SectionUpdate

logger = logging.getLogger(__name__)

class NoteService:
    """
    Service for note and section management
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize with a database session
        
        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session
    
    def get_notes_by_user(self, user_id: int) -> List[Note]:
        """
        Get all notes for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of notes
        """
        try:
            notes = self.db.query(Note).filter(Note.user_id == user_id).all()
            return notes
        except SQLAlchemyError as e:
            logger.error(f"Error getting notes for user {user_id}: {str(e)}")
            return []
    
    def get_note_by_id(self, note_id: int) -> Optional[Note]:
        """
        Get a note by ID
        
        Args:
            note_id: Note ID
            
        Returns:
            Note or None if not found
        """
        try:
            note = self.db.query(Note).filter(Note.id == note_id).first()
            return note
        except SQLAlchemyError as e:
            logger.error(f"Error getting note {note_id}: {str(e)}")
            return None
    
    def create_note(self, note_data: NoteCreate) -> Optional[Note]:
        """
        Create a new note
        
        Args:
            note_data: Note data
            
        Returns:
            Created note or None if failed
        """
        try:
            # Create note
            note = Note(
                user_id=note_data.user_id,
                patient_id=note_data.patient_id,
                title=note_data.title,
                encounter_date=note_data.encounter_date
            )
            
            self.db.add(note)
            self.db.flush()  # Get ID without committing
            
            # Create sections if any
            if note_data.sections:
                for section_data in note_data.sections:
                    self.add_section_to_note(note.id, section_data)
            
            self.db.commit()
            self.db.refresh(note)
            return note
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating note: {str(e)}")
            return None
    
    def update_note(self, note_id: int, note_data: NoteUpdate) -> Optional[Note]:
        """
        Update a note
        
        Args:
            note_id: Note ID
            note_data: Note update data
            
        Returns:
            Updated note or None if failed
        """
        try:
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note {note_id} not found for update")
                return None
            
            # Update fields if provided
            if note_data.title is not None:
                note.title = note_data.title
            if note_data.patient_id is not None:
                note.patient_id = note_data.patient_id
            if note_data.encounter_date is not None:
                note.encounter_date = note_data.encounter_date
            
            self.db.commit()
            self.db.refresh(note)
            return note
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating note {note_id}: {str(e)}")
            return None
    
    def delete_note(self, note_id: int) -> bool:
        """
        Delete a note
        
        Args:
            note_id: Note ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note {note_id} not found for deletion")
                return False
            
            self.db.delete(note)
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting note {note_id}: {str(e)}")
            return False
    
    def add_section_to_note(self, note_id: int, section_data: SectionCreate) -> Optional[Section]:
        """
        Add a section to a note
        
        Args:
            note_id: Note ID
            section_data: Section data
            
        Returns:
            Created section or None if failed
        """
        try:
            # First check if note exists
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note {note_id} not found for adding section")
                return None
            
            # Prepare section data
            section_dict = section_data.model_dump(exclude={"id"})
            section_dict["note_id"] = note_id
            # Create section
            section = Section(**section_dict)

            self.db.add(section)
            self.db.commit()
            self.db.refresh(section)
            return section
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error adding section to note {note_id}: {str(e)}")
            return None
    
    def update_section(self, section_id: int, section_data: SectionUpdate) -> Optional[Section]:
        """
        Update a section
        
        Args:
            section_id: Section ID
            section_data: Section update data
            
        Returns:
            Updated section or None if failed
        """
        try:
            section = self.db.query(Section).filter(Section.id == section_id).first()
            if not section:
                logger.warning(f"Section {section_id} not found for update")
                return None
            
            # Update fields if provided
            if section_data.title is not None:
                section.title = section_data.title
            if section_data.template_id is not None:
                section.template_id = section_data.template_id
            if section_data.soap_category is not None:
                section.soap_category = section_data.soap_category
            if section_data.content is not None:
                section.content = section_data.content
            if section_data.is_visible_to_patient is not None:
                section.is_visible_to_patient = section_data.is_visible_to_patient
            if section_data.display_order is not None:
                section.display_order = section_data.display_order
            
            section.updated_at = datetime.now()
            
            self.db.commit()
            self.db.refresh(section)
            return section
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating section {section_id}: {str(e)}")
            return None
    
    def delete_section(self, section_id: int) -> bool:
        """
        Delete a section
        
        Args:
            section_id: Section ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            section = self.db.query(Section).filter(Section.id == section_id).first()
            if not section:
                logger.warning(f"Section {section_id} not found for deletion")
                return False
            
            self.db.delete(section)
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting section {section_id}: {str(e)}")
            return False
    