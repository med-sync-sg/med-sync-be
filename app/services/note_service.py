import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError

from app.models.models import Note, Section, User, SectionType
from app.schemas.note import NoteCreate, NoteRead, NoteUpdate
from app.schemas.section import SectionCreate
from app.services.report_generation.section_management_service import SectionManagementService

# Configure logger
logger = logging.getLogger(__name__)

class NoteService:
    """
    Service for managing medical notes and their sections.
    Handles CRUD operations and business logic for notes.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize the note service with a database session
        
        Args:
            db_session: SQLAlchemy session for database operations
        """
        self.db = db_session
        
    def create_note(self, note_data: NoteCreate) -> Optional[Note]:
        """
        Create a new note with sections
        
        Args:
            note_data: Data for creating the note
            
        Returns:
            Created Note object or None if creation failed
        """
        try:
            # Validate user exists
            user = self.db.query(User).filter(User.id == note_data.user_id).first()
            if not user:
                logger.error(f"User {note_data.user_id} not found when creating note")
                return None
            
            # Create note
            db_note = Note(
                title=note_data.title,
                patient_id=note_data.patient_id,
                user_id=note_data.user_id,
                encounter_date=note_data.encounter_date
            )
            
            self.db.add(db_note)
            self.db.flush()  # Get ID without committing
            
            # Create sections
            if note_data.sections:
                for section_data in note_data.sections:
                    db_section = Section(
                        **section_data
                    )
                    self.db.add(db_section)
            
            # Commit transaction
            self.db.commit()
            self.db.refresh(db_note)
            
            logger.info(f"Note created with ID {db_note.id} for user {note_data.user_id}")
            return db_note
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error creating note: {str(e)}")
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating note: {str(e)}")
            return None
    
    def get_note_by_id(self, note_id: int) -> Optional[Note]:
        """
        Get a note by ID with its sections
        
        Args:
            note_id: ID of the note to retrieve
            
        Returns:
            Note object with sections or None if not found
        """
        try:
            note = self.db.query(Note).options(
                joinedload(Note.sections)
            ).filter(Note.id == note_id).first()
            
            if not note:
                logger.warning(f"Note with ID {note_id} not found")
                
            return note
            
        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {str(e)}")
            return None
    
    def get_notes_by_user(self, user_id: int) -> List[Note]:
        """
        Get all notes for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of Note objects
        """
        try:
            notes = self.db.query(Note).filter(
                Note.user_id == user_id
            ).order_by(Note.encounter_date.desc()).all()
            
            logger.info(f"Retrieved {len(notes)} notes for user {user_id}")
            return notes
            
        except Exception as e:
            logger.error(f"Error retrieving notes for user {user_id}: {str(e)}")
            return []
    
    def get_notes_by_patient(self, patient_id: int) -> List[Note]:
        """
        Get all notes for a patient
        
        Args:
            patient_id: ID of the patient
            
        Returns:
            List of Note objects
        """
        try:
            notes = self.db.query(Note).filter(
                Note.patient_id == patient_id
            ).order_by(Note.encounter_date.desc()).all()
            
            logger.info(f"Retrieved {len(notes)} notes for patient {patient_id}")
            return notes
            
        except Exception as e:
            logger.error(f"Error retrieving notes for patient {patient_id}: {str(e)}")
            return []
    
    def update_note(self, note_id: int, note_data: NoteUpdate) -> Optional[Note]:
        """
        Update an existing note
        
        Args:
            note_id: ID of the note to update
            note_data: Updated data for the note
            
        Returns:
            Updated Note object or None if update failed
        """
        try:
            # Get existing note
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note with ID {note_id} not found for update")
                return None
            
            # Update note fields if provided
            if note_data.title is not None:
                note.title = note_data.title
            if note_data.patient_id is not None:
                note.patient_id = note_data.patient_id
            if note_data.encounter_date is not None:
                note.encounter_date = note_data.encounter_date
                
            # Update sections if provided
            if note_data.sections:
                # Handle section updates
                self._update_note_sections(note, note_data.sections)
            
            self.db.commit()
            self.db.refresh(note)
            
            logger.info(f"Note {note_id} updated successfully")
            return note
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error updating note {note_id}: {str(e)}")
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating note {note_id}: {str(e)}")
            return None
    
    def _update_note_sections(self, note: Note, sections_data: List[Any]) -> None:
        """
        Update sections for a note (helper method)
        
        Args:
            note: Note object to update
            sections_data: List of section update data
        """
        # Implementation depends on how you want to handle section updates
        # This is a simplified approach that just adds new sections
        for section_data in sections_data:
            # Check if it's an update to existing section or a new one
            if hasattr(section_data, 'id') and section_data.id:
                # Update existing section
                existing_section = next(
                    (s for s in note.sections if s.id == section_data.id), 
                    None
                )
                if existing_section:
                    if section_data.title:
                        existing_section.title = section_data.title
                    if section_data.content:
                        existing_section.content = section_data.content
                    if section_data.section_type:
                        existing_section.section_type = section_data.section_type
            else:
                # Create new section
                new_section = Section(
                    **section_data
                )
                self.db.add(new_section)
    
    def delete_note(self, note_id: int) -> bool:
        """
        Delete a note and its sections
        
        Args:
            note_id: ID of the note to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Get note
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note with ID {note_id} not found for deletion")
                return False
            
            # Delete note (sections will be deleted by cascade)
            self.db.delete(note)
            self.db.commit()
            
            logger.info(f"Note {note_id} deleted successfully")
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error deleting note {note_id}: {str(e)}")
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting note {note_id}: {str(e)}")
            return False
    
    def add_section_to_note(self, note_id: int, section_data: SectionCreate) -> Optional[Section]:
        """
        Add a new section to an existing note
        
        Args:
            note_id: ID of the note
            section_data: Data for the new section
            
        Returns:
            Created Section object or None if creation failed
        """
        try:
            # Verify note exists
            note = self.db.query(Note).filter(Note.id == note_id).first()
            if not note:
                logger.warning(f"Note with ID {note_id} not found when adding section")
                return None
            
            # Verify section type exists
            section_type = self.db.query(SectionType).filter(SectionType.id == section_data.section_type_id).first()
            if not section_type:
                section_type_service = SectionManagementService(self.db)
                section_type = section_type_service.get_default_section_type()
                section_data.section_type_id = section_type.id
            
            # Create section
            section = Section(
                note_id=note_id,
                user_id=section_data.user_id,
                title=section_data.title,
                content=section_data.content,
                section_type_id=section_data.section_type_id,
                section_type_code=section_type.code
            )
            
            self.db.add(section)
            self.db.commit()
            self.db.refresh(section)
            
            logger.info(f"Section added to note {note_id} with ID {section.id}")
            return section
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error adding section to note {note_id}: {str(e)}")
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding section to note {note_id}: {str(e)}")
            return None
    
    def get_sections_by_note(self, note_id: int) -> List[Section]:
        """
        Get all sections for a note
        
        Args:
            note_id: ID of the note
            
        Returns:
            List of Section objects
        """
        try:
            sections = self.db.query(Section).filter(
                Section.note_id == note_id
            ).all()
            
            logger.info(f"Retrieved {len(sections)} sections for note {note_id}")
            return sections
            
        except Exception as e:
            logger.error(f"Error retrieving sections for note {note_id}: {str(e)}")
            return []