from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column, Session
from app.schemas.section import TextCategoryEnum, SectionCreate
from typing import Optional, List
import datetime
from sqlalchemy.sql import func
import datetime
import numpy as np
import pickle
import json
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    middle_name: Mapped[str] = mapped_column(String, nullable=True)
    last_name: Mapped[str] = mapped_column(String, nullable=False)

    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    # Use back_populates for bidirectional relationships.
    notes: Mapped[list["Note"]] = relationship("Note", back_populates="user", cascade="all, delete-orphan")
    note_templates: Mapped[list["NoteTemplate"]] = relationship("NoteTemplate", back_populates="user", cascade="all, delete-orphan")
    speaker_profiles: Mapped[list["SpeakerProfile"]] = relationship("SpeakerProfile", back_populates="user", cascade="all, delete-orphan")
    calibration_recordings: Mapped[list["CalibrationRecording"]]= relationship("CalibrationRecording", back_populates="user", cascade="all, delete-orphan")
    report_templates: Mapped[list["ReportTemplate"]] = relationship("ReportTemplate", back_populates="user", cascade="all, delete-orphan")
class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    encounter_date: Mapped[str] = mapped_column(String)  # Can also be Date if preferred
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="notes")
    sections: Mapped[list["Section"]] = relationship("Section", back_populates="note", cascade="all, delete-orphan")

class Section(Base):
    __tablename__ = "sections"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(Integer, ForeignKey("notes.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String, default="")
    content = Column(JSONB)
    section_type: Mapped[str] = mapped_column(String, default=TextCategoryEnum.OTHERS.name)
    section_description: Mapped[str] = mapped_column(String, default=TextCategoryEnum.OTHERS.value)
    note: Mapped["Note"] = relationship("Note", back_populates="sections")

class NoteTemplate(Base):
    __tablename__ = "note_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship("User", back_populates="note_templates")
    section_templates: Mapped[list["SectionTemplate"]] = relationship("SectionTemplate", back_populates="note_template", cascade="all, delete-orphan")

class SectionTemplate(Base):
    __tablename__ = "section_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    note_template_id: Mapped[int] = mapped_column(Integer, ForeignKey("note_templates.id"), nullable=False)
    section_type: Mapped[str] = mapped_column(String, default=TextCategoryEnum.OTHERS.name)
    section_description: Mapped[str] = mapped_column(String, default=TextCategoryEnum.OTHERS.value)
    metadata_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    content_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    note_template: Mapped["NoteTemplate"] = relationship("NoteTemplate", back_populates="section_templates")

def post_section(pydantic_section: SectionCreate, db: Session) -> Section:
    """
    Uploads a BaseSection (Pydantic model) to the database.
    The database will assign the ID automatically.
    
    Args:
        pydantic_section (BaseSection): The section instance to upload.
        db (Session): A SQLAlchemy session.
    
    Returns:
        Section: The newly created ORM Section object with its assigned ID.
    """
    # Convert the Pydantic model to a dictionary. Exclude the id since it is auto-generated.
    section_data = pydantic_section.model_dump(exclude={"id"})
    
    # Create an ORM Section object.
    orm_section = Section(**section_data)
    
    # Add and commit the record.
    db.add(orm_section)
    db.commit()
    db.refresh(orm_section)  # Refresh to load the auto-assigned id and any defaults.
    
    return orm_section

class SpeakerProfile(Base):
    """Model for storing speaker profiles used for voice calibration"""
    __tablename__ = "speaker_profiles"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Profile data - serialized as binary
    profile_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    
    # Metadata for quick reference
    feature_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    training_phrases_count: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Additional meta information
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stats: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="speaker_profiles")
    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship("CalibrationRecording", back_populates="speaker_profile")
    
    @classmethod
    def create_from_data(cls, user_id: int, profile_dict: dict) -> "SpeakerProfile":
        """Create a SpeakerProfile instance from profile dictionary"""
        # Pickle the profile data
        profile_data = pickle.dumps(profile_dict)
        
        # Extract metadata
        feature_dimension = profile_dict.get("feature_dimension", 0)
        training_phrases_count = profile_dict.get("training_phrases", 0)
        
        # Extract basic stats for quick reference
        stats = {}
        if "mean_vector" in profile_dict and isinstance(profile_dict["mean_vector"], np.ndarray):
            stats["mean_min"] = float(np.min(profile_dict["mean_vector"]))
            stats["mean_max"] = float(np.max(profile_dict["mean_vector"]))
            stats["mean_avg"] = float(np.mean(profile_dict["mean_vector"]))
        
        return cls(
            user_id=user_id,
            profile_data=profile_data,
            feature_dimension=feature_dimension,
            training_phrases_count=training_phrases_count,
            stats=stats
        )
    
    def get_profile_dict(self) -> dict:
        """Deserialize and return profile data"""
        try:
            return pickle.loads(self.profile_data)
        except Exception as e:
            # Return empty profile if deserialization fails
            return {"error": str(e)}


class CalibrationPhrase(Base):
    """Model for storing calibration phrases"""
    __tablename__ = "calibration_phrases"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    # Text properties for filtering
    difficulty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    medical_terms: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    
    # Relationships
    recordings: Mapped[list["CalibrationRecording"]] = relationship("CalibrationRecording", back_populates="phrase")


class CalibrationRecording(Base):
    """Model for storing calibration recordings"""
    __tablename__ = "calibration_recordings"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    phrase_id: Mapped[int] = mapped_column(Integer, ForeignKey("calibration_phrases.id"), nullable=False)
    speaker_profile_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("speaker_profiles.id"), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Features extracted from recording - serialized as binary
    features: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    
    # Optional metadata about the recording
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feature_type: Mapped[Optional[str]] = mapped_column(String, nullable=True, default="mfcc")
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="calibration_recordings")
    phrase: Mapped["CalibrationPhrase"] = relationship("CalibrationPhrase", back_populates="recordings")
    speaker_profile: Mapped[Optional["SpeakerProfile"]] = relationship("SpeakerProfile", back_populates="calibration_recordings")

class ReportTemplate(Base):
    __tablename__ = "report_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    report_type: Mapped[str] = mapped_column(String, nullable=False)  # "doctor", "patient", "custom"
    template_data: Mapped[dict] = mapped_column(JSONB, nullable=False)  # Stores section ordering, formatting rules
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="report_templates")