from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column, Session
from app.schemas.section import SectionCreate
from typing import Optional, List
import datetime
from sqlalchemy.sql import func
import datetime
import numpy as np
import pickle
import json
from enum import Enum
Base = declarative_base()

class SOAPCategory(str, Enum):
    """Enum representing SOAP categories"""
    OTHER = "OTHER"  # For administrative or other non-SOAP sections
    SUBJECTIVE = "SUBJECTIVE"
    OBJECTIVE = "OBJECTIVE"
    ASSESSMENT = "ASSESSMENT"
    PLAN = "PLAN"
    MIXED = "MIXED"  # For sections that span multiple categories
    
class SectionType(Base):
    """Model for defining types of sections in medical documentation"""
    __tablename__ = "section_types"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)  # Unique code for this section type
    name: Mapped[str] = mapped_column(String, nullable=False)  # Display name
    description: Mapped[str] = mapped_column(String, nullable=True)  # Detailed description
    
    # SOAP categorization
    soap_category: Mapped[str] = mapped_column(String, nullable=False, default=SOAPCategory.OTHER)
    
    # Hierarchical structure
    parent_id: Mapped[int] = mapped_column(Integer, ForeignKey("section_types.id"), nullable=True)
    parent: Mapped["SectionType"] = relationship("SectionType", remote_side=[id], backref="children")
    
    # Content schema - defines expected structure for this section type
    content_schema = Column(JSONB, nullable=True)  # JSON Schema for content validation
    
    # Display and behavior properties
    is_required: Mapped[bool] = mapped_column(Boolean, default=False)  # Whether this section is required in notes
    default_title: Mapped[str] = mapped_column(String, nullable=True)  # Default title when creating new section
    default_order: Mapped[int] = mapped_column(Integer, default=100)  # Default ordering within parent
    is_visible_to_patient: Mapped[bool] = mapped_column(Boolean, default=True)  # Whether visible in patient-facing reports
    
    # Template reference - default template for this section type
    default_template_id: Mapped[int] = mapped_column(Integer, ForeignKey("section_templates.id"), nullable=True)
    default_template = relationship("SectionTemplate")
    
    # Sections of this type
    sections = relationship("Section", back_populates="section_type")
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    @property
    def full_path(self):
        """Get full hierarchical path of section type"""
        if self.parent:
            return f"{self.parent.full_path}/{self.code}"
        return self.code
    
    def __repr__(self):
        return f"<SectionType {self.code}: {self.name}>"

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
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(Integer, ForeignKey("notes.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Basic section attributes
    title: Mapped[str] = mapped_column(String, default="")
    content = Column(JSONB)  # Structured content
    
    # Section type reference
    section_type_id = Column(Integer, ForeignKey("section_types.id"), nullable=False)
    section_type = relationship("SectionType", back_populates="sections")
    
    # Metadata
    is_visible_to_patient = Column(Boolean, default=True)  # Override visibility setting
    display_order = Column(Integer, default=100)  # Order within note
    
    # Timestamps and tracking
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_modified_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    last_modified_by = relationship("User", foreign_keys=[last_modified_by_id])
    
    # Relationships
    note: Mapped["Note"] = relationship("Note", back_populates="sections")
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    
    # Section relationships (self-referential for hierarchical sections)
    parent_id: Mapped[int] = mapped_column(Integer, ForeignKey("sections.id"), nullable=True)
    parent: Mapped["Section"] = relationship("Section", remote_side=[id], backref="subsections")
    
    # Related sections (for cross-references)
    related_sections = relationship(
        "Section",
        secondary="section_relationships",
        primaryjoin="Section.id==SectionRelationship.source_section_id",
        secondaryjoin="Section.id==SectionRelationship.target_section_id",
        backref="referenced_by"
    )
    
    def __init__(self, **kwargs):
        super(Section, self).__init__(**kwargs)
        # Auto-populate legacy fields from section_type for backward compatibility
        if hasattr(self, 'section_type') and self.section_type:
            self.section_type_code = self.section_type.code
            self.section_description = self.section_type.description
    
    @property
    def soap_category(self):
        """Get the SOAP category for this section"""
        if self.section_type:
            return self.section_type.soap_category
        return SOAPCategory.OTHER
    
    def __repr__(self):
        return f"<Section {self.id}: {self.title} ({self.section_type_code})>"

class SectionRelationship(Base):
    """Model for tracking relationships between sections"""
    __tablename__ = "section_relationships"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_section_id: Mapped[int] = mapped_column(Integer, ForeignKey("sections.id"), nullable=False)
    target_section_id: Mapped[int] = mapped_column(Integer, ForeignKey("sections.id"), nullable=False)
    
    # Relationship type (reference, dependency, etc.)
    relationship_type: Mapped[String] = mapped_column(String, nullable=False)
    description: Mapped[String] = mapped_column(String, nullable=True)
    
    # Metadata
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    created_by_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_by: Mapped["User"] = relationship("User")
    
    def __repr__(self):
        return f"<SectionRelationship {self.source_section_id} -> {self.target_section_id}: {self.relationship_type}>"

class SectionTemplate(Base):
    """Model for section templates that can be reused across notes"""
    __tablename__ = "section_templates"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    
    # Template content structure
    content_template = Column(JSONB, nullable=False)  # Template structure with placeholders
    
    # Template metadata
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    author: Mapped["User"] = relationship("User")
    is_system_template: Mapped[bool] = mapped_column(Boolean, default=False)  # System-provided vs user-created
    specialty: Mapped[str] = mapped_column(String, nullable=True)  # Medical specialty this template is for
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Section types using this as default
    section_types: Mapped[list["SectionType"]] = relationship("SectionType", back_populates="default_template")
    
    def __repr__(self):
        return f"<SectionTemplate {self.id}: {self.name}>"

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