from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column, Session
from app.schemas.section import TextCategoryEnum, SectionCreate
from typing import Optional 

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