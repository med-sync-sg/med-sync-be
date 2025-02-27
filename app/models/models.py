from sqlalchemy import Column, String
from typing import List
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from app.schemas.note import BaseNote
from app.schemas.section import BaseSectionCreate, TextCategoryEnum
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base, Session
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    notes: Mapped[List["Note"]] = relationship()
    note_templates: Mapped[List["NoteTemplate"]] = relationship()
    hashed_password: Mapped[str] = mapped_column(nullable=False)

class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(nullable=False)
    # The entire Pydantic-defined note structure stored as JSON
    title: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="notes")
    sections: Mapped[List["Section"]] = relationship()

class Section(Base):
    __tablename__ = "sections"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    note : Mapped["Note"] = relationship(back_populates="sections")
    title : Mapped[str] = mapped_column(default="")
    content = Column(JSONB)
    section_type : Mapped[str] = mapped_column(default=TextCategoryEnum.OTHERS.name)
    section_description : Mapped[str] = mapped_column(default=TextCategoryEnum.OTHERS.value)
    order : Mapped[int] = mapped_column(default=1)

class NoteTemplate(Base):
    __tablename__ = "note_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped["int"] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="note_templates")
    section_templates : Mapped[List["SectionTemplate"]] = relationship(back_populates="note_template")
    description: Mapped[str] = mapped_column(nullable=False)

class SectionTemplate(Base):
    __tablename__ = "section_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    description: Mapped[str] = mapped_column(nullable=False)
    note_template_id : Mapped[int] = mapped_column(ForeignKey("note_templates.id"), nullable=False)
    note_template: Mapped["NoteTemplate"] = relationship(back_populates="section_templates")
    title: Mapped[str] = mapped_column(nullable=False)
    section_type : Mapped[str] = mapped_column(default=TextCategoryEnum.OTHERS.name)
    section_description : Mapped[str] = mapped_column(default=TextCategoryEnum.OTHERS.value)
    metadata_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    content_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)


def upload_section(pydantic_section: BaseSectionCreate, db: Session) -> Section:
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