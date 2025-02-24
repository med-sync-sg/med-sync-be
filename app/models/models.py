from sqlalchemy import Column, String
from typing import List
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from app.schemas.note import BaseNote
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base

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

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[str] = mapped_column(nullable=False)
    # The entire Pydantic-defined note structure stored as JSON
    title: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="notes")
    sections: Mapped[List["Section"]] = relationship()

class Section(Base):
    __tablename__ = "sections"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    note_id : Mapped[int] = mapped_column(ForeignKey("notes.id"), nullable=False)
    note : Mapped["Note"] = relationship(back_populates="sections")
    title = Column(String)
    content = Column(JSONB)

class NoteTemplate(Base):
    __tablename__ = "note_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped["int"] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="note_templates")
    section_templates : Mapped[List["SectionTemplate"]] = relationship(back_populates="note_template")
    description: Mapped[str] = mapped_column(nullable=False)

class SectionTemplate(Base):
    __tablename__ = "section_templates"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(nullable=False)
    note_template_id : Mapped[int] = mapped_column(ForeignKey("note_templates.id"), nullable=False)
    note_template: Mapped["NoteTemplate"] = relationship(back_populates="section_templates")
    title: Mapped[str] = mapped_column(nullable=False)
    metadata_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    content_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
