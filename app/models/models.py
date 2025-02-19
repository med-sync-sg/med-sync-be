from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB
from app.schemas.notes import BaseNote
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)


class Section(Base):
    __tablename__ = "sections"
    id = Column(Integer, primary_key=True)
    note_id = Column(Integer, ForeignKey("notes.id"), nullable=False)
    title = Column(String)
    content = Column(String)


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, unique=True, primary_key=True)
    patient_id = Column(String)
    # The entire Pydantic-defined note structure stored as JSON
    title = Column(String, nullable=False)
    sections = Column(Integer, ForeignKey("sections.id"), nullable=True)
