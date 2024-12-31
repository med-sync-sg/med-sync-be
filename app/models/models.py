import json
from pydantic import BaseModel, Field
from typing import Union, Optional, List, Any
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Concept(db.Model):
    __tablename__ = 'concepts'
    id = db.Column(db.Integer, primary_key=True)
    cui = db.Column(db.String(8), index=True, unique=True)
    name = db.Column(db.String)
    definition = db.Column(db.String)
    semantic_types = db.relationship('SemanticType', backref='concept', lazy='dynamic')
    relations = db.relationship('Relation', backref='concept', lazy='dynamic')

class Term(db.Model):
    __tablename__ = 'terms'
    id = db.Column(db.Integer, primary_key=True)
    cui = db.Column(db.String(8), db.ForeignKey('concepts.cui'))
    language = db.Column(db.String(3))
    term_status = db.Column(db.String(1))
    is_preferred = db.Column(db.String(1))
    string = db.Column(db.String)
    source = db.Column(db.String)
    term_type = db.Column(db.String)

class Definition(db.Model):
    __tablename__ = 'definitions'
    id = db.Column(db.Integer, primary_key=True)
    cui = db.Column(db.String(8), db.ForeignKey('concepts.cui'))
    definition = db.Column(db.String)
    source = db.Column(db.String)

class SemanticType(db.Model):
    __tablename__ = 'semantic_types'
    id = db.Column(db.Integer, primary_key=True)
    cui = db.Column(db.String(8), db.ForeignKey('concepts.cui'))
    semantic_type = db.Column(db.String)
    tui = db.Column(db.String(4))

class Relation(db.Model):
    __tablename__ = 'relations'
    id = db.Column(db.Integer, primary_key=True)
    cui1 = db.Column(db.String(8), db.ForeignKey('concepts.cui'))
    cui2 = db.Column(db.String(8))
    relationship = db.Column(db.String)
    additional_relationship = db.Column(db.String)
    source = db.Column(db.String)


class MedicalEntry(BaseModel):
    topic: str = Field(default="")
    entry: Optional[Union[str, int, float, List, 'MedicalEntry', List['MedicalEntry']]] = None
    parent: Optional['MedicalEntry'] = None

    class Config:
        json_encoders = {
            'MedicalEntry': lambda v: v.to_dict(exclude={'parent'})
        }

class ConsultationModel(BaseModel):
    entries: List[MedicalEntry]
    def __init__(self):
        self.entries = []

class SOAPModel(ConsultationModel):
    subjective_entries: List[MedicalEntry]
    objective_entries: List[MedicalEntry]
    assessment_entries: List[MedicalEntry]
    plan_entries: List[MedicalEntry]
    
    def __init__(self):
        super(self)
        self.subjective_entries = []
        self.objective_entries = []
        self.assessment_entries = []
        self.plan_entries = []