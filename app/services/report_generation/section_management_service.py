import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.utils.nlp.nlp_utils import embed_text, cosine_similarity
from app.models.models import SectionType, SOAPCategory
import copy

logger = logging.getLogger(__name__)

class SectionManagementService:
    """Service for managing section types"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_section_type(self, section_type_id: int) -> Optional[SectionType]:
        """Get section type by ID"""
        return self.db.query(SectionType).filter(SectionType.id == section_type_id).first()
    
    def get_section_type_by_code(self, code: str) -> Optional[SectionType]:
        """Get section type by code"""
        return self.db.query(SectionType).filter(SectionType.code == code).first()
    
    def get_all_section_types(self) -> List[SectionType]:
        """Get all section types"""
        return self.db.query(SectionType).order_by(SectionType.default_order).all()
    
    def get_section_types_by_soap(self, soap_category: str) -> List[SectionType]:
        """Get section types by SOAP category"""
        return self.db.query(SectionType).filter(
            SectionType.soap_category == soap_category
        ).order_by(SectionType.default_order).all()
    
    def get_default_section_type(self) -> SectionType:
        """Get default section type (falls back to OTHER or creates one)"""
        default_type = self.db.query(SectionType).filter(SectionType.code == "OTHER").first()
        
        if not default_type:
            # Create a default OTHER type if not found
            default_type = SectionType(
                code="OTHER",
                name="Other",
                description="Uncategorized section",
                soap_category=SOAPCategory.OTHER,
                default_order=999,
                is_required=False
            )
            
            self.db.add(default_type)
            self.db.commit()
            self.db.refresh(default_type)
            
        return default_type
    
    def find_content_dictionary(self, keyword_dict: Dict[str, Any], section_type_code: str) -> Dict[str, Any]:
        """Find appropriate content dictionary template based on section type"""
        section_type = self.get_section_type_by_code(section_type_code)
        
        # Use section type's content schema if available
        if section_type and section_type.content_schema:
            return copy.deepcopy(section_type.content_schema)
        
        # Fallback to a basic template
        return {
            "Main Symptom": {
                "name": "",
                "description": "",
                "duration": "",
                "severity": ""
            },
            "additional_content": {}
        }

    def get_semantic_section_type(self, section_title: str, section_content: Dict[str, Any], db=Session) -> Tuple[int, str]:
        """
        Determine section type based on semantic similarity
        
        Args:
            section_title: Title of the section
            section_content: Content of the section
            db_session: Database session for accessing SectionType models
            
        Returns:
            Predicted section type code and ID in a tuple.
        """
        # If no database session, use a fallback approach
        if not db:
            # Create a default mapping to use when DB is unavailable
            default_mapping = {
                "chief_complaint": "CHIEF_COMPLAINT",
                "medical history": "PMH",
                "physical exam": "PHYSICAL_EXAM",
                "assessment": "ASSESSMENT",
                "plan": "TREATMENT_PLAN",
                "vital signs": "VITALS"
            }
            
            # Simple keyword matching for fallback
            content_text = section_title.lower()
            for keyword, section_type in default_mapping.items():
                if keyword in content_text:
                    return section_type
                    
            return (0, "OTHER")  # Default fallback
            
        # Get all section types from database
        section_types = db.query(SectionType).all()
        
        # Create reference text from section content
        content_text = section_title.lower()
        
        # Add key content terms if available
        if isinstance(section_content, dict):
            # Add main keys as content
            content_text += " " + " ".join(section_content.keys()).lower()
        
        # Get embedding for content
        content_embedding = embed_text(content_text)
        
        # Calculate similarities with each section type
        best_match = None
        best_match_id = -1
        best_score = 0.0
        
        for section_type in section_types:
            # Create reference text for the section type
            type_text = f"{section_type.name} {section_type.description or ''}"
            type_embedding = embed_text(type_text)
            
            # Calculate similarity
            similarity = cosine_similarity(content_embedding, type_embedding)
            
            # Update best match if better
            if similarity > best_score:
                best_score = similarity
                best_match = section_type.code
                best_match_id = section_type.id
        
        # If no good match found, return a default
        if not best_match or best_score < 0.3:
            # Try to find a generic "OTHER" type
            other_type = db.query(SectionType).filter(SectionType.code == "OTHER").first()
            if other_type:
                return other_type.code
            return (0, "OTHER")  # Fallback if no match and no OTHER type
        
        return (best_match_id, best_match)