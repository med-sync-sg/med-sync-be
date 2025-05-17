import logging
import copy
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session

from app.utils.nlp.spacy_utils import find_medical_modifiers
from app.utils.nlp.nlp_utils import merge_keywords_into_template
from app.schemas.section import SectionCreate, SectionRead
from app.services.section_template_service import SectionTemplateService
from datetime import datetime
# Configure logger
logger = logging.getLogger(__name__)

class KeywordExtractService:
    """
    Service for extracting and processing medical keywords from text.
    Manages keyword extraction, classification, and template mapping.
    """
    
    def __init__(self):
        """
        Initialize keyword service
        
        Args:
        """
        self.section_template_service = SectionTemplateService()
        self.buffer_keywords = []
        
        logger.info("KeywordService initialized")
    
    def extract_keywords_from_doc(self, doc) -> List[Dict[str, Any]]:
        """
        Extract keywords from a spaCy document
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of extracted keyword dictionaries
        """
        try:
            keywords = find_medical_modifiers(doc=doc)
            logger.info(f"Extracted {len(keywords)} keyword sets from document")
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def process_and_buffer_keywords(self, new_keywords: List[Dict[str, Any]]) -> None:
        """
        Process new keywords and add to buffer
        
        Args:
            new_keywords: List of new keyword dictionaries
        """
        for keyword in new_keywords:
            self.buffer_keywords.append(keyword)
            
        logger.debug(f"Added {len(new_keywords)} keywords to buffer (total: {len(self.buffer_keywords)})")
    
    def create_section_from_keywords(self) -> Tuple[list, List[SectionCreate]]:
        """
        Create sections based on keyword dictionaries by finding and populating templates
        
        Returns:
            Tuple of (templates, sections)
        """
        sections = []
        found_templates = []
        
        for keyword_dict in self.buffer_keywords:
            # Extract the term from the keyword dictionary
            term = keyword_dict.get("term", "")
            if not term:
                logger.warning("Keyword dictionary missing 'term' field")
                continue
                
            # Find relevant template for this term
            templates = self.section_template_service.find_templates_by_text(term, similarity_threshold=0.0)
            if not templates:
                logger.info(f"No template found for term: {term}")
                continue
                
            # Select best template based on similarity score
            template = max(templates, key=lambda t: t.get("similarity_score", 0))
            found_templates.append(template)
            
            # Get template fields
            template_fields = self.section_template_service.get_template_with_fields(template)
            if template_fields is None or len(template_fields) < 1:
                template_fields = self.section_template_service.get_base_fields()

            # Match template fields to keyword data
            match_tuples = self.section_template_service.match_template_fields(
                template_fields=template_fields, 
                keyword_dict=keyword_dict
            )
            
            # Initialize the content dictionary for this section
            section_content = {}
            
            # Process each match tuple and add to section content
            for match_tuple in match_tuples:
                # Merge the keyword and its modifiers into the template
                merged_result = merge_keywords_into_template(match_tuple=match_tuple)
                
                # Merge the content into our section content
                for field_id, field_obj in merged_result["content"].items():
                    if field_id in section_content:
                        # Handle cases where field already exists in content
                        if isinstance(section_content[field_id], list):
                            # If already a list, append new field(s)
                            if isinstance(field_obj, list):
                                section_content[field_id].extend(field_obj)
                            else:
                                section_content[field_id].append(field_obj)
                        else:
                            # Convert to list and add new field(s)
                            if isinstance(field_obj, list):
                                section_content[field_id] = [section_content[field_id], *field_obj]
                            else:
                                section_content[field_id] = [section_content[field_id], field_obj]
                    else:
                        # New field, add directly
                        section_content[field_id] = field_obj
            
            # Create the section data structure
            section_data = {
                "title": template.get("name", ""),
                "template_id": template.get("id", ""),
                "soap_category": template.get("soap_category", "OTHER"),
                "content": section_content,
                "user_id": 1,  # Default user ID, should be updated when actually creating
                "note_id": 0   # Will be set when added to a note
            }
            
            # Convert to section
            section = convert_to_section_create(section_data)
            sections.append(section)
        self.clear()
        
        return (found_templates, sections)
            

    def clear(self) -> None:
        """Clear all keywords"""
        self.buffer_keywords = []
        logger.info("Cleared all keywords")
    
    def get_keyword_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current keywords
        
        Returns:
            Dictionary with keyword summary
        """
        terms = [kw.get("term", "") for kw in self.buffer_keywords]
        
        return {
            "buffer_count": len(self.buffer_keywords),
            "unique_terms": terms,
        }


def convert_to_section_create(template_content: Dict[str, Any]) -> SectionCreate:
    """
    Convert template content to a SectionRead object
    
    Args:
        template_content: Merged template content with field values
        
    Returns:
        SectionRead object
    """
    try:
        # Create the SectionRead object
        section = SectionCreate(
            note_id=1,
            user_id=template_content.get("user_id", 1),
            title=template_content.get("title", ""),
            template_id=template_content.get("template_id", ""),
            soap_category=template_content.get("soap_category", "OTHER"),
            content=template_content.get("content", {}),
            is_visible_to_patient=template_content.get("is_visible_to_patient", True),
            display_order=template_content.get("display_order", 100),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return section
        
    except Exception as e:
        logger.error(f"Error converting to SectionRead: {str(e)}")
        # Return a minimal valid SectionRead
        return SectionCreate(
            note_id=1,
            user_id=1,
            title="Error",
            template_id="",
            soap_category="OTHER",
            content={"error": str(e)},
            is_visible_to_patient=True,
            display_order=100,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )