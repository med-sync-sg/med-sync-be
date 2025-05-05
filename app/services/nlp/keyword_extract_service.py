import logging
import copy
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session

from app.utils.nlp.spacy_utils import find_medical_modifiers
from app.utils.nlp.nlp_utils import merge_keywords_into_template
from app.schemas.section import SectionCreate
from app.services.section_template_service import SectionTemplateService

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
        Create a section based on a keyword dictionary by finding and populating templates
        
        Args:
            keyword_dict: Dictionary containing the keyword term and its modifiers
            
        Returns:
            SectionCreate object if a template is found, None otherwise
        """
        sections = []
        templates = []
        for keyword_dict in self.buffer_keywords:
            # Extract the term from the keyword dictionary
            term = keyword_dict.get("term", "")
            if not term:
                logger.warning("Keyword dictionary missing 'term' field")
                return ([], [])
                
            # Find relevant template for this term
            templates = self.section_template_service.find_templates_by_text(term, similarity_threshold=0.0)
            if not templates:
                logger.info(f"No template found for term: {term}")
                return ([], [])
                
                
            def getMax(value):
                if "similarity_score" in value:
                    return value["similarity_score"]
                else:
                    return 0.0
                
            # Use the first template found (could be enhanced to select best match)
            template = max(templates, key=getMax)
            
            # Create a copy of the template to avoid modifying the original
            section_data = copy.deepcopy(template)
            
            template_fields = self.section_template_service.get_template_with_fields(template)
            logger.info(template_fields)
            if template_fields == None: 
                template_fields = self.section_template_service.get_base_fields()
            elif len(template_fields) < 1:
                template_fields = self.section_template_service.get_base_fields()

            # Merge the keyword and its modifiers into the template
            content = merge_keywords_into_template(
                keywords=keyword_dict,
                template_fields=template_fields,
                template_id=template.get("id"),
            )
            logger.info(f"Resulting content of section: {content}")
            try:
                # Create section with populated content
                section = SectionCreate(
                    user_id=1,
                    title=section_data.get("name", ""),
                    field_values=content,
                    content=content,
                    template_id=template.get("id", ""),
                    soap_category=template.get("id", "OTHER")
                )
                
                logger.info(f"Created section from keyword: {term}")
                sections.append(section)
            except Exception as e:
                logger.error(f"Error creating section from keyword: {str(e)}")
                continue

        return (templates, sections)
            

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
