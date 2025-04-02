import logging
import copy
from typing import List, Dict, Any, Optional
from app.utils.nlp.keyword_extractor import find_medical_modifiers
from app.schemas.section import SectionCreate, TextCategoryEnum
from app.db.local_session import DatabaseManager
# Configure logger
logger = logging.getLogger(__name__)

class KeywordService:
    """
    Service for extracting and processing medical keywords from text.
    Manages keyword extraction, classification, and template mapping.
    """
    
    def __init__(self, data_store: Optional[DatabaseManager] = None):
        """
        Initialize keyword service
        
        Args:
            data_store: IrisDataStore instance or None to use singleton
        """
        self.data_store = data_store or DatabaseManager()
        self.buffer_keywords = []
        self.final_keywords = []
        
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
    
    def merge_keywords(self) -> None:
        """
        Merge buffered keywords with final keywords
        """
        for keyword_dict in self.buffer_keywords:
            self._merge_or_add_keyword(keyword_dict)
            
        # Clear buffer after merging
        self.buffer_keywords = []
        logger.info(f"Merged keywords, now have {len(self.final_keywords)} unique keyword sets")
    
    def _merge_or_add_keyword(self, keyword_dict: Dict[str, Any]) -> None:
        """
        Merge keyword with existing one or add as new
        
        Args:
            keyword_dict: Keyword dictionary to merge or add
        """
        term = keyword_dict.get("term", "")
        if not term:
            logger.warning("Skipping keyword with empty term")
            return
            
        # Look for matching term in final keywords
        found = False
        for i, existing_dict in enumerate(self.final_keywords):
            if keyword_dict["term"] == existing_dict["term"]:
                # Merge with existing entry
                try:
                    self.final_keywords[i] = self.data_store.merge_flat_keywords_into_template(
                        existing_dict, keyword_dict
                    )
                    logger.debug(f"Merged keyword for term: {term}")
                    found = True
                    break
                except Exception as e:
                    logger.error(f"Error merging keywords for {term}: {str(e)}")
                    found = True  # Mark as found to avoid adding duplicate
                    break
                    
        # Add as new entry if not found
        if not found:
            self.final_keywords.append(keyword_dict)
            logger.debug(f"Added new keyword for term: {term}")
    
    def create_sections(self, user_id: int, note_id: int) -> List[SectionCreate]:
        """
        Create sections from the final keywords
        
        Args:
            user_id: User ID for the sections
            note_id: Note ID for the sections
            
        Returns:
            List of SectionCreate objects
        """
        sections = []
        
        try:
            # Get content dictionaries for each keyword
            content_dicts = self.fill_content_dictionaries()
            
            # Create sections from content dictionaries
            for index, content in enumerate(content_dicts):
                if index >= len(self.final_keywords):
                    logger.warning(f"No keyword found for content at index {index}")
                    continue
                    
                keyword = self.final_keywords[index]
                term = keyword.get("term", "")
                
                # Classify text to determine category
                category = self.data_store.classify_text_category(term)
                
                # Create section
                section = SectionCreate(
                    user_id=user_id,
                    note_id=note_id,
                    title=keyword.get("label", term or "Section"),
                    content=content,
                    section_type=category,
                    section_description=TextCategoryEnum[category].value
                )
                
                sections.append(section)
                
            logger.info(f"Created {len(sections)} sections from keywords")
            return sections
            
        except Exception as e:
            logger.error(f"Error creating sections: {str(e)}")
            return []
    
    def fill_content_dictionaries(self) -> List[Dict[str, Any]]:
        """
        Fill content dictionaries based on the final keywords
        
        Returns:
            List of content dictionaries
        """
        result = []
        
        for keyword_dict in self.final_keywords:
            try:
                term = keyword_dict.get("term", "")
                
                # Classify text to determine category
                category = self.data_store.classify_text_category(term)
                
                # Find matching template
                template = self.data_store.find_content_dictionary(keyword_dict, category)
                
                # Fill template with keyword data
                content = self.data_store.recursive_fill_content_dictionary(
                    keyword_dict, template
                )
                
                result.append(content)
                
            except Exception as e:
                logger.error(f"Error filling content dictionary for term '{keyword_dict.get('term', '')}': {str(e)}")
                # Add an empty dictionary to maintain index correspondence
                result.append({})
        
        logger.info(f"Filled {len(result)} content dictionaries from keywords")
        return result
    
    def clear(self) -> None:
        """Clear all keywords"""
        self.buffer_keywords = []
        self.final_keywords = []
        logger.info("Cleared all keywords")
    
    def get_keyword_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current keywords
        
        Returns:
            Dictionary with keyword summary
        """
        terms = [kw.get("term", "") for kw in self.final_keywords]
        
        return {
            "buffer_count": len(self.buffer_keywords),
            "final_count": len(self.final_keywords),
            "unique_terms": terms,
        }