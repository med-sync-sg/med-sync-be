import logging
import copy
import re
from typing import List, Dict, Any, Tuple, Set
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
    Manages keyword extraction, classification, and template mapping with consolidation.
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
    
    def consolidate_keywords(self) -> None:
        """
        Consolidate keywords in buffer to reduce redundancy and merge related information
        
        This implements the three-stage consolidation approach:
        1. Within-keyword consolidation (clean each keyword dict internally)
        2. Cross-keyword consolidation (merge similar keywords)
        3. Final cleanup
        """
        if not self.buffer_keywords:
            return
            
        logger.info(f"Starting consolidation of {len(self.buffer_keywords)} keywords")
        
        # Stage 1: Clean each keyword dictionary internally
        for i, keyword_dict in enumerate(self.buffer_keywords):
            self.buffer_keywords[i] = self._consolidate_within_keyword(keyword_dict)
        
        # Stage 2: Merge similar keywords across dictionaries
        self.buffer_keywords = self._merge_similar_keywords(self.buffer_keywords)
        
        # Stage 3: Final cleanup - remove any remaining duplicates
        self.buffer_keywords = self._final_cleanup(self.buffer_keywords)
        
        logger.info(f"Consolidation complete, {len(self.buffer_keywords)} keywords remaining")
    
    def _consolidate_within_keyword(self, keyword_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate terms within a single keyword dictionary
        
        Args:
            keyword_dict: Single keyword dictionary
            
        Returns:
            Consolidated keyword dictionary
        """
        result = copy.deepcopy(keyword_dict)
        
        # Clean each category individually
        categories = ['temporal', 'locations', 'quantities', 'modifiers']
        for category in categories:
            if category in result and isinstance(result[category], list):
                result[category] = self._clean_category_terms(result[category])
        
        # Remove cross-category redundancy (e.g., remove "severe" from modifiers if main term is "severe headache")
        result = self._remove_cross_category_redundancy(result)
        
        return result
    
    def _clean_category_terms(self, terms_list: List[str]) -> List[str]:
        """
        Clean terms within a single category (remove duplicates, substrings, word-subsets)
        
        Args:
            terms_list: List of terms in a category
            
        Returns:
            Cleaned list of terms
        """
        if not terms_list:
            return []
        
        # Step 1: Remove exact duplicates (case-insensitive)
        unique_terms = []
        seen_terms = set()
        for term in terms_list:
            normalized = self._normalize_term(term)
            if normalized not in seen_terms:
                unique_terms.append(term)
                seen_terms.add(normalized)
        
        # Step 2: Remove substring relationships
        filtered_terms = []
        for i, term1 in enumerate(unique_terms):
            is_subset = False
            for j, term2 in enumerate(unique_terms):
                if i != j and self._is_substring(term1, term2):
                    # term1 is a substring of term2, so term2 is more comprehensive
                    is_subset = True
                    break
            if not is_subset:
                filtered_terms.append(term1)
        
        # Step 3: Remove word-subset relationships
        final_terms = []
        for i, term1 in enumerate(filtered_terms):
            is_word_subset = False
            for j, term2 in enumerate(filtered_terms):
                if i != j and self._is_word_subset(term1, term2):
                    # All words in term1 appear in term2, keep term2
                    is_word_subset = True
                    break
            if not is_word_subset:
                final_terms.append(term1)
        
        # Sort by priority (longer, more descriptive terms first)
        final_terms.sort(key=self._calculate_term_priority, reverse=True)
        
        return final_terms
    
    def _remove_cross_category_redundancy(self, keyword_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove redundancy between main term and other categories
        
        Args:
            keyword_dict: Keyword dictionary
            
        Returns:
            Dictionary with cross-category redundancy removed
        """
        result = copy.deepcopy(keyword_dict)
        main_term = result.get('term', '').lower()
        
        if not main_term:
            return result
        
        # Split main term into words for comparison
        main_words = set(self._normalize_term(main_term).split())
        
        # Clean each category
        categories = ['temporal', 'locations', 'quantities', 'modifiers']
        for category in categories:
            if category in result and isinstance(result[category], list):
                cleaned_category = []
                for term in result[category]:
                    term_words = set(self._normalize_term(term).split())
                    
                    # Skip if all words already appear in main term
                    if not term_words.issubset(main_words):
                        cleaned_category.append(term)
                    else:
                        logger.debug(f"Removed redundant {category} term '{term}' (already in main term '{main_term}')")
                
                result[category] = cleaned_category
        
        return result
    
    def _merge_similar_keywords(self, keywords_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge keyword dictionaries with similar main terms
        
        Args:
            keywords_list: List of keyword dictionaries
            
        Returns:
            List with similar keywords merged
        """
        if len(keywords_list) <= 1:
            return keywords_list
        
        # Group keywords by similarity
        groups = self._group_keywords_by_similarity(keywords_list)
        
        # Merge each group
        merged_keywords = []
        for group in groups:
            if len(group) == 1:
                merged_keywords.append(group[0])
            else:
                merged_keyword = self._merge_keyword_group(group)
                merged_keywords.append(merged_keyword)
        
        return merged_keywords
    
    def _group_keywords_by_similarity(self, keywords_list: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group keywords with similar main terms
        
        Args:
            keywords_list: List of keyword dictionaries
            
        Returns:
            List of groups (each group is a list of similar keywords)
        """
        groups = []
        used_indices = set()
        
        for i, keyword1 in enumerate(keywords_list):
            if i in used_indices:
                continue
                
            # Start a new group with this keyword
            group = [keyword1]
            used_indices.add(i)
            
            # Find similar keywords
            term1 = keyword1.get('term', '')
            for j, keyword2 in enumerate(keywords_list):
                if j in used_indices:
                    continue
                    
                term2 = keyword2.get('term', '')
                if self._terms_are_similar(term1, term2):
                    group.append(keyword2)
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def _merge_keyword_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a group of similar keywords into one
        
        Args:
            group: List of similar keyword dictionaries
            
        Returns:
            Merged keyword dictionary
        """
        if not group:
            return {}
        
        if len(group) == 1:
            return group[0]
        
        # Choose the best main term (most comprehensive)
        best_term = max((kw.get('term', '') for kw in group), key=self._calculate_term_priority)
        
        # Merge all categories
        merged = {'term': best_term}
        categories = ['temporal', 'locations', 'quantities', 'modifiers']
        
        for category in categories:
            all_terms = []
            for keyword in group:
                if category in keyword and isinstance(keyword[category], list):
                    all_terms.extend(keyword[category])
            
            # Clean the merged category
            merged[category] = self._clean_category_terms(all_terms)
        
        return merged
    
    def _final_cleanup(self, keywords_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Final cleanup pass to remove any remaining duplicates
        
        Args:
            keywords_list: List of keyword dictionaries
            
        Returns:
            Final cleaned list
        """
        # Remove keyword dictionaries with identical main terms
        seen_terms = set()
        final_keywords = []
        
        for keyword in keywords_list:
            term = self._normalize_term(keyword.get('term', ''))
            if term and term not in seen_terms:
                final_keywords.append(keyword)
                seen_terms.add(term)
        
        return final_keywords
    
    def _normalize_term(self, term: str) -> str:
        """
        Normalize a term for comparison (lowercase, remove extra spaces)
        
        Args:
            term: Term to normalize
            
        Returns:
            Normalized term
        """
        if not term or not isinstance(term, str):
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', term.lower().strip())
        return normalized
    
    def _is_substring(self, term1: str, term2: str) -> bool:
        """
        Check if term1 is a substring of term2
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            True if term1 is contained in term2
        """
        norm1 = self._normalize_term(term1)
        norm2 = self._normalize_term(term2)
        
        if not norm1 or not norm2 or norm1 == norm2:
            return False
        
        return norm1 in norm2
    
    def _is_word_subset(self, term1: str, term2: str) -> bool:
        """
        Check if all words in term1 appear in term2
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            True if all words in term1 appear in term2
        """
        words1 = set(self._normalize_term(term1).split())
        words2 = set(self._normalize_term(term2).split())
        
        if not words1 or not words2 or words1 == words2:
            return False
        
        return words1.issubset(words2)
    
    def _terms_are_similar(self, term1: str, term2: str) -> bool:
        """
        Check if two main terms are similar enough to be merged
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            True if terms should be merged
        """
        if not term1 or not term2:
            return False
        
        # Check for substring relationship
        if self._is_substring(term1, term2) or self._is_substring(term2, term1):
            return True
        
        # Check for word overlap
        words1 = set(self._normalize_term(term1).split())
        words2 = set(self._normalize_term(term2).split())
        
        # If one is a subset of the other, they're similar
        if words1.issubset(words2) or words2.issubset(words1):
            return True
        
        # Check for significant word overlap (at least 50% overlap)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) > 0:
            overlap_ratio = len(intersection) / len(union)
            return overlap_ratio >= 0.5
        
        return False
    
    def _calculate_term_priority(self, term: str) -> int:
        """
        Calculate priority score for a term (higher score = higher priority)
        
        Args:
            term: Term to score
            
        Returns:
            Priority score
        """
        if not term:
            return 0
        
        # Base score from word count (more words = more descriptive)
        word_count = len(term.split())
        score = word_count * 10
        
        # Bonus for medical-sounding terms
        medical_indicators = ['pain', 'ache', 'syndrome', 'disease', 'disorder', 'condition', 'symptoms']
        for indicator in medical_indicators:
            if indicator in term.lower():
                score += 5
        
        # Bonus for descriptive adjectives
        descriptive_words = ['severe', 'acute', 'chronic', 'mild', 'sharp', 'dull', 'throbbing']
        for word in descriptive_words:
            if word in term.lower():
                score += 3
        
        return score
    
    def create_section_from_keywords(self) -> Tuple[list, List[SectionCreate]]:
        """
        Create sections based on keyword dictionaries by finding and populating templates
        
        Returns:
            Tuple of (templates, sections)
        """
        # First consolidate keywords to reduce redundancy
        self.consolidate_keywords()
        
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