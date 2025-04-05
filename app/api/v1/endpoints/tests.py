from fastapi import APIRouter, Depends, Body, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
import time
import os
import traceback

from sqlalchemy.orm import Session
from app.db.local_session import DatabaseManager

from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers
from app.utils.nlp.summarizer import generate_summary
from app.utils.nlp.nlp_utils import merge_flat_keywords_into_template
from app.utils.speech_processor import SpeechProcessor
from app.db.data_loader import classify_text_category, find_content_dictionary
from app.services.nlp.keyword_extract_service import KeywordExtractService

from app.models.models import Note, Section
from app.schemas.section import SectionCreate, TextCategoryEnum
from app.schemas.note import NoteCreate

# Configure logger
logger = logging.getLogger(__name__)


router = APIRouter()
get_session = DatabaseManager().get_session
keyword_extract_service = KeywordExtractService(DatabaseManager())

# Sample transcripts for testing
demo_transcript = """
Patient: Doctor, I've had a sore throat for the past three days, and it's getting worse. It feels scratchy, and swallowing is uncomfortable.
Doctor: I see. Has it been painful enough to affect eating or drinking?
Patient: No, but I also have a mild cough and keep sneezing a lot. My nose has been running non-stop.
Doctor: Sounds like you're experiencing some nasal irritation. Have you noticed any thick or discolored mucus?
Patient: Yeah, sometimes I feel some mucus at the back of my throat.
Doctor: Alright. Do you feel any tightness in your chest or shortness of breath when coughing?
Patient: No, but I've been feeling a bit feverish since last night. I haven't checked my temperature though. My body feels tired too.
Doctor: Fatigue and feverishness can be common with viral infections. Any chills or sweating?
Patient: No.
Doctor: Understood. Based on your symptoms, it looks like an upper respiratory tract infection, likely viral. Let me examine your throat to confirm.
"""

class TranscriptFragment(BaseModel):
    """Request model for text transcript processing"""
    transcript: str

class BasicTestRequest(BaseModel):
    """Request model for basic test endpoint"""
    text: str

class MetricsResponse(BaseModel):
    """Response model with metrics information"""
    success: bool
    entity_count: int
    entities: List[Dict[str, Any]]
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the tests service
    
    Returns:
        Status of the test service
    """
    try:
        # Test the NLP pipeline with a simple text
        test_text = "Patient has a fever and cough."
        doc = process_text(test_text)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "nlp_pipeline": "working",
            "entities_detected": len(doc.ents),
            "entity_samples": [{"text": ent.text, "label": ent.label_} for ent in doc.ents][:3]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.post("/basic-test")
async def basic_test(request: BasicTestRequest):
    """
    Basic test endpoint for NLP processing
    
    Process a text through the NLP pipeline and return entities with metrics
    
    Args:
        request: Request with text to process
        
    Returns:
        Processing results with entities and metrics
    """
    try:
        # Process the text
        start_time = time.time()
        doc = process_text(request.text)
        processing_time = time.time() - start_time
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            is_medical = getattr(ent, "_.is_medical_term", False)
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "is_medical": is_medical
            })
        
        # Calculate basic metrics
        metrics = {
            "processing_time_ms": processing_time * 1000,
            "medical_entity_ratio": sum(1 for e in entities if e.get("is_medical", False)) / len(entities) if entities else 0
        }
        
        return {
            "success": True,
            "entity_count": len(entities),
            "entities": entities,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error in basic test: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "entity_count": 0,
            "entities": [],
            "error": str(e)
        }

@router.post("/text-transcript", response_model=List[str])
async def process_text_transcript(fragment: TranscriptFragment):
    """
    Process a text transcript and extract medical entities and sections
    
    Args:
        fragment: TranscriptFragment with the transcript text
        
    Returns:
        List of JSON-serialized section objects
    """
    try:
        text = fragment.transcript
        
        # Input validation
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty transcript provided")
            
        # Limit very long transcripts to prevent overloading
        max_length = 10000  # Character limit
        if len(text) > max_length:
            logger.warning(f"Transcript truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Process text through NLP pipeline
        transcription_doc = process_text(text)
        logger.info(f"Entities found: {len(transcription_doc.ents)}")
        
        # Extract keywords safely
        try:
            extracted_keywords = find_medical_modifiers(doc=transcription_doc)
            keyword_extract_service.buffer_keywords = extracted_keywords if isinstance(extracted_keywords, list) else []
            logger.info(f"Extracted keywords: {len(keyword_extract_service.buffer_keywords)}")
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            logger.error(traceback.format_exc())
            keyword_extract_service.buffer_keywords = []
        
        # Ensure buffer_keyword_dicts is a list of dictionaries
        if not isinstance(keyword_extract_service.buffer_keywords, list):
            keyword_extract_service.buffer_keywords = []
        
        # Remove duplicates
        result_dicts = []
        for keyword_dict in keyword_extract_service.buffer_keywords:
            if not isinstance(keyword_dict, dict):
                logger.warning(f"Skipping non-dictionary keyword: {keyword_dict}")
                continue
                
            found = False
            for i, existing_dict in enumerate(result_dicts):
                if keyword_dict.get("term", "") == existing_dict.get("term", ""):
                    found = True
                    break
            if not found:
                result_dicts.append(keyword_dict)
        
        keyword_extract_service.final_keywords = result_dicts
        
        # Create sections based on final_keyword_dicts
        sections = []
        for result_keyword_dict in keyword_extract_service.final_keywords:
            try:
                category = classify_text_category(result_keyword_dict.get("term", ""))
                template = find_content_dictionary(result_keyword_dict, category)
                merged_content = merge_flat_keywords_into_template(result_keyword_dict, template, threshold=0.5)
                
                if merged_content.get("Main Symptom") is not None:
                    if "name" in merged_content["Main Symptom"] and merged_content["Main Symptom"]["name"]:
                        sections.append(merged_content)
                    else:
                        logger.info(f"Content not added as it has no name: {result_keyword_dict.get('term', '')}")
            except Exception as section_error:
                # Log the error but continue processing other sections
                logger.error(f"Error processing section for term '{result_keyword_dict.get('term', '')}': {str(section_error)}")
                continue
        
        # Convert sections to JSON strings
        sections_json = []
        for section in sections:
            try:
                sections_json.append(json.dumps(section))
            except Exception as json_error:
                logger.error(f"Error serializing section to JSON: {str(json_error)}")
                # Include a simplified version instead
                sections_json.append(json.dumps({
                    "error": "Serialization error", 
                    "term": result_keyword_dict.get("term", "unknown")
                }))
        
        return sections_json
    
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing transcript: {str(e)}"
        )

@router.post("/advanced-test")
async def advanced_test(
    request: BasicTestRequest,
    calculate_metrics: bool = True,
    include_raw_data: bool = False
):
    """
    Advanced test endpoint with detailed metrics
    
    Args:
        request: Request with text to process
        calculate_metrics: Whether to calculate detailed metrics
        include_raw_data: Whether to include raw NLP data
        
    Returns:
        Detailed processing results
    """
    try:
        # Process the text
        start_time = time.time()
        doc = process_text(request.text)
        processing_time = time.time() - start_time
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            is_medical = getattr(ent, "_.is_medical_term", False)
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "is_medical": is_medical
            })
        
        # Extract keywords
        keyword_start = time.time()
        try:
            keywords = find_medical_modifiers(doc=doc) if calculate_metrics else []
            if not isinstance(keywords, list):
                keywords = []
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            keywords = []
        keyword_time = time.time() - keyword_start
        
        # Calculate metrics
        metrics = {
            "processing_time_ms": processing_time * 1000,
            "keyword_extraction_time_ms": keyword_time * 1000,
            "total_time_ms": (processing_time + keyword_time) * 1000,
            "entity_count": len(entities),
            "medical_entity_count": sum(1 for e in entities if e.get("is_medical", False)),
            "keyword_count": len(keywords)
        }
        
        if len(entities) > 0:
            metrics["medical_entity_ratio"] = metrics["medical_entity_count"] / metrics["entity_count"]
        
        # Prepare response
        response = {
            "success": True,
            "entity_count": len(entities),
            "entities": entities,
            "metrics": metrics
        }
        
        # Include keywords if requested
        if include_raw_data:
            response["keywords"] = keywords
            response["tokens"] = [{"text": token.text, "pos": token.pos_, "dep": token.dep_} 
                                 for token in doc]
            
        return response
        
    except Exception as e:
        logger.error(f"Error in advanced test: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "entity_count": 0,
            "entities": [],
            "error": str(e),
            "traceback": traceback.format_exc() if include_raw_data else None
        }

@router.post("/entity-metrics")
async def calculate_entity_metrics(
    gold_standard: List[Dict[str, Any]] = Body(...),
    predicted: List[Dict[str, Any]] = Body(...)
):
    """
    Calculate metrics between gold standard and predicted entities
    
    Args:
        gold_standard: List of gold standard entities
        predicted: List of predicted entities
        
    Returns:
        Dictionary with precision, recall, F1 and error rate
    """
    try:
        # Convert to lowercase for comparison
        gold_texts = [entity['text'].lower() for entity in gold_standard]
        pred_texts = [entity['text'].lower() for entity in predicted]
        
        # Count true positives, false positives, false negatives
        true_positives = sum(1 for text in pred_texts if text in gold_texts)
        false_positives = sum(1 for text in pred_texts if text not in gold_texts)
        false_negatives = sum(1 for text in gold_texts if text not in pred_texts)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Error rate (1 - F1)
        error_rate = 1.0 - f1
        
        return {
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "error_rate": error_rate
            },
            "counts": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "gold_count": len(gold_texts),
                "pred_count": len(pred_texts)
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            "error": str(e),
            "metrics": {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "error_rate": 1.0
            }
        }