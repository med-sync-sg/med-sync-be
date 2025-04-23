from fastapi import APIRouter, Depends, Body, HTTPException, Request, File, UploadFile, Form
from tempfile import NamedTemporaryFile
import wave
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import time
import os
import traceback
import numpy as np
from fastapi.responses import HTMLResponse
import datetime

from sqlalchemy.orm import Session
from app.db.local_session import DatabaseManager

from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers
from app.utils.nlp.summarizer import generate_summary
from app.utils.nlp.nlp_utils import merge_flat_keywords_into_template

from app.models.models import Note, Section, ReportTemplate, User
from app.schemas.section import SectionCreate, SectionRead
from app.schemas.note import NoteCreate
from app.services.transcription_service import TranscriptionService
from app.services.audio_service import AudioService
from app.services.nlp.keyword_extract_service import KeywordExtractService
from app.services.note_service import NoteService
from app.services.report_generation.report_service import ReportService
from app.services.report_generation.section_management_service import SectionManagementService
# Configure logger
logger = logging.getLogger(__name__)


router = APIRouter()
get_session = DatabaseManager().get_session

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

@router.post("/test-report")
async def generate_report_for_note(
    note_id: int,
    user_id: int,
    report_type: str = "doctor",
    db: Session = Depends(get_session)
):
    """
    Generate a report for an existing note
    
    Args:
        note_id: ID of the note to generate report for
        report_type: Type of report to generate ('doctor' or 'patient')
        template_id: Optional template ID to use
        db: Database session
        
    Returns:
        HTML report as a string
    """
    try:
        # Check if note exists
        note = db.query(Note).filter(Note.id == note_id, Note.user_id == user_id).first()
        if not note:
            raise HTTPException(
                status_code=404,
                detail="Note not found"
            )
        
        # Initialize report service
        report_service = ReportService(db)
        

        if report_type == "patient":
            # Generate default patient report
            report_html = report_service.generate_patient_report(note_id)
        else:
            # Generate default doctor report
            report_html = report_service.generate_doctor_report(note_id)
        
        if not report_html:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate report"
            )
            
        # Return the report as HTML
        return HTMLResponse(content=report_html, media_type="text/html")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )

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
            is_medical = ent._.get("is_medical_term")
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

@router.post("/text-transcript")
async def process_text_transcript(fragment: TranscriptFragment, db: Session = Depends(get_session)):
    """
    Process a text transcript and extract medical entities and sections
    
    Args:
        fragment: TranscriptFragment with the transcript text
        
    Returns:
        Dictionary with transcription, entities, and sections in the same format as process-audio
    """
    try:
        text = fragment.transcript
        keyword_extract_service = KeywordExtractService(db)
        section_management_service = SectionManagementService(db)
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
        
        # Extract entities
        entities = []
        for ent in transcription_doc.ents:
            is_medical = ent._.get("is_medical_term")
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "is_medical": is_medical
            })
        
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
        section_objects = []
        for i, result_keyword_dict in enumerate(keyword_extract_service.final_keywords):
            try:
                
                section_type_id, section_type_code = section_management_service.get_semantic_section_type(result_keyword_dict.get("term", ""))
                template = section_management_service.find_content_dictionary(result_keyword_dict, section_type_code)
                merged_content = merge_flat_keywords_into_template(result_keyword_dict, template, threshold=0.5)
                
                if merged_content.get("Main Symptom") is not None:
                    if "name" in merged_content["Main Symptom"] and merged_content["Main Symptom"]["name"]:
                        # Create a SectionRead object
                        section_read = SectionRead(
                            id=i,  # Dummy ID
                            note_id=0,  # Dummy note ID
                            user_id=0,  # Dummy user ID
                            title=result_keyword_dict.get("term", "Section"),
                            content=merged_content,
                            section_type_id=section_type_id,
                            section_type_code=section_type_code
                        )
                        section_objects.append(section_read.model_dump())
                    else:
                        logger.info(f"Content not added as it has no name: {result_keyword_dict.get('term', '')}")
            except Exception as section_error:
                # Log the error but continue processing other sections
                logger.error(f"Error processing section for term '{result_keyword_dict.get('term', '')}': {str(section_error)}")
                continue
        
        # Prepare the response in the same format as process-audio
        response = {
            "success": True,
            "transcription": text,
            "entity_count": len(entities),
            "entities": entities,
            "sections": section_objects
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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
            is_medical = ent._.get("is_medical_term")
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
        
@router.post("/text-chunk")
async def process_text_chunk(
    chunk: str = Body(..., embed=True),
    user_id: int = Body(0, embed=True),
    note_id: int = Body(0, embed=True),
    db: Session = Depends(get_session)
):
    """
    Process a text chunk to simulate real-time transcription
    
    Args:
        chunk: Text chunk to process
        user_id: User ID for the session
        note_id: Note ID for the session
        db: Database session
        
    Returns:
        Processed text with any extracted sections
    """
    try:
        # Input validation
        if not chunk or len(chunk.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty text chunk provided")
            
        # Limit very long texts
        max_length = 5000  # Character limit
        if len(chunk) > max_length:
            logger.warning(f"Text chunk truncated from {len(chunk)} to {max_length} characters")
            chunk = chunk[:max_length]
        
        
        # Initialize services
        transcription_service = TranscriptionService()
        keyword_service = KeywordExtractService()
        note_service = NoteService(db)
        
        # Manually update transcription service state
        transcription_service.full_transcript = chunk
        transcription_service.transcript_segments.append(chunk)
        
        # Extract keywords
        keywords = transcription_service.extract_keywords()
        
        # Process keywords using the existing service methods
        keyword_service.process_and_buffer_keywords(keywords)
        keyword_service.merge_keywords()
        sections = keyword_service.create_sections(user_id, note_id)
        
        # Add sections to note and format for response
        sections_data = []
        for section in sections:
            # Save section to database
            db_section = note_service.add_section_to_note(note_id, section)
            if db_section:
                # Convert to JSON for response
                sections_data.append({
                    'id': db_section.id,
                    'title': db_section.title,
                    'content': db_section.content,
                    'section_type': db_section.section_type
                })
        
        return {
            'text': chunk,
            'entities_count': len(keywords),
            'sections': sections_data
        }
        
    except Exception as e:
        logger.error(f"Error processing text chunk: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text chunk: {str(e)}"
        )
        