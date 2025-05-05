from fastapi import APIRouter, Depends, Body, HTTPException, Request, File, UploadFile, Form, Query
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
from app.db.neo4j_session import get_neo4j_session, Neo4jSession
from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers
from app.utils.nlp.summarizer import generate_summary

from app.models.models import Note, Section, ReportTemplate, User
from app.schemas.section import SectionCreate, SectionRead
from app.schemas.note import NoteCreate
from app.services.section_template_service import SectionTemplateService
from app.services.transcription_service import TranscriptionService
from app.services.audio_service import AudioService
from app.services.nlp.keyword_extract_service import KeywordExtractService
from app.services.note_service import NoteService
from app.services.report_generation.report_service import ReportService
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

@router.post("/test-adaptation")
async def test_adaptation(
    audio_file: UploadFile = File(...),
    user_id: int = Form(...),
    use_adaptation: bool = Form(True),
    db: Session = Depends(get_session)
):
    """
    Test the speaker adaptation feature with an uploaded audio file
    
    Args:
        audio_file: The audio file to transcribe
        user_id: User ID for speaker adaptation profile
        use_adaptation: Whether to use speaker adaptation
        db: Database session
        
    Returns:
        Transcription results with and without adaptation for comparison
    """
    try:
        logger.info(f"Testing adaptation for user {user_id}, use_adaptation={use_adaptation}")
        
        # Create a temporary file to store the uploaded audio
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            # Write the uploaded file content to the temporary file
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f"Saved audio to temporary file: {temp_path}")
        
        try:
            # Initialize necessary services
            transcription_service = TranscriptionService()
            speech_processor = transcription_service.speech_processor
            
            # Load the audio data
            with wave.open(temp_path, 'rb') as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read all frames at once
                frames = wav_file.readframes(n_frames)
            
            # Convert to numpy array for processing
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize to [-1.0, 1.0]
            
            # Check if user has a speaker profile
            profile = None
            if use_adaptation:
                try:
                    profile = speech_processor._get_speaker_profile(user_id, db)
                    if profile is None:
                        logger.warning(f"No speaker profile found for user {user_id}")
                except Exception as profile_error:
                    logger.error(f"Error getting speaker profile: {str(profile_error)}")
            
            # Start timing
            start_time_standard = time.time()
            
            # First transcribe without adaptation
            standard_transcription = speech_processor.transcribe(audio_data, sample_rate=frame_rate)
            
            standard_time = time.time() - start_time_standard
            
            # Now transcribe with adaptation if requested
            adapted_transcription = None
            adaptation_time = None
            adaptation_info = None
            
            if use_adaptation:
                start_time_adapted = time.time()
                
                # Transcribe with adaptation
                adapted_transcription = speech_processor.transcribe_with_adaptation(
                    audio_data, user_id, db, sample_rate=frame_rate
                )
                
                adaptation_time = time.time() - start_time_adapted
                
                # Get information about the adaptation
                transformer = speech_processor.adaptation_cache.get(user_id)
                if transformer:
                    adaptation_info = {
                        "vtln_warp_factor": getattr(transformer, "vtln_warp_factor", None),
                        "feature_dimension": len(transformer.mean_vector) if hasattr(transformer, "mean_vector") else None,
                        "transform_applied": transformer.transform_matrix is not None
                    }
            
            # Prepare response
            response = {
                "standard_transcription": standard_transcription,
                "standard_processing_time_ms": round(standard_time * 1000, 2),
                "use_adaptation": use_adaptation,
                "user_id": user_id,
                "audio_info": {
                    "channels": channels,
                    "sample_width": sample_width,
                    "frame_rate": frame_rate,
                    "duration_seconds": n_frames / frame_rate,
                    "samples": len(audio_data)
                }
            }
            
            if use_adaptation:
                response.update({
                    "adapted_transcription": adapted_transcription,
                    "adaptation_processing_time_ms": round(adaptation_time * 1000, 2),
                    "adaptation_info": adaptation_info,
                    "has_speaker_profile": profile is not None,
                    # "word_difference": count_word_differences(standard_transcription, adapted_transcription)
                })
            
            return response
            
        finally:
            # Clean up - delete the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Deleted temporary file: {temp_path}")
                
    except Exception as e:
        logger.error(f"Error testing adaptation: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def count_word_differences(text1: str, text2: str) -> Dict[str, Any]:
    """
    Count the differences between two transcription texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with difference metrics
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    # Word error rate calculation
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, words1, words2)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    
    total_words = max(len(words1), len(words2))
    if total_words == 0:
        return {"word_error_rate": 0, "word_match_rate": 1.0, "word_count_difference": 0}
    
    word_error_rate = 1.0 - (matches / total_words)
    
    return {
        "word_error_rate": word_error_rate,
        "word_match_rate": 1.0 - word_error_rate,
        "word_count_difference": len(words2) - len(words1)
    }

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
async def process_text_transcript(
    fragment: TranscriptFragment, 
    db: Session = Depends(get_session),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """
    Process a text transcript and extract medical entities, creating content 
    that can be used with templates
    
    Args:
        fragment: TranscriptFragment with the transcript text
        
    Returns:
        Dictionary with transcription, entities, and processed content
    """
    try:

            
        # Limit very long transcripts to prevent overloading
        text = fragment.transcript
        
        # Input validation
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty transcript provided")

        max_length = 10000  # Character limit
        if len(text) > max_length:
            logger.warning(f"Transcript truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        keyword_service = KeywordExtractService()
        transcription_service = TranscriptionService()

        
        # Manually update transcription service state
        transcription_service.full_transcript = fragment.transcript
        transcription_service.transcript_segments.append(fragment.transcript)
        
        # Extract keywords
        keywords = transcription_service.extract_keywords()
        
        # Process keywords using the existing service methods
        keyword_service.process_and_buffer_keywords(keywords)
        
        templates, sections = keyword_service.create_section_from_keywords()
        
        # Prepare the response
        response = {
            "success": True,
            "transcription": text,
            "entity_count": len(keyword_service.buffer_keywords),
            "entities": keyword_service.buffer_keywords,
            "template_suggestions": templates,
            "processed_content": sections
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

@router.get("/find-template")
async def find_template(
    query: str = Query(..., description="Text to search for similar templates"),
    threshold: float = Query(0.65, description="Similarity threshold (0-1)"),
    limit: int = Query(5, description="Maximum number of results"),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """
    Find section templates similar to the provided text
    
    Args:
        query: Text to find similar templates for
        threshold: Similarity threshold (0-1)
        limit: Maximum number of results
        
    Returns:
        List of similar templates with similarity scores
    """
    try:
        start_time = time.time()
        
        # Create template service
        template_service = SectionTemplateService(neo4j)
        
        # Get embedding for query
        query_embedding = neo4j.get_embedding(query)
        
        # Search for templates
        templates = template_service.find_templates_by_text(
            query, 
            similarity_threshold=threshold,
            limit=limit
        )
        
        # Search for fields
        fields = template_service.find_fields_by_text(
            query,
            similarity_threshold=threshold,
            limit=limit
        )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "query": query,
            "templates": templates,
            "fields": fields,
            "processing_time_ms": round(processing_time * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Error finding templates: {str(e)}")
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
        sections = keyword_service.create_section_from_keywords()
        
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
        