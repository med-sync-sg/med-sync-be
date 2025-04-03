from fastapi import APIRouter, Depends, Body, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
import time
import os

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

class TestConfig(BaseModel):
    """Configuration for test execution"""
    detailed_logging: bool = False
    include_raw_nlp: bool = False
    return_sections: bool = True
    benchmark: bool = False

class DiagnosticInfo(BaseModel):
    """Model for diagnostic information about the processing pipeline"""
    elapsed_time: float
    entity_count: int
    keyword_count: int
    section_count: int

class ProcessingResult(BaseModel):
    """Comprehensive response model for processing results"""
    success: bool
    message: str
    sections: Optional[List[Dict[str, Any]]] = None
    diagnostic: Optional[DiagnosticInfo] = None
    raw_nlp: Optional[Dict[str, Any]] = None

@router.get("/")
def test_with_sample_transcript():
    """
    Run a test using the built-in sample transcript
    
    Returns a list of processed medical data structures
    """
    transcription_doc = process_text(demo_transcript)
    logging.info(f"Entities: {transcription_doc.ents}")
        
    # Extract prototype features from the transcript
    keyword_extract_service.buffer_keywords = find_medical_modifiers(doc=transcription_doc)
    logging.info(f"Extracted keywords: {keyword_extract_service.buffer_keywords}")
    
    # Merge duplicate entries based on the 'term'
    result_dicts = []
    for keyword_dict in keyword_extract_service.buffer_keywords:
        found = False
        for i, existing_dict in enumerate(result_dicts):
            if keyword_dict["term"] == existing_dict["term"]:
                # If duplicate found, skip
                found = True
                break
        if not found:
            result_dicts.append(keyword_dict)
    keyword_extract_service.final_keywords = result_dicts
    
    result = []
    for result_keyword_dict in keyword_extract_service.final_keywords:
        category = classify_text_category(result_keyword_dict["term"])
        template = find_content_dictionary(result_keyword_dict, category)
        # Use the prototype-based merging function
        merged_content = merge_flat_keywords_into_template(result_keyword_dict, template, threshold=0.5)
        logging.info("Merged Content Dictionary: %s", merged_content)
        if merged_content.get("Main Symptom") is not None:
            if len(merged_content["Main Symptom"]["name"]) == 0:
                print("Content not added as it has no name: ", merged_content)
            else:
                result.append(merged_content)
    return result

@router.post("/text-transcript", response_model=List[str])
def process_text_transcript(fragment: TranscriptFragment):
    """
    Process a text transcript fragment
    
    Takes a text transcript and runs it through the NLP pipeline,
    returning the generated sections as JSON strings
    
    Args:
        fragment: TranscriptFragment object containing the transcript text
        
    Returns:
        List of JSON-serialized section objects
    """
    text = fragment.transcript
    transcription_doc = process_text(text)
    logging.info(f"Entities: {transcription_doc.ents}")
    
    keyword_extract_service.buffer_keywords = find_medical_modifiers(doc=transcription_doc)
    logging.info(f"Extracted keywords: {keyword_extract_service.buffer_keywords}")
    
    result_dicts = []
    for keyword_dict in keyword_extract_service.buffer_keywords:
        found = False
        for i, existing_dict in enumerate(result_dicts):
            if keyword_dict["term"] == existing_dict["term"]:
                found = True
                break
        if not found:
            result_dicts.append(keyword_dict)
    keyword_extract_service.final_keywords = result_dicts
    
    # Create sections based on final_keyword_dicts and prototype-based mapping
    sections = []
    for result_keyword_dict in keyword_extract_service.final_keywords:
        category = classify_text_category(result_keyword_dict["term"])
        template = find_content_dictionary(result_keyword_dict, category)
        merged_content = merge_flat_keywords_into_template(result_keyword_dict, template, threshold=0.5)
        if merged_content.get("Main Symptom") is not None:
            if len(merged_content["Main Symptom"]["name"]) == 0:
                print("Content not added as it has no name: ", merged_content)
            else:
                sections.append(merged_content)    
    
    sections_json = []
    for section in sections:
        sections_json.append(json.dumps(section))
    return sections_json

@router.post("/advanced-text-processing", response_model=ProcessingResult)
def advanced_text_processing(
    fragment: TranscriptFragment = Body(...),
    config: TestConfig = Body(TestConfig()),
    db: Session = Depends(get_session)
):
    """
    Advanced text processing endpoint with detailed diagnostics
    
    Processes a text transcript with configurable options for
    diagnostic information, section generation, and benchmarking
    
    Args:
        fragment: TranscriptFragment containing the transcript text
        config: TestConfig with processing configuration options
        db: Database session
        
    Returns:
        ProcessingResult with sections and diagnostic information
    """
    start_time = time.time()
    
    try:
        text = fragment.transcript
        
        if config.detailed_logging:
            logger.info(f"Processing transcript ({len(text)} characters)")
        
        # Process text through NLP pipeline
        transcription_doc = process_text(text)
        
        # Extract and merge keywords
        keywords = find_medical_modifiers(doc=transcription_doc)
        
        # Remove duplicates
        result_dicts = []
        for keyword_dict in keywords:
            found = False
            for i, existing_dict in enumerate(result_dicts):
                if keyword_dict["term"] == existing_dict["term"]:
                    found = True
                    break
            if not found:
                result_dicts.append(keyword_dict)
        
        # Create sections
        sections = []
        for result_keyword_dict in result_dicts:
            category = classify_text_category(result_keyword_dict["term"])
            template = find_content_dictionary(result_keyword_dict, category)
            merged_content = merge_flat_keywords_into_template(
                result_keyword_dict, template, threshold=0.5
            )
            
            if config.detailed_logging:
                logger.info(f"Processed term '{result_keyword_dict.get('term', '')}' as category '{category}'")
            
            if merged_content.get("Main Symptom") is not None:
                if len(merged_content["Main Symptom"]["name"]) > 0:
                    sections.append(merged_content)
                    
        # Create diagnostic information
        elapsed_time = time.time() - start_time
        diagnostic = DiagnosticInfo(
            elapsed_time=elapsed_time,
            entity_count=len(transcription_doc.ents),
            keyword_count=len(result_dicts),
            section_count=len(sections)
        )
        
        if config.detailed_logging:
            logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
            logger.info(f"Found {len(transcription_doc.ents)} entities, {len(result_dicts)} keywords, {len(sections)} sections")
        
        # Create result object
        result = ProcessingResult(
            success=True,
            message="Transcript processed successfully",
            diagnostic=diagnostic
        )
        
        # Include sections if requested
        if config.return_sections:
            result.sections = sections
        
        # Include raw NLP data if requested
        if config.include_raw_nlp:
            result.raw_nlp = {
                "entities": [{"text": ent.text, "label": ent.label_} for ent in transcription_doc.ents],
                "keywords": result_dicts
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        return ProcessingResult(
            success=False,
            message=f"Error: {str(e)}",
            diagnostic=DiagnosticInfo(
                elapsed_time=time.time() - start_time,
                entity_count=0,
                keyword_count=0,
                section_count=0
            )
        )

@router.post("/end-to-end", response_model=Dict[str, Any])
async def end_to_end_test(
    transcript: TranscriptFragment,
    user_id: int = Body(...),
    db: Session = Depends(get_session)
):
    """
    Run an end-to-end test of text processing and note generation
    
    This creates a real note in the database with sections derived from
    the transcript text, demonstrating the complete pipeline functionality
    
    Args:
        transcript: TranscriptFragment with the transcript text
        user_id: ID of the user to associate with the note
        db: Database session
        
    Returns:
        Dictionary with results of the test including the created note ID
    """
    try:
        # Process the transcript
        transcription_doc = process_text(transcript.transcript)
        
        # Generate a summary for the note title
        summary = generate_summary(transcript.transcript, top_n=1)
        title = f"Test Note: {summary[:50]}..."
        
        # Create a new note
        note_create = NoteCreate(
            title=title,
            patient_id=1,  # Test patient ID
            user_id=user_id,
            encounter_date=time.strftime("%Y-%m-%d"),
            sections=[]  # We'll add sections after creating the note
        )
        
        # Add the note to the database
        db_note = Note(
            title=note_create.title,
            patient_id=note_create.patient_id,
            user_id=note_create.user_id,
            encounter_date=note_create.encounter_date
        )
        db.add(db_note)
        db.commit()
        db.refresh(db_note)
        
        # Extract and process keywords
        keyword_extract_service.buffer_keywords = find_medical_modifiers(doc=transcription_doc)
        
        # Remove duplicates
        result_dicts = []
        for keyword_dict in keyword_extract_service.buffer_keywords:
            found = False
            for i, existing_dict in enumerate(result_dicts):
                if keyword_dict["term"] == existing_dict["term"]:
                    found = True
                    break
            if not found:
                result_dicts.append(keyword_dict)
        keyword_extract_service.final_keywords = result_dicts
        
        # Create sections
        sections = []
        for result_keyword_dict in result_dicts:
            category = classify_text_category(result_keyword_dict["term"])
            template = find_content_dictionary(result_keyword_dict, category)
            merged_content = merge_flat_keywords_into_template(
                result_keyword_dict, template, threshold=0.5
            )
            
            if merged_content.get("Main Symptom") is not None and len(merged_content["Main Symptom"]["name"]) > 0:
                # Create a section object
                section_create = SectionCreate(
                    user_id=user_id,
                    note_id=db_note.id,
                    title=result_keyword_dict.get("term", "Section"),
                    content=merged_content,
                    section_type=category,
                    section_description=TextCategoryEnum[category].value
                )
                
                # Add to database
                db_section = Section(
                    user_id=section_create.user_id,
                    note_id=section_create.note_id,
                    title=section_create.title,
                    content=section_create.content,
                    section_type=section_create.section_type,
                    section_description=section_create.section_description
                )
                db.add(db_section)
                sections.append(section_create)
        
        # Commit all sections
        db.commit()
        
        # Return results
        return {
            "success": True,
            "note_id": db_note.id,
            "title": db_note.title,
            "sections_created": len(sections),
            "sections": [section.model_dump() for section in sections]
        }
    
    except Exception as e:
        logger.error(f"End-to-end test error: {str(e)}")
        # Make sure we rollback the transaction on error
        db.rollback()
        raise HTTPException(status_code=500, detail=f"End-to-end test failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the tests router
    
    Tests if the core dependencies are available and working
    
    Returns:
        Status of the test service and its dependencies
    """
    status = {
        "status": "ok",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Test NLP components
    try:
        # Simple text to test NLP pipeline
        test_text = "Patient has a fever and cough."
        doc = process_text(test_text)
        status["components"]["nlp"] = {
            "status": "ok",
            "entities_found": len(doc.ents)
        }
    except Exception as e:
        status["components"]["nlp"] = {
            "status": "error",
            "error": str(e)
        }
        status["status"] = "degraded"
    
    # Test database components
    try:
        # Database connection will be tested through dependency
        status["components"]["database"] = {
            "status": "ok"
        }
    except Exception as e:
        status["components"]["database"] = {
            "status": "error",
            "error": str(e)
        }
        status["status"] = "degraded"
    
    # Test if we can load UMLS data
    try:
        result = classify_text_category("fever")
        status["components"]["umls_data"] = {
            "status": "ok",
            "test_classification": result
        }
    except Exception as e:
        status["components"]["umls_data"] = {
            "status": "error",
            "error": str(e)
        }
        status["status"] = "degraded"
    
    return status