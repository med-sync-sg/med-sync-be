# tests/unit/test_spacy_utils.py
import pytest
from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers

def test_process_text():
    # Simple test for the NLP pipeline
    test_text = "Patient has a headache and sore throat."
    doc = process_text(test_text)
    
    # Verify basic functionality
    assert doc is not None
    assert len(doc.ents) > 0
    
    # Check that it found medical entities
    medical_entities = [ent for ent in doc.ents if getattr(ent, "_.is_medical_term", False)]
    assert len(medical_entities) > 0
    
    # Verify text was correctly processed
    assert any(ent.text.lower() in ["headache", "sore throat"] for ent in medical_entities)

def test_find_medical_modifiers():
    # Test keyword extraction
    test_text = "Patient has had a severe headache for three days."
    doc = process_text(test_text)
    
    modifiers = find_medical_modifiers(doc)
    
    # Verify the structure
    assert isinstance(modifiers, list)
    assert len(modifiers) > 0
    
    # Find the headache term
    headache_modifier = next((mod for mod in modifiers if "headache" in mod.get("term", "").lower()), None)
    assert headache_modifier is not None
    
    # Check modifiers were extracted
    assert "severe" in headache_modifier.get("modifiers", [])
    assert any("three days" in q for q in headache_modifier.get("quantities", []))