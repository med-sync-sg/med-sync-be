# tests/integration/test_endpoints.py
import pytest
from fastapi.testclient import TestClient
from app.app import app
import json

client = TestClient(app)

# def test_health_check():
#     response = client.get("/tests/health")
#     assert response.status_code == 200
#     assert response.json()["status"] == "healthy"

# def test_basic_text_processing():
#     test_text = "Patient reports fever and cough for two days."
#     response = client.post(
#         "/tests/basic-test",
#         json={"text": test_text}
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert data["success"] is True
#     assert data["entity_count"] > 0
    
#     # Verify it found medical entities
#     entities = data["entities"]
#     medical_terms = [ent["text"] for ent in entities]
#     assert any(term in ["fever", "cough"] for term in medical_terms)

def test_text_transcript_processing():
    sample_transcript = """
    Patient: Doctor, I've had a sore throat for three days.
    Doctor: Any other symptoms?
    Patient: Yes, I have a mild fever and headache.
    """
    
    response = client.post(
        "/tests/text-transcript",
        json={"transcript": sample_transcript}
    )
    assert response.status_code == 200
    sections = response.json()
    
    # Verify sections were created
    assert len(sections) > 0
    
    # Try to parse the sections
    parsed_sections = [json.loads(section) for section in sections]
    symptoms = []
    for section in parsed_sections:
        if "Main Symptom" in section and section["Main Symptom"].get("name"):
            symptoms.append(section["Main Symptom"]["name"].lower())
    
    # Verify expected symptoms were found
    expected_symptoms = ["sore throat", "fever", "headache"]
    for symptom in expected_symptoms:
        assert any(symptom in s for s in symptoms), f"Failed to find symptom: {symptom}"