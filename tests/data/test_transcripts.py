# tests/data/test_transcripts.py
"""Test data for transcript processing"""

SAMPLE_TRANSCRIPTS = [
    """
    Patient: I've been having a headache for three days now. It's mostly on the right side.
    Doctor: Is it constant or does it come and go?
    Patient: It comes and goes, but it's very intense when it's there.
    """,
    
    """
    Patient: I've had a sore throat and fever since yesterday.
    Doctor: Any cough or runny nose?
    Patient: Yes, I started coughing this morning and my nose is stuffy.
    """
]

EXPECTED_SYMPTOMS = [
    ["headache"],
    ["sore throat", "fever", "cough", "runny nose"]
]