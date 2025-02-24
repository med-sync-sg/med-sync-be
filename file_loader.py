import os
from app.utils.nlp.spacy_init import process_text, categorize_doc
from app.utils.nlp.summarizer import generate_summary
from app.utils.nlp.report_generator import generate_doctor_report
from spacy import displacy
import nltk
import datetime
TRANSCRIPT_PATH = "D:\medsync\primock57\\texts"
TAGGED_DOCS_PATH = "D:\medsync\primock57\\tagged"
nltk.download('punkt_tab')


example_data = {
    "report_date": datetime.date.today().strftime("%Y-%m-%d"),
    "patient_info": {
        "name": "John Doe",
        "age": 45,
        "gender": "Male"
    },
    "sections": [
        {
            "title": "Chief Complaint",
            "summary": "Patient reports severe headache for 3 days.",
            "evidence": [
                "Extracted sentence: 'I have had a severe headache for 3 days...'",
                "Identified entity: headache"
            ]
        },
        {
            "title": "Patient Information",
            "summary": "Patient is a 45-year-old teacher with a history of hypertension.",
            "evidence": [
                "Extracted details: '45-year-old', 'teacher', 'hypertension'"
            ]
        }
    ]
}

def get_transcript_text(file):
    full_text = []
    for line in file:
        full_text.append(line)
    return full_text

def load_test_transcripts():
    transcript_texts = []
    for filename in os.listdir(TRANSCRIPT_PATH):
        with open(os.path.join(TRANSCRIPT_PATH, filename)) as f:
            transcript_texts.append(get_transcript_text(f))
            
    return transcript_texts



def load():
    transcript_files = load_test_transcripts()
    file = transcript_files[0]
    text_list = get_transcript_text(file)
    full_text = ""
    for line in text_list:
        full_text = full_text + line
    tagged_doc = process_text(full_text)
    html = displacy.render(tagged_doc)
    categorized_texts = categorize_doc(tagged_doc)
    
    for topic, texts in categorized_texts.items():
        text = ""
        for line in texts:
            text = text + line
        
        print(topic, generate_summary(text))
    generate_doctor_report(data=example_data, is_doctor_report=True)
    generate_doctor_report(data=example_data, is_doctor_report=False)
            
load()