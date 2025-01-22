import os
from app.utils.nlp import process_text
from spacy import displacy
TRANSCRIPT_PATH = "D:\medsync\primock57\\texts"
TAGGED_DOCS_PATH = "D:\medsync\primock57\\tagged"

def get_transcript_text(file):
    full_text = ""
    for line in file:
        full_text = full_text + line
    return full_text

def load_test_transcripts():
    transcript_texts = []
    for filename in os.listdir(TRANSCRIPT_PATH):
        with open(os.path.join(TRANSCRIPT_PATH, filename)) as f:
            transcript_texts.append(get_transcript_text(f))
            
    return transcript_texts



def load():
    transcript_files = load_test_transcripts()
    number = 1
    file = transcript_files[0]
    # for file in transcript_files:
    text = get_transcript_text(file)
    tagged_doc = process_text(text)
    html = displacy.render(tagged_doc)
    with open(os.path.join(TAGGED_DOCS_PATH + '_{number}'), "w") as f:
        f.write(html)
    f.close()
    # number = number + 1
        
        
load()