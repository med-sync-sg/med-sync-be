from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db_app.session import DataStore, DRUGS_AND_MEDICINES_TUI, PATIENT_INFORMATION_TUI
from pyarrow import feather
import io
import pandas as pd
from fastapi.responses import StreamingResponse

'''
The db_app.py is run separately from app.py through db_main.py such that heavy and stationary DB data loading is offloaded to
a different process.
'''
def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    return app

# Instantiate your DataStore once
data_store = DataStore()
print(data_store.concepts_df.head())

# drugs_and_medicines_df = data_store.get_concepts_with_sty_def(DRUGS_AND_MEDICINES_TUI)
patient_information_df = data_store.get_concepts_with_sty_def(PATIENT_INFORMATION_TUI)
app = create_app()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialize_dataframe_to_feather(df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    feather.write_feather(df, buffer)
    buffer.seek(0)
    return buffer

@app.get("/umls-data/drugs")
def get_drugs():
    buffer = serialize_dataframe_to_feather(data_store.concepts_with_sty_def_df)
    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.get("/umls-data/symptoms-and-diseases")
def get_symptoms_and_diseases():
    buffer = serialize_dataframe_to_feather(data_store.concepts_with_sty_def_df)
    return StreamingResponse(buffer, media_type="application/octet-stream")

@app.get("/umls-data/patient-information")
def get_patient_information():
    buffer = serialize_dataframe_to_feather(patient_information_df)
    return StreamingResponse(buffer, media_type="application/octet-stream")