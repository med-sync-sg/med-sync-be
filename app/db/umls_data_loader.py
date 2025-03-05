import json
import requests
from app.models.models import Base
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from os import environ
from typing import Dict
import pandas as pd
import pyarrow as pa
from pyarrow.feather import read_feather
from io import BytesIO
DB_USER = environ.get('DB_USER')
DB_PASSWORD = environ.get('DB_PASSWORD')
DB_HOST = environ.get('DB_HOST')
DB_PORT = environ.get('DB_PORT')
DB_NAME = environ.get('DB_NAME')

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATA_LOADER_URL = "http://127.0.0.1:8002"

print("Loading UMLS data...")

umls_df_dict: dict = {
    "combined_df": None,
    "concepts_with_sty_def_df": None,
    "patient_information_df": None,
}

combined = BytesIO(requests.get(f"{DATA_LOADER_URL}/umls-data/combined").content)
umls_df_dict["combined_df"] = read_feather(combined)
combined.close()

concepts_with_sty_def = BytesIO(requests.get(f"{DATA_LOADER_URL}/umls-data/symptoms-and-diseases").content)
umls_df_dict["concepts_with_sty_def_df"] = read_feather(concepts_with_sty_def)
concepts_with_sty_def.close()

patient_information = BytesIO(requests.get(f"{DATA_LOADER_URL}/umls-data/patient-information").content)
umls_df_dict["patient_information_df"] = read_feather(patient_information)
patient_information.close()

print("Loaded UMLS data!")

engine: Engine = create_engine(DATABASE_URL)
SessionMaker: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
