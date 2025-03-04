from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db_app.session import DataStore
from pyarrow import feather
import io
import pandas as pd
from fastapi.responses import StreamingResponse

def create_app() -> FastAPI:
    app = FastAPI(title="Backend Connection", version="1.0.0")
    return app

# Instantiate your DataStore once
data_store = DataStore()
print(data_store.concepts_df.head())

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


@app.get("/umls-data/combined")
def get_combined():
    buffer = serialize_dataframe_to_feather(data_store.combined_df)
    return StreamingResponse(buffer, media_type="application/octet-stream")

@app.get("/umls-data/concepts-with-sty-def-df")
def get_concepts_with_sty_def():
    buffer = serialize_dataframe_to_feather(data_store.concepts_with_sty_def_df)
    return StreamingResponse(buffer, media_type="application/octet-stream")