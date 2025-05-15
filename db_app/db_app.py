from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from db_app.session import DataStore, DRUGS_AND_MEDICINES_TUI, PATIENT_INFORMATION_TUI, SYMPTOMS_AND_DISEASES_TUI
from db_app.neo4j.neo4j import Neo4jInitializer

from pyarrow import feather
import io
import pandas as pd
import logging
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from graphql import graphql_sync
import json
import os
# Configure logger
logger = logging.getLogger(__name__)

'''
The db_app.py is run separately from app.py through db_main.py such that heavy and stationary DB data loading is offloaded to
a different process.
'''
def create_app() -> FastAPI:
    """Create and configure the FastAPI app for database operations"""
    app = FastAPI(
        title="MedSync Data Service", 
        description="Service for providing UMLS and vector database access",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# Create the connection
uri = os.environ.get("NEO4J_URI")
user = os.environ.get("NEO4J_USER")
password = os.environ.get("NEO4J_PASSWORD")
neo4j = Neo4jInitializer()
neo4j.initialize()
neo4j.test_vector_search()
neo4j_connection = neo4j.neo4j_connection
try:
    result = neo4j_connection.run_query("MATCH (n) RETURN count(n) as count")
    print(f"Connected to Neo4j! Node count: {result[0]['count']}")
except Exception as e:
    print(f"Failed to connect to Neo4j: {str(e)}")

# Create the app
app = create_app()

def get_db():
    db = data_store.SessionMaker()
    try:
        yield db
    finally:
        if db:
            db.close()

# Initialize DataStore singleton - logs will indicate initialization progress
try:
    logger.info("Initializing DataStore for API endpoints")
    data_store = DataStore()
    logger.info("DataStore initialization complete")
except Exception as e:
    logger.error(f"Error initializing DataStore: {str(e)}")
    # Continue anyway - the app will handle errors in the endpoints

def serialize_dataframe_to_feather(df: pd.DataFrame) -> io.BytesIO:
    """
    Serialize a pandas DataFrame to feather format in memory
    
    Args:
        df: DataFrame to serialize
        
    Returns:
        BytesIO buffer with serialized data
    """
    try:
        buffer = io.BytesIO()
        feather.write_feather(df, buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Error serializing DataFrame: {str(e)}")
        raise

@app.get("/")
def root():
    """Root endpoint with service info"""
    return {
        "service": "MedSync Data Service",
        "status": "running",
        "endpoints": [
            "/umls-data/drugs",
            "/umls-data/symptoms-and-diseases",
            "/status"
        ]
    }

@app.get("/status")
def get_status():
    """Get service status and information"""
    try:
        # Check if DataStore is initialized
        if not hasattr(data_store, '_initialized') or not data_store._initialized:
            return {
                "status": "initializing",
                "message": "DataStore is still initializing"
            }
        
        # Get database schema info
        table_status = {}
        # Return status info
        return {
            "status": "ready",
            "concepts_loaded": len(data_store.concepts_with_sty_def_df),
            "database_schema": data_store.schema_name,
            "tables": table_status
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/umls-data/drugs")
def get_drugs():
    """
    Get drugs and medicines data as a feather-serialized dataframe
    """
    try:
        # Filter the concepts dataframe to only include drugs/medicines
        drugs_df = data_store.concepts_with_sty_def_df[
            data_store.concepts_with_sty_def_df["TUI"].isin(DRUGS_AND_MEDICINES_TUI)
        ]
        
        if len(drugs_df) == 0:
            logger.warning("No drug data found")
            
        # Serialize and return
        buffer = serialize_dataframe_to_feather(drugs_df)
        return StreamingResponse(buffer, media_type="application/octet-stream")
    
    except Exception as e:
        logger.error(f"Error processing drugs request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing drugs data: {str(e)}"
        )

@app.get("/umls-data/symptoms-and-diseases")
def get_symptoms_and_diseases():
    """
    Get symptoms and diseases data as a feather-serialized dataframe
    """
    try:
        # Filter to symptoms and diseases
        symptoms_df = data_store.concepts_with_sty_def_df[
            data_store.concepts_with_sty_def_df["TUI"].isin(SYMPTOMS_AND_DISEASES_TUI)
        ]
        
        if len(symptoms_df) == 0:
            logger.warning("No symptoms/diseases data found")
            
        # Serialize and return
        buffer = serialize_dataframe_to_feather(symptoms_df)
        return StreamingResponse(buffer)
    
    except Exception as e:
        logger.error(f"Error processing symptoms request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing symptoms data: {str(e)}"
        )

@app.get("/umls-data/patient-information")
def get_patient_information():
    """
    Get patient information data as a feather-serialized dataframe
    """
    try:
        # Filter to patient information
        patient_info_df = data_store.concepts_with_sty_def_df[
            data_store.concepts_with_sty_def_df["TUI"].isin(PATIENT_INFORMATION_TUI)
        ]
        
        if len(patient_info_df) == 0:
            logger.warning("No patient information data found")
            
        # Serialize and return
        buffer = serialize_dataframe_to_feather(patient_info_df)
        return StreamingResponse(buffer, media_type="application/octet-stream")
    
    except Exception as e:
        logger.error(f"Error processing patient info request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing patient information data: {str(e)}"
        )