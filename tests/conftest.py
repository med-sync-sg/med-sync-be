# tests/conftest.py
import pytest
import os
import sys
from app.db.local_session import DatabaseManager
from fastapi.testclient import TestClient
from app.app import app  # Adjust this import to match your app structure


@pytest.fixture
def client():
    """Test client for the FastAPI app"""
    return TestClient(app)

# Add fixtures for database dependency
@pytest.fixture
def db_session():
    """Creates a test database session"""
    # Create a test session with in-memory SQLite
    test_db = DatabaseManager()
    db = next(test_db.get_session())
    yield db
    db.close()

# Mock authentication fixture
@pytest.fixture
def auth_headers():
    """Returns mock authentication headers"""
    return {"Authorization": "Bearer test_token"}