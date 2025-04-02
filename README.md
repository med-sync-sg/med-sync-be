# MedSync Backend API

A FastAPI-based backend for medical transcription and analysis, designed to process audio streams, transcribe them, and extract medical information.

## Project Structure

```
app/
├── api/
│   └── v1/
│       ├── endpoints/
│       │   ├── auth.py         # Authentication endpoints
│       │   ├── notes.py        # Note management endpoints
│       │   ├── reports.py      # Report generation endpoints
│       │   ├── tests.py        # Test endpoints for debugging
│       │   └── users.py        # User management endpoints
├── db/
│   ├── local_session.py        # PostgreSQL database connection
│   └── umls_data_loader.py     # UMLS medical terminology loader
├── models/
│   └── models.py               # SQLAlchemy ORM models
├── schemas/
│   ├── base.py                 # Base Pydantic schemas
│   ├── note.py                 # Note schemas
│   ├── section.py              # Section schemas
│   └── user.py                 # User schemas
├── utils/
│   ├── auth_utils.py           # Authentication utilities
│   ├── constants.py            # Constant definitions
│   ├── report_generator.py     # Report generation utilities
│   ├── speech_processor.py     # Audio processing and transcription
│   ├── text_utils.py           # Text processing utilities
│   ├── websocket_manager.py    # WebSocket connection management
│   └── nlp/
│       ├── keyword_extractor.py # Medical keyword extraction
│       ├── spacy_utils.py       # SpaCy NLP pipeline utilities
│       ├── summarizer.py        # Text summarization
│       └── report_templates/    # HTML templates for reports
├── app.py                      # FastAPI application setup
└── __init__.py                 # Package initialization
main.py                         # Application entry point
```

## Key Components

### Speech Processing

The speech processing pipeline includes:

- **AudioCollector**: A singleton class that accumulates audio chunks in real-time, detects silence (end of utterance), and triggers transcription.
- **SpeechProcessor**: Handles the actual audio-to-text transcription using the Wav2Vec2 model.

### NLP Pipeline

The NLP pipeline processes transcribed text to extract medical information:

- **SpaCy Utils**: Configure SpaCy with custom components for medical entity recognition
- **Keyword Extractor**: Extracts medical terms and their modifiers from the text
- **Text Utils**: Provides text cleaning and normalization utilities

### Data Storage

Two database systems are used:

- **PostgreSQL**: Relational database for user data, notes, and sections

### WebSocket API

Real-time communication for audio streaming:

- **WebSocketManager**: Handles WebSocket connections, validates users, and processes audio chunks
- **Authentication**: Token-based authentication for secure connections

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL
- UMLS API access

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   export DB_USER=your_postgres_user
   export DB_PASSWORD=your_postgres_password
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=medsync_db
   
   export JWT_SECRET_KEY=your_secret_key
   export JWT_ALGORITHM=HS256
   ```

### Running the API

Run the API with default settings:

```
python main.py
```

With custom host and port:

```
python main.py --host 0.0.0.0 --port 8080
```

Enable auto-reload during development:

```
python main.py --reload
```

## API Endpoints

### Authentication

- `POST /auth/login`: Log in with username and password
- `POST /auth/sign-up`: Create a new user account

### Notes

- `GET /notes`: Get all notes
- `GET /notes/{note_id}`: Get a specific note
- `POST /notes`: Create a new note
- `PUT /notes/{note_id}`: Update a note
- `DELETE /notes/{note_id}`: Delete a note

### Users

- `GET /users`: Get all users
- `GET /users/{user_id}`: Get a specific user
- `PUT /users/{user_id}`: Update a user
- `DELETE /users/{user_id}`: Delete a user

### Reports

- `GET /reports/{user_id}/{report_id}`: Get a specific report
- `POST /reports`: Create a new report

### WebSocket

- `/ws`: WebSocket endpoint for real-time audio streaming and transcription