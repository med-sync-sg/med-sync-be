# MedSync Backend

MedSync is a medical transcription and analysis system designed to help healthcare professionals efficiently document and process patient interactions. The backend provides a complete API for real-time audio transcription, medical entity extraction, note management, and report generation.

## Key Features

- Real-time speech-to-text transcription through WebSocket connection
- Medical entity recognition and extraction using NLP
- Secure authentication with JWT tokens
- Patient and doctor report generation with customizable templates
- Structured note and section management
- RESTful API for CRUD operations

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT with bcrypt password hashing
- **NLP Processing**: spaCy with custom medical entity components
- **Audio Processing**: Wav2Vec2 model for transcription
- **Template Engine**: Jinja2 for report generation

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Virtual environment tool (optional but recommended)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd medsync-backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Environment Variables:
   Create a `.env` file in the project root with the following variables:
   ```
   DB_USER=root
   DB_PASSWORD=medsync!
   DB_HOST=localhost
   DB_PORT=9000
   DB_NAME=medsync_db
   JWT_SECRET_KEY=your-secret-key
   JWT_ALGORITHM=HS256
   ```

2. Run database migrations:
   ```bash
   alembic upgrade head
   ```

### Running the Application

1. Start the application:
   ```bash
   python db_main.py
   python main.py
   ```

2. The API will be available at `http://127.0.0.1:8001`

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

## Project Structure

```
app/
├── api/                  # API routes and endpoints
│   └── v1/
│       └── endpoints/    # API endpoint implementations
├── db/                   # Database configuration and utilities
├── models/               # SQLAlchemy ORM models
├── schemas/              # Pydantic schemas for validation
├── services/             # Business logic services
├── utils/                # Utility functions and helpers
│   └── nlp/              # NLP processing utilities
└── app.py                # FastAPI application initialization

main.py                   # Application entry point
```

## Key Components

### Authentication

The system uses JWT tokens for authentication. Users can register and login through the `/auth/sign-up` and `/auth/login` endpoints.

```python
# Example authentication flow
POST /auth/sign-up  # Register a new user
POST /auth/login    # Login and receive JWT token
```

### Note Management

Notes are the primary data structure for storing patient interactions:

```python
# Example note operations
POST /notes/create  # Create a new note
GET /notes/{id}     # Get a specific note
GET /notes/         # List all notes
```

### Real-time Transcription

Real-time transcription is available through WebSocket connection:

```
WebSocket: /ws?token={jwt_token}&user_id={user_id}&note_id={note_id}
```

The client sends audio chunks, and the server responds with transcribed text and extracted medical entities.

### Report Generation

The system can generate both patient-friendly and doctor-focused reports from notes:

```python
# Example report generation
GET /reports/{note_id}/doctor   # Generate doctor report
GET /reports/{note_id}/patient  # Generate patient report
```

## Development Guidelines

### Adding New Endpoints

1. Create a new file in `app/api/v1/endpoints/` or add to an existing one
2. Define the route using FastAPI decorators
3. Register the router in `app/app.py`

### Database Changes

1. Create a new model in `app/models/`
2. Create corresponding Pydantic schemas in `app/schemas/`
3. Generate and run migrations:
   ```bash
   alembic revision --autogenerate -m "Description"
   alembic upgrade head
   ```

### NLP Pipeline Extensions

To extend the NLP capabilities:

1. Add new utilities in `app/utils/nlp/`
2. Register custom pipeline components in `app/utils/nlp/spacy_utils.py`
3. Update the keyword extraction in `app/services/nlp/keyword_extract_service.py`

## Testing

Run tests with pytest:

```bash
pytest
```

For API testing, use the test endpoints provided in `app/api/v1/endpoints/tests.py`.

## Contributing

1. Create a feature branch
2. Make changes and add tests
3. Submit a pull request

## License

[License details here]