from app.models.models import Base
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from os import environ
from sqlalchemy.pool import QueuePool

DB_USER = environ.get('DB_USER', "root")
DB_PASSWORD = environ.get('DB_PASSWORD', "medsync!")
DB_HOST = environ.get('DB_HOST', "localhost")
DB_PORT = environ.get('DB_PORT', "9000")
DB_NAME = environ.get('DB_NAME', "medsync_db")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_engine_with_proper_pooling():
    """Create SQLAlchemy engine with optimized connection pooling"""
    return create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,  # Adjust based on your workload
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True  # Check connection validity before using
    )

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.engine = create_engine_with_proper_pooling()
            cls._instance.session_factory = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=cls._instance.engine
            )
            # Initialize database if needed
            cls._instance._initialize_database()
        return cls._instance
    
    def _initialize_database(self):
        """Initialize database schema and tables"""
        # Only create tables once at startup
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session"""
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()