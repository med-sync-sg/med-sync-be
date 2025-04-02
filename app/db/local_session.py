from app.models.models import Base
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from os import environ

DB_USER = environ.get('DB_USER', "root")
DB_PASSWORD = environ.get('DB_PASSWORD', "medsync!")
DB_HOST = environ.get('DB_HOST', "localhost")
DB_PORT = environ.get('DB_PORT', "9000")
DB_NAME = environ.get('DB_NAME', "medsync_db")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
def create_session() -> sessionmaker:
    engine: Engine = create_engine(DATABASE_URL)
    SessionMaker: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionMaker

'''
LocalDataStore should be used when communicating with the local database, currently hosted on docker with PostgreSQL.
'''
class LocalDataStore:
    _instance = None
    engine: Engine = create_engine(DATABASE_URL)
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    def __new__(cls):
        if cls._instance is None:
            print("Loading DataFrame...")
            with cls.engine.connect() as conn:
                Base.metadata.create_all(bind=cls.engine)

                # SQL DB loading
                cls._instance = super(LocalDataStore, cls).__new__(cls)
                print("Local session loading completed.")
        return cls._instance
    
    def get_db(self):
        db = self.SessionMaker()
        try:
            yield db
        finally:
            db.close()