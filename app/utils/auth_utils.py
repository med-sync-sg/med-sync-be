from jose import jwt
from datetime import datetime, timedelta
import bcrypt
from os import environ

# JWT settings
SECRET_KEY = environ.get("JWT_SECRET_KEY")  # Replace with something strong from env
ALGORITHM = environ.get("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour

def hash_password(plain_password: str) -> str:
    # bcrypt.gensalt() automatically generates a salt
    hashed_bytes = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    # Convert bytes to a UTF-8 string for storage in your DB
    return hashed_bytes.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # hashed_password was stored as a UTF-8 string, so encode it back to bytes
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_bytes)


def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    # Raises jose.exceptions.JWTError if invalid
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload