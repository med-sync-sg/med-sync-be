from pydantic import BaseModel

# A model class inherited by any requests that require authentication.
class BaseAuthModel(BaseModel):
    user_id: int