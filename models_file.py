# models_file.py
from sqlalchemy import Column, Integer, String, Text
from database import Base

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(36), nullable=True, index=True)
    filename = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
