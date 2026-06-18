import uuid
from pydantic import BaseModel, ConfigDict
from typing import Optional


class TenantCreate(BaseModel):
    name: str
    system_prompt: str
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    phone_normal: Optional[str] = None
    phone_emergency: Optional[str] = None
    business_hours: Optional[str] = None
    emergency_keywords: Optional[str] = None   # JSON文字列
    topic_keywords: Optional[str] = None       # JSON文字列


class TenantUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    client_id: Optional[str] = None
    phone_normal: Optional[str] = None
    phone_emergency: Optional[str] = None
    business_hours: Optional[str] = None
    emergency_keywords: Optional[str] = None
    topic_keywords: Optional[str] = None


class TenantResponse(BaseModel):
    id: str
    name: str
    client_id: Optional[str] = None
    system_prompt: str
    api_key: str
    phone_normal: Optional[str] = None
    phone_emergency: Optional[str] = None
    business_hours: Optional[str] = None
    emergency_keywords: Optional[str] = None
    topic_keywords: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
