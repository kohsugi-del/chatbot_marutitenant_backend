import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime
from database import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    client_id = Column(String(64), nullable=True, index=True)   # フロント設定ファイル識別子
    system_prompt = Column(Text, nullable=False)
    api_key = Column(String(64), unique=True, nullable=False, index=True)

    # クライアント設定（フロント config/clients/*.ts の代替）
    phone_normal = Column(String, nullable=True)
    phone_emergency = Column(String, nullable=True)
    business_hours = Column(String, nullable=True)
    emergency_keywords = Column(Text, nullable=True)   # JSON配列文字列
    topic_keywords = Column(Text, nullable=True)       # JSON配列文字列

    created_at = Column(DateTime(timezone=True), default=_now)
