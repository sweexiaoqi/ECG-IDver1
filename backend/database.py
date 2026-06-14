import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, TypeDecorator
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from backend.config import DATABASE_URL

# Setup SQLAlchemy Base and Engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Custom Type Decorator to store lists of floats as JSON text
class FloatListJSON(TypeDecorator):
    impl = Text

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                return json.loads(value)
            except Exception:
                return []
        return []

# Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    templates = relationship("EcgTemplate", back_populates="user", cascade="all, delete-orphan")
    replay_samples = relationship("ReplaySample", back_populates="user", cascade="all, delete-orphan")


class EcgTemplate(Base):
    __tablename__ = "ecg_templates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(FloatListJSON, nullable=False)  # 128-dim float array
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="templates")


class ReplaySample(Base):
    __tablename__ = "replay_samples"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    signal_data = Column(FloatListJSON, nullable=False)  # Segmented heartbeat window (e.g. 200 floats)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="replay_samples")


class AuthLog(Base):
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(20), nullable=False)  # 'AUTHENTICATION', 'REGISTRATION', 'FAILED_ATTEMPT', 'CALIBRATION'
    status = Column(String(20), nullable=False)      # 'AUTH_APPROVED', 'FAILED', 'VERIFICATION_ERROR', 'SUCCESS'
    username = Column(String(100), index=True)
    accuracy = Column(Float)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# Helper to initialize database
def init_db():
    Base.metadata.create_all(bind=engine)


# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
