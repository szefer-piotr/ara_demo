"""Database models for ARA Demo API"""

from sqlalchemy import Column, String, Text, JSON, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

# Create Base class for declarative models
Base = declarative_base()


class Session(Base):
    """Session table for storing user sessions"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    data = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    files = relationship("File", back_populates="session", cascade="all, delete-orphan")
    hypotheses = relationship("Hypothesis", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Session(id={self.id}, session_id={self.session_id}, created_at={self.created_at})>"


class File(Base):
    """File table for storing uploaded files metadata"""
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    storage_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=True)
    file_type = Column(String(100), nullable=True)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="files")
    
    def __repr__(self):
        return f"<File(id={self.id}, file_id={self.file_id}, filename={self.filename})>"


class Hypothesis(Base):
    """Hypothesis table for storing research hypotheses"""
    __tablename__ = "hypotheses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    data = Column(JSON, nullable=True, default=dict)
    status = Column(String(50), nullable=False, default="draft")
    confidence_level = Column(Integer, nullable=True, default=50)  # 0-100
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="hypotheses")
    analysis_plans = relationship("AnalysisPlan", back_populates="hypothesis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Hypothesis(id={self.id}, hypothesis_id={self.hypothesis_id}, title={self.title})>"


class AnalysisPlan(Base):
    """Analysis plan table for storing generated analysis plans"""
    __tablename__ = "analysis_plans"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(255), unique=True, nullable=False, index=True)
    hypothesis_id = Column(String(255), ForeignKey("hypotheses.hypothesis_id"), nullable=False, index=True)
    plan_data = Column(JSON, nullable=False, default=dict)
    accepted = Column(Boolean, nullable=False, default=False)
    status = Column(String(50), nullable=False, default="draft")
    total_steps = Column(Integer, nullable=True, default=0)
    completed_steps = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    hypothesis = relationship("Hypothesis", back_populates="analysis_plans")
    
    def __repr__(self):
        return f"<AnalysisPlan(id={self.id}, plan_id={self.plan_id}, accepted={self.accepted})>"


class ApiKey(Base):
    """API key table for storing authentication keys"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(255), unique=True, nullable=False, index=True)
    key_hash = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    
    def __repr__(self):
        return f"<ApiKey(id={self.id}, key_id={self.key_id}, active={self.active})>"


class StepRun(Base):
    """Step run table for storing execution runs"""
    __tablename__ = "step_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), unique=True, nullable=False, index=True)
    plan_id = Column(String(255), ForeignKey("analysis_plans.plan_id"), nullable=False, index=True)
    step_id = Column(String(255), nullable=False, index=True)
    step_number = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    input_data = Column(JSON, nullable=True, default=dict)
    output_data = Column(JSON, nullable=True, default=dict)
    results = Column(JSON, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<StepRun(id={self.id}, run_id={self.run_id}, status={self.status})>"


class Report(Base):
    """Report table for storing generated reports"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id"), nullable=False, index=True)
    report_type = Column(String(50), nullable=False, default="final")
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    format = Column(String(50), nullable=False, default="html")
    file_path = Column(Text, nullable=True)
    citation_count = Column(Integer, nullable=False, default=0)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Report(id={self.id}, report_id={self.report_id}, format={self.format})>"
